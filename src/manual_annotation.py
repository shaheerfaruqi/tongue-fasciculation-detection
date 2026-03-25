
"""
Fasciculation annotation & comparison tool (mac-friendly)

Usage:
  # Install deps:
  #   pip install opencv-python numpy pandas

  # Annotate a video
  # python manualAnnotation.py annotate --video tongueFasciculationsResized.mp4 --out ann.csv
  #   (Optional resume) --load ann.csv

  # Compare manual annotations vs model detections
  #   python fasciculation_tool.py compare \
  #       --annotations ann.csv --detections model.csv \
  #       --fps 60 --tol_frames 2 --tol_pixels 15 \
  #       --out var_vs_manual
  #
  #   Writes var_vs_manual_summary.csv and var_vs_manual_matches.csv
"""

import argparse
import os
import sys
import cv2
import time
import numpy as np
import pandas as pd
from collections import defaultdict, deque

# ---------- Utilities ----------

def draw_hud_bar(frame_width, text_lines, y0=22):
    """Create a separate HUD bar image (not drawn on the video frame)."""
    line_step = 26
    bar_h = max(y0 + line_step * len(text_lines) + 10, 30)
    bar = np.zeros((bar_h, frame_width, 3), dtype=np.uint8)
    y = y0
    for line in text_lines:
        cv2.putText(bar, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        y += line_step
    return bar

def load_annotations(csv_path):
    if (csv_path is None) or (not os.path.exists(csv_path)):
        return defaultdict(list)
    df = pd.read_csv(csv_path)
    ann = defaultdict(list)
    for _, r in df.iterrows():
        ann[int(r["frame_idx"])].append({
            "x": int(r["x"]),
            "y": int(r["y"]),
            "note": str(r.get("note", "")),
        })
    return ann

def save_annotations(ann_dict, video_path, fps, out_csv):
    rows = []
    for fidx, pts in ann_dict.items():
        t = fidx / fps if fps and fps > 0 else 0.0
        for p in pts:
            rows.append({
                "video_path": video_path,
                "frame_idx": fidx,
                "time_sec": round(t, 6),
                "x": p["x"],
                "y": p["y"],
                "note": p.get("note", "")
            })
    df = pd.DataFrame(rows, columns=["video_path","frame_idx","time_sec","x","y","note"])
    if not df.empty:
        df.sort_values(["frame_idx","time_sec"], inplace=True)
    df.to_csv(out_csv, index=False)
    return out_csv

# ---------- Annotate mode (mac-robust) ----------

class Annotator:
    def __init__(self, video_path, out_csv, load_csv=None, point_radius=6, circle_thickness=2):
        self.video_path = video_path
        self.out_csv = out_csv

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        # FPS for timing only (we manage frame index ourselves)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 0:
            self.fps = 30.0  # sensible default for delay timing

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.total_frames or self.total_frames < 1:
            # Some mac/AVFoundation builds struggle to report frame count;
            # we still work because we hard-seek; but keep >=1 to avoid negatives.
            self.total_frames = max(1, self.total_frames)

        self.curr_frame_idx = 0
        self.frame_height = 0  # set after first frame read, used for mouse filtering
        self.playing = False

        self.window = "Fasciculation Annotator"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        try:
            cv2.startWindowThread()  # improves event handling on some mac builds
        except Exception:
            pass
        cv2.setMouseCallback(self.window, self.on_mouse)

        self.point_radius = point_radius
        self.circle_thickness = circle_thickness
        self.ann = load_annotations(load_csv)
        self.undo_stack = defaultdict(lambda: deque())  # per-frame undo

        # Preload first frame to initialize UI
        self.last_frame = None  # keep last valid frame for boundary recovery
        f = self._read_frame(self.curr_frame_idx)
        if f is not None:
            self.last_frame = f

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Ignore clicks on the HUD bar below the video
            if self.frame_height > 0 and y >= self.frame_height:
                return
            p = {"x": int(x), "y": int(y), "note": ""}
            self.ann[self.curr_frame_idx].append(p)
            self.undo_stack[self.curr_frame_idx].append(("add", p))

    def _read_frame(self, idx):
        # Clamp and hard-seek every time so our counter is the source of truth
        idx = int(max(0, min(self.total_frames - 1, idx)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            # Retry once (boundary quirk)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return None
        self.curr_frame_idx = idx
        self.frame_height = frame.shape[0]
        self.last_frame = frame  # always save last good frame
        # Optional: show index in window title
        try:
            cv2.setWindowTitle(self.window, f"{self.window} — Frame {self.curr_frame_idx+1}/{self.total_frames}")
        except Exception:
            pass
        return frame

    def _draw_points(self, frame, points):
        for p in points:
            cv2.circle(frame, (p["x"], p["y"]), self.point_radius, (0, 255, 0), self.circle_thickness, cv2.LINE_AA)
            if p.get("note"):
                cv2.putText(frame, p["note"], (p["x"] + 8, p["y"] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    def _hud(self, frame):
        """Build HUD bar and stack it below the video frame."""
        t = self.curr_frame_idx / self.fps if self.fps else 0.0
        lines = [
            f"Video: {os.path.basename(self.video_path)}",
            f"Frame {self.curr_frame_idx+1}/{self.total_frames} | Time {t:.3f}s | FPS {self.fps:.3f}",
            "Controls: [Space]=Play/Pause  [<-/->]=Step  [,/.]=+-10  Click=Add point",
            "           [d]=Undo  [c]=Clear frame  [n]=Note last point  [s]=Save  [q]=Save+Quit",
            f"Points on this frame: {len(self.ann.get(self.curr_frame_idx, []))}"
        ]
        hud_bar = draw_hud_bar(frame.shape[1], lines)
        return np.vstack([frame, hud_bar])

    def _add_note_last_point(self):
        pts = self.ann.get(self.curr_frame_idx, [])
        if not pts:
            print("No points to annotate on this frame.")
            return
        try:
            note = input("Note (for last point on current frame): ").strip()
        except EOFError:
            note = ""
        pts[-1]["note"] = note
        self.undo_stack[self.curr_frame_idx].append(("note", pts[-1], ""))  # minimal tracking

    def _undo(self):
        stk = self.undo_stack[self.curr_frame_idx]
        if not stk:
            return
        op = stk.pop()
        if op[0] == "add":
            p = op[1]
            pts = self.ann.get(self.curr_frame_idx, [])
            if pts and pts[-1] is p:
                pts.pop()

    def _clear_frame(self):
        if self.ann.get(self.curr_frame_idx):
            self.ann[self.curr_frame_idx] = []
            self.undo_stack[self.curr_frame_idx].clear()

    def run(self):
        last_save_time = time.time()

        # Common arrow codes across mac/Qt/X11/Windows
        LEFT_KEYS  = {81, 2424832, 65361, 123}
        RIGHT_KEYS = {83, 2555904, 65363, 124}

        while True:
            # Advance our own pointer during play
            if self.playing:
                if self.curr_frame_idx < self.total_frames - 1:
                    self.curr_frame_idx += 1
                else:
                    self.playing = False  # pause at last frame

            frame = self._read_frame(self.curr_frame_idx)
            if frame is None:
                # Boundary read failure — use last valid frame instead of exiting
                if self.last_frame is not None:
                    frame = self.last_frame.copy()
                    self.playing = False
                    # Shrink total_frames so we don't keep hitting the bad frame
                    if self.curr_frame_idx > 0:
                        self.total_frames = self.curr_frame_idx
                        self.curr_frame_idx = self.total_frames - 1
                else:
                    break  # truly no frames available

            # Draw current frame annotations + HUD
            pts = self.ann.get(self.curr_frame_idx, [])
            self._draw_points(frame, pts)
            canvas = self._hud(frame)

            cv2.imshow(self.window, canvas)
            delay = int(1000 / self.fps) if (self.playing and self.fps > 0) else 0

            key = cv2.waitKeyEx(delay)
            key8 = (key & 0xFF) if key != -1 else -1

            # --- handle keys ---
            if key in LEFT_KEYS:
                self.playing = False
                self.curr_frame_idx = max(self.curr_frame_idx - 1, 0)

            elif key in RIGHT_KEYS:
                self.playing = False
                self.curr_frame_idx = min(self.curr_frame_idx + 1, self.total_frames - 1)

            elif key8 == ord(','):
                self.playing = False
                self.curr_frame_idx = max(self.curr_frame_idx - 10, 0)

            elif key8 == ord('.'):
                self.playing = False
                self.curr_frame_idx = min(self.curr_frame_idx + 10, self.total_frames - 1)

            elif key8 == ord(' '):  # toggle play/pause
                self.playing = not self.playing

            elif key8 == ord('q'):
                save_annotations(self.ann, self.video_path, self.fps, self.out_csv)
                print(f"Saved annotations to: {self.out_csv}")
                break

            elif key8 == ord('s'):
                save_annotations(self.ann, self.video_path, self.fps, self.out_csv)
                print(f"Saved annotations to: {self.out_csv}")

            elif key8 == ord('d'):
                self._undo()

            elif key8 == ord('c'):
                self._clear_frame()

            elif key8 == ord('n'):
                self._add_note_last_point()

            # Autosave every ~60s
            if time.time() - last_save_time > 60:
                save_annotations(self.ann, self.video_path, self.fps, self.out_csv)
                last_save_time = time.time()

        self.cap.release()
        cv2.destroyAllWindows()

# ---------- Comparison mode ----------

def _to_df(path):
    df = pd.read_csv(path)
    if "frame_idx" not in df.columns:
        raise ValueError(f"{path} must contain a 'frame_idx' column.")
    for col in ["x","y"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

def compare_annotations_with_detections(ann_csv, det_csv, fps, tol_frames=2, tol_pixels=15, out_csv=None):
    """
    Greedy one-to-one matching:
      - A detection matches a manual point if |frame_ann - frame_det| <= tol_frames
        AND (if x,y exist for both) Euclidean distance <= tol_pixels.
      - If model lacks x,y, we do frame-only matching.
    """
    ann = _to_df(ann_csv).copy()
    det = _to_df(det_csv).copy()

    ann["matched"] = False
    det["matched"] = False
    matches = []

    # Index detections by frame for speed
    det_by_frame = defaultdict(list)
    for i, r in det.iterrows():
        det_by_frame[int(r.frame_idx)].append((i, r))

    def dist(a, b):
        if np.isnan(a.x) or np.isnan(a.y) or np.isnan(b.x) or np.isnan(b.y):
            return None
        return float(np.hypot(a.x - b.x, a.y - b.y))

    for ai, ar in ann.iterrows():
        f = int(ar.frame_idx)
        candidate_rows = []
        # collect within frame tolerance
        for dfra in range(f - tol_frames, f + tol_frames + 1):
            for (di, dr) in det_by_frame.get(dfra, []):
                if det.at[di, "matched"]:
                    continue
                dxy = dist(ar, dr)
                # if either side lacks x,y -> allow match without spatial constraint
                if dxy is None or dxy <= tol_pixels:
                    candidate_rows.append((di, dr, abs(dfra - f), (dxy if dxy is not None else 0.0)))

        if candidate_rows:
            # best: min frame gap, then min distance
            candidate_rows.sort(key=lambda t: (t[2], t[3]))
            di, dr, _, _ = candidate_rows[0]
            ann.at[ai, "matched"] = True
            det.at[di, "matched"] = True
            matches.append({
                "ann_frame": int(ar.frame_idx),
                "det_frame": int(dr.frame_idx),
                "ann_x": ar.x, "ann_y": ar.y,
                "det_x": dr.x, "det_y": dr.y,
                "frame_gap": int(abs(dr.frame_idx - ar.frame_idx)),
                "pixel_dist": (dist(ar, dr) if dist(ar, dr) is not None else np.nan)
            })

    tp = int(ann["matched"].sum())
    fn = int((~ann["matched"]).sum())
    fp = int((~det["matched"]).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0

    report_rows = [
        {"metric":"TP","value":tp},
        {"metric":"FP","value":fp},
        {"metric":"FN","value":fn},
        {"metric":"precision","value":prec},
        {"metric":"recall","value":rec},
        {"metric":"f1","value":f1},
    ]
    report = pd.DataFrame(report_rows)

    matches_df = pd.DataFrame(matches, columns=[
        "ann_frame","det_frame","ann_x","ann_y","det_x","det_y","frame_gap","pixel_dist"
    ])

    if out_csv:
        base, _ = os.path.splitext(out_csv)
        summary_path = f"{base}_summary.csv"
        matches_path = f"{base}_matches.csv"
        report.to_csv(summary_path, index=False)
        matches_df.to_csv(matches_path, index=False)
        print(f"Saved summary: {summary_path}")
        print(f"Saved matches: {matches_path}")

    return report, matches_df

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Fasciculation annotation & comparison tool")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ann = sub.add_parser("annotate", help="Annotate fasciculations in a video")
    p_ann.add_argument("--video", required=True, help="Path to MP4 video")
    p_ann.add_argument("--out", required=True, help="Output annotations CSV")
    p_ann.add_argument("--load", default=None, help="Optional: load existing annotations CSV to resume")

    p_cmp = sub.add_parser("compare", help="Compare annotations vs model detections")
    p_cmp.add_argument("--annotations", required=True, help="Manual annotations CSV")
    p_cmp.add_argument("--detections", required=True, help="Model detections CSV")
    p_cmp.add_argument("--fps", type=float, required=True, help="Video FPS used to derive time from frames")
    p_cmp.add_argument("--tol_frames", type=int, default=2, help="Temporal tolerance in frames (default 2)")
    p_cmp.add_argument("--tol_pixels", type=int, default=15, help="Spatial tolerance in pixels (default 15)")
    p_cmp.add_argument("--out", default=None, help="Base path for output CSVs (will write *_summary.csv and *_matches.csv)")

    args = parser.parse_args()

    if args.cmd == "annotate":
        tool = Annotator(args.video, args.out, args.load)
        try:
            tool.run()
            # Always save on exit (just in case)
            save_annotations(tool.ann, tool.video_path, tool.fps, tool.out_csv)
        except KeyboardInterrupt:
            save_annotations(tool.ann, tool.video_path, tool.fps, tool.out_csv)
        except Exception as e:
            print("Error:", e)
            sys.exit(1)

    elif args.cmd == "compare":
        try:
            report, matches = compare_annotations_with_detections(
                args.annotations, args.detections, args.fps,
                tol_frames=args.tol_frames, tol_pixels=args.tol_pixels,
                out_csv=args.out
            )
            print("\n=== Summary ===")
            print(report.to_string(index=False))
            if not matches.empty:
                print("\nFirst few matches:")
                print(matches.head().to_string(index=False))
        except Exception as e:
            print("Error:", e)
            sys.exit(1)

if __name__ == "__main__":
    main()
