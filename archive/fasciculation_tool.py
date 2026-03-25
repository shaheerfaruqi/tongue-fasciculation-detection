#!/usr/bin/env python3
"""
Fasciculation annotation & comparison tool
- Loads video via plain cv2.VideoCapture + CAP_PROP_POS_FRAMES seeking (like your first script)
- Robust key handling on Mac (waitKeyEx + wide arrow codes)
- We clamp at ends and the HUD always reflects the actual shown frame
- Includes compare utility (manual vs model detections)

Usage
-----
pip install opencv-python numpy pandas

Annotate:
  python fasciculation_tool.py annotate --video path/to/video.mp4 --out ann.csv
  # resume:
  python fasciculation_tool.py annotate --video path/to/video.mp4 --out ann.csv --load ann.csv

Compare:
  python fasciculation_tool.py compare --annotations ann.csv --detections model.csv \
      --fps 60 --tol_frames 2 --tol_pixels 15 --out results_base
"""

import argparse
import os
import sys
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd

# ============== Utilities ==============

def draw_hud_bar(frame_width, text_lines, y0=22):
    """Render a detached HUD bar (no video occlusion) and return it."""
    line_step = 26
    bar_h = max(int(y0 + line_step * len(text_lines) + 10), 30)
    bar = np.zeros((bar_h, frame_width, 3), dtype=np.uint8)
    cv2.rectangle(bar, (0, 0), (frame_width, bar_h), (0, 0, 0), -1)
    y = y0
    for line in text_lines:
        cv2.putText(bar, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
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

# ============== Annotator (loads like the first script) ==============

class Annotator:
    def __init__(self, video_path, out_csv, load_csv=None, point_radius=6, circle_thickness=2):
        self.video_path = video_path
        self.out_csv = out_csv

        # --- SIMPLE loader, like your first script ---
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 0:
            self.fps = 30.0  # delay timing only

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.total_frames or self.total_frames < 1:
            self.total_frames = 1

        self.curr_frame_idx = 0
        self.playing = False

        self.window = "Fasciculation Annotator"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        try:
            cv2.startWindowThread()
        except Exception:
            pass
        cv2.setMouseCallback(self.window, self.on_mouse)

        self.point_radius = point_radius
        self.circle_thickness = circle_thickness
        self.ann = load_annotations(load_csv)
        self.undo_stack = defaultdict(lambda: deque())  # per-frame undo
        self.last_save_time = time.time()
        self.last_frame = None  # keep last valid frame so UI stays up at EOF

        # Prime first frame (exactly as first script: hard-seek then read)
        frame = self._read_frame(self.curr_frame_idx)
        if frame is None:
            raise RuntimeError("Could not read initial frame.")
        self._render(frame)

    # --- frame IO identical in spirit to your first script ---
    def _read_frame(self, idx):
        idx = int(max(0, min(self.total_frames - 1, idx)))
        # Hard-seek to requested frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            # retry once (boundary quirk)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return None
        self.curr_frame_idx = idx
        self.last_frame = frame
        try:
            cv2.setWindowTitle(self.window, f"{self.window} — Frame {self.curr_frame_idx+1}/{self.total_frames}")
        except Exception:
            pass
        return frame

    # --- mouse & edit ops ---
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            p = {"x": int(x), "y": int(y), "note": ""}
            self.ann[self.curr_frame_idx].append(p)
            self.undo_stack[self.curr_frame_idx].append(("add", p))

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
        self.undo_stack[self.curr_frame_idx].append(("note", pts[-1], ""))

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

    # --- drawing ---
    def _draw_points(self, frame, points):
        for p in points:
            cv2.circle(frame, (p["x"], p["y"]), self.point_radius, (0, 255, 0), self.circle_thickness, cv2.LINE_AA)
            if p.get("note"):
                cv2.putText(frame, p["note"], (p["x"] + 8, p["y"] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    def _render(self, frame):
        base = frame.copy()
        pts = self.ann.get(self.curr_frame_idx, [])
        self._draw_points(base, pts)

        t = self.curr_frame_idx / self.fps if self.fps else 0.0
        lines = [
            f"Video: {os.path.basename(self.video_path)}",
            f"Frame {self.curr_frame_idx+1}/{self.total_frames} | Time {t:.3f}s | FPS {self.fps:.3f}",
            "Controls: [Space]=Play/Pause  [Left/Right or A/D]=Prev/Next frame  "
            "[ '[' or ',' ]=Back 10  [']' or '.']=Fwd 10  Click=Add point",
            "           [d]=Undo  [c]=Clear frame  [n]=Note last point  [s]=Save  [q]=Save+Quit",
            f"Points on this frame: {len(pts)}"
        ]

        # Stack HUD below the video to avoid covering the frame
        hud = draw_hud_bar(base.shape[1], lines)
        canvas = np.vstack([base, hud])
        cv2.imshow(self.window, canvas)

    # --- main loop ---
    def run(self):
        # Arrow variants across mac/Qt/Win/X11
        LEFT_KEYS  = {81, 2424832, 65361, 123}
        RIGHT_KEYS = {83, 2555904, 65363, 124}

        while True:
            # If playing, advance our index and clamp at end
            if self.playing:
                if self.curr_frame_idx < self.total_frames - 1:
                    self.curr_frame_idx += 1
                else:
                    self.playing = False
                frame = self._read_frame(self.curr_frame_idx)
                if frame is None:
                    # Pause on last valid frame instead of closing
                    self.playing = False
                    frame = self.last_frame
                if frame is not None:
                    self._render(frame)

            delay = int(1000 / self.fps) if self.playing else 1
            key = cv2.waitKeyEx(delay)
            key8 = (key & 0xFF) if key != -1 else -1

            if key in LEFT_KEYS or key8 == ord('a'):
                self.playing = False
                new_idx = max(self.curr_frame_idx - 1, 0)
                frame = self._read_frame(new_idx)
                if frame is None:
                    frame = self.last_frame
                if frame is not None:
                    self._render(frame)

            elif key in RIGHT_KEYS or key8 == ord('d'):
                self.playing = False
                new_idx = min(self.curr_frame_idx + 1, self.total_frames - 1)
                frame = self._read_frame(new_idx)
                if frame is None:
                    frame = self.last_frame
                if frame is not None:
                    self._render(frame)

            elif key8 in (ord('['), ord(',')):
                self.playing = False
                new_idx = max(self.curr_frame_idx - 10, 0)
                frame = self._read_frame(new_idx)
                if frame is None:
                    frame = self.last_frame
                if frame is not None:
                    self._render(frame)

            elif key8 in (ord(']'), ord('.')):
                self.playing = False
                new_idx = min(self.curr_frame_idx + 10, self.total_frames - 1)
                frame = self._read_frame(new_idx)
                if frame is None:
                    frame = self.last_frame
                if frame is not None:
                    self._render(frame)

            elif key8 == ord(' '):
                if self.curr_frame_idx < self.total_frames - 1:
                    self.playing = not self.playing
                else:
                    self.playing = False

            elif key8 == ord('q'):
                save_annotations(self.ann, self.video_path, self.fps, self.out_csv)
                print(f"Saved annotations to: {self.out_csv}")
                break

            elif key8 == ord('s'):
                save_annotations(self.ann, self.video_path, self.fps, self.out_csv)
                print(f"Saved annotations to: {self.out_csv}")

            elif key8 == ord('d'):
                self._undo()
                frame = self._read_frame(self.curr_frame_idx)
                if frame is None:
                    frame = self.last_frame
                if frame is not None:
                    self._render(frame)

            elif key8 == ord('c'):
                self._clear_frame()
                frame = self._read_frame(self.curr_frame_idx)
                if frame is None:
                    frame = self.last_frame
                if frame is not None:
                    self._render(frame)

            elif key8 == ord('n'):
                self._add_note_last_point()
                frame = self._read_frame(self.curr_frame_idx)
                if frame is None:
                    frame = self.last_frame
                if frame is not None:
                    self._render(frame)

            # periodic autosave
            if time.time() - self.last_save_time > 60:
                save_annotations(self.ann, self.video_path, self.fps, self.out_csv)
                self.last_save_time = time.time()

        self.cap.release()
        cv2.destroyAllWindows()

# ============== Comparison ==============

def _to_df(path):
    df = pd.read_csv(path)
    if "frame_idx" not in df.columns:
        raise ValueError(f"{path} must contain a 'frame_idx' column.")
    for col in ["x","y"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

def compare_annotations_with_detections(ann_csv, det_csv, fps, tol_frames=2, tol_pixels=15, out_csv=None):
    ann = _to_df(ann_csv).copy()
    det = _to_df(det_csv).copy()

    ann["matched"] = False
    det["matched"] = False
    matches = []

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
        for dfra in range(f - tol_frames, f + tol_frames + 1):
            for (di, dr) in det_by_frame.get(dfra, []):
                if det.at[di, "matched"]:
                    continue
                dxy = dist(ar, dr)
                if dxy is None or dxy <= tol_pixels:
                    candidate_rows.append((di, dr, abs(dfra - f), (dxy if dxy is not None else 0.0)))

        if candidate_rows:
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
        report.to_csv(f"{base}_summary.csv", index=False)
        matches_df.to_csv(f"{base}_matches.csv", index=False)
        print(f"Saved {base}_summary.csv and {base}_matches.csv")

    return report, matches_df

# ============== CLI ==============

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
    p_cmp.add_argument("--fps", type=float, required=True, help="Video FPS for time conversion")
    p_cmp.add_argument("--tol_frames", type=int, default=2, help="Temporal tolerance in frames")
    p_cmp.add_argument("--tol_pixels", type=int, default=15, help="Spatial tolerance in pixels")
    p_cmp.add_argument("--out", default=None, help="Base path for output CSVs (writes *_summary.csv & *_matches.csv)")

    args = parser.parse_args()

    if args.cmd == "annotate":
        tool = Annotator(args.video, args.out, args.load)
        try:
            tool.run()
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
