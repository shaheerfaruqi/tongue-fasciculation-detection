#!/usr/bin/env python3
"""
Compare manual vs automated fasciculation CSV exports.

Example:
  python compare_excel_files.py \
      --manual manual_marks.csv --auto automated.csv \
      --fps 60 --tol-frames 10 --tol-pixels 15 --out-base manual_vs_auto

Outputs two CSVs: <out_base>_summary.csv and <out_base>_matches.csv.
The tool tries to auto-detect columns named like frame_idx/frame, x, y.
For automated CSVs with a panel/source column, it keeps only "denoised" rows by default.
If your headers differ, pass explicit overrides (e.g. --manual-frame-col Frame #).
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# Column name fallbacks (matched case-insensitively)
FRAME_CANDS = {"frame_idx", "frame", "frame index", "frame number", "frame #"}
TIME_CANDS = {"time", "time_sec", "time (s)", "seconds", "sec"}
X_CANDS = {"x", "x_px", "x coordinate", "xcoord", "x-coord"}
Y_CANDS = {"y", "y_px", "y coordinate", "ycoord", "y-coord"}
PANEL_CANDS = {"panel", "source", "stream"}


def _read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext != ".csv":
        raise ValueError(f"Expected a CSV file, got {path}")
    return pd.read_csv(path)


def _resolve_column(df: pd.DataFrame, override: Optional[str], candidates) -> Optional[str]:
    cols_lower = {str(c).lower(): c for c in df.columns}
    if override:
        key = override if override in df.columns else override.lower()
        if key in cols_lower:
            return cols_lower[key]
        raise ValueError(f"Column '{override}' not found. Available: {list(df.columns)}")
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def _prepare_points(
    path: Path,
    role: str,
    fps: Optional[float],
    frame_override: Optional[str],
    time_override: Optional[str],
    x_override: Optional[str],
    y_override: Optional[str],
    panel_override: Optional[str] = None,
    panel_value: Optional[str] = None,
) -> pd.DataFrame:
    df = _read_table(path)
    if df.empty:
        raise ValueError(f"{role}: {path} is empty.")

    if role.lower() == "auto" and panel_value and panel_value.strip().lower() != "all":
        panel_col = _resolve_column(df, panel_override, PANEL_CANDS)
        if panel_col is None:
            print(f"Warning: auto panel column not found in {path}; using all automated rows.")
        else:
            keep_value = panel_value.strip().lower()
            panel_vals = df[panel_col].astype(str).str.strip().str.lower()
            df = df.loc[panel_vals == keep_value].copy()
            if df.empty:
                raise ValueError(
                    f"auto: no rows found with {panel_col}='{panel_value}' in {path}."
                )

    frame_col = _resolve_column(df, frame_override, FRAME_CANDS)
    time_col = _resolve_column(df, time_override, TIME_CANDS)
    x_col = _resolve_column(df, x_override, X_CANDS)
    y_col = _resolve_column(df, y_override, Y_CANDS)

    if frame_col is None:
        if time_col is None:
            raise ValueError(f"{role}: need a frame column or time column (with --fps).")
        if fps is None or fps <= 0:
            raise ValueError(f"{role}: FPS is required when using a time column.")
        df["frame_idx"] = np.rint(df[time_col].astype(float) * fps).astype(int)
    else:
        df["frame_idx"] = df[frame_col].astype(float).round().astype(int)

    if x_col is None or y_col is None:
        raise ValueError(f"{role}: could not find both x and y columns.")

    points = df[["frame_idx", x_col, y_col]].copy()
    points.columns = ["frame_idx", "x", "y"]
    return points


def _collapse_nearby_points(df: pd.DataFrame, tol_frames: int, tol_pixels: int) -> pd.DataFrame:
    """Merge near-duplicate detections into event-level points."""
    if df.empty:
        return df.copy()

    work = df[["frame_idx", "x", "y"]].copy().reset_index(drop=True)
    work["frame_idx"] = work["frame_idx"].astype(int)
    work = work.sort_values("frame_idx", kind="mergesort").reset_index(drop=True)

    n = len(work)
    parent = np.arange(n, dtype=int)
    rank = np.zeros(n, dtype=int)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    frames = work["frame_idx"].to_numpy()
    xs = work["x"].to_numpy(dtype=float)
    ys = work["y"].to_numpy(dtype=float)

    for i in range(n):
        j = i + 1
        while j < n and (frames[j] - frames[i]) <= tol_frames:
            if np.isfinite(xs[i]) and np.isfinite(ys[i]) and np.isfinite(xs[j]) and np.isfinite(ys[j]):
                if np.hypot(xs[i] - xs[j], ys[i] - ys[j]) <= tol_pixels:
                    union(i, j)
            j += 1

    roots = np.array([find(i) for i in range(n)], dtype=int)
    grouped = work.assign(_root=roots).groupby("_root", sort=False, as_index=False)
    collapsed = grouped.agg(
        frame_idx=("frame_idx", lambda s: int(np.rint(np.median(s.to_numpy(dtype=float))))),
        x=("x", "mean"),
        y=("y", "mean"),
        merged_count=("frame_idx", "size"),
    )
    return collapsed[["frame_idx", "x", "y", "merged_count"]]


def compare_points(
    manual_df: pd.DataFrame,
    auto_df: pd.DataFrame,
    tol_frames: int,
    tol_pixels: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ann = manual_df.copy()
    det = auto_df.copy()

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
                    candidate_rows.append((di, dr, abs(dfra - f), dxy if dxy is not None else 0.0))

        if candidate_rows:
            candidate_rows.sort(key=lambda t: (t[2], t[3]))
            di, dr, _, _ = candidate_rows[0]
            ann.at[ai, "matched"] = True
            det.at[di, "matched"] = True
            matches.append(
                {
                    "ann_frame": int(ar.frame_idx),
                    "det_frame": int(dr.frame_idx),
                    "ann_x": ar.x,
                    "ann_y": ar.y,
                    "det_x": dr.x,
                    "det_y": dr.y,
                    "frame_gap": int(abs(dr.frame_idx - ar.frame_idx)),
                    "pixel_dist": dist(ar, dr) if dist(ar, dr) is not None else np.nan,
                }
            )

    tp = int(ann["matched"].sum())
    fn = int((~ann["matched"]).sum())
    fp = int((~det["matched"]).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    report = pd.DataFrame(
        [
            {"metric": "TP", "value": tp},
            {"metric": "FP", "value": fp},
            {"metric": "FN", "value": fn},
            {"metric": "precision", "value": prec},
            {"metric": "recall", "value": rec},
            {"metric": "f1", "value": f1},
        ]
    )
    matches_df = pd.DataFrame(
        matches,
        columns=["ann_frame", "det_frame", "ann_x", "ann_y", "det_x", "det_y", "frame_gap", "pixel_dist"],
    )
    return report, matches_df


def main():
    parser = argparse.ArgumentParser(description="Compare manual and automated fasciculation CSV files.")
    parser.add_argument("--manual", required=True, type=Path, help="Manual markings CSV path")
    parser.add_argument("--auto", required=True, type=Path, help="Automated detections CSV path")
    parser.add_argument("--fps", type=float, required=True, help="Video FPS (needed to convert time to frame)")
    parser.add_argument("--tol-frames", type=int, default=10, help="Temporal tolerance in frames")
    parser.add_argument("--tol-pixels", type=int, default=15, help="Spatial tolerance in pixels")
    parser.add_argument("--out-base", default="comparison", help="Base name for output CSVs")
    parser.add_argument("--manual-frame-col", default=None, help="Override manual frame column header")
    parser.add_argument("--manual-time-col", default=None, help="Override manual time column header")
    parser.add_argument("--manual-x-col", default=None, help="Override manual x column header")
    parser.add_argument("--manual-y-col", default=None, help="Override manual y column header")
    parser.add_argument("--auto-frame-col", default=None, help="Override auto frame column header")
    parser.add_argument("--auto-time-col", default=None, help="Override auto time column header")
    parser.add_argument("--auto-x-col", default=None, help="Override auto x column header")
    parser.add_argument("--auto-y-col", default=None, help="Override auto y column header")
    parser.add_argument("--auto-panel-col", default=None, help="Override auto panel/source column header")
    parser.add_argument(
        "--auto-panel-value",
        default="denoised",
        help="Auto panel value to keep (default: denoised). Use 'all' to disable filtering.",
    )
    parser.add_argument(
        "--no-merge-auto-nearby",
        action="store_true",
        help="Disable collapsing near-duplicate automated detections before scoring.",
    )

    args = parser.parse_args()

    manual_df = _prepare_points(
        args.manual,
        role="manual",
        fps=args.fps,
        frame_override=args.manual_frame_col,
        time_override=args.manual_time_col,
        x_override=args.manual_x_col,
        y_override=args.manual_y_col,
    )
    auto_df = _prepare_points(
        args.auto,
        role="auto",
        fps=args.fps,
        frame_override=args.auto_frame_col,
        time_override=args.auto_time_col,
        x_override=args.auto_x_col,
        y_override=args.auto_y_col,
        panel_override=args.auto_panel_col,
        panel_value=args.auto_panel_value,
    )
    if not args.no_merge_auto_nearby:
        before_n = len(auto_df)
        auto_df = _collapse_nearby_points(auto_df, tol_frames=args.tol_frames, tol_pixels=args.tol_pixels)
        after_n = len(auto_df)
        print(
            f"Auto dedupe: collapsed {before_n} detections into {after_n} event(s) "
            f"using {args.tol_frames} frame / {args.tol_pixels}px thresholds."
        )
        auto_df = auto_df[["frame_idx", "x", "y"]]

    report, matches = compare_points(
        manual_df,
        auto_df,
        tol_frames=args.tol_frames,
        tol_pixels=args.tol_pixels,
    )

    summary_path = f"{args.out_base}_summary.csv"
    matches_path = f"{args.out_base}_matches.csv"
    report.to_csv(summary_path, index=False)
    matches.to_csv(matches_path, index=False)

    print("\n=== Summary ===")
    print(report.to_string(index=False))
    if not matches.empty:
        print(f"\nSaved {summary_path} and {matches_path}. First few matches:")
        print(matches.head().to_string(index=False))
    else:
        print(f"\nSaved {summary_path}. No matches found; {matches_path} is empty.")


if __name__ == "__main__":
    main()
