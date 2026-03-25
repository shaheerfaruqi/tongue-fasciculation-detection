#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ======================
# Default config (can be overridden by CLI)
# ======================

# Visualization (does not affect detection sensitivity)
STEP = 10                  # grid spacing for arrows
SCALE = 3                  # arrow length multiplier
ARROW_COLOR = (0, 255, 0)  # BGR

# ---- Detection thresholds (you will tune these) ----
MIN_BLOB_AREA = 7          # px^2; ignore tiny noise (raise to be stricter)
MAX_BLOB_FRAC = 0.05       # reject blobs >5% of frame area (lower to be stricter)
THR_ORIG = 140             # binarization threshold for "original" panel (lower = more sensitive)
THR_DENO = 140             # binarization threshold for "denoised" panel (lower = more sensitive)
INCOHERENCE_MIN = 0.11     # circular variance cutoff; increase to be stricter about incoherence
# Region-specific tuning (central 3/5 of bottom third gets stricter thresholds)
ROI_THR_ORIG = 180         # stricter binarization for ROI (orig panel)
ROI_THR_DENO = 175         # stricter binarization for ROI (denoised panel)
ROI_MIN_BLOB_AREA = 7      # slightly larger min area in ROI
BORDER_FRAC = 0.10         # ignore detections within 10% of frame border

# Denoising strength (affects sensitivity: larger blur = fewer small detections)
MEDIAN_BLUR_KSIZE = 3      # must be odd; try 3,5,7

# Farnebäck parameters (affect flow magnitude/noise)
FARNE_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=20,           # increase for smoother/stronger flow, but slower
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)


# ======================
# Helpers
# ======================

def contour_direction_incoherence(cnt, angle_map):
    """
    Returns circular variance (0..1). Higher => more incoherent directions.
    """
    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 0 or h <= 0:
        return 0.0
    patch = angle_map[y:y+h, x:x+w]
    if patch.size == 0:
        return 0.0
    ux = np.cos(patch)
    uy = np.sin(patch)
    mean_vec = np.array([ux.mean(), uy.mean()], dtype=np.float64)
    mean_len = np.linalg.norm(mean_vec) + 1e-8
    circ_var = 1.0 - mean_len  # 0=coherent, 1=incoherent
    return float(circ_var)


def contour_centroid(cnt):
    """Return (cx, cy) for a contour using image moments; falls back to bbox center."""
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w / 2.0
        cy = y + h / 2.0
    return float(cx), float(cy)


# ======================
# Core detection function
# ======================

def run_variance_based_detection(
    video_path: Path,
    out_csv: Path,
    out_video_combined: Path,
    out_video_orig: Path,
    out_video_deno: Path,
    out_video_flow: Path,
    step: int = STEP,
    scale: int = SCALE,
    arrow_color=ARROW_COLOR,
    min_blob_area: float = MIN_BLOB_AREA,
    max_blob_frac: float = MAX_BLOB_FRAC,
    thr_orig: int = THR_ORIG,
    thr_deno: int = THR_DENO,
    incoherence_min: float = INCOHERENCE_MIN,
    median_blur_ksize: int = MEDIAN_BLUR_KSIZE,
    farne_params: dict = FARNE_PARAMS,
):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    ret, prev = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read first frame from {video_path}")

    height, width = prev.shape[:2]
    frame_area = height * width
    out_width = width * 3
    # Detection should ignore a border around the frame
    margin_x = int(width * BORDER_FRAC)
    margin_y = int(height * BORDER_FRAC)
    central_mask = np.zeros((height, width), dtype=np.uint8)
    central_mask[margin_y:height - margin_y, margin_x:width - margin_x] = 255
    # Region masks: bottom half is ROI with stricter thresholds (full width)
    y0_roi = height // 2
    y1_roi = height
    x0_roi = 0
    x1_roi = width
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    roi_mask[y0_roi:y1_roi, x0_roi:x1_roi] = 255
    roi_mask = cv2.bitwise_and(roi_mask, central_mask)
    nonroi_mask = central_mask.copy()
    nonroi_mask[y0_roi:y1_roi, x0_roi:x1_roi] = 0

    # Video writer for visualization (lossless)
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    out_combined = cv2.VideoWriter(str(out_video_combined), fourcc, fps, (out_width, height)) if out_video_combined else None
    out_orig = cv2.VideoWriter(str(out_video_orig), fourcc, fps, (width, height))
    out_deno = cv2.VideoWriter(str(out_video_deno), fourcc, fps, (width, height))
    out_flow = cv2.VideoWriter(str(out_video_flow), fourcc, fps, (width, height))

    # Stabilization tools (used for denoised/stabilized path only)
    orb = cv2.ORB_create(5000)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Separate previous frames for visualization (unblurred) vs flow (blurred)
    prev_frame_raw_vis = prev.copy()
    prev_frame_stab_vis = prev.copy()
    prev_gray_raw_flow = cv2.medianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), median_blur_ksize)
    prev_gray_stab_flow = prev_gray_raw_flow.copy()

    detections = []  # rows for CSV
    frame_idx = 0    # 0-based, to match fasciculation_tool.py

    def detect_region(motion_energy_map, angle_map, mask, thr_value, min_area, max_area_frac):
        """Threshold within mask and return filtered contours + stats."""
        _, motion_thresh = cv2.threshold(motion_energy_map, thr_value, 255, cv2.THRESH_BINARY)
        if mask is not None:
            motion_thresh = cv2.bitwise_and(motion_thresh, mask)
        contours, _ = cv2.findContours(
            motion_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > (max_area_frac * frame_area):
                continue
            incoh = contour_direction_incoherence(cnt, angle_map)
            if incoh < incoherence_min:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = contour_centroid(cnt)
            patch = motion_energy_map[y:y+h, x:x+w]
            mean_energy = float(patch.mean()) if patch.size > 0 else 0.0
            results.append((cnt, area, incoh, (cx, cy), (x, y, w, h), mean_energy))
        return results

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1  # this frame's index (prev_gray corresponds to frame_idx-1)
        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- 1a) Non-stabilized path (orig panel) ----
        gray_raw_denoised = cv2.medianBlur(gray_raw, median_blur_ksize)
        flow_raw = cv2.calcOpticalFlowFarneback(prev_gray_raw_flow, gray_raw_denoised, None, **farne_params)
        fx_full_raw = flow_raw[..., 0]
        fy_full_raw = flow_raw[..., 1]
        global_fx_raw = np.median(fx_full_raw)
        global_fy_raw = np.median(fy_full_raw)
        fx_res_raw = fx_full_raw - global_fx_raw
        fy_res_raw = fy_full_raw - global_fy_raw
        mag_res_raw, ang_res_raw = cv2.cartToPolar(fx_res_raw, fy_res_raw)
        motion_energy_raw = cv2.normalize(mag_res_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # ---- 1b) Stabilized path (denoised panel) ----
        gray_stab = gray_raw.copy()
        frame_stab = frame.copy()
        matrix = None
        kp1, des1 = orb.detectAndCompute(prev_gray_stab_flow, None)
        kp2, des2 = orb.detectAndCompute(gray_raw, None)
        if des1 is not None and des2 is not None and len(kp1) > 10 and len(kp2) > 10:
            try:
                matches = bf_matcher.match(des1, des2)
                if len(matches) > 10:
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    matrix, _ = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC)
                    if matrix is not None:
                        gray_stab = cv2.warpAffine(gray_raw, matrix, (gray_raw.shape[1], gray_raw.shape[0]))
                        frame_stab = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))
            except cv2.error:
                pass

        gray_denoised = cv2.medianBlur(gray_stab, median_blur_ksize)
        flow_stab = cv2.calcOpticalFlowFarneback(prev_gray_stab_flow, gray_denoised, None, **farne_params)
        fx_full = flow_stab[..., 0]
        fy_full = flow_stab[..., 1]
        global_fx = np.median(fx_full)
        global_fy = np.median(fy_full)
        fx_res = fx_full - global_fx
        fy_res = fy_full - global_fy
        mag_res, ang_res = cv2.cartToPolar(fx_res, fy_res)
        motion_energy = cv2.normalize(mag_res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Panels for visualization
        original_disp = prev_frame_raw_vis.copy()
        candidate_disp = prev_frame_stab_vis.copy()
        # Visualize ROI bounds (faint overlay) clipped to the central detection mask
        roi_color = (50, 100, 255)  # light magenta-ish
        roi_alpha = 0.25
        x0_vis = margin_x
        x1_vis = width - margin_x
        y0_vis = max(y0_roi, margin_y)
        y1_vis = height - margin_y
        for disp in (original_disp, candidate_disp):
            overlay = disp.copy()
            cv2.rectangle(overlay, (x0_vis, y0_vis), (x1_vis, y1_vis), roi_color, -1)
            disp[:] = cv2.addWeighted(overlay, roi_alpha, disp, 1 - roi_alpha, 0)

        # ======================
        # 4) Detect candidates with region-specific thresholds
        # ======================
        orig_regions = [
            ("nonroi", nonroi_mask, thr_orig, min_blob_area),
            ("roi", roi_mask, ROI_THR_ORIG, ROI_MIN_BLOB_AREA),
        ]
        deno_regions = [
            ("nonroi", nonroi_mask, thr_deno, min_blob_area),
            ("roi", roi_mask, ROI_THR_DENO, ROI_MIN_BLOB_AREA),
        ]

        for region_label, mask, thr_val, min_area in orig_regions:
            results = detect_region(motion_energy_raw, ang_res_raw, mask, thr_val, min_area, max_blob_frac)
            for cnt, area, incoh, (cx, cy), (x, y, w, h), mean_energy in results:
                cv2.rectangle(original_disp, (x, y), (x + w, y + h), (0, 0, 255), 1)
                detections.append(
                    {
                        "frame_idx": frame_idx,
                        "x": cx,
                        "y": cy,
                        "panel": "orig",
                        "region": region_label,
                        "area": float(area),
                        "incoherence": incoh,
                        "mean_energy": mean_energy,
                    }
                )

        for region_label, mask, thr_val, min_area in deno_regions:
            results = detect_region(motion_energy, ang_res, mask, thr_val, min_area, max_blob_frac)
            for cnt, area, incoh, (cx, cy), (x, y, w, h), mean_energy in results:
                cv2.rectangle(candidate_disp, (x, y), (x + w, y + h), (0, 0, 255), 1)
                detections.append(
                    {
                        "frame_idx": frame_idx,
                        "x": cx,
                        "y": cy,
                        "panel": "denoised",
                        "region": region_label,
                        "area": float(area),
                        "incoherence": incoh,
                        "mean_energy": mean_energy,
                    }
                )

        # ======================
        # 5) Flow arrows (for visualization)
        # ======================
        flow_vis = cv2.cvtColor(gray_denoised, cv2.COLOR_GRAY2BGR)
        h, w = gray_raw.shape
        yy, xx = np.mgrid[step // 2 : h : step, step // 2 : w : step].astype(int)
        fx_samp = fx_res[yy, xx]
        fy_samp = fy_res[yy, xx]
        for (xi, yi, fxi, fyi) in zip(xx.flatten(), yy.flatten(), fx_samp.flatten(), fy_samp.flatten()):
            end_pt = (int(xi + scale * fxi), int(yi + scale * fyi))
            cv2.arrowedLine(flow_vis, (xi, yi), end_pt, arrow_color, 1, tipLength=0.3)

        # ======================
        # 6) Compose and save visualization frame
        # ======================
        if out_combined is not None:
            combined = np.hstack([original_disp, candidate_disp, flow_vis])
            out_combined.write(combined)
        out_orig.write(original_disp)
        out_deno.write(candidate_disp)
        out_flow.write(flow_vis)

        # Advance reference frame
        prev_gray_raw_flow = gray_raw_denoised
        prev_gray_stab_flow = gray_denoised
        prev_frame_raw_vis = frame
        prev_frame_stab_vis = frame_stab

    # ======================
    # Cleanup & write CSV
    # ======================
    cap.release()
    if out_combined is not None:
        out_combined.release()
    out_orig.release()
    out_deno.release()
    out_flow.release()

    df = pd.DataFrame(detections, columns=[
        "frame_idx", "x", "y", "panel", "region", "area", "incoherence", "mean_energy"
    ])
    df.to_csv(out_csv, index=False)
    print(f"Saved detections CSV to: {out_csv}")
    if out_combined is not None:
        print(f"Saved combined visualization video to: {out_video_combined}")
    print(f"Saved original-panel video to: {out_video_orig}")
    print(f"Saved denoised-panel video to: {out_video_deno}")
    print(f"Saved flow video to: {out_video_flow}")


# ======================
# CLI
# ======================

def main():
    parser = argparse.ArgumentParser(
        description="Variance-based fasciculation detector (full-frame) "
                    "producing an annotated video and detection CSV."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument("--out_csv", type=Path, default=Path("detections_variance.csv"),
                        help="Output detections CSV")
    parser.add_argument("--out_video_combined", type=Path,
                        default=Path("tongue_fasciculations_variance_fullframe.avi"),
                        help="Output combined annotated video (set --no_combined to skip)")
    parser.add_argument("--no_combined", action="store_true",
                        help="Skip writing the combined 3-panel video")
    parser.add_argument("--out_video_orig", type=Path,
                        default=Path("variance_orig_panel.avi"),
                        help="Output video for original panel")
    parser.add_argument("--out_video_deno", type=Path,
                        default=Path("variance_denoised_panel.avi"),
                        help="Output video for denoised panel")
    parser.add_argument("--out_video_flow", type=Path,
                        default=Path("variance_flow_panel.avi"),
                        help="Output video for flow panel")

    args = parser.parse_args()

    run_variance_based_detection(
        video_path=args.video,
        out_csv=args.out_csv,
        out_video_combined=(None if args.no_combined else args.out_video_combined),
        out_video_orig=args.out_video_orig,
        out_video_deno=args.out_video_deno,
        out_video_flow=args.out_video_flow,
    )


if __name__ == "__main__":
    main()
