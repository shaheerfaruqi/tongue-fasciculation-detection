#!/usr/bin/env python3
"""
variancebased.py — variance/flow-based fasciculation detector with 5-second temporal integration.

Key change vs your current script:
- Builds per-frame motion masks (region-specific thresholds),
- Maintains a rolling 5-second "hit-count" map (how many frames each pixel was active),
- Detects blobs on the temporally integrated map (pixels active >= min_hits in last window),
  which suppresses 1-frame blur/noise and can recover multi-frame weak twitches.

Offline-friendly: detections are delayed by up to the window length (rolling window), but you
still output per-frame detections (frame_idx where the integrated condition becomes true).
"""

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ======================
# Defaults (override by CLI if you want)
# ======================

# Visualization (does not affect detection sensitivity)
STEP = 10
SCALE = 3
ARROW_COLOR = (0, 255, 0)  # BGR

# ---- Detection thresholds ----
MIN_BLOB_AREA = 7
MAX_BLOB_FRAC = 0.05
THR_ORIG = 140
THR_DENO = 140
INCOHERENCE_MIN = 0.11

# Region-specific tuning
ROI_THR_ORIG = 180
ROI_THR_DENO = 175
ROI_MIN_BLOB_AREA = 7
BORDER_FRAC = 0.10

# Denoising strength
MEDIAN_BLUR_KSIZE = 3  # must be odd

# Temporal integration
WINDOW_SEC = 1.0       # integrate over last N seconds
MIN_HITS = 3           # pixel must be active in >= MIN_HITS frames within window

# Farnebäck parameters
FARNE_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=20,
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
    circ_var = 1.0 - mean_len
    return float(circ_var)


def contour_centroid(cnt):
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w / 2.0
        cy = y + h / 2.0
    return float(cx), float(cy)


def build_region_threshold_mask(
    motion_energy_u8: np.ndarray,
    nonroi_mask_u8: np.ndarray,
    roi_mask_u8: np.ndarray,
    thr_nonroi: int,
    thr_roi: int,
    morph_ksize: int = 3
) -> np.ndarray:
    """
    Build a binary mask of motion candidates using different thresholds in ROI vs non-ROI.
    motion_energy_u8: uint8 0..255.
    Returns uint8 mask (0 or 255).
    """
    m = np.zeros_like(motion_energy_u8, dtype=np.uint8)

    # Non-ROI
    if nonroi_mask_u8 is not None:
        nonroi_pixels = cv2.bitwise_and(motion_energy_u8, nonroi_mask_u8)
    else:
        nonroi_pixels = motion_energy_u8
    _, nonroi_bin = cv2.threshold(nonroi_pixels, thr_nonroi, 255, cv2.THRESH_BINARY)
    m = cv2.bitwise_or(m, nonroi_bin)

    # ROI
    if roi_mask_u8 is not None:
        roi_pixels = cv2.bitwise_and(motion_energy_u8, roi_mask_u8)
        _, roi_bin = cv2.threshold(roi_pixels, thr_roi, 255, cv2.THRESH_BINARY)
        m = cv2.bitwise_or(m, roi_bin)

    # Morphology to remove single-pixel sparkle + fill tiny holes
    if morph_ksize and morph_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    return m


def region_label_from_point(cx, cy, roi_mask_u8):
    x = int(round(cx))
    y = int(round(cy))
    h, w = roi_mask_u8.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return "nonroi"
    return "roi" if roi_mask_u8[y, x] > 0 else "nonroi"


def contours_from_mask(binary_u8: np.ndarray):
    contours, _ = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# ======================
# Core
# ======================

def run_variance_based_detection(
    video_path: Path,
    out_csv: Path,
    out_video_combined: Path,
    out_video_orig: Path,
    out_video_deno: Path,
    out_video_flow: Path,
    window_sec: float = WINDOW_SEC,
    min_hits: int = MIN_HITS,
    step: int = STEP,
    scale: int = SCALE,
    arrow_color=ARROW_COLOR,
    min_blob_area: float = MIN_BLOB_AREA,
    roi_min_blob_area: float = ROI_MIN_BLOB_AREA,
    max_blob_frac: float = MAX_BLOB_FRAC,
    thr_orig: int = THR_ORIG,
    thr_deno: int = THR_DENO,
    roi_thr_orig: int = ROI_THR_ORIG,
    roi_thr_deno: int = ROI_THR_DENO,
    incoherence_min: float = INCOHERENCE_MIN,
    median_blur_ksize: int = MEDIAN_BLUR_KSIZE,
    farne_params: dict = FARNE_PARAMS,
):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    ret, prev = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read first frame from {video_path}")

    height, width = prev.shape[:2]
    frame_area = height * width
    out_width = width * 3

    # Central detection mask (ignore border)
    margin_x = int(width * BORDER_FRAC)
    margin_y = int(height * BORDER_FRAC)
    central_mask = np.zeros((height, width), dtype=np.uint8)
    central_mask[margin_y:height - margin_y, margin_x:width - margin_x] = 255

    # ROI: bottom half (you can change this later)
    y0_roi = height // 2
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    roi_mask[y0_roi:height, 0:width] = 255
    roi_mask = cv2.bitwise_and(roi_mask, central_mask)

    nonroi_mask = central_mask.copy()
    nonroi_mask[y0_roi:height, 0:width] = 0

    # Video writers
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    out_combined = cv2.VideoWriter(str(out_video_combined), fourcc, fps, (out_width, height)) if out_video_combined else None
    out_orig_w = cv2.VideoWriter(str(out_video_orig), fourcc, fps, (width, height))
    out_deno_w = cv2.VideoWriter(str(out_video_deno), fourcc, fps, (width, height))
    out_flow_w = cv2.VideoWriter(str(out_video_flow), fourcc, fps, (width, height))

    # Stabilization tools
    orb = cv2.ORB_create(5000)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Prev frames for flow
    prev_frame_raw_vis = prev.copy()
    prev_frame_stab_vis = prev.copy()
    prev_gray_raw_flow = cv2.medianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), median_blur_ksize)
    prev_gray_stab_flow = prev_gray_raw_flow.copy()

    # Temporal integration state: per panel maintain rolling sum of binary masks
    win_len = max(1, int(round(window_sec * fps)))
    min_hits = max(1, int(min_hits))

    orig_mask_queue = deque(maxlen=win_len)
    deno_mask_queue = deque(maxlen=win_len)
    orig_hit_sum = np.zeros((height, width), dtype=np.uint16)  # counts 0..win_len
    deno_hit_sum = np.zeros((height, width), dtype=np.uint16)

    detections = []
    frame_idx = 0

    def update_rolling_hits(queue, hit_sum_u16, new_mask_u8):
        """
        queue stores uint8 binary masks (0/255).
        hit_sum_u16 stores counts of how many frames pixel was active.
        """
        # Convert 0/255 to 0/1
        new01 = (new_mask_u8 > 0).astype(np.uint16)

        if len(queue) == queue.maxlen:
            old = queue[0]
            old01 = (old > 0).astype(np.uint16)
            hit_sum_u16[:] = hit_sum_u16 - old01

        queue.append(new_mask_u8)
        hit_sum_u16[:] = hit_sum_u16 + new01

    def detect_from_temporal_hits(
        hit_sum_u16: np.ndarray,
        angle_map: np.ndarray,
        motion_energy_u8: np.ndarray,
        panel_label: str,
        disp_img_bgr: np.ndarray
    ):
        """
        Convert rolling hit-count map into a temporal binary map and run contour detection.
        Detections are added only if blob area + incoherence + size constraints pass.
        """
        # Require pixel active in >= effective_min_hits frames in current (possibly not full) queue
        effective_window = max(1, len(orig_mask_queue) if panel_label == "orig" else len(deno_mask_queue))
        effective_min_hits = min(min_hits, effective_window)

        temporal_bin = ((hit_sum_u16 >= effective_min_hits).astype(np.uint8) * 255)

        # Find contours on temporal-bin map
        contours = contours_from_mask(temporal_bin)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= 0:
                continue

            # Region label decided by centroid
            cx, cy = contour_centroid(cnt)
            region = region_label_from_point(cx, cy, roi_mask)

            # Region-specific min area
            min_area = roi_min_blob_area if region == "roi" else min_blob_area
            if area < min_area:
                continue

            if area > (max_blob_frac * frame_area):
                continue

            incoh = contour_direction_incoherence(cnt, angle_map)
            if incoh < incoherence_min:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            patch = motion_energy_u8[y:y+h, x:x+w]
            mean_energy = float(patch.mean()) if patch.size > 0 else 0.0

            # Summarize persistence as mean hits within bbox (optional but useful)
            hit_patch = hit_sum_u16[y:y+h, x:x+w]
            mean_hits = float(hit_patch.mean()) if hit_patch.size > 0 else float(effective_min_hits)

            # Draw box
            cv2.rectangle(disp_img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(
                disp_img_bgr,
                f"{panel_label} {region} hits~{mean_hits:.1f}",
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

            detections.append(
                {
                    "frame_idx": frame_idx,
                    "x": cx,
                    "y": cy,
                    "panel": panel_label,
                    "region": region,
                    "area": float(area),
                    "incoherence": incoh,
                    "mean_energy": mean_energy,
                    "mean_hits_5s": mean_hits,
                    "window_sec": float(window_sec),
                    "min_hits": int(min_hits),
                }
            )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- 1a) Non-stabilized path ----
        gray_raw_denoised = cv2.medianBlur(gray_raw, median_blur_ksize)
        flow_raw = cv2.calcOpticalFlowFarneback(prev_gray_raw_flow, gray_raw_denoised, None, **farne_params)
        fx_full_raw = flow_raw[..., 0]
        fy_full_raw = flow_raw[..., 1]
        fx_res_raw = fx_full_raw - np.median(fx_full_raw)
        fy_res_raw = fy_full_raw - np.median(fy_full_raw)
        mag_res_raw, ang_res_raw = cv2.cartToPolar(fx_res_raw, fy_res_raw)
        motion_energy_raw = cv2.normalize(mag_res_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # ---- 1b) Stabilized path ----
        gray_stab = gray_raw.copy()
        frame_stab = frame.copy()

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
                        gray_stab = cv2.warpAffine(gray_raw, matrix, (width, height))
                        frame_stab = cv2.warpAffine(frame, matrix, (width, height))
            except cv2.error:
                pass

        gray_denoised = cv2.medianBlur(gray_stab, median_blur_ksize)
        flow_stab = cv2.calcOpticalFlowFarneback(prev_gray_stab_flow, gray_denoised, None, **farne_params)
        fx_full = flow_stab[..., 0]
        fy_full = flow_stab[..., 1]
        fx_res = fx_full - np.median(fx_full)
        fy_res = fy_full - np.median(fy_full)
        mag_res, ang_res = cv2.cartToPolar(fx_res, fy_res)
        motion_energy = cv2.normalize(mag_res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # ======================
        # 2) Per-frame region-thresholded motion masks (0/255)
        # ======================
        orig_mask = build_region_threshold_mask(
            motion_energy_u8=motion_energy_raw,
            nonroi_mask_u8=nonroi_mask,
            roi_mask_u8=roi_mask,
            thr_nonroi=thr_orig,
            thr_roi=roi_thr_orig,
            morph_ksize=3
        )
        deno_mask = build_region_threshold_mask(
            motion_energy_u8=motion_energy,
            nonroi_mask_u8=nonroi_mask,
            roi_mask_u8=roi_mask,
            thr_nonroi=thr_deno,
            thr_roi=roi_thr_deno,
            morph_ksize=3
        )

        # ======================
        # 3) Update rolling 5-second hit-count maps
        # ======================
        update_rolling_hits(orig_mask_queue, orig_hit_sum, orig_mask)
        update_rolling_hits(deno_mask_queue, deno_hit_sum, deno_mask)

        # Panels for visualization
        original_disp = prev_frame_raw_vis.copy()
        candidate_disp = prev_frame_stab_vis.copy()

        # Optional: visualize ROI area (faint overlay)
        roi_color = (50, 100, 255)
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
        # 4) Detect on temporally integrated map (key change)
        # ======================
        detect_from_temporal_hits(
            hit_sum_u16=orig_hit_sum,
            angle_map=ang_res_raw,
            motion_energy_u8=motion_energy_raw,
            panel_label="orig",
            disp_img_bgr=original_disp
        )
        detect_from_temporal_hits(
            hit_sum_u16=deno_hit_sum,
            angle_map=ang_res,
            motion_energy_u8=motion_energy,
            panel_label="denoised",
            disp_img_bgr=candidate_disp
        )

        # ======================
        # 5) Flow arrows (visualization)
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
        # 6) Write outputs
        # ======================
        if out_combined is not None:
            combined = np.hstack([original_disp, candidate_disp, flow_vis])
            out_combined.write(combined)
        out_orig_w.write(original_disp)
        out_deno_w.write(candidate_disp)
        out_flow_w.write(flow_vis)

        # Advance reference frames
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
    out_orig_w.release()
    out_deno_w.release()
    out_flow_w.release()

    df = pd.DataFrame(
        detections,
        columns=[
            "frame_idx", "x", "y", "panel", "region",
            "area", "incoherence", "mean_energy",
            "mean_hits_5s", "window_sec", "min_hits"
        ],
    )
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
        description="Variance-based fasciculation detector with 5-second temporal integration."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument("--out_csv", type=Path, default=Path("detections_variance_temporal.csv"),
                        help="Output detections CSV")
    parser.add_argument("--out_video_combined", type=Path,
                        default=Path("tongue_fasciculations_variance_temporal.avi"),
                        help="Output combined annotated video (set --no_combined to skip)")
    parser.add_argument("--no_combined", action="store_true",
                        help="Skip writing the combined 3-panel video")
    parser.add_argument("--out_video_orig", type=Path,
                        default=Path("variance_temporal_orig_panel.avi"),
                        help="Output video for original panel")
    parser.add_argument("--out_video_deno", type=Path,
                        default=Path("variance_temporal_denoised_panel.avi"),
                        help="Output video for denoised panel")
    parser.add_argument("--out_video_flow", type=Path,
                        default=Path("variance_temporal_flow_panel.avi"),
                        help="Output video for flow panel")

    parser.add_argument("--window_sec", type=float, default=WINDOW_SEC,
                        help="Temporal integration window in seconds (default 5)")
    parser.add_argument("--min_hits", type=int, default=MIN_HITS,
                        help="Pixel must be active in >= min_hits frames within window (default 2)")

    args = parser.parse_args()

    run_variance_based_detection(
        video_path=args.video,
        out_csv=args.out_csv,
        out_video_combined=(None if args.no_combined else args.out_video_combined),
        out_video_orig=args.out_video_orig,
        out_video_deno=args.out_video_deno,
        out_video_flow=args.out_video_flow,
        window_sec=args.window_sec,
        min_hits=args.min_hits,
    )


if __name__ == "__main__":
    main()
