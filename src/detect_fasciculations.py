#!/usr/bin/env python3
"""
variance_improved.py — improved variance/flow-based fasciculation detector for tongue ultrasound.

Improvements implemented (no deep learning):
- Robust stabilization (ORB; translation-only; fallback to ECC translation-only)
- Speckle-aware denoising: median + optional light bilateral
- Artifact rejection for blur/probe jolt:
    * focus drop check via variance of Laplacian
    * global motion spike check via robust flow magnitude percentiles
- Adaptive thresholding:
    * NO per-frame min-max normalization for detection
    * per-region robust thresholds based on median + k*MAD of residual flow magnitude
- Region-specific detection masks + morphology before contouring
- Temporal integration using rolling hit-count map (suppresses 1-frame noise, recovers weak multi-frame twitches)
- Direction metric: coherence-first by default (tunable)
- Optional object-level temporal confirmation (2/3 frames near same location)

Fixes added based on your new video:
- Stabilization now forbids rotation (translation-only) to avoid spurious rotation artifacts.
- Warps use BORDER_CONSTANT (no reflected “ghost” fill).

Extras:
- Skip detections in the first/last N frames (videos remain full-length).
- Optional FPS override for output writing (fixes “videos look 0.5s long” when input fps is misread).

Outputs:
- Annotated videos (orig, denoised/stabilized, flow)
- CSV with detections including persistence, thresholds, artifact flags
"""

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# ======================
# Defaults
# ======================

# Visualization
STEP = 10
SCALE = 3
ARROW_COLOR = (0, 255, 0)

# Detection geometry
BORDER_FRAC = 0.10
MIN_BLOB_AREA = 15
ROI_MIN_BLOB_AREA = 12
MAX_BLOB_FRAC = 0.03

# ROI definition (bottom half by default)
ROI_Y_FRAC_START = 0.5

# Denoising
MEDIAN_BLUR_KSIZE = 3          # odd
USE_BILATERAL = False
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 25
BILATERAL_SIGMA_SPACE = 25

# Farnebäck
FARNE_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=20,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

# Adaptive thresholding (robust)
K_NONROI = 6.0
K_ROI = 7.0

# Direction criterion
USE_COHERENCE = True
COHERENCE_MIN = 0.50
INCOHERENCE_MIN = 0.11

# Morphology
MORPH_KSIZE = 3

# Temporal integration (pixel-wise)
WINDOW_SEC_DEFAULT = 1.0
MIN_HITS_DEFAULT = 3

# Optional: object confirmation 2-of-3 frames
USE_OBJECT_CONFIRM_DEFAULT = True
OBJECT_CONFIRM_K = 3
OBJECT_CONFIRM_WINDOW = 5
OBJECT_CONFIRM_DIST_PX = 25

# Artifact rejection
FOCUS_DROP_RATIO_DEFAULT = 0.65
FOCUS_MEDIAN_WINDOW = 31
GLOBAL_MOTION_P95_MAX_DEFAULT = 2.5
GLOBAL_MOTION_MED_MAX_DEFAULT = 0.8

# Skip detection in head/tail frames
SKIP_HEAD_FRAMES_DEFAULT = 5
SKIP_TAIL_FRAMES_DEFAULT = 5

# Output FPS override (0 = auto/clamped)
FPS_OUT_DEFAULT = 0.0

# Minimum mean flow magnitude for a detected blob (rejects near-zero residual motion)
MIN_MEAN_MAG_DEFAULT = 0.10


# ======================
# Utilities
# ======================

def ensure_odd(x: int) -> int:
    return x if x % 2 == 1 else x + 1

def mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)

def robust_threshold(values: np.ndarray, k: float) -> float:
    med = float(np.median(values))
    sigma = 1.4826 * mad(values)
    return med + k * sigma

def focus_measure(gray_u8: np.ndarray) -> float:
    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
    return float(lap.var())

def contour_centroid(cnt) -> Tuple[float, float]:
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(cnt)
    return float(x + w / 2.0), float(y + h / 2.0)

def direction_coherence(cnt, angle_map: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 0 or h <= 0:
        return 0.0
    patch = angle_map[y:y+h, x:x+w]
    if patch.size == 0:
        return 0.0
    ux = np.cos(patch)
    uy = np.sin(patch)
    return float(np.hypot(float(ux.mean()), float(uy.mean())))

def build_masks(height: int, width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    margin_x = int(width * BORDER_FRAC)
    margin_y = int(height * BORDER_FRAC)
    central = np.zeros((height, width), dtype=np.uint8)
    central[margin_y:height - margin_y, margin_x:width - margin_x] = 255

    y0_roi = int(height * ROI_Y_FRAC_START)
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[y0_roi:height, 0:width] = 255
    roi = cv2.bitwise_and(roi, central)

    nonroi = central.copy()
    nonroi[y0_roi:height, 0:width] = 0
    return central, nonroi, roi, y0_roi

def apply_morph(binary_u8: np.ndarray, ksize: int) -> np.ndarray:
    if not ksize or ksize < 3:
        return binary_u8
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    out = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, k, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
    return out

def region_label_from_point(cx: float, cy: float, roi_mask: np.ndarray) -> str:
    x = int(round(cx))
    y = int(round(cy))
    if x < 0 or y < 0 or y >= roi_mask.shape[0] or x >= roi_mask.shape[1]:
        return "nonroi"
    return "roi" if roi_mask[y, x] > 0 else "nonroi"

def clamp_fps(x: float) -> float:
    if (x is None) or (not np.isfinite(x)) or (x < 5.0) or (x > 120.0):
        return 20.0
    return float(x)


# ======================
# Stabilization (translation-only)
# ======================

@dataclass
class StabilizationResult:
    gray_stab: np.ndarray
    frame_stab: np.ndarray
    matrix: Optional[np.ndarray]
    method: str  # "ORB_TRANSLATION", "ECC_TRANSLATION", "NONE"

def stabilize_frame(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    curr_frame_bgr: np.ndarray,
    orb: cv2.ORB,
    bf: cv2.BFMatcher,
    ecc_iters: int = 50,
    ecc_eps: float = 1e-6
) -> StabilizationResult:
    h, w = curr_gray.shape[:2]

    # ORB -> robust translation-only via median displacement of matches
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if des1 is not None and des2 is not None and len(kp1) >= 12 and len(kp2) >= 12:
        try:
            matches = bf.match(des1, des2)
            if len(matches) >= 20:
                matches = sorted(matches, key=lambda m: m.distance)[:200]
                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)  # prev
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)  # curr

                d = (pts1 - pts2).reshape(-1, 2)  # displacement vectors
                dx = float(np.median(d[:, 0]))
                dy = float(np.median(d[:, 1]))

                M = np.array([[1.0, 0.0, dx],
                              [0.0, 1.0, dy]], dtype=np.float32)

                gray_stab = cv2.warpAffine(
                    curr_gray, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                frame_stab = cv2.warpAffine(
                    curr_frame_bgr, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                return StabilizationResult(gray_stab, frame_stab, M, "ORB_TRANSLATION")
        except cv2.error:
            pass

    # ECC fallback -> translation-only
    try:
        warp = np.array([[1, 0, 0],
                         [0, 1, 0]], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iters, ecc_eps)

        prev_f = prev_gray.astype(np.float32)
        curr_f = curr_gray.astype(np.float32)

        _, warp = cv2.findTransformECC(
            prev_f, curr_f, warp,
            cv2.MOTION_TRANSLATION,
            criteria, None, 5
        )

        gray_stab = cv2.warpAffine(
            curr_gray, warp, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        frame_stab = cv2.warpAffine(
            curr_frame_bgr, warp, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        return StabilizationResult(gray_stab, frame_stab, warp, "ECC_TRANSLATION")
    except cv2.error:
        return StabilizationResult(curr_gray, curr_frame_bgr, None, "NONE")


# ======================
# Temporal integration helpers
# ======================

def update_rolling_hits(queue: Deque[np.ndarray], hit_sum: np.ndarray, new_mask_u8: np.ndarray):
    new01 = (new_mask_u8 > 0).astype(np.uint16)
    if len(queue) == queue.maxlen:
        old01 = (queue[0] > 0).astype(np.uint16)
        hit_sum[:] = hit_sum - old01
    queue.append(new_mask_u8)
    hit_sum[:] = hit_sum + new01


# ======================
# Core
# ======================

def run(
    video_path: Path,
    out_csv: Path,
    out_video_combined: Optional[Path],
    out_video_orig: Path,
    out_video_deno: Path,
    out_video_flow: Path,
    window_sec: float,
    min_hits: int,
    use_coherence: bool,
    coh_min: float,
    incoh_min: float,
    k_nonroi: float,
    k_roi: float,
    focus_drop_ratio: float,
    p95_max: float,
    med_max: float,
    use_object_confirm: bool,
    skip_head: int,
    skip_tail: int,
    fps_out: float,
    min_mean_mag: float,
):
    cap = cv2.VideoCapture(str(video_path))
    fps_in = clamp_fps(float(cap.get(cv2.CAP_PROP_FPS) or 0.0))
    fps = float(fps_out) if (fps_out and fps_out > 0) else fps_in
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev_bgr = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read first frame from {video_path}")

    height, width = prev_bgr.shape[:2]
    frame_area = height * width
    out_width = width * 3

    central_mask, nonroi_mask, roi_mask, y0_roi = build_masks(height, width)

    # Writers
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    out_combined_w = cv2.VideoWriter(str(out_video_combined), fourcc, fps, (out_width, height)) if out_video_combined else None
    out_orig_w = cv2.VideoWriter(str(out_video_orig), fourcc, fps, (width, height))
    out_deno_w = cv2.VideoWriter(str(out_video_deno), fourcc, fps, (width, height))
    out_flow_w = cv2.VideoWriter(str(out_video_flow), fourcc, fps, (width, height))

    if out_combined_w is not None and (not out_combined_w.isOpened()):
        raise RuntimeError("Failed to open combined VideoWriter (codec/container issue).")
    if not out_orig_w.isOpened():
        raise RuntimeError("Failed to open orig VideoWriter (codec/container issue).")
    if not out_deno_w.isOpened():
        raise RuntimeError("Failed to open deno VideoWriter (codec/container issue).")
    if not out_flow_w.isOpened():
        raise RuntimeError("Failed to open flow VideoWriter (codec/container issue).")

    orb = cv2.ORB_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_gray_raw = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    prev_gray_stab = prev_gray_raw.copy()

    k_med = ensure_odd(MEDIAN_BLUR_KSIZE)
    prev_gray_raw_flow = cv2.medianBlur(prev_gray_raw, k_med)
    prev_gray_stab_flow = prev_gray_raw_flow.copy()

    prev_frame_raw_vis = prev_bgr.copy()
    prev_frame_stab_vis = prev_bgr.copy()

    win_len = max(1, int(round(window_sec * fps)))
    min_hits = max(1, int(min_hits))

    orig_q: Deque[np.ndarray] = deque(maxlen=win_len)
    deno_q: Deque[np.ndarray] = deque(maxlen=win_len)
    orig_hits = np.zeros((height, width), dtype=np.uint16)
    deno_hits = np.zeros((height, width), dtype=np.uint16)

    focus_hist: Deque[float] = deque(maxlen=FOCUS_MEDIAN_WINDOW)
    recent_objs: Deque[List[Tuple[float, float, str]]] = deque(maxlen=OBJECT_CONFIRM_WINDOW)

    detections: List[Dict] = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        in_head = frame_idx <= int(skip_head)
        in_tail = (total_frames > 0) and (frame_idx > (total_frames - int(skip_tail)))
        skip_detection = bool(in_head or in_tail)

        gray_raw = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        stab = stabilize_frame(prev_gray_stab, gray_raw, frame_bgr, orb, bf)
        gray_stab = stab.gray_stab
        frame_stab = stab.frame_stab

        # Denoising
        gray_raw_d = cv2.medianBlur(gray_raw, k_med)
        gray_stab_d = cv2.medianBlur(gray_stab, k_med)
        if USE_BILATERAL:
            gray_raw_d = cv2.bilateralFilter(gray_raw_d, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
            gray_stab_d = cv2.bilateralFilter(gray_stab_d, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

        # Flow + residual
        flow_raw = cv2.calcOpticalFlowFarneback(prev_gray_raw_flow, gray_raw_d, None, **FARNE_PARAMS)
        fx_r = flow_raw[..., 0] - np.median(flow_raw[..., 0])
        fy_r = flow_raw[..., 1] - np.median(flow_raw[..., 1])
        mag_r, ang_r = cv2.cartToPolar(fx_r, fy_r)

        flow_stab = cv2.calcOpticalFlowFarneback(prev_gray_stab_flow, gray_stab_d, None, **FARNE_PARAMS)
        fx_s = flow_stab[..., 0] - np.median(flow_stab[..., 0])
        fy_s = flow_stab[..., 1] - np.median(flow_stab[..., 1])
        mag_s, ang_s = cv2.cartToPolar(fx_s, fy_s)

        # Artifact checks
        f = focus_measure(gray_stab_d)
        focus_hist.append(f)
        focus_med = float(np.median(np.array(focus_hist))) if len(focus_hist) >= 5 else f
        blur_artifact = (focus_med > 0) and (f < focus_drop_ratio * focus_med)

        p95 = float(np.percentile(mag_s, 95))
        med_mag = float(np.median(mag_s))
        motion_spike = (p95 > p95_max) or (med_mag > med_max)

        artifact_skip = bool(blur_artifact or motion_spike)

        # Adaptive thresholds
        def region_thr(mag: np.ndarray, mask: np.ndarray, k: float) -> float:
            vals = mag[mask > 0]
            if vals.size < 50:
                vals = mag[central_mask > 0]
            return robust_threshold(vals.astype(np.float32), k)

        thr_raw_nonroi = region_thr(mag_r, nonroi_mask, k_nonroi)
        thr_raw_roi = region_thr(mag_r, roi_mask, k_roi)
        thr_stab_nonroi = region_thr(mag_s, nonroi_mask, k_nonroi)
        thr_stab_roi = region_thr(mag_s, roi_mask, k_roi)

        def build_motion_mask(mag: np.ndarray, thr_nonroi: float, thr_roi: float) -> np.ndarray:
            mask_u8 = np.zeros((height, width), dtype=np.uint8)
            nonroi = ((mag > thr_nonroi) & (nonroi_mask > 0))
            roi = ((mag > thr_roi) & (roi_mask > 0))
            mask_u8[nonroi | roi] = 255
            return apply_morph(mask_u8, MORPH_KSIZE)

        raw_mask = build_motion_mask(mag_r, thr_raw_nonroi, thr_raw_roi)
        stab_mask = build_motion_mask(mag_s, thr_stab_nonroi, thr_stab_roi)

        # Update rolling hits
        update_rolling_hits(orig_q, orig_hits, raw_mask)
        update_rolling_hits(deno_q, deno_hits, stab_mask)

        eff_raw_win = max(1, len(orig_q))
        eff_stab_win = max(1, len(deno_q))
        eff_min_hits_raw = min(min_hits, eff_raw_win)
        eff_min_hits_stab = min(min_hits, eff_stab_win)

        raw_temporal = (orig_hits >= eff_min_hits_raw).astype(np.uint8) * 255
        stab_temporal = (deno_hits >= eff_min_hits_stab).astype(np.uint8) * 255

        # Visualization frames
        original_disp = prev_frame_raw_vis.copy()
        candidate_disp = prev_frame_stab_vis.copy()

        # ROI overlay
        margin_x = int(width * BORDER_FRAC)
        margin_y = int(height * BORDER_FRAC)
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

        # Flow arrows (stabilized)
        flow_vis = cv2.cvtColor(gray_stab_d, cv2.COLOR_GRAY2BGR)
        yy, xx = np.mgrid[STEP // 2 : height : STEP, STEP // 2 : width : STEP].astype(int)
        fx_samp = fx_s[yy, xx]
        fy_samp = fy_s[yy, xx]
        for (xi, yi, fxi, fyi) in zip(xx.flatten(), yy.flatten(), fx_samp.flatten(), fy_samp.flatten()):
            end_pt = (int(xi + SCALE * fxi), int(yi + SCALE * fyi))
            cv2.arrowedLine(flow_vis, (xi, yi), end_pt, ARROW_COLOR, 1, tipLength=0.3)

        # Artifact text
        status = []
        if blur_artifact:
            status.append("BLUR")
        if motion_spike:
            status.append("MOTION_SPIKE")
        if status:
            for disp in (original_disp, candidate_disp, flow_vis):
                cv2.putText(disp, "ARTIFACT: " + ",".join(status), (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        frame_objects: List[Tuple[float, float, str]] = []

        if (not skip_detection) and (not artifact_skip):
            def detect_on_temporal(
                temporal_u8: np.ndarray,
                ang_map: np.ndarray,
                mag_map: np.ndarray,
                panel: str,
                disp: np.ndarray,
                thr_nonroi: float,
                thr_roi: float,
                eff_min_hits: int,
                hit_map: np.ndarray,
            ):
                contours, _ = cv2.findContours(temporal_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = float(cv2.contourArea(cnt))
                    if area <= 0:
                        continue
                    if area > (MAX_BLOB_FRAC * frame_area):
                        continue

                    cx, cy = contour_centroid(cnt)
                    region = region_label_from_point(cx, cy, roi_mask)
                    min_area = ROI_MIN_BLOB_AREA if region == "roi" else MIN_BLOB_AREA
                    if area < min_area:
                        continue

                    coh = direction_coherence(cnt, ang_map)
                    incoh = 1.0 - coh

                    if use_coherence:
                        if coh < coh_min:
                            continue
                    else:
                        if incoh < incoh_min:
                            continue

                    x, y, w, h = cv2.boundingRect(cnt)
                    patch_mag = mag_map[y:y+h, x:x+w]
                    mean_mag = float(patch_mag.mean()) if patch_mag.size else 0.0

                    # Reject blobs with negligible actual flow magnitude
                    if mean_mag < min_mean_mag:
                        continue

                    patch_hits = hit_map[y:y+h, x:x+w]
                    mean_hits = float(patch_hits.mean()) if patch_hits.size else float(eff_min_hits)

                    frame_objects.append((cx, cy, panel))

                    cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.putText(
                        disp,
                        f"{panel} {region} hits~{mean_hits:.1f} coh~{coh:.2f}",
                        (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA
                    )

                    detections.append({
                        "frame_idx": frame_idx,
                        "x": cx,
                        "y": cy,
                        "panel": panel,
                        "region": region,
                        "area": area,
                        "coherence": coh,
                        "incoherence": incoh,
                        "mean_mag": mean_mag,
                        "mean_hits": mean_hits,
                        "eff_min_hits": int(eff_min_hits),
                        "thr_nonroi": float(thr_nonroi),
                        "thr_roi": float(thr_roi),
                        "focus": float(f),
                        "focus_med": float(focus_med),
                        "blur_artifact": int(blur_artifact),
                        "motion_spike": int(motion_spike),
                        "stab_method": stab.method,
                    })

            detect_on_temporal(raw_temporal, ang_r, mag_r, "orig", original_disp,
                               thr_raw_nonroi, thr_raw_roi, eff_min_hits_raw, orig_hits)
            detect_on_temporal(stab_temporal, ang_s, mag_s, "denoised", candidate_disp,
                               thr_stab_nonroi, thr_stab_roi, eff_min_hits_stab, deno_hits)

        # Object confirmation (only for objects detected)
        if use_object_confirm:
            recent_objs.append(frame_objects)
            history = [obj for fr in recent_objs for obj in fr]
            confirmed = {}
            for (cx, cy, panel) in frame_objects:
                count = 0
                for (hx, hy, hp) in history:
                    if hp != panel:
                        continue
                    if (hx - cx) ** 2 + (hy - cy) ** 2 <= OBJECT_CONFIRM_DIST_PX ** 2:
                        count += 1
                confirmed[(cx, cy, panel)] = int(count >= OBJECT_CONFIRM_K)

            i = len(detections) - 1
            while i >= 0 and detections[i]["frame_idx"] == frame_idx:
                key = (detections[i]["x"], detections[i]["y"], detections[i]["panel"])
                detections[i]["confirmed_2of3"] = confirmed.get(key, 0)
                i -= 1

        # Write videos (always full length)
        if out_combined_w is not None:
            combined = np.hstack([original_disp, candidate_disp, flow_vis])
            out_combined_w.write(combined)
        out_orig_w.write(original_disp)
        out_deno_w.write(candidate_disp)
        out_flow_w.write(flow_vis)

        # Advance
        prev_gray_raw = gray_raw
        prev_gray_stab = gray_stab
        prev_gray_raw_flow = gray_raw_d
        prev_gray_stab_flow = gray_stab_d
        prev_frame_raw_vis = frame_bgr
        prev_frame_stab_vis = frame_stab

    cap.release()
    if out_combined_w is not None:
        out_combined_w.release()
    out_orig_w.release()
    out_deno_w.release()
    out_flow_w.release()

    df = pd.DataFrame(detections)
    if not df.empty:
        cols = [
            "frame_idx", "x", "y", "panel", "region",
            "area", "coherence", "incoherence",
            "mean_mag", "mean_hits", "eff_min_hits",
            "thr_nonroi", "thr_roi",
            "focus", "focus_med", "blur_artifact", "motion_spike",
            "stab_method", "confirmed_2of3"
        ]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    df.to_csv(out_csv, index=False)
    print(f"fps_in={fps_in:.3f}, fps_out={fps:.3f}, total_frames={total_frames}")
    print(f"Saved detections CSV to: {out_csv}")
    if out_video_combined:
        print(f"Saved combined video to: {out_video_combined}")
    print(f"Saved original-panel video to: {out_video_orig}")
    print(f"Saved denoised-panel video to: {out_video_deno}")
    print(f"Saved flow video to: {out_video_flow}")


# ======================
# CLI
# ======================

def main():
    parser = argparse.ArgumentParser(description="Improved variance/flow fasciculation detector (no DL).")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, default=Path("detections_variance_improved.csv"))
    parser.add_argument("--out_video_combined", type=Path, default=Path("variance_improved_combined.avi"))
    parser.add_argument("--no_combined", action="store_true")
    parser.add_argument("--out_video_orig", type=Path, default=Path("variance_improved_orig.avi"))
    parser.add_argument("--out_video_deno", type=Path, default=Path("variance_improved_deno.avi"))
    parser.add_argument("--out_video_flow", type=Path, default=Path("variance_improved_flow.avi"))

    parser.add_argument("--window_sec", type=float, default=WINDOW_SEC_DEFAULT)
    parser.add_argument("--min_hits", type=int, default=MIN_HITS_DEFAULT)

    parser.add_argument("--use_coherence", action="store_true", help="Force coherence filter on.")
    parser.add_argument("--use_incoherence", action="store_true", help="Force incoherence filter on.")
    parser.add_argument("--coh_min", type=float, default=COHERENCE_MIN)
    parser.add_argument("--incoh_min", type=float, default=INCOHERENCE_MIN)

    parser.add_argument("--k_nonroi", type=float, default=K_NONROI)
    parser.add_argument("--k_roi", type=float, default=K_ROI)

    parser.add_argument("--focus_drop_ratio", type=float, default=FOCUS_DROP_RATIO_DEFAULT)
    parser.add_argument("--p95_max", type=float, default=GLOBAL_MOTION_P95_MAX_DEFAULT)
    parser.add_argument("--med_max", type=float, default=GLOBAL_MOTION_MED_MAX_DEFAULT)

    parser.add_argument("--skip_head", type=int, default=SKIP_HEAD_FRAMES_DEFAULT)
    parser.add_argument("--skip_tail", type=int, default=SKIP_TAIL_FRAMES_DEFAULT)

    parser.add_argument("--fps_out", type=float, default=FPS_OUT_DEFAULT,
                        help="Output FPS. 0 = use input FPS (clamped).")

    parser.add_argument("--disable_object_confirm", action="store_true")
    parser.add_argument("--min_mean_mag", type=float, default=MIN_MEAN_MAG_DEFAULT,
                        help="Minimum mean flow magnitude for a blob to count as a detection.")

    args = parser.parse_args()

    if args.use_incoherence and args.use_coherence:
        raise SystemExit("Choose only one: --use_coherence or --use_incoherence")
    if args.use_incoherence:
        use_coh = False
    elif args.use_coherence:
        use_coh = True
    else:
        use_coh = USE_COHERENCE

    run(
        video_path=args.video,
        out_csv=args.out_csv,
        out_video_combined=(None if args.no_combined else args.out_video_combined),
        out_video_orig=args.out_video_orig,
        out_video_deno=args.out_video_deno,
        out_video_flow=args.out_video_flow,
        window_sec=args.window_sec,
        min_hits=args.min_hits,
        use_coherence=use_coh,
        coh_min=args.coh_min,
        incoh_min=args.incoh_min,
        k_nonroi=args.k_nonroi,
        k_roi=args.k_roi,
        focus_drop_ratio=args.focus_drop_ratio,
        p95_max=args.p95_max,
        med_max=args.med_max,
        use_object_confirm=(not args.disable_object_confirm),
        skip_head=args.skip_head,
        skip_tail=args.skip_tail,
        fps_out=args.fps_out,
        min_mean_mag=args.min_mean_mag,
    )

if __name__ == "__main__":
    main()
