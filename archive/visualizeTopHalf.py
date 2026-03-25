import cv2
import numpy as np
from pathlib import Path

# === CONFIG ===
video_path = Path("tongue fasciculations_resized.mp4")
step = 5
scale = 2
arrow_color = (0, 255, 0)

# Farnebäck optical flow parameters
farne_params = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# === LOAD VIDEO ===
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS) or 20
ret, prev = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame.")

height, width = prev.shape[:2]
out_width = width * 3

# === Video writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_path = "tongue_fasciculations_annotated.mp4"
out = cv2.VideoWriter(out_path, fourcc, fps, (out_width, height))

# === Tools for stabilization ===
orb = cv2.ORB_create(5000)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Precompute ROI boundary (top half)
roi_h = height // 2

# === MAIN LOOP ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === 1. Motion Stabilization ===
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(gray, None)
    if des1 is not None and des2 is not None and len(kp1) > 10 and len(kp2) > 10:
        try:
            matches = bf_matcher.match(des1, des2)
            if len(matches) > 10:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                matrix, _ = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC)
                if matrix is not None:
                    gray = cv2.warpAffine(gray, matrix, (gray.shape[1], gray.shape[0]))
        except cv2.error:
            pass

    # === 2. Denoising ===
    gray_denoised = cv2.medianBlur(gray, 5)

    # === 3. Optical Flow ===
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_denoised, None, **farne_params)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_energy = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Create copies for masking to top-half ROI
    me_for_orig = motion_energy.copy()
    me_for_deno = motion_energy.copy()

    # Zero out bottom half so detection only happens in top half
    me_for_orig[roi_h:, :] = 0
    me_for_deno[roi_h:, :] = 0

    # === 4. Detect candidates on original frame (TOP HALF ONLY) ===
    _, motion_thresh_orig = cv2.threshold(me_for_orig, 100, 255, cv2.THRESH_BINARY)
    contours_orig, _ = cv2.findContours(motion_thresh_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    original_disp = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)
    for cnt in contours_orig:
        if cv2.contourArea(cnt) > 5:
            x, y, w, h = cv2.boundingRect(cnt)
            # Clip rectangle to top half just in case
            if y < roi_h:
                y2 = min(y + h, roi_h)
                cv2.rectangle(original_disp, (x, y), (x + w, y2), (0, 0, 255), 1)

    # === 5. Detect candidates on denoised frame (TOP HALF ONLY) ===
    _, motion_thresh_denoised = cv2.threshold(me_for_deno, 240, 255, cv2.THRESH_BINARY)
    contours_denoised, _ = cv2.findContours(motion_thresh_denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_disp = cv2.cvtColor(gray_denoised, cv2.COLOR_GRAY2BGR)
    for cnt in contours_denoised:
        if cv2.contourArea(cnt) > 5:
            x, y, w, h = cv2.boundingRect(cnt)
            if y < roi_h:
                y2 = min(y + h, roi_h)
                cv2.rectangle(candidate_disp, (x, y), (x + w, y2), (0, 0, 255), 1)

    # Draw ROI boundary line (visual cue)
    cv2.line(original_disp, (0, roi_h), (width, roi_h), (255, 0, 0), 1)
    cv2.line(candidate_disp, (0, roi_h), (width, roi_h), (255, 0, 0), 1)

    # === 6. Flow Overlay (full frame arrows; restrict to top-half if you prefer) ===
    flow_vis = cv2.cvtColor(gray_denoised, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape
    yy, xx = np.mgrid[step//2:h:step, step//2:w:step].astype(int)
    fx, fy = flow[yy, xx].T
    for (xi, yi, fxi, fyi) in zip(xx.flatten(), yy.flatten(), fx.flatten(), fy.flatten()):
        end_pt = (int(xi + scale*fxi), int(yi + scale*fyi))
        cv2.arrowedLine(flow_vis, (xi, yi), end_pt, arrow_color, 1, tipLength=0.3)

    # Optional: if you want arrows only in top half, uncomment:
    # mask = (yy.flatten() < roi_h)
    # for (xi, yi, fxi, fyi, m) in zip(xx.flatten(), yy.flatten(), fx.flatten(), fy.flatten(), mask):
    #     if not m: continue
    #     end_pt = (int(xi + scale*fxi), int(yi + scale*fyi))
    #     cv2.arrowedLine(flow_vis, (xi, yi), end_pt, arrow_color, 1, tipLength=0.3)

    # === 7. Combine & Save ===
    combined = np.hstack([original_disp, candidate_disp, flow_vis])
    out.write(combined)

    # Optional preview
    # cv2.imshow("Original | Denoised | Flow", combined)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    prev_gray = gray_denoised

# === CLEANUP ===
cap.release()
out.release()
cv2.destroyAllWindows()
