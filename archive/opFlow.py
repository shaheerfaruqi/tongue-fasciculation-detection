import cv2
import numpy as np
from pathlib import Path

# === CONFIG ===
video_path = Path("tongue fasciculations.mp4")
step = 10     # spacing between flow vectors
scale = 2     # flow vector length scale
arrow_color = (0, 255, 0)

# Farnebäck parameters
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

# === INIT ORB for global alignment ===
orb = cv2.ORB_create(5000)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# === MAIN LOOP ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === 1. Motion Stabilization (global affine alignment) ===
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(gray, None)
    if des1 is not None and des2 is not None and len(kp1) > 10 and len(kp2) > 10:
        matches = bf_matcher.match(des1, des2)
        if len(matches) > 10:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            matrix, _ = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC)
            if matrix is not None:
                gray = cv2.warpAffine(gray, matrix, (gray.shape[1], gray.shape[0]))

    # === 2. Denoising ===
    gray_denoised = cv2.medianBlur(gray, 5)

    # === 3. Optical Flow ===
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_denoised, None, **farne_params)

    # === 4. Flow Arrow Overlay ===
    flow_vis = cv2.cvtColor(gray_denoised, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape
    y, x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)
    fx, fy = flow[y, x].T

    for (xi, yi, fxi, fyi) in zip(x.flatten(), y.flatten(), fx.flatten(), fy.flatten()):
        end_pt = (int(xi + scale*fxi), int(yi + scale*fyi))
        cv2.arrowedLine(flow_vis, (xi, yi), end_pt, arrow_color, 1, tipLength=0.3)

    # === 5. Side-by-side Display ===
    original_disp = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)
    denoised_disp = cv2.cvtColor(gray_denoised, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([original_disp, denoised_disp, flow_vis])
    cv2.imshow("Original | Denoised+Stabilized | Optical Flow", combined)

    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

    prev_gray = gray_denoised

cap.release()
cv2.destroyAllWindows()
