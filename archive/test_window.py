# save as test_window.py
import cv2
import numpy as np

img = np.zeros((300, 300, 3), np.uint8)
cv2.putText(img, "Fasciculation!", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()