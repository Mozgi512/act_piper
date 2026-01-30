
import cv2
import numpy as np

print("Testing cv2.imshow...")
img = np.zeros((480, 640, 3), dtype=np.uint8)
# Draw a red rectangle
cv2.rectangle(img, (100, 100), (400, 400), (0, 0, 255), -1)

cv2.imshow("Test Window", img)
print("Window created. Press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Done.")
