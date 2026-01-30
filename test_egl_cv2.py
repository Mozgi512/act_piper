
import os
os.environ['MUJOCO_GL'] = 'egl'

import cv2
import numpy as np
from dm_control import mujoco

print("Initializing EGL...")
physics = mujoco.Physics.from_xml_string('<mujoco/>')
print("EGL Initialized.")

print("Rendering frame...")
pixels = physics.render(height=480, width=640)
# Convert to BGR
img_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

print("Showing window...")
cv2.imshow("EGL + CV2 Test", img_bgr)
print("Window shown. Press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Done.")
