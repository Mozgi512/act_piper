
import os
import dm_control.mujoco
import cv2

# Force EGL to check what it is using
os.environ['MUJOCO_GL'] = 'egl'

try:
    print("Checking EGL Renderer...")
    physics = dm_control.mujoco.Physics.from_xml_string('<mujoco/>')
    # Render to trigger context creation
    pixels = physics.render(height=100, width=100, camera_id=-1)
    
    # Introspect (This is hard with dm_control directly, but we can infer from speed or logs)
    # Actually, let's try to use OpenGL directly to query GL_RENDERER if possible, or just time it.
    
    import time
    start = time.time()
    for _ in range(10):
        physics.render(height=480, width=640, camera_id=-1)
    end = time.time()
    print(f"10 frames render time: {end-start:.4f}s")
    
except Exception as e:
    print(f"EGL Check failed: {e}")

print("\nChecking GLFW/GLX context availability...")
# Unset EGL to let it try default (GLFW)
del os.environ['MUJOCO_GL']
try:
    from dm_control import _render
    # Force reload or re-import if possible? No, backend is set on import.
    # We might need a separate process to test GLFW.
    print("Cannot check GLFW in same process as EGL. Will check in separate call.")
except Exception as e:
    print(e)
