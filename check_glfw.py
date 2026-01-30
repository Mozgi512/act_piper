
import os
# Force GLFW
os.environ['MUJOCO_GL'] = 'glfw'
print(f"MUJOCO_GL set to: {os.environ['MUJOCO_GL']}")

try:
    from dm_control import mujoco
    print("Imported mujoco")
    physics = mujoco.Physics.from_xml_string('<mujoco/>')
    print("Created physics")
    # This should trigger GLFW context creation
    pixels = physics.render(height=480, width=640, camera_id=-1)
    print(f"Render successful. Shape: {pixels.shape}")
except Exception as e:
    print(f"GLFW Render failed: {e}")
