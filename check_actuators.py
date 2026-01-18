
import os
from dm_control import mujoco
from piper_constants import XML_DIR

def check_actuators():
    xml_path = os.path.join(XML_DIR, 'bimanual_piper_ee_many_cubes.xml')
    print(f"Loading: {xml_path}")
    physics = mujoco.Physics.from_xml_path(xml_path)
    
    print(f"Total Actuators: {physics.model.nu}")
    print("Actuator Names (in order):")
    for i in range(physics.model.nu):
        name = physics.model.id2name(i, 'actuator')
        print(f"  [{i}] {name}")

if __name__ == "__main__":
    check_actuators()
