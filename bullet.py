import pybullet as p
import pybullet_data
import time

# Connect to PyBullet physics server
p.connect(p.GUI)  # or p.GUI for graphical version

# Set the simulation search path to include PyBullet's data assets
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set the gravity in the simulation
p.setGravity(0, 0, -9.81)
kuka_id = p.loadURDF("/Users/francesco/Desktop/pingpong/urdf/braccioLight/braccioLight.urdf", [0, 0, 0], useFixedBase=True)
table_id = p.loadURDF("urdf/table/table.urdf", [3, 1, 0], useFixedBase=True, globalScaling=1.5)
ball_id = p.loadURDF("urdf/ball.urdf", [2, 0, 1], useFixedBase=False)
# Load the KUKA IIWA model
#kuka_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
# Get the end-effector link index (KUKA IIWA's end effector is usually link 6)
end_effector_index = 2 # !!!! IMPORTANT  va aggiunto l'end effector nel file urdf

target_position = [0.1, 0, 3]  # x, y, z coordinates
# Define the target orientation for the end-effector in quaternion
target_orientation = p.getQuaternionFromEuler([0, 0, 0])  # No rotation initially
print(target_orientation)

# Calculate the inverse kinematics for the target position and orientation
joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=kuka_id,
    endEffectorLinkIndex=end_effector_index,
    targetPosition=target_position,
    targetOrientation=target_orientation
)

#serial communication of the joint angles
print(joint_angles)

# Set the joint angles from IK as targets for the KUKA arm joints
for i, angle in enumerate(joint_angles):
    p.setJointMotorControl2(
        bodyUniqueId=kuka_id,
        jointIndex=i,
        controlMode=p.POSITION_CONTROL,
        targetPosition=angle,
        force=200  # Adjust force as needed
    )

# Run the simulation
for _ in range(1000):   # Run for 1000 timesteps
    p.stepSimulation()
    time.sleep(1 / 240)   # Sleep to simulate real-time (240 Hz)

# Disconnect from the physics server
p.disconnect()
