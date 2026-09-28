import numpy as np
import matplotlib.pyplot as plt

filepath = r"F:\Krishna Kummari\STP\Projects\My_Project\claude_project_code\neural_ik_level1\neural_ik\data\xarm6_ik_dataset.npz"

with np.load(filepath) as data:
    # Extract the arrays into separate variables
    joints = data['joints']
    positions = data['positions']
    orientations = data['orientations']
    collision_labels = data['collision_labels']
    joint_limits = data['joint_limits']

# Print the shapes and summaries of the data
print(f"Joints shape:           {joints.shape} (likely joint angles for the 6 DoF arm)")
print(f"Positions shape:        {positions.shape} (likely X, Y, Z coordinates)")
print(f"Orientations shape:     {orientations.shape} (likely quaternions or rotation matrices)")
print(f"Collision labels shape: {collision_labels.shape} (boolean or binary safety mask)")
print(f"Joint limits shape:     {joint_limits.shape} (min/max bounds for the actuators)")

# Peek at the first sample
print("\nFirst joint configuration sample:\n", joints[0])

plt.plot(joints[:,5])

plt.show()