import numpy as np
import matplotlib.pyplot as plt
import pickle



# Basis vectors of the hexagonal lattice
a1 = np.array([3, np.sqrt(3)]) / 2
a2 = np.array([3, -np.sqrt(3)]) / 2

# Sublattice vectors
b = np.array([[1, 0], [2, 0]])

# Rotation angle (30 degrees)
theta = np.pi / 6  # 30 degrees in radians
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

a1 = a1 @ rotation_matrix.T
a2 = a2 @ rotation_matrix.T
b = b @ rotation_matrix.T


# Define the size of the lattice
n = 8  

# Load the color vector array of shape (n, n, 2)
color_vector = np.load('data/config_mcts_final.npy')

# Initialize lists to store all lattice point positions and corresponding colors
positions = []
colors = []

# Generate lattice points and determine their colors
for i in range(n):
    for j in range(n):
        for k in range(2):
            position = i * a1 + j * a2 + b[k, :]
            positions.append(position)
            colors.append(color_vector[i, j, k])

positions = np.array(positions)
colors = np.array(colors)

fig = plt.figure(figsize=(8, 6))


blue_legend_added = False
red_legend_added = False

# Booleans to track if legend for each type has been added
legend_added = {0: False, 1: False, 2: False}
color_map = {0: 'lightblue', 1: 'tomato', 2: 'mediumseagreen'}
label_map = {0: 'C', 1: 'B', 2: 'N'}

# Create a dictionary to store plot handles for each occupancy type
handles = {0: None, 1: None, 2: None}

for i, pos in enumerate(positions):
    c = int(colors[i])  # occupancy
    if not legend_added[c]:
        h, = plt.plot(pos[0], pos[1], 'o', color=color_map[c], markersize=23, label=label_map[c])
        handles[c] = h
        legend_added[c] = True
    else:
        plt.plot(pos[0], pos[1], 'o', color=color_map[c], markersize=23)

# Specify legend order: C (0), B (1), N (2)
order = [0, 1, 2]
ordered_handles = [handles[i] for i in order]
ordered_labels = [label_map[i] for i in order]

plt.legend(handles=ordered_handles, labels=ordered_labels, loc="lower right")

plt.gca().set_aspect('equal', adjustable='box')

# Remove outer spines
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# Remove axes
plt.axis('off')
#plt.title("Best configurtion with 16 Si by MH", fontsize=16, fontweight="bold")

# Save as PNG file with 300 DPI resolution
plt.savefig('configuration.png', dpi=300, bbox_inches='tight', pad_inches=0.5)  

plt.show()





