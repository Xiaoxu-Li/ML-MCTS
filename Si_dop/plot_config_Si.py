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
color_vector = np.load('data/config_mlmcts.npy')

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

# Convert to numpy arrays
positions = np.array(positions)
colors = np.array(colors)

fig = plt.figure(figsize=(8, 6))


blue_legend_added = False
red_legend_added = False

# Plot the points based on color_vector and add legend labels
for i, pos in enumerate(positions):
    if colors[i] == 1:
        if not blue_legend_added:
            plt.plot(pos[0], pos[1], 'o', color="lightblue", markersize=12, label='C') 
            blue_legend_added = True
        else:
            plt.plot(pos[0], pos[1], 'o', color="lightblue", markersize=12)
    else:
        if not red_legend_added:
            plt.plot(pos[0], pos[1], 'o', color="tomato", markersize=12, label='Si') 
            red_legend_added = True
        else:
            plt.plot(pos[0], pos[1], 'o', color="tomato", markersize=12) 

# Add legend
plt.legend(loc="lower right")

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





