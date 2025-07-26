import matplotlib.pyplot as plt
import numpy as np
import math
import pickle


# Round a float value down to a specified number of decimal places
def floor_to_decimal(value, decimal_places):
    factor = 10 ** decimal_places
    return math.floor(value * factor) / factor

# Reference value
ref_value = floor_to_decimal(-11.185857, 6) - 1e-2


colors = ['black', 'red', 'lime', 'blue', 'yellow', 'cyan', 
          'Magenta', 'Gray', 'Green', 'Purple', 'Navy', 'Orange', 'Violet', 'Indigo', 'Lavender',
          'Gold', 'Brown', 'Teal']

fig = plt.figure()


# --- Load MH data ---
output_path_mh = "data/result_mh.txt"
data_mh = np.loadtxt(output_path_mh, delimiter='\t')
x_mh = data_mh[:, 0]    # Number of evaluations
y_mh = data_mh[:, 1]    # Objective function values
z_mh = y_mh - ref_value
plt.plot(x_mh, z_mh, color=colors[2], linestyle="-", linewidth=2, label = "MH")

y2_mh = data_mh[:, 2]    # Objective value at current node
z2_mh = y2_mh - ref_value
#plt.plot(x_mh, z2_mh, color=colors[4], linewidth=2, label = "MH path")


# --- Load MCTS data ---
output_path_mcts = "data/result_mcts.txt"
data_mcts = np.loadtxt(output_path_mcts, delimiter='\t')
x_mcts = data_mcts[:, 0]    # Number of evaluations
y_mcts = data_mcts[:, 1]    # Objective function values
z_mcts = y_mcts - ref_value
plt.plot(x_mcts, z_mcts, color=colors[1], linestyle="-", linewidth=2, label = "MCTS")

y2_mcts = data_mcts[:, 2]    # Objective value at current node
z2_mcts = y2_mcts - ref_value
#plt.plot(x_mcts, z2_mcts, color=colors[3], linestyle="-", linewidth=2, label = "MCTS path")


# --- Load MLMCTS data ---
output_path_mlmcts = "data/result_mlmcts.txt"
data_mlmcts = np.loadtxt(output_path_mlmcts, delimiter='\t')
x_mlmcts = data_mlmcts[:, 0]    # Number of evaluations
y_mlmcts = data_mlmcts[:, 1]    # Objective function values
z_mlmcts = y_mlmcts - ref_value
plt.plot(x_mlmcts, z_mlmcts, color=colors[3], linestyle="-", linewidth=2, label = "ML-MCTS")

y2_mlmcts = data_mlmcts[:, 2]    # Objective value at current node
z2_mlmcts = y2_mlmcts - ref_value
#plt.plot(x_mlmcts, z2_mlmcts, color=colors[3], linestyle="-", linewidth=2, label = "MLMCTS path")

plt.legend(fontsize=16)
plt.yscale('log')

plt.xlabel("Number of evaluations", fontsize=16)
plt.ylabel("Objective function value", fontsize=16)


# Manually set y-axis tick labels (corrected by adding reference value)
yticks = np.array([5e-1, 1e-1, 5e-2, 1e-2]) 
yticks_show = yticks + ref_value
plt.yticks(yticks, [f'{tick:.4f}' for tick in yticks_show])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


plt.tight_layout()

plt.savefig('converge.png', dpi=300) 

plt.show()