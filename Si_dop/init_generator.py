import numpy as np

"""
Generate an initial state

Input: 
length of supercell n, n*n cells are under consideration;

Concentration of the component labeled as "0"

"""

# Generate a random state 
def generate_random_state(n, c):  
    n_atom = 2 * n * n
    structure = np.ones((n, n, 2), dtype=int)   
    n0 = round(c * n_atom)
    indices = np.random.choice(structure.size, n0, replace=False)
    structure.flat[indices] = 0
    return structure


# Generate a specified state 
def generate_state(n, c):  
    n_atom = 2 * n * n
    structure = np.ones((n, n, 2), dtype=int)   
    n0 = round(c * n_atom)
    structure.flat[:n0] = 0
    return structure


initial_structure = generate_state(8, 1/16)
np.save('config_init.npy', initial_structure)