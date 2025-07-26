import numpy as np

# Generate a random state with n1 atoms of type 1 and n2 atoms of type 2
def generate_random_state(n, n1, n2):
    n_atom = 2 * n * n    
    structure = np.zeros((n, n, 2), dtype=int)
    indices = np.random.choice(n_atom, n1 + n2, replace=False)    
    # Assign 1 to the first n1 randomly chosen positions
    structure.flat[indices[:n1]] = 1
    # Assign 2 to the next n2 positions
    structure.flat[indices[n1:]] = 2
    return structure


# Generate a specified state with first n1 atoms as 1 and next n2 atoms as 2
def generate_state(n, n1, n2):
    n_atom = 2 * n * n    
    structure = np.zeros((n, n, 2), dtype=int)
    # Assign 1 to the first n1 positions
    structure.flat[:n1] = 1
    # Assign 2 to the next n2 positions
    structure.flat[n1:n1+n2] = 2    
    return structure

# B corresponding to 1, N corresponding to 2
initial_structure = generate_state(8, 3, 6)
np.save('config_init_triple.npy', initial_structure)