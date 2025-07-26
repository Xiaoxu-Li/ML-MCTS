import numpy as np
import random
import copy
import math
import os
from contextlib import contextmanager
import cProfile
import pstats
import io
import mcts_triple



"""
Metropolis for 2D graphene with B and N doping

2025/7/25
"""

K_BT = 0.01  # Boltzman constant


# Find the index sets for each occupied state
def find_elements_with_occupancy(data_structure, occupancy):
    n = data_structure.shape[0]
    element = []
    for i in range(n):
        for j in range(n):
            for k in range(2):
                if data_structure[i, j, k] == occupancy:
                    element.append([i, j, k])
    return element


# Exchange the occupancy corresponding to index1 and index2
def change_element(data_structure, index1, index2):
    if len(index1) != 3 or len(index2) != 3:
        raise ValueError("Both index1 and index2 must have three elements each.")
    data_structure[index1[0], index1[1], index1[2]], data_structure[index2[0], index2[1], index2[2]] = \
        data_structure[index2[0], index2[1], index2[2]], data_structure[index1[0], index1[1], index1[2]]
    

# Calculate the value of the Boltzmann distribution
def boltzmann_distribution(F_new, F_old, K_BT): 
    y = np.exp((F_old - F_new) / K_BT)
    return y


"""
Metropolis sampling

Begin the Metropolis sampling, note that n is the number of our sample, 
n1 and n2 are the number of Atom 1 and Atom 2, respectively, and N is the number of Monte-Carlo times.
"""

# Generate a random state 
def generate_random_state(n, n1, n2):  
    n_atom = 2 * n * n
    structure = np.zeros((n, n, 2), dtype=int)
    # Randomly select (n1 + n2) unique indices from n_atom
    indices = np.random.choice(n_atom, n1+n2, replace=False)
    # Set value 1 at the first n1 selected indices
    for index in indices[:n1]:
        structure.flat[index] = 1
    # Set value 2 at the next n2 selected indices
    for index in indices[n1:]:
        structure.flat[index] = 2
    return structure


def metropolis_algorithm(n, N, initial_structure):

    n_atom = 2 * n * n

    # Create lists between 1D index to 3D index, to save computational cost
    OneToThree_list = []
    for index_1d in range(n_atom):
        i, j, k = np.unravel_index(index_1d, (n, n, 2))
        OneToThree_list.append([i, j, k]) 
    OneToThree_list = np.array(OneToThree_list)

    ThreeToOne_list = np.zeros((n, n, 2), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(2):
                index_1d = np.ravel_multi_index((i, j, k), (n, n, 2))
                ThreeToOne_list[i, j, k] = index_1d

    
    # Create a list to store relative positions after appropriate periodic shift
    shift_list = np.zeros((2 * n - 1, 2 * n - 1, 2, 2, 2))
    for dx in range(-n + 1, n):
        x_translations = np.array([dx, dx + n, dx - n])
        for dy in range(-n + 1, n):    
            y_translations = np.array([dy, dy + n, dy - n])
            for sub_idx1 in range(2):
                for sub_idx2 in range(2):
                    # Calculate all possible distances
                    x_translated = x_translations[:, None] * mcts_triple.x_basis[0] \
                        + y_translations[None, :] * mcts_triple.y_basis[0] \
                        + mcts_triple.sub_basis[sub_idx1, 0] - mcts_triple.sub_basis[sub_idx2, 0]
                    y_translated = x_translations[:, None] * mcts_triple.x_basis[1] \
                        + y_translations[None, :] * mcts_triple.y_basis[1] \
                        + mcts_triple.sub_basis[sub_idx1, 1] - mcts_triple.sub_basis[sub_idx2, 1]                    
                    distances = np.sqrt(x_translated ** 2 + y_translated ** 2)
                    min_position = np.unravel_index(np.argmin(distances), distances.shape)
                    x_shift = x_translated[min_position]
                    y_shift = y_translated[min_position]
                    shift_list[dx + n - 1, dy + n - 1, sub_idx1, sub_idx2] = np.array([x_shift, y_shift])


    # Coordinate list for lattice
    coordinate_list = np.zeros((n_atom, 2))
    for i in range(n):
        for j in range(n):
            for k in range(2):
                index_1d = ThreeToOne_list[i, j, k]
                coordinate_list[index_1d, :] = i * mcts_triple.x_basis \
                    + j * mcts_triple.y_basis + mcts_triple.sub_basis[k, :]
                

    # Create a set to store the indices of neighbouring atoms
    radius_max = np.max(mcts_triple.R_list + mcts_triple.D_list) 
    num_neighbour = 0
    index1 = OneToThree_list[0]
    for index2_1d in range(1, n_atom):
        index2 = OneToThree_list[index2_1d]     
        dx = index1[0] - index2[0]
        dy = index1[1] - index2[1]
        vec = shift_list[dx + n - 1, dy + n - 1, index1[2], index2[2]]
        r = np.linalg.norm(vec)  
        if r <= radius_max:
            num_neighbour += 1

    neighbour_atoms = np.zeros((n_atom, num_neighbour, 2))  
    for index1_1d in range(n_atom):
        index1 = OneToThree_list[index1_1d]
        t = 0
        for index2_1d in range(n_atom):
            # neighbour but without itself
            if index2_1d != index1_1d:                    
                index2 = OneToThree_list[index2_1d]                    
                dx = index1[0] - index2[0]
                dy = index1[1] - index2[1]
                vec = shift_list[dx + n - 1, dy + n - 1, index1[2], index2[2]]
                r = np.linalg.norm(vec)  
                if r <= radius_max:
                    # store the index paired with the distance 
                    neighbour_atoms[index1_1d, t, 0] = index2_1d
                    neighbour_atoms[index1_1d, t, 1] = r
                    t += 1
    

    # Bond order function
    # Note that Bond order function depends the order of indices
    def Bond_order(structure, index1, index2):
        occupancy1 = structure[index1[0], index1[1], index1[2]] 
        beta = mcts_triple.beta_list[occupancy1]
        n_ = mcts_triple.n_list[occupancy1]
        sum = 0.0
        index1_1d = ThreeToOne_list[index1[0], index1[1], index1[2]]  # central atom
        index2_1d = ThreeToOne_list[index2[0], index2[1], index2[2]]   
        vec1 = shift_list[index2[0] - index1[0] + n - 1, index2[1] - index1[1] + n - 1, index2[2], index1[2]]
        for index_1d in range(n_atom):                   
            if index_1d != index1_1d and index_1d != index2_1d:
                i, j, k = OneToThree_list[index_1d]    
                occupancy3 = structure[i, j, k]
                dx = i - index1[0]
                dy = j - index1[1]
                vec2 = shift_list[dx + n - 1, dy + n - 1, k, index1[2]]
                r = np.linalg.norm(vec2)
                R = mcts_triple.R_list[occupancy1, occupancy3]
                D = mcts_triple.D_list[occupancy1, occupancy3]
                if r <= R + D:
                    sum += mcts_triple.fc(r, R, D) * mcts_triple.Angle(vec1, vec2, occupancy1)
        bo = (1 + (beta * sum) ** n_) ** (-1 / (2 * n_))
        return bo
    
    
    # Energy per particle based on Tersoff potential
    def energy(structure):
        E = 0.0
        for index1_1d in range(n_atom):
            index1 = OneToThree_list[index1_1d]
            occupancy1 = structure[index1[0], index1[1], index1[2]]
            for j in range(num_neighbour):
                index2_1d = neighbour_atoms[index1_1d, j, 0]
                index2_1d = int(index2_1d)
                r = neighbour_atoms[index1_1d, j, 1]    
                index2 = OneToThree_list[index2_1d]                    
                occupancy2 = structure[index2[0], index2[1], index2[2]]
                R = mcts_triple.R_list[occupancy1, occupancy2]
                D = mcts_triple.D_list[occupancy1, occupancy2]
                if r <= R + D:
                    E += mcts_triple.fc(r, R, D) * (mcts_triple.fR(r, occupancy1, occupancy2) + \
                        mcts_triple.fA(r, occupancy1, occupancy2))
                    #E += mcts_triple.fc(r, R, D) * (mcts_triple.fR(r, occupancy1, occupancy2) + \
                    #   Bond_order(structure, index1, index2) * mcts_triple.fA(r, occupancy1, occupancy2))
        E = E / (2 * n_atom)
        E_round = np.round(E, 12)   # rounding error
        return E_round

    structure = initial_structure
    f_old = energy(structure)

    
    # Record the evaluation time of objective function
    i = 0
    t = []  # times of evaluation
    m = []  # minimum value in each loop

    value = []  # current value in each loop

    t.append(0)
    m.append(f_old)
    value.append(f_old)

    opt_structure = copy.deepcopy(structure)
    value_min = f_old

    output_path = 'data/result_mh.txt'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # MC START
    while(N > 0):
        N = N - 1
        data_structure = copy.deepcopy(structure)

        # Exchange a pair of indices with randomly chosen different occupancies
        occupancies = [0, 1, 2]
        # Randomly choose two different occupancy values
        occ1, occ2 = random.sample(occupancies, 2)
        set1 = find_elements_with_occupancy(data_structure, occ1)
        set2 = find_elements_with_occupancy(data_structure, occ2)
        i1 = random.randint(0, len(set1) - 1)
        i2 = random.randint(0, len(set2) - 1)
        index1 = set1[i1]
        index2 = set2[i2]
        change_element(data_structure, index1, index2)

        # Evaluate the energy of new structure
        f_new = energy(data_structure)
        i = i + 1

        # update the lowest energy and corresponding structure
        if f_new < value_min:
            value_min = f_new
            opt_structure = copy.deepcopy(data_structure) 

        t.append(i)
        m.append(value_min)

        # Metroplois algorithm
        if f_new < f_old:
            structure = copy.deepcopy(data_structure)
            f = f_new
        else:
            acceptance_probability = boltzmann_distribution(f_new, f_old, K_BT)
            P = random.random()
            if P < acceptance_probability:
                structure = copy.deepcopy(data_structure)
                f = f_new
            else:
                f = f_old
                value.append(f)
                continue
        f_old = f
        value.append(f)

        print("----------------------")
        print("Iteration %d" % i)
        print("The current energy is %.10f" % f)
        print("The current optimal energy is %.10f" % value_min)
        print("----------------------")

        
        if i % 200 == 0 and k > 1: 
            index = k // 200
            np.save(f"data/config_mh{index}.npy", opt_structure)
        

    data = np.column_stack((t, m, value))
    # Write data into the file
    np.savetxt(output_path, data, fmt=['%d', '%.16f', '%.16f'], delimiter='\t')
    np.save('data/config_mh_final.npy', opt_structure)

    assert energy(opt_structure) == value_min, "opt_structure does not match value_min!"
    return value


# Run example
if __name__ == "__main__": 
    # Profiling
    pr = cProfile.Profile()
    pr.enable()  
    initial_structure = np.load('config_init_triple.npy')
    value = metropolis_algorithm(8, 10000, initial_structure)
    pr.disable()   

    # Create a StringIO object to store performance analysis results
    s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')  # sorted by cumulative time
    ps = pstats.Stats(pr, stream=s).sort_stats('time')  # sorted by total time

    # Only show first 10 lines
    ps.print_stats(10)   
    # Print results
    print(s.getvalue())

