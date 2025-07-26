#!/usr/bin/env python
import random
import math
import numpy as np
import copy
import cProfile
import pstats
import io
import os


"""
Standard MCTS a 2D graphene with Si doping

2025/7/25
"""


"""
Tersoff potential:
"0" = "Si"
"1" = "C"
"""

# Truncation potential
def fc(r, R, D):     
    if r < R - D:
        fc = 1
    else:
        fc = 1/2 + 1/2 * math.cos(math.pi * (r - R + D) / (2 * D))
    return fc

# Repulsion potential
def fR(r, occupancy1, occupancy2): 
    A = A_list[occupancy1, occupancy2]
    lambda_ = lambda_list[occupancy1, occupancy2]
    fR = A * math.exp(-lambda_ * r)
    return fR

# Attraction potential
def fA(r, occupancy1, occupancy2):
    B = B_list[occupancy1, occupancy2]
    mu = mu_list[occupancy1, occupancy2]
    fA = - B * math.exp(-mu * r)
    return fA

# Angel fucntion
def Angle(vec1, vec2, occupancy):
    # input: two vectors
    # occupancy: type of central atom
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    c = c_list[occupancy]
    d = d_list[occupancy]
    h = h_list[occupancy]
    g = 1 + (c / d) ** 2 - c ** 2 / (d ** 2 + (h - cos_theta) ** 2) 
    return g


# Parameter lists for Tensoff potential
# data from Tersoff. Chemical order in amorphous silicon carbide. PRB, 1994.
# double-index parameters
R_list = np.array([[2.85, 2.35726], [2.35726, 1.95]])
D_list = np.array([[0.15, 0.15271], [0.15271, 0.15]])
A_list = np.array([[1830.8, 1681.731], [1681.731, 1544.8]]) 
lambda_list = np.array([[2.4799, 2.9726], [2.9726, 3.4653]]) 
B_list = np.array([[471.18, 432.154], [432.154, 389.63]]) 
mu_list = np.array([[1.7322, 2.0193], [2.0193, 2.3064]]) 

# single-index parameters
c_list = np.array([100390, 19981])
d_list = np.array([16.217, 7.034])
h_list = np.array([-0.59825, -0.33953])
beta_list = np.array([1.1e-6, 4.1612e-6])
n_list = np.array([0.78734, 0.99054])

# Lattice information
#lattice_const = 1.42
lattice_const = 1.56

x_basis = np.array([3, np.sqrt(3)]) * lattice_const / 2
y_basis = np.array([3, -np.sqrt(3)]) * lattice_const / 2
# sublattice
sub_basis = np.array([[lattice_const, 0], [2 * lattice_const, 0]])



# Create a decorator to count method invocation
def call_counter(func):
    def wrapper(self, *args, **kwargs):
        self.__class__._total_calls += 1
        return func(self, *args, **kwargs)
    return wrapper

m = 3
n = 2 ** m    
n_atom = 2 * n ** 2
# consider a 2D system with n*n unit cells (but 2*n*n atoms)
concentration = 1/8
n1 = round(concentration * n_atom)  # corresponding to "0" or "Si"
n2 = int(n_atom - n1)    # corresponding to "1" or "C"   


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


# Generate a random state 
# Note that sublattice is included in tri-index
VECTOR = np.zeros((n, n, 2), dtype=int)
indices = np.random.choice(n_atom, n2, replace=False)
for index in indices:
    i, j, k = OneToThree_list[index]
    VECTOR[i, j, k] = 1


coordinate_list = np.zeros((n_atom, 2))
for i in range(n):
    for j in range(n):
        for k in range(2):
            index_1d = ThreeToOne_list[i, j, k]
            coordinate_list[index_1d, :] = i * x_basis + j * y_basis + sub_basis[k, :]


# Create a list to store relative positions after appropriate periodic shift
shift_list = np.zeros((2 * n - 1, 2 * n - 1, 2, 2, 2))
for dx in range(-n + 1, n):
    x_translations = np.array([dx, dx + n, dx - n])
    for dy in range(-n + 1, n):    
        y_translations = np.array([dy, dy + n, dy - n])
        for sub_idx1 in range(2):
            for sub_idx2 in range(2):
                # Calculate all possible distances
                x_translated = x_translations[:, None] * x_basis[0] + y_translations[None, :] * y_basis[0] \
                    + sub_basis[sub_idx1, 0] - sub_basis[sub_idx2, 0]
                y_translated = x_translations[:, None] * x_basis[1] + y_translations[None, :] * y_basis[1] \
                    + sub_basis[sub_idx1, 1] - sub_basis[sub_idx2, 1]                    
                distances = np.sqrt(x_translated ** 2 + y_translated ** 2)
                min_position = np.unravel_index(np.argmin(distances), distances.shape)
                x_shift = x_translated[min_position]
                y_shift = y_translated[min_position]
                shift_list[dx + n - 1, dy + n - 1, sub_idx1, sub_idx2] = np.array([x_shift, y_shift])


# Create a set to store the indices of neighbouring atoms
radius_max = np.max(R_list + D_list)
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


class State():
    # Class attribute, used to record the total number of method (objective function) invocations
    _total_calls = 0  

    # Initialize
    def __init__(self, vector = VECTOR):
        self.vector = vector
        # state.exchange_set is updated on the fly
        self.exchange_set = self.find_exchange_atom()


    # Find the index sets for each occupied state
    def find_elements_with_occupancy(self, occupancy):
        elements = []
        for i in range(n):
            for j in range(n):
                for k in range(2):
                    if self.vector[i, j, k] == occupancy:
                        elements.append([i, j, k])
        elements = np.array(elements)
        return elements
    

    # Generate a pair set containing all possible pairs of 
    # positions to exchange
    def find_exchange_atom(self):
        occupancy1 = 0
        occupancy2 = 1
        element1 = self.find_elements_with_occupancy(occupancy1)
        element2 = self.find_elements_with_occupancy(occupancy2)
        pair_set = [[i, j] for i in element1 for j in element2]
        np.random.shuffle(pair_set)
        return pair_set

    # Move to next state 
    def next_state(self):
        pair_set = self.exchange_set
        num_set = len(pair_set)
        i = random.randint(0, num_set - 1)
        new_vector = copy.deepcopy(self.vector)
        indice1, indice2 = pair_set[i]            
        new_vector[indice1[0], indice1[1], indice1[2]], new_vector[indice2[0], indice2[1], indice2[2]] = \
            new_vector[indice2[0], indice2[1], indice2[2]], new_vector[indice1[0], indice1[1], indice1[2]]
        next = State(vector = new_vector) 
        del self.exchange_set[i]
        return next


    # Bond order function
    # Note that Bond order function depends the order of indices
    def Bond_order(self, index1, index2):
        occupancy1 = self.vector[index1[0], index1[1], index1[2]] 
        beta = beta_list[occupancy1]
        n_ = n_list[occupancy1]
        sum = 0.0
        index1_1d = ThreeToOne_list[index1[0], index1[1], index1[2]]  # central atom
        index2_1d = ThreeToOne_list[index2[0], index2[1], index2[2]]   
        vec1 = shift_list[index2[0] - index1[0] + n - 1, index2[1] - index1[1] + n - 1, index2[2], index1[2]]
        for j in range(num_neighbour):
            index_1d = neighbour_atoms[index1_1d, j, 0]   
            r = neighbour_atoms[index1_1d, j, 1]                 
            if index_1d != index1_1d and index_1d != index2_1d:
                i, j, k = OneToThree_list[index_1d]    
                occupancy3 = self.vector[i, j, k]
                dx = i - index1[0]
                dy = j - index1[1]
                vec2 = shift_list[dx + n - 1, dy + n - 1, k, index1[2]]
                R = R_list[occupancy1, occupancy3]
                D = D_list[occupancy1, occupancy3]
                if r <= R + D:
                    sum += fc(r, R, D) * Angle(vec1, vec2, occupancy1)
        bo = (1 + (beta * sum) ** n_) ** (-1 / (2 * n_))
        return bo
    
    # Energy per particle based on Tersoff potential
    # Wrap the objective function using a decorator
    @call_counter
    def energy(self):
        E = 0.0
        for index1_1d in range(n_atom):
            index1 = OneToThree_list[index1_1d]
            occupancy1 = self.vector[index1[0], index1[1], index1[2]]
            for j in range(num_neighbour):
                index2_1d = neighbour_atoms[index1_1d, j, 0]
                index2_1d = int(index2_1d)
                r = neighbour_atoms[index1_1d, j, 1]
                index2 = OneToThree_list[index2_1d]                
                occupancy2 = self.vector[index2[0], index2[1], index2[2]]
                R = R_list[occupancy1, occupancy2]
                D = D_list[occupancy1, occupancy2]
                if r <= R + D:
                    E += fc(r, R, D) * (fR(r, occupancy1, occupancy2) + fA(r, occupancy1, occupancy2))  
                    #E += fc(r, R, D) * (fR(r, occupancy1, occupancy2) + self.Bond_order(index1, index2) \
                    #                    * fA(r, occupancy1, occupancy2))                        
        E = E / (2 * n_atom)
        E_round = np.round(E, 12)   # rounding error
        return E_round
    

    @classmethod
    def get_total_calls(cls):
        return cls._total_calls
    

    # Print the state information
    def __repr__(self):
        s = "vector: %s" % str(self.vector)
        return s
    
    # Define the same state
    def __eq__(self, other):
        if isinstance(self.vector, np.ndarray) and isinstance(other.vector, np.ndarray):
                return np.array_equal(self.vector, other.vector)
        

class Node():
    def __init__(self, state, parent = None):
        self.visits = 0
        self.energy = state.energy()
        self.value = -self.energy
        self.state = state
        self.children = [] 
        self.parent = parent	
    
    # Add a child node
    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)
    
    # Check if the node is fully expanded
    def fully_expanded(self, layer_width):        
        if len(self.children) == layer_width or len(self.state.exchange_set) == 0:
            return True
        return False
    
    # Print the node information
    def __repr__(self):
        s = "value: %f; energy: %.16f; children: %d; visits: %d" \
            % (self.value, self.energy, len(self.children), self.visits)
        return s
    
    # Define the same node
    def __eq__(self, other):
        return self.state == other.state
    



# whole MCTS process
def MCTS(times:int, root, layer_width:int): 

    t = []  # record the times of function evaluation
    v = []  # record the minimum function value
    v_current = [] # record the current function value
    K = 0  # number of iterations achieving the current optimal energy

    f_opt = root.energy
    t.append(0)
    v.append(f_opt)
    v_current.append(f_opt)

    for K in range(times):
        scalar = max(0.1 * np.exp(-0.001 * K), 0.01) 
        # set the scalar adaptively
        node_expandable = TRAVERSE(root, layer_width, scalar)
        leaf = EXPAND(node_expandable)                
        BACKPROPAGATE(leaf, leaf.value)
        if leaf.energy < f_opt:
            vector_opt = copy.deepcopy(leaf.state.vector)
            f_opt = leaf.energy
            print("--------------------------------")	
            print("K = %d" % K)
            #print("scalar = %.6f" % scalar)
            print("f_opt = %.6f" % f_opt)
            
        """
        if K % 500 == 0 and K > 1: 
            index = K // 500
            np.save(f"data/config_mcts{index}.npy", vector_opt)
        """

        v_current.append(leaf.energy) # current energy
        v.append(f_opt)    # optimal energy
        # Get total number of calls of objective function
        total_calls = State.get_total_calls()
        t.append(total_calls)

    data = np.column_stack((t, v, v_current))
    return data, vector_opt


# Traverse to a node, which is urgent to expand
# a not fully expanded node means the number of child nodes not greater than layer_width
def TRAVERSE(node, layer_width, scalar):
    while node.fully_expanded(layer_width):
        node = BESTCHILD(node, scalar)
    return node

# Expand a new node
def EXPAND(node):     
    new_state = node.state.next_state()
    node.add_child(new_state)
    return node.children[-1]

# Backpropagation, update all the visited nodes
def BACKPROPAGATE(node, reward):
    while node is not None:
        node.value = max(reward, node.value)
        node.visits += 1        
        node = node.parent
    return

# Find the best child node
def BESTCHILD(node, scalar):
    bestchildren = []      
    for child in node.children:
        exploit = child.value
        explore = np.sqrt(2.0 * math.log(node.visits) / float(child.visits))	
        score = exploit + scalar * explore     
        if not bestchildren:
            bestscore = score
        if score == bestscore:
            bestchildren.append(child)
        elif score > bestscore: 
            bestchildren = [child]
            bestscore = score 
    node_best = random.choice(bestchildren)                       
    return node_best




if __name__ == "__main__": 
    # Profiling
    pr = cProfile.Profile()
    pr.enable()  

    output_path = 'data/result_mcts.txt'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    init_vector = np.load('config_init.npy')
    root = Node(State(vector=init_vector))
    data, vector_opt = MCTS(10000, root, 50)
    # MCTS(times, root, layer_width)

    np.savetxt(output_path, data, fmt=['%d', '%.12f', '%.12f'], delimiter='\t')
    np.save('data/config_mcts_final.npy', vector_opt)

    pr.disable()   

    # Create a StringIO object to store performance analysis results
    s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')  # sorted by cumulative time
    ps = pstats.Stats(pr, stream=s).sort_stats('time')  # sorted by total time
 
    # Only show first 10 lines
    ps.print_stats(10)   
    # Print results
    print(s.getvalue())
	
	