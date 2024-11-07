import random
import copy
import numpy as np
from graphviz import Digraph

# ----------------------------
# 1. Define the Primitives
# ----------------------------

# Functions (Primitives)
def flip_horizontal(grid):
    return np.fliplr(grid)

def flip_vertical(grid):
    return np.flipud(grid)

def rotate_90(grid):
    return np.rot90(grid, k=-1)

def rotate_180(grid):
    return np.rot90(grid, k=2)

def rotate_270(grid):
    return np.rot90(grid, k=1)

def identity(grid):
    return grid

# Terminal
def input_grid():
    pass  # Placeholder; the actual grid will be passed during execution

# List of available functions and terminals
FUNCTIONS = [
    ('flip_horizontal', flip_horizontal, 1),
    ('flip_vertical', flip_vertical, 1),
    ('rotate_90', rotate_90, 1),
    ('rotate_180', rotate_180, 1),
    ('rotate_270', rotate_270, 1),
    ('identity', identity, 1)
]

TERMINALS = [
    ('input_grid', input_grid, 0)
]

# ----------------------------
# 2. Define Program Representation
# ----------------------------

class Node:
    _id_counter = 0  # Static variable to assign unique IDs to nodes

    def __init__(self, value, children=None):
        self.id = Node._id_counter  # Unique identifier for the node
        Node._id_counter += 1
        self.value = value  # Function name or terminal
        self.children = children if children is not None else []

    def __str__(self):
        if self.children:
            return f"{self.value}({', '.join(str(child) for child in self.children)})"
        else:
            return self.value

# ----------------------------
# 3. Implement Program Generation
# ----------------------------

def generate_random_program(max_depth, current_depth=0):
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.3):
        # Choose a terminal
        terminal_name, terminal_func, arity = random.choice(TERMINALS)
        return Node(terminal_name)
    else:
        # Choose a function
        func_name, func, arity = random.choice(FUNCTIONS)
        children = [generate_random_program(max_depth, current_depth + 1) for _ in range(arity)]
        return Node(func_name, children)

# ----------------------------
# 4. Implement Program Execution
# ----------------------------

def execute_program(program, input_grid):
    if program.value in [name for name, _, _ in TERMINALS]:
        if program.value == 'input_grid':
            return input_grid
    else:
        # Find the function
        func = next(func for name, func, _ in FUNCTIONS if name == program.value)
        # Execute child nodes
        args = [execute_program(child, input_grid) for child in program.children]
        return func(*args)

# ----------------------------
# 5. Implement Fitness Evaluation
# ----------------------------

def fitness(program, input_output_pairs):
    total_score = 0
    for input_grid, expected_output in input_output_pairs:
        try:
            result = execute_program(program, input_grid)
            if not isinstance(result, np.ndarray):
                return float('-inf')
            if result.shape != expected_output.shape:
                return float('-inf')
            if np.array_equal(result, expected_output):
                total_score += 1
        except Exception:
            return float('-inf')
    return total_score

# ----------------------------
# 6. Implement Genetic Operators
# ----------------------------

def crossover(parent1, parent2):
    # Deep copy the parents
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    # Get all nodes
    nodes1 = get_all_nodes(child1)
    nodes2 = get_all_nodes(child2)

    # Randomly select crossover points
    crossover_point1 = random.choice(nodes1)
    crossover_point2 = random.choice(nodes2)

    # Swap the subtrees
    crossover_point1.value, crossover_point1.children = crossover_point2.value, crossover_point2.children

    return child1, child2

def mutation(program, max_depth):
    # Deep copy the program
    mutant = copy.deepcopy(program)

    # Get all nodes
    nodes = get_all_nodes(mutant)

    # Randomly select a mutation point
    mutation_point = random.choice(nodes)

    # Replace the subtree at the mutation point
    new_subtree = generate_random_program(max_depth=max_depth, current_depth=0)
    mutation_point.value = new_subtree.value
    mutation_point.children = new_subtree.children

    return mutant

def get_all_nodes(program):
    nodes = [program]
    for child in program.children:
        nodes.extend(get_all_nodes(child))
    return nodes

def selection(population, scores, k=3):
    selected = random.choices(list(zip(population, scores)), k=k)
    selected = sorted(selected, key=lambda x: x[1], reverse=True)
    return selected[0][0]

# ----------------------------
# 7. Implement the GP Algorithm
# ----------------------------

def genetic_programming(input_output_pairs, population_size=50, generations=20, max_depth=5, mutation_rate=0.3, crossover_rate=0.7):
    # Initialize population
    population = [generate_random_program(max_depth) for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate fitness
        scores = [fitness(prog, input_output_pairs) for prog in population]
        best_score = max(scores)
        print(f"Generation {generation}: Best Fitness = {best_score}/{len(input_output_pairs)}")
        if best_score == len(input_output_pairs):
            print("Solution found in generation", generation)
            best_program = population[scores.index(best_score)]
            return best_program

        # Selection and reproduction
        new_population = []
        while len(new_population) < population_size:
            parent1 = selection(population, scores)
            if random.random() < crossover_rate:
                parent2 = selection(population, scores)
                child1, child2 = crossover(parent1, parent2)
                new_population.extend([child1, child2])
            else:
                child = copy.deepcopy(parent1)
                new_population.append(child)
        population = new_population[:population_size]

        # Mutation
        for i in range(len(population)):
            if random.random() < mutation_rate:
                population[i] = mutation(population[i], max_depth)
    # Return the best program found
    scores = [fitness(prog, input_output_pairs) for prog in population]
    best_program = population[scores.index(max(scores))]
    return best_program

# ----------------------------
# 8. Implement Tree Visualization
# ----------------------------

def visualize_tree(program, graph=None):
    if graph is None:
        graph = Digraph()
    # Add the current node
    graph.node(str(program.id), program.value)
    for child in program.children:
        # Add child node and edge
        graph.node(str(child.id), child.value)
        graph.edge(str(program.id), str(child.id))
        # Recursively add the subtree
        visualize_tree(child, graph)
    return graph

def save_tree_as_dot(program, filename):
    # Reset Node IDs to ensure unique IDs for each run
    Node._id_counter = 0
    # Reassign IDs
    reassign_ids(program)
    graph = visualize_tree(program)
    # Save the .dot file
    graph.save(filename + '.dot')
    # Render the graph to a PNG file
    graph.render(filename, format='png', cleanup=False)
    print(f"Program tree saved as {filename}.png and {filename}.dot")

def reassign_ids(program):
    program.id = Node._id_counter
    Node._id_counter += 1
    for child in program.children:
        reassign_ids(child)

# ----------------------------
# 9. Main Function
# ----------------------------

def main():
    # Define training examples
    input_output_pairs = [
        # Example 1
        (
            np.array([[1, 2],
                      [3, 4]]),
            np.array([[2, 1],
                      [4, 3]])
        ),
        # Example 2
        (
            np.array([[0, 1, 2],
                      [3, 4, 5],
                      [6, 7, 8]]),
            np.array([[2, 1, 0],
                      [5, 4, 3],
                      [8, 7, 6]])
        )
    ]

    # Run the GP algorithm
    best_program = genetic_programming(input_output_pairs)

    print("\nBest Program Found:")
    print(str(best_program))

    # Visualize and save the program tree
    save_tree_as_dot(best_program, 'program_tree')

    # Test on a new input
    test_input = np.array([[9, 8, 7],
                           [6, 5, 4],
                           [3, 2, 1]])
    expected_output = np.array([[7, 8, 9],
                                [4, 5, 6],
                                [1, 2, 3]])

    result = execute_program(best_program, test_input)
    print("\nTest Input:")
    print(test_input)
    print("\nExpected Output:")
    print(expected_output)
    print("\nProgram Output:")
    print(result)
    print("\nMatch:", np.array_equal(result, expected_output))

if __name__ == "__main__":
    main()
