import numpy as np
"program:instance of the node class-sequence of functions or onyl terminal node is also possible"
"input_output_pairs: list of tuples containing both the input and the output grids"
"fitness: int representing the fittness score of the program"
"input_grid: numpy array representing the input grid"
"expected_output: numpy array representing the ground truth ouutput grid"
"output: numpy array representing the output grid obtained by applying the program on the input grid"
"evaluate_fittness: function to evaluate the fittness of the program"
"evaluate: Node class method to evaluate the program tree in depth first manner"
def evaluate_fitness(program, input_output_pairs):
    fitness = 0
    for input_grid, expected_output in input_output_pairs:
        output = program.evaluate(input_grid)
        if np.array_equal(output, expected_output):
            fitness += 1
    return fitness

#how can we modify the fittness function to measure the edit distance 
