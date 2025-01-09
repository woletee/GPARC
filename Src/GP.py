
from Nods import *
from fitness import *
from utils import *
#notes
"input_output_pairs: list of tuples containing both the input and the output grids "
"population size: int representing the number of programs in the population"
"generations: int representing the number of generations or the number of iteratioons the genetic programming will have to run"
"mutation_rate: float representing the probability of mutation"
"crossover_rate: float representing the probability of the crossover"
"max_dpth: int representing the maximum depth of the program tree to be formed"
"population: list of programs representing the colletion of programs generated randomly at the start of the genetic programming"
"all_generations: list of Generation objects containing the best fittness, population(list of programs), mutation_Rate, crossover_rate and the max_depth"
"best_program: instance of the Node class representing the best program found by the genetic programming"
"gen :int representing the current generation number which is the iteration number"
"fittness_score: list of integers representing the fittness score of each program in the generated population"
"total_fittness: int representing the sum of the fittness scores of all the program in the population-actually it is not used anywhere in the code"
"best_fittness:int representing the maximum fittness score in the population"
"best_index: int representing the index of the best program in the population list"
"programs_with_fittness: list of tuples containing both the program and the fittness score of each program in the population"
"selected: list of programs selected from the population based on the torunament selection metod"
"next_generation: list of programs representing the next generation of programs->it is basically the list of evolved programs obtained after finishing single generation process"
"child1, child2: instances of the Node class representing the children programs obtained after crossover of the parent programs"
"parent1, parent2: instances of the Node class representing the parent programs selected for the crossover->basically ranodomly"
"child:instance of the Node class obtained after mutation of the parent program"
"generation_info: instance of the Gneration class containing the best fittness, population, mutation rate , crossover rate and the max depth of single generation"
"all_generations:list of generation_info objects"
"best_fittness: int representing the best fittness score of the best program found by the GP"
"best_program:instance of the best program found by the GP"
# Genetic Programming Algorithm
def genetic_programming(input_output_pairs, population_size, generations, mutation_rate, crossover_rate, max_depth):
    population = [generate_random_program(max_depth, current_depth=0) for _ in range(population_size)]
    all_generations = []
    best_program = None
    for gen in range(generations):
        generateed_programs=[str(program) for program in population]
        fitness_scores = [evaluate_fitness(program, input_output_pairs) for program in population]
        total_fitness = sum(fitness_scores)
        best_fitness = max(fitness_scores)
        best_index = fitness_scores.index(best_fitness)
        best_program = population[best_index]
        programs_with_fitness = list(zip(population, fitness_scores))

        selected = tournament_selection(population,programs_with_fitness, k=3)
        selected_programs=[str(program) for program in selected]
        next_generation = []
        crossover_info = []
        mutation_info=[]
        while len(next_generation) < population_size:
            if random.random() < crossover_rate and len(selected) >= 2:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 , crossover_point1, crossover_point2= crossover(parent1, parent2)
                next_generation.extend([child1, child2])
                crossover_info.append({
                    "parent1": str(parent1),
                    "parent2": str(parent2),
                    "child1": str(child1),
                    "child2": str(child2),
                    "crossover_point1":str(crossover_point1),
                    "crossover_point2":str(crossover_point2)
                })
            else:
                parent = random.choice(selected)
                child, nodes = mutation(parent, max_depth, mutation_rate)
                next_generation.append(child)
                mutation_info.append({
                    "parent":str(parent),
                    "child":str(child),
                    "nodes":str([str(node) for node in nodes])
                })

        population = next_generation[:population_size]
        generation_info = {
            "gen _number": gen+1,
            "best_fitness": best_fitness,
            "population": [f"{str(program)}/{fittness} "for program, fittness in zip( population, fitness_scores)],
            "selected_info":selected_programs,
            "crossover_info": crossover_info,
            "mutation_info":mutation_info,
            "starter_generation_programs": generateed_programs
        }
        all_generations.append(generation_info)

    return best_program, all_generations
