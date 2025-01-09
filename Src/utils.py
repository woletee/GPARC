import copy
import random
from Nods import *

#Note
"crossover: function to perform crossover between two parent progams"
"mutation: function to perform mutation on a program"
"tournamnet_selection: function to perform tournamnet selection on the population"

def crossover(parent1, parent2):
    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    nodes1, nodes2 = get_all_nodes(child1), get_all_nodes(child2)
    if not nodes1 or not nodes2:
        return child1, child2
    crossover_point1 = random.choice(nodes1)
    crossover_point2 = random.choice(nodes2)
    crossover_point1.value, crossover_point1.children, crossover_point2.value, crossover_point2.children = \
        crossover_point2.value, crossover_point2.children, crossover_point1.value, crossover_point1.children
    return child1, child2, crossover_point1, crossover_point2

def mutation(program, max_depth, mutation_rate):
    mutant = copy.deepcopy(program)
    nodes = get_all_nodes(mutant)
    for node in nodes:
        if random.random() < mutation_rate:
            new_subtree = generate_random_program(max_depth=max_depth, current_depth=0)
            node.value = new_subtree.value
            node.children = new_subtree.children
    return mutant, nodes

def tournament_selection(population,programs_with_fittness, k):
    selected = []
    population_size = len(population)
    for _ in range(population_size):
        participants = random.sample(programs_with_fittness, k)
        winner = max(participants, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected
