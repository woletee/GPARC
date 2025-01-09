import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from task_loader import *
from visualization import *
from GP import *
#Notes 
" directory_path:str keeps the path containing the tasks"
"tasks:list of tuples containing the the task filename and the task data "
"task_idx:index of the current task in the current iteration"
"task_file name:str name of the task file being processed"
"task_data: dictionary containing the task data basically the input and the output grids"
"input_output_pairs: list of tuples containing the input and the output grids"
"best program: Instance of the Node class->object  of the best program which is found by using the genetic programming -"
"all_generations: Instance of the Generation Class->list of Generation objects containing the best fitness, population , mutatuion rate, crossover rate and the max depth"
"input_grid: numpy arrau represeting the input grid for the task"
"expected_output: numpy array representing the expected output grid for the task"
"predicted_output: numpy array representing the predicted output grid for the task obtained by applying the best program obtained from the genetic programming on the input grid"
"results: list of dictionaries where each dictionary contains details about a specific task information including the task filename, the best program, and all generaion data "
"result_file: file object to write the results of the genetic programming on the tasks to a json file"




" directory_path:str keeps the path containing the tasks"
"tasks:list of tuples containing the the task filename and the task data "
"task_idx:index of the current task in the current iteration"
"task_file name:str name of the task file being processed"
"task_data: dictionary containing the task data basically the input and the output grids"
"input_output_pairs: list of tuples containing the input and the output grids"
"best program: Instance of the Node class->object  of the best program which is found by using the genetic programming -"
"all_generations: Instance of the Generation Class->list of Generation objects containing the best fitness, population , mutatuion rate, crossover rate and the max depth"
"input_grid: numpy arrau represeting the input grid for the task"
"expected_output: numpy array representing the expected output grid for the task"
"predicted_output: numpy array representing the predicted output grid for the task obtained by applying the best program obtained from the genetic programming on the input grid"
"results: list of dictionaries where each dictionary contains details about a specific task information including the task filename, the best program, and all generaion data "
"result_file: file object to write the results of the genetic programming on the tasks to a json file"



def genetic_programming_to_tasks(directory_path):
    tasks = load_tasks_from_directory(directory_path)
    results = []

    for task_index, (task_filename, task_data) in enumerate(tasks):
        print(f"Processing Task: {task_filename}")
        input_output_pairs = prepare_input_output_pairs(task_data)
        best_program, all_generations = genetic_programming(
            input_output_pairs=input_output_pairs,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            max_depth=max_depth
        )
        for i, (input_grid, expected_output) in enumerate(input_output_pairs):
            predicted_output = best_program.evaluate(input_grid)
            plot_comparison(input_grid, expected_output, predicted_output, task_number=f"{task_index + 1}-{i + 1}")
        results.append({
            "task_filename": task_filename,
            "best_program": str(best_program),
            "all_generations": all_generations
        })

        print(f"Completed Task: {task_filename}")
        print(f"Best Fitness: {all_generations[-1]['best_fitness']}")
        print(f"Best Program: {best_program}\n")

    save_results_to_json(results, 'genetic_programming_results.json')
    return results

if __name__ == "__main__":
    directory_path = r"C:\Users\gebre\Downloads\GPARC_3\GPARC (2)\GPARC\GPARC\Abstraction_and_reasoning_corpus\training"
    population_size = 10
    generations = 5
    mutation_rate = 0.1
    crossover_rate = 0.7
    max_depth = 5
    results = genetic_programming_to_tasks(
        directory_path=directory_path
    )
    with open("genetic_programming_results.json", "w") as result_file:
        json.dump(results, result_file, indent=4)

