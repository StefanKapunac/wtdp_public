from copy import deepcopy
import random
import statistics
from time import perf_counter
import os
import csv
import sys

from utils import *
 
def grasp(g, initial_solution, initial_solution_value, sorted_external_edges, cutoff):
    solution_value = initial_solution_value
    improved = True
    while improved:
        improved = False
        min_cost = solution_value
        best_improvement = None
        nodes_shuffled = list(initial_solution)
        random.shuffle(nodes_shuffled)
        for node in nodes_shuffled:
            new_num_incorrect_nodes, new_cost = recalc_objective_function_node_removed(initial_solution, 0, solution_value, node, g, sorted_external_edges)
            if new_num_incorrect_nodes == 0 and new_cost < min_cost and random.randrange(0, 100) > cutoff:
                min_cost = new_cost
                best_improvement = node
                improved = True
        if improved:
            solution_value = min_cost
            fix_external_edges_after_node_removed(best_improvement, g, sorted_external_edges)
            initial_solution.remove(best_improvement)
    return initial_solution, solution_value, sorted_external_edges

def same_quality_solution_inside(population, old_population_len, new_solution, new_solution_value):
    for i in range(old_population_len):
        if len(population[i][0]) == len(new_solution) and population[i][1] == new_solution_value:
            return True
    return False

def select(population, size):
    sorted_population = sorted(population, key=lambda x: x[1])
    return sorted_population[:size]

def crossover(g, solution_i, solution_i_value, sorted_external_edges_i, solution_j, cutoff):
    # copy is needed because new individual is created
    new_solution = deepcopy(solution_i)
    new_sorted_external_edges = deepcopy(sorted_external_edges_i)
    new_num_incorrect_nodes = 0
    new_cost = solution_i_value
    for node in solution_j:
        if node not in new_solution:
            new_num_incorrect_nodes, new_cost = recalc_objective_function_node_added(new_solution, new_num_incorrect_nodes, new_cost, node, g, new_sorted_external_edges)
            fix_external_edges_after_node_added(node, g, new_sorted_external_edges)
            new_solution.add(node)

    return grasp(g, new_solution, new_cost, new_sorted_external_edges, cutoff=cutoff)

def mutation(solution, solution_value, sorted_external_edges, m_l, m_u, g, all_nodes_sorted_by_degree):
    new_num_incorrect_nodes = 0
    new_cost = solution_value
    m = random.randint(m_l, m_u)
    chosen_nodes = random.sample(solution, m)
    for node in chosen_nodes:
        new_num_incorrect_nodes, new_cost = recalc_objective_function_node_removed(solution, new_num_incorrect_nodes, new_cost, node, g, sorted_external_edges)
        fix_external_edges_after_node_removed(node, g, sorted_external_edges)
        solution.remove(node)

    # get to feasible solution using 'primal heuristic'
    if new_num_incorrect_nodes == 0:
        return new_cost
    else:
        for node in all_nodes_sorted_by_degree:
            if node not in solution:
                new_num_incorrect_nodes, new_cost = recalc_objective_function_node_added(solution, new_num_incorrect_nodes, new_cost, node, g, sorted_external_edges)
                fix_external_edges_after_node_added(node, g, sorted_external_edges)
                solution.add(node)

                if new_num_incorrect_nodes == 0:
                    return new_cost

    # it should never come to this point, we can always add all nodes in the solution
    return None

def local_search(solution, solution_value, sorted_external_edges, g):
    curr_num_incorrect_nodes = 0
    curr_cost = solution_value
    all_nodes = set(g.nodes)
    improved = True
    while improved:
        improved = False
        nodes_outside = all_nodes.difference(solution)
        for node_out in nodes_outside:
            new_num_incorrect_nodes, new_cost = recalc_objective_function_node_added(solution, curr_num_incorrect_nodes, curr_cost, node_out, g, sorted_external_edges)
            if new_num_incorrect_nodes == 0 and new_cost < curr_cost:
                curr_cost = new_cost
                fix_external_edges_after_node_added(node_out, g, sorted_external_edges)
                solution.add(node_out)
                improved = True
                break
        if not improved:
            nodes_inside = list(solution)
            for node_in in nodes_inside:
                new_num_incorrect_nodes, new_cost = recalc_objective_function_node_removed(solution, curr_num_incorrect_nodes, curr_cost, node_in, g, sorted_external_edges)
                if new_num_incorrect_nodes == 0 and new_cost < curr_cost:
                    curr_cost = new_cost
                    fix_external_edges_after_node_removed(node_in, g, sorted_external_edges)
                    solution.remove(node_in)
                    improved = True
                    break

    return curr_cost

def ga(g, time_seconds, initial_population_size=100, population_size=20, cutoff=30, m_l=1, m_u=4, num_iters=20, grasp_only=False):
    start_time = perf_counter()
    initial_solution = set(g.nodes)
    num_incorrect, initial_solution_value, initial_sorted_external_edges = calc_initial_solution_cost(initial_solution, g)
    population = []
    for i in range(initial_population_size):
        # copy is needed because new individual is created
        solution_i = deepcopy(initial_solution)
        sorted_external_edges_i = deepcopy(initial_sorted_external_edges)
        new_solution, new_solution_value, new_sorted_external_edges = grasp(g, solution_i, initial_solution_value, sorted_external_edges_i, cutoff)
        if not same_quality_solution_inside(population, len(population), new_solution, new_solution_value):
            population.append((new_solution, new_solution_value, new_sorted_external_edges))
        if perf_counter()-start_time>time_seconds:
            break
    print(len(population))
    if grasp_only:
        solution = select(population, 1)
        return solution[0][0], solution[0][1]
    population = select(population, population_size)
    all_nodes_sorted_by_degree = sorted(g.degree, key=lambda x: x[1], reverse=True)
    all_nodes_sorted_by_degree = [x[0] for x in all_nodes_sorted_by_degree]
    # total number of iterations = 20 * (20 * 19 / 2) = 3800
    iter = 0
    while iter < num_iters and perf_counter()-start_time<time_seconds:
        iter+=1
        old_population_len = len(population) # there can be less than 20 individuals in the population because we keep only different
        for i in range(old_population_len):
            for j in range(i + 1, old_population_len):
                new_solution, new_solution_value, new_sorted_external_edges = crossover(g, population[i][0], population[i][1], population[i][2], population[j][0], cutoff)
                new_solution_value = mutation(new_solution, new_solution_value, new_sorted_external_edges, m_l, m_u, g, all_nodes_sorted_by_degree)
                new_solution_value = local_search(new_solution, new_solution_value, new_sorted_external_edges, g)
                if not same_quality_solution_inside(population, old_population_len, new_solution, new_solution_value):
                    population.append((new_solution, new_solution_value, new_sorted_external_edges))
        population = select(population, population_size)
    solution = select(population, 1)
    return solution[0][0], solution[0][1]

def solve(file_path, num_runs,time_seconds, grasp_only):
    g = read_graph_from_file(file_path)
    times = []
    solution_values = []
    num_evaluations = []
    for i in range(num_runs):
        gl.num_evals = 0
        start_time = perf_counter()
        solution, solution_value = ga(g,time_seconds, grasp_only=grasp_only)
        end_time = perf_counter()
        time_elapsed = end_time - start_time
        times.append(time_elapsed)
        solution_values.append(solution_value)
        num_evaluations.append(gl.num_evals)
    if num_runs > 1:
        return statistics.mean(times), statistics.mean(solution_values), statistics.stdev(solution_values), min(solution_values), statistics.mean(num_evaluations)
    else:
        return times[0], solution_values[0], 0, solution_values[0], num_evaluations[0]

if __name__ == '__main__':
    if len(sys.argv)!=6:
        print(f"Incorrect usage, expected parameters are: <DIR_PATH> <NAME_MASK> <GRASP_ONLY> <TIME_SECONDS> <NUM_RUNS>")
        sys.exit()

    random.seed(12345)
    DIR_PATH = sys.argv[1] # '../instances/Ma' for example
    NAME_MASK = sys.argv[2] # '75' for example
    GRASP_ONLY_STR = sys.argv[3] # when true GRASP only, otherwise GRASP+GA
    TIME_SECONDS = int(sys.argv[4]) # 1800
    NUM_RUNS = int(sys.argv[5]) #1

    INSTANCE_NAME = (DIR_PATH+'_'+NAME_MASK).replace('/','_').replace('\\','_').replace('.','')
    if GRASP_ONLY_STR == 'True':
        METHOD = 'grasp'
        GRASP_ONLY = True
    elif GRASP_ONLY_STR =='False':
        METHOD = 'grasp-ga'
        GRASP_ONLY = False
    else:
        raise Exception("Incorrect GRASP_ONLY value, only allowed values are True or False.")
    OUTPUT_PATH = f'output/{METHOD}_{INSTANCE_NAME}_{NUM_RUNS}runs.csv'

    print(OUTPUT_PATH)
    dir_path = os.path.abspath(DIR_PATH)
    with open(OUTPUT_PATH, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['file_path', 'avg_time', 'avg_solution_value', 'stdev_solution_value', 'best_solution_value', 'avg_num_incorrect_nodes', 'avg_num_evaluations'])
    
    i = 0
    for file_path in sorted(os.listdir(dir_path)):
        if NAME_MASK=="" or NAME_MASK in file_path:
            abs_path = os.path.join(dir_path, file_path)
            avg_time, avg_solution_value, stdev_solution_value, best_solution_value, avg_num_evaluations = solve(abs_path, NUM_RUNS,TIME_SECONDS, grasp_only=GRASP_ONLY)
            print(f'{i}: {file_path}, {avg_time}, {avg_solution_value}, {best_solution_value}, {avg_num_evaluations}')
            with open(OUTPUT_PATH, 'a', newline='') as out:
                writer = csv.writer(out)
                writer.writerow([file_path, f'{avg_time:.2f}', f'{avg_solution_value:.2f}', f'{stdev_solution_value:.2f}', f'{best_solution_value}', f'{avg_num_evaluations:.2f}'])
            i += 1