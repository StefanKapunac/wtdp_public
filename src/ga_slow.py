import random
import os
import csv
import statistics
from time import perf_counter

from utils import read_graph_from_file

def objective_function(solution, g):
    res = 0
    # node weights
    node_weights = 0
    for i in solution:
        node_weights += g.nodes[i]['weight']
    # internal edges
    internal_edges = 0
    for i in solution:
        for j in solution:
            if j > i and j in g[i]:
                internal_edges += g[i][j]['weight']
    # external edges
    external_edges = 0
    for i in set(g.nodes).difference(solution):
        external_edges += min((j[1]['weight'] for j in g[i].items() if j[0] in solution))

    res = node_weights + internal_edges + external_edges
    return res

def is_total_dominating_set(g, removed_node, num_neighbors_in_solution):
    for i, k in enumerate(num_neighbors_in_solution):
        k_new = k
        if i in g[removed_node]:
            k_new -= 1
        if k_new < 1:
            return False
    return True

def get_num_neighbors_in_solution(solution, g):
    num_neighbors_in_solution = [0 for _ in g.nodes]
    for i in solution:
        for j in g[i]:
            num_neighbors_in_solution[j] += 1
    return num_neighbors_in_solution

def grasp(g, initial_solution, cutoff=30):
    all_nodes = set(g.nodes)
    solution = initial_solution
    scores = [float('-inf') for _ in g.nodes]
    num_neighbors_in_solution = get_num_neighbors_in_solution(solution, g)

    improving_move_exists = True
    while improving_move_exists:
        improving_move_exists = False
        vertex_to_remove = None
        best_score = 0
        for i in solution:
            if not is_total_dominating_set(g, i, num_neighbors_in_solution):
                continue
            if scores[i] == float('-inf'):
                scores[i] = g.nodes[i]['weight']
                for j in set(g[i]).intersection(solution):
                    scores[i] += g.edges[i,j]['weight']
                neighbors_of_i_in_solution = [n for n in g[i].items() if n[0] in solution]
                w_star = min(neighbors_of_i_in_solution, key=lambda x: x[1]['weight'])[1]['weight']
                scores[i] -= w_star
                for j in set(g[i]).intersection(all_nodes.difference(solution)):
                    # is i the closest neighbor of j
                    neighbors_of_j_in_solution = [n for n in g[j].items() if n[0] in solution]
                    min_cost = min(neighbors_of_j_in_solution, key=lambda x: x[1]['weight'])[1]['weight']
                    nodes_with_min_cost = [n[0] for n in neighbors_of_j_in_solution if n[1]['weight'] == min_cost]
                    if i in nodes_with_min_cost:
                        neighbors_of_j_in_solution_without_i = [n for n in g[j].items() if n[0] in solution and n[0] != i]
                        wj_star = min(neighbors_of_j_in_solution_without_i, key=lambda x: x[1]['weight'])[1]['weight']
                        scores[i] -= wj_star
            if scores[i] > best_score and random.randrange(0, 100) > cutoff:
                best_score = scores[i]
                vertex_to_remove = i
                improving_move_exists = True
        if improving_move_exists:
            solution.remove(vertex_to_remove)
            for j in g[vertex_to_remove]:
                num_neighbors_in_solution[j] -= 1
                # this is not in the pseudocode, but is in the text
                scores[j] = float('-inf')
                for k in g[j]:
                    scores[k] = float('-inf')

    return solution, objective_function(solution, g)

def same_quality_solution_inside(population, new_solution, new_solution_value):
    for solution, solution_value in population:
        if len(solution) == len(new_solution) and solution_value == new_solution_value:
            return True
    return False

def select(population, size):
    sorted_population = sorted(population, key=lambda x: x[1])
    return sorted_population[:size]

def crossover(g, solution_i, solution_j, cutoff):
    return grasp(g, initial_solution=solution_i.union(solution_j), cutoff=cutoff)

def is_feasible(num_neighbors_in_solution):
    for i in num_neighbors_in_solution:
        if i == 0:
            return False
    return True

def mutation(new_solution, m_l, m_u, g):
    m = random.randint(m_l, m_u)
    chosen_nodes = random.sample(new_solution, m)
    for node in chosen_nodes:
        new_solution.remove(node)
    
    num_neighbors_in_solution = get_num_neighbors_in_solution(new_solution, g)
    if is_feasible(num_neighbors_in_solution):
        return new_solution 
    else:
        # sort (by degree) the nodes outside the solution, then add them until the solution is feasible
        all_nodes_sorted = sorted(g.degree, key=lambda x: x[1], reverse=True)
        all_nodes_sorted = [x[0] for x in all_nodes_sorted]
        for node in all_nodes_sorted:
            if node not in new_solution:
                new_solution.add(node)
                for j in g[node]:
                    num_neighbors_in_solution[j] += 1
                    if is_feasible(num_neighbors_in_solution):
                        return new_solution

    # it should never come to this point, we can always add all nodes in the solution
    return None

def test_add_vertex(solution, solution_value, i, g):
    solution_with_i = set(solution)
    solution_with_i.add(i)
    solution_with_i_value = objective_function(solution_with_i, g)
    if solution_with_i_value < solution_value:
        return True
    return False

def test_remove_vertex(new_solution, new_solution_value, i, g):
    new_solution_without_i = set(new_solution)
    new_solution_without_i.remove(i)

    num_neighbors_in_solution = [0 for _ in g.nodes]
    for k in new_solution_without_i:
        for j in g[k]:
            num_neighbors_in_solution[j] += 1

    if not is_feasible(num_neighbors_in_solution):
        return False

    if objective_function(new_solution_without_i, g) < new_solution_value:
        return True
    return False

def local_search(new_solution, g):
    all_nodes = set(g.nodes)
    improving_move_exists = True
    while improving_move_exists:
        improving_move_exists = False
        nodes_out = all_nodes.difference(new_solution)
        for i in nodes_out:
            if test_add_vertex(new_solution, i, g):
                new_solution.add(i)
                improving_move_exists = True
                break
        if not improving_move_exists:
            for i in new_solution:
                if test_remove_vertex(new_solution, i, g):
                    new_solution.remove(i)
                    improving_move_exists = True
                    break
    return new_solution

def ga(g, initial_population_size=100, population_size=20, cutoff=30, m_l=1, m_u=4, num_iters=20, grasp_only=False):
    population = []
    for i in range(initial_population_size):
        new_solution, new_solution_value = grasp(g, set(g.nodes), cutoff)
        if not same_quality_solution_inside(population, new_solution, g):
            population.append((new_solution, new_solution_value))
    if grasp_only:
        solution = select(population, 1, g)
        return solution[0]
    population = select(population, population_size, g)
    for iter in range(num_iters):
        old_population = list(population)
        for i, (solution_i, solution_value_i) in enumerate(old_population):
            for j in range(i + 1, len(old_population)):
                new_solution, new_solution_value = crossover(g, solution_i, old_population[j][0], cutoff)
                new_solution = mutation(new_solution, m_l, m_u, g)
                new_solution = local_search(new_solution, g)
                if not same_quality_solution_inside(population, new_solution, new_solution_value):
                    population.append((new_solution, new_solution_value))
        population = select(population, population_size, g)
    solution = select(population, 1, g)
    return solution[0]

def solve(file_path, num_runs, grasp_only):
    g = read_graph_from_file(file_path)
    times = []
    solution_values = []
    for i in range(num_runs):
        start_time = perf_counter()
        solution, solution_value = ga(g, grasp_only=grasp_only)
        end_time = perf_counter()
        time_elapsed = end_time - start_time
        times.append(time_elapsed)
        solution_values.append(solution_value)
    if num_runs > 1:
        return statistics.mean(times), statistics.mean(solution_values), statistics.stdev(solution_values), min(solution_values)
    else:
        return times[0], solution_values[0], 0, solution_values[0]

def get_results():
    DIR_PATH = '/home/pc/Desktop/ri/projekat/wtdp/instances/New'
    V = 75
    NUM_RUNS = 1
    GRASP_ONLY = False
    NAME = 'grasp' if GRASP_ONLY else 'ga' 
    OUTPUT_PATH = f'/home/pc/Desktop/ri/projekat/wtdp/test/their_results/new/new_75_{NAME}_reimpl_{NUM_RUNS}runs.csv'
    dir_path = os.path.abspath(DIR_PATH)
    with open(OUTPUT_PATH, 'w') as out:
        writer = csv.writer(out)
        writer.writerow(['avg_time', 'avg_solution_value', 'stdev_solution_value', 'best_solution_value'])
        for file_path in sorted(os.listdir(dir_path)):
            if str(V) in file_path:
                abs_path = os.path.join(dir_path, file_path)
                avg_time, avg_solution_value, stdev_solution_value, best_solution_value = solve(abs_path, NUM_RUNS, grasp_only=GRASP_ONLY)
                writer.writerow([f'{avg_time:.2f}', f'{avg_solution_value:.2f}', f'{stdev_solution_value:.2f}', f'{best_solution_value}'])

if __name__ == '__main__':
    get_results()