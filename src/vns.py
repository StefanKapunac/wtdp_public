from cmath import sqrt
import math
import random
import os
from copy import deepcopy
from time import perf_counter
import csv
import statistics
import sys
from utils import *

def initialize(g, prob):
    solution = {i for i in range(len(g.nodes)) if random.random() < prob}
    num_incorrect_nodes, cost, sorted_external_edges = calc_initial_solution_cost(solution, g)
    
    sum_all_node_weights = sum(w['weight'] for _,w in g.nodes(data=True))
    sum_all_edge_weights = sum(w['weight'] for _,_,w in g.edges(data=True))
    total_sum = sum_all_node_weights + sum_all_edge_weights

    return solution, num_incorrect_nodes, cost, total_sum, sorted_external_edges

def obj_func(num_incorrect_nodes, cost, total_sum):
    # we want to compare infeasible solutions
    # num_incorrect_nodes = num nodes violaiting the total domination condition
    # cost = objective function
    # total_sum = sum of all nodes and edges in the graph
    # => any feasible solution will have smaller cost than any infeasible
    # if the number of violating nodes is the same, the solution with smaller cost is better
    return num_incorrect_nodes + cost / total_sum

def shaking_only_reduce(solution, k, g, num_incorrect_nodes, cost, sorted_external_edges):
    new_solution = deepcopy(solution)
    new_sorted_external_edges = deepcopy(sorted_external_edges)
    new_num_incorrect_nodes = num_incorrect_nodes
    new_cost = cost
    if len(solution) == 0:
        return new_solution, new_num_incorrect_nodes, new_cost, new_sorted_external_edges
    # remove k random nodes from the solution
    to_choose = min(k, len(new_solution) - 1)
    chosen_nodes = random.sample(new_solution, to_choose)
    for node in chosen_nodes:
        new_num_incorrect_nodes, new_cost = recalc_objective_function_node_removed(new_solution, new_num_incorrect_nodes, new_cost, node, g, new_sorted_external_edges)
        fix_external_edges_after_node_removed(node, g, new_sorted_external_edges)
        new_solution.remove(node)

    return new_solution, new_num_incorrect_nodes, new_cost, new_sorted_external_edges

def local_search_first_improvement(solution, num_incorrect_nodes, cost, total_sum, g, sorted_external_edges):
    num_nodes = len(g.nodes)
    new_solution = deepcopy(solution)
    curr_num_incorrect_nodes, curr_cost = num_incorrect_nodes, cost
    all_nodes_list = list(range(num_nodes))
    improved = True
    while improved:
        improved = False
        nodes_inside = list(new_solution)
        
        random.shuffle(all_nodes_list)
        for node in all_nodes_list:
            if curr_num_incorrect_nodes > 0 and node in nodes_inside:
                continue
            if node in nodes_inside:
                new_num_incorrect_nodes, new_cost = recalc_objective_function_node_removed(new_solution,
                                                                                           curr_num_incorrect_nodes,
                                                                                           curr_cost,
                                                                                           node,
                                                                                           g,
                                                                                           sorted_external_edges)
                if obj_func(new_num_incorrect_nodes, new_cost, total_sum) < obj_func(curr_num_incorrect_nodes, curr_cost, total_sum):
                    curr_num_incorrect_nodes, curr_cost = new_num_incorrect_nodes, new_cost
                    fix_external_edges_after_node_removed(node, g, sorted_external_edges)
                    new_solution.remove(node)
                    improved = True
                    break
            else:
                new_num_incorrect_nodes, new_cost = recalc_objective_function_node_added(new_solution,
                                                                                         curr_num_incorrect_nodes,
                                                                                         curr_cost,
                                                                                         node,
                                                                                         g,
                                                                                         sorted_external_edges)
                if obj_func(new_num_incorrect_nodes, new_cost, total_sum) < obj_func(curr_num_incorrect_nodes, curr_cost, total_sum):
                    curr_num_incorrect_nodes, curr_cost = new_num_incorrect_nodes, new_cost
                    fix_external_edges_after_node_added(node, g, sorted_external_edges)
                    new_solution.add(node)
                    improved = True
                    break

    return new_solution, curr_num_incorrect_nodes, curr_cost

# local search based on swaps
def local_search2_first_improvement(solution, num_incorrect_nodes, cost, total_sum, g, sorted_external_edges):
    num_nodes = len(g.nodes)
    new_solution = deepcopy(solution)
    new_sorted_external_edges = deepcopy(sorted_external_edges)
    curr_num_incorrect_nodes, curr_cost = num_incorrect_nodes, cost
    all_nodes = set(range(num_nodes))
    improved = True
    while improved:
        improved = False

        # add one node
        nodes_outside = list(all_nodes.difference(new_solution))
        random.shuffle(nodes_outside)
        for node_out in nodes_outside:
            new_num_incorrect_nodes, new_cost = recalc_objective_function_node_added(new_solution, curr_num_incorrect_nodes, curr_cost, node_out, g, new_sorted_external_edges)
            if obj_func(new_num_incorrect_nodes, new_cost, total_sum) < obj_func(curr_num_incorrect_nodes, curr_cost, total_sum):
                curr_num_incorrect_nodes, curr_cost = new_num_incorrect_nodes, new_cost
                fix_external_edges_after_node_added(node_out, g, new_sorted_external_edges)
                new_solution.add(node_out)
                improved = True
                break
        if improved:
            continue
        old_solution = list(new_solution)
        random.shuffle(old_solution)
        for node_in in old_solution:
            # remove node_in
            new_num_incorrect_nodes, new_cost = recalc_objective_function_node_removed(new_solution, curr_num_incorrect_nodes, curr_cost, node_in, g, new_sorted_external_edges)
            fix_external_edges_after_node_removed(node_in, g, new_sorted_external_edges)
            new_solution.remove(node_in)
            # maybe that is enough
            if obj_func(new_num_incorrect_nodes, new_cost, total_sum) < obj_func(curr_num_incorrect_nodes, curr_cost, total_sum):
                curr_num_incorrect_nodes, curr_cost = new_num_incorrect_nodes, new_cost
                improved = True
                break

            # actually swap
            random.shuffle(nodes_outside)
            for node_out in nodes_outside:
                # add node_out
                newer_num_incorrect_nodes, newer_cost = recalc_objective_function_node_added(new_solution, new_num_incorrect_nodes, new_cost, node_out, g, new_sorted_external_edges)
                if obj_func(newer_num_incorrect_nodes, newer_cost, total_sum) < obj_func(curr_num_incorrect_nodes, curr_cost, total_sum):
                    curr_num_incorrect_nodes, curr_cost = newer_num_incorrect_nodes, newer_cost
                    fix_external_edges_after_node_added(node_out, g, new_sorted_external_edges)
                    new_solution.add(node_out)
                    improved = True
                    break
            if improved:
                break
            else:
                # revert the changes
                fix_external_edges_after_node_added(node_in, g, new_sorted_external_edges)
                new_solution.add(node_in)

    return new_solution, curr_num_incorrect_nodes, curr_cost, new_sorted_external_edges

def vns(g, num_iters, time_seconds, min_neighbors, max_neighbors, init_prob, move_prob, perform_ls2):
    start_time = perf_counter()
    solution, num_incorrect_nodes, cost, total_sum, sorted_external_edges = initialize(g, init_prob)
    curr_value = obj_func(num_incorrect_nodes, cost, total_sum)
    i = 0
    lastLS2Iter = 0
    lastImprIter = 0
    while i < num_iters and perf_counter() - start_time < time_seconds:
        for k in range(min_neighbors, max_neighbors + 1):
            i += 1

            if i%100==0:
                print(str(i)+'. k='+str(k)+' best='+str(cost))

            if k > len(solution) or k > len(g.nodes) - len(solution):
                break
            new_solution, new_num_incorrect_nodes, new_cost, new_sorted_external_edges = shaking_only_reduce(solution,
                                                                                                             k,
                                                                                                             g,
                                                                                                             num_incorrect_nodes,
                                                                                                             cost,
                                                                                                             sorted_external_edges) 
            new_solution, new_num_incorrect_nodes, new_cost = local_search_first_improvement(new_solution,
                                                                                             new_num_incorrect_nodes,
                                                                                             new_cost,
                                                                                             total_sum,
                                                                                             g,
                                                                                             new_sorted_external_edges)
                                 
            if perform_ls2 and i >= lastLS2Iter + 10 and i >= lastImprIter + 100:
                lastLS2Iter = i
                new_solution, new_num_incorrect_nodes, new_cost, new_sorted_external_edges = local_search2_first_improvement(new_solution,
                                                                                                                             new_num_incorrect_nodes,
                                                                                                                             new_cost,
                                                                                                                             total_sum,
                                                                                                                             g,
                                                                                                                             new_sorted_external_edges)

            if new_num_incorrect_nodes > 0:
                print('infeasible solution after local search')
                print(solution)
                print(new_solution)
                print(new_num_incorrect_nodes)
                print(new_cost)
            new_value = obj_func(new_num_incorrect_nodes, new_cost, total_sum)

            if new_value < curr_value:
                lastImprIter = i
            if new_value < curr_value or (new_value == curr_value and random.random() < move_prob):
                curr_value = new_value
                solution = deepcopy(new_solution)
                sorted_external_edges = deepcopy(new_sorted_external_edges)
                num_incorrect_nodes, cost = new_num_incorrect_nodes, new_cost
                break
                
    return solution, sorted_external_edges, num_incorrect_nodes, cost, total_sum

def solve(file_path, num_iters,time_seconds,kmin, kmax_func, kmax_max, init_prob, move_prob, prob_in, perform_ls2, log=False):
    g = read_graph_from_file(file_path)
    start_time = perf_counter()
    max_neighborhoods = 0
    n = len(g.nodes)
    if kmax_func=="sqrtn":
        max_neighborhoods = int(round(math.sqrt(n),0))
    elif kmax_func=="logn":
        max_neighborhoods = int(round(math.log2(n),0))
    elif kmax_func=="ndiv10":
        max_neighborhoods = int(round(n/10,0))
    elif kmax_func=="ndiv5":
        max_neighborhoods = int(round(n/5,0))
    else:
        raise Exception("Unrecognized kmax_func "+kmax_func)
    
    if max_neighborhoods>kmax_max:
        max_neighborhoods = kmax_max

    print(f'n is {n} and corresponding kmax = min({kmax_func}, {kmax_max}) = {max_neighborhoods}')
    solution, sorted_external_edges, num_incorrect_nodes, cost, total_sum = vns(g, num_iters, time_seconds,kmin, max_neighborhoods, init_prob, move_prob, prob_in, perform_ls2)
    end_time = perf_counter()
    internal_edges, external_edges = get_internal_and_external_edges(solution, g, sorted_external_edges)
    time_elapsed = end_time - start_time
    solution_value = obj_func(num_incorrect_nodes, cost, total_sum)
    if log:
        print(f'Solution: {solution}')
        print(f'len(solution): {len(solution)}')
        if num_incorrect_nodes > 0:
            print(f'Unfeasible, num incorrect nodes {num_incorrect_nodes}')
        else:
            print(f'Feasible')
        print(f'Cost: {cost}')
        
        print(f'obj_func(solution) = {solution_value}')
        print(f'Execution time: {time_elapsed:0.4f} seconds')
    return solution, internal_edges, external_edges, num_incorrect_nodes, cost, solution_value, time_elapsed 


def solve_n_times(n, file_path, num_iters,time_seconds,kmin, kmax_func, kmax_max, init_prob, move_prob, prob_in, perform_ls2, log=False):
    costs = []
    times = []
    num_incorrects = []
    num_evaluations = []
    for i in range(n):
        gl.num_evals = 0
        solution, internal_edges, external_edges, num_incorrect_nodes, cost, solution_value, time_elapsed = solve(file_path, num_iters,time_seconds,kmin, kmax_func, kmax_max, init_prob, move_prob, prob_in,perform_ls2, log)
        costs.append(cost)
        times.append(time_elapsed)
        num_incorrects.append(num_incorrect_nodes)
        num_evaluations.append(gl.num_evals)
        # now detailed results
        with open(file_path+'.out', 'a', newline='') as out:
            out.write('file_path: '+file_path+os.linesep)
            out.write('num_iters: '+str(num_iters)+os.linesep)
            out.write('time_seconds: '+str(time_seconds)+os.linesep)
            out.write('kmin: '+str(kmin)+os.linesep)
            out.write('kmax_func: '+str(kmax_func)+os.linesep)
            out.write('kmax_max: '+str(kmax_max)+os.linesep)
            out.write('perform_ls2: '+str(perform_ls2)+os.linesep)
            out.write('solution_value: '+str(solution_value)+os.linesep)
            out.write('cost: '+str(cost)+os.linesep)
            out.write('num_incorrect_nodes: '+str(num_incorrect_nodes)+os.linesep)
            out.write('time_elapsed: '+str(time_elapsed)+os.linesep)
            out.write('solution_nodes:'+os.linesep)
            for node in solution:
                out.write(str(node)+os.linesep)
            out.write('solution_internal_edges:'+os.linesep)
            for edge in internal_edges:
                out.write(str(edge)+os.linesep)
            out.write('solution_external_edges:'+os.linesep)
            for edge in external_edges:
                out.write(str(edge)+os.linesep)
    if n > 1:
        return statistics.mean(times), statistics.mean(costs), statistics.stdev(costs), min(costs), statistics.mean(num_incorrects), statistics.mean(num_evaluations)
    else:
        return times[0], costs[0], 0, costs[0], num_incorrects[0], num_evaluations[0]


def main():
    if len(sys.argv)!=10:
        print(f"Incorrect usage, expected parameters are: <DIR_PATH> <NAME_MASK> <KMIN> <KMAX_FUNC> <KMAX_MAX> <NUM_ITERS> <TIME_SECONDS> <NUM_RUNS> <PERFORM_LS2 (y/n)>")
        sys.exit()
    
    random.seed(12345)
    DIR_PATH = sys.argv[1] # '../instances/Ma' for example
    NAME_MASK = sys.argv[2] # '75' for example
    KMIN = int(sys.argv[3]) # 1 for example
    KMAX_FUNC = sys.argv[4] # allowed are logn, sqrtn, ndiv5, ndiv10
    if KMAX_FUNC!="logn" and KMAX_FUNC!="sqrtn" and KMAX_FUNC!="ndiv10" and KMAX_FUNC!="ndiv5":
        print("Incorrect value for KMAX_FUNC, allowed values are: logn, sqrtn, ndiv5 and ndiv10.")
        sys.exit()
    KMAX_MAX = int(sys.argv[5]) # max value for kmax, kmax is obtained as min(KMAX_FUNC, KMAX_MAX)
    NUM_ITERS = int(sys.argv[6]) # 3900 # num_iters = initialPopSize + nIter * (popSize * (popSize-1))/2 = 100 + 20*20*19/2 = 3900 
    TIME_SECONDS = int(sys.argv[7]) # 1800
    NUM_RUNS = int(sys.argv[8]) #1
    if sys.argv[9]=='yes':
        PERFORM_LS2 = True
    elif sys.argv[9]=='no':
        PERFORM_LS2 = False
    else:
        print('Incorrect value for PERFORM_LS2, allowed values are yes or no.')
        sys.exit() 

    # these bellow are basically problem-specific constants, not parameters, except MOVE_PROB which is standard part of VNS, but we can keep it at 0.5
    INIT_PROB = 0.2
    MOVE_PROB = 0.5
    PROB_IN = 0.95
    INSTANCE_NAME = (DIR_PATH+'_'+NAME_MASK).replace('/','_').replace('\\','_').replace('.','')
    OUTPUT_PATH = f'output/vns_{INSTANCE_NAME}_{NUM_RUNS}runs_{NUM_ITERS}iter_{KMIN}kmin_{KMAX_FUNC}kmaxfunc_{KMAX_MAX}_kmaxmax_{PERFORM_LS2}ls2.csv'
    print(OUTPUT_PATH)
    dir_path = os.path.abspath(DIR_PATH)
    with open(OUTPUT_PATH, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['file_path', 'avg_time', 'avg_solution_value', 'stdev_solution_value', 'best_solution_value', 'avg_num_incorrect_nodes', 'avg_num_evaluations'])

    i = 0
    for file_path in sorted(os.listdir(dir_path)):
        if NAME_MASK=="" or NAME_MASK in file_path:
            abs_path = os.path.join(dir_path, file_path)
            print('Doing instance '+abs_path)
            avg_time, avg_solution_value, stdev_solution_value, best_solution_value, avg_num_incorrect_nodes, avg_num_evaluations = solve_n_times(NUM_RUNS, abs_path, NUM_ITERS, TIME_SECONDS,KMIN, KMAX_FUNC, KMAX_MAX, INIT_PROB, MOVE_PROB, PROB_IN, PERFORM_LS2)
            print(f'{i}: {file_path}, {avg_time}, {avg_solution_value}, {best_solution_value}, {avg_num_incorrect_nodes}, {avg_num_evaluations}')
            with open(OUTPUT_PATH, 'a', newline='') as out:
                writer = csv.writer(out)
                writer.writerow([file_path, f'{avg_time:.2f}', f'{avg_solution_value:.2f}', f'{stdev_solution_value:.2f}', best_solution_value, f'{avg_num_incorrect_nodes:.2f}', f'{avg_num_evaluations:.2f}'])
            i += 1

if __name__ == '__main__':
    main()