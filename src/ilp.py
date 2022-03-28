import os
from docplex.mp.model import Model

from utils import read_graph_from_file

def create_model_ma1(graph):
    model = Model(name='ma1')
    x = model.binary_var_list(len(graph.nodes), name='x')
    y = model.binary_var_list(len(graph.edges), name='y')
    z = model.binary_var_list(len(graph.edges), name='z')

    ij_to_edge_id = {(i,j): edge_id for edge_id, (i,j) in enumerate(graph.edges)}
    for (i,j),edge_id in list(ij_to_edge_id.items()):
        ij_to_edge_id[j,i] = edge_id

    for i in graph.nodes:
        model.add_constraint(sum(x[j] for j in graph[i]) >= 1) # TDOM
        model.add_constraint(x[i] + sum(y[ij_to_edge_id[i, j]] for j in graph[i]) >= 1) # WTD1.7

    for edge_id, (i,j) in enumerate(graph.edges):
        model.add_constraint(x[i] + x[j] >= y[edge_id]) # WTD1.3
        model.add_constraint(x[i] >= z[edge_id]) # WTD1.4
        model.add_constraint(x[j] >= z[edge_id]) # WTD1.4
        model.add_constraint(z[edge_id] >= x[i] + x[j] - 1) # WTD1.5
        model.add_constraint(y[edge_id] >= z[edge_id]) # WTD1.6

    sum_node_weights = sum(x[i] * w['weight'] for i, w in graph.nodes(data=True))
    sum_edge_weights = sum(y[ij_to_edge_id[i,j]] * w['weight'] for i, j, w in graph.edges(data=True))
    model.minimize(sum_node_weights + sum_edge_weights)

    model.print_information()

    return model

def create_model_ma2(graph, M):
    model = Model(name='ma2')
    x = model.binary_var_list(len(graph.nodes), name='x')
    y = model.binary_var_list(len(graph.edges), name='y')

    ij_to_edge_id = {(i,j): edge_id for edge_id, (i,j) in enumerate(graph.edges)}
    for (i,j),edge_id in list(ij_to_edge_id.items()):
        ij_to_edge_id[j,i] = edge_id

    for i in graph.nodes:
        model.add_constraint(sum(x[j] for j in graph[i]) >= 1) # TDOM
        sum_incident_edges = sum(y[ij_to_edge_id[i, j]] for j in graph[i])
        model.add_constraint(x[i] + sum_incident_edges >= 1) # WTD1.7
        model.add_constraint(sum_incident_edges <= 1 + M*x[i]) # WTD2.4

    for edge_id, (i,j) in enumerate(graph.edges):
        model.add_constraint(x[i] + x[j] >= y[edge_id]) # WTD1.3
        model.add_constraint(y[edge_id] >= x[i] + x[j] - 1) # WTD2.3

    sum_node_weights = sum(x[i] * w['weight'] for i, w in graph.nodes(data=True))
    sum_edge_weights = sum(y[ij_to_edge_id[i,j]] * w['weight'] for i, j, w in graph.edges(data=True))
    model.minimize(sum_node_weights + sum_edge_weights)

    model.print_information()

    return model

def create_model_ma3(graph, L):
    model = Model(name='ma3')
    num_nodes = len(graph.nodes)
    x = model.binary_var_list(num_nodes, name='x')
    q = model.integer_var_list(num_nodes, lb=0, ub=num_nodes * L, name='q')

    for i in graph.nodes:
        model.add_constraint(sum(x[j] for j in graph[i]) >= 1) # TDOM
        model.add_constraint(q[i] >= sum(graph[i][j]['weight'] * (x[i] + x[j] - 1) for j in graph[i])) # WTD3.3
        for j in graph[i]:
            # from the original paper:
            model.add_constraint(q[i] >= 2 * (graph[i][j]['weight'] * x[j] - L * x[i] - 
                                              sum(L * x[k] for k in graph[i] if graph[i][k]['weight'] < graph[i][j]['weight']))) # WTD3.2

            # not the same thing from the other paper
            # model.add_constraint(q[i] >= 2 * (graph[i][j]['weight'] * x[i] - L * x[j] - 
            #                                   sum(L * x[k] for k in graph[i] if graph[i][k]['weight'] <= graph[i][j]['weight'])))

    total_cost = sum(x[i] * w['weight'] + q[i] / 2 for i, w in graph.nodes(data=True))
    model.minimize(total_cost)

    model.print_information()

    return model

def create_model_f1(graph):
    model = Model(name='f1')
    num_nodes = len(graph.nodes)
    x = model.binary_var_list(num_nodes, name='x')
    num_edges = len(graph.edges)
    y = model.binary_var_list(num_edges, name='y')
    z = model.binary_var_list(2 * num_edges, name='z')

    ij_to_arc_id = {(i,j): arc_id for arc_id, (i,j) in enumerate(graph.edges)}
    k = 0
    for (i,j), _ in list(ij_to_arc_id.items()):
        ij_to_arc_id[j,i] = num_edges + k
        k += 1

    for i in graph.nodes:
        model.add_constraint(sum(x[j] for j in graph[i]) >= 1) # TDOM
        model.add_constraint(x[i] + sum(z[ij_to_arc_id[j,i]] for j in graph[i]) == 1) # XZLINK1
        

    for (i,j), arc_id in ij_to_arc_id.items():
        model.add_constraint(z[arc_id] <= x[i]) # XZLINK2

    for edge_id, (i,j) in enumerate(graph.edges):
        model.add_constraint(y[edge_id] >= x[i] + x[j] - 1) # YZLINK

    nodes_cost = sum(x[i] * w['weight'] for i, w in graph.nodes(data=True))
    internal_edges_cost = sum(y[edge_id] * w['weight'] for edge_id, (i, j, w) in enumerate(graph.edges(data=True)))
    external_edges_cost = sum(z[arc_id] * graph[i][j]['weight'] for (i, j), arc_id in ij_to_arc_id.items())

    model.minimize(nodes_cost + internal_edges_cost + external_edges_cost)

    model.print_information()

    return model

def create_model_f2(graph):
    model = Model(name='f2')
    num_nodes = len(graph.nodes)
    x = model.binary_var_list(num_nodes, name='x')
    num_edges = len(graph.edges)
    y = model.binary_var_list(num_edges, name='y')
    q = model.continuous_var_list(num_nodes, lb=0, name='q')

    for i in graph.nodes:
        model.add_constraint(sum(x[j] for j in graph[i]) >= 1) # TDOM

        # sort neighbors by edge weight
        adjacent_nodes = [x[0] for x in sorted(graph[i].items(), key=lambda x: x[1]['weight'])]
        for idx, k in enumerate(adjacent_nodes):
            c_ki = graph[k][i]['weight']
            model.add_constraint(q[i] >= c_ki - sum((c_ki - graph[l][i]['weight']) * x[l] for l in adjacent_nodes[:idx]) - c_ki * x[i]) # EXTCOSTS-i

    for edge_id, (i,j) in enumerate(graph.edges):
        model.add_constraint(y[edge_id] >= x[i] + x[j] - 1) # YZLINK

    nodes_cost = sum(x[i] * w['weight'] + q[i] for i, w in graph.nodes(data=True))
    edges_cost = sum(y[edge_id] * w['weight'] for edge_id, (i, j, w) in enumerate(graph.edges(data=True)))

    model.minimize(nodes_cost + edges_cost)

    model.print_information()

    return model

def create_model(model_name, g):
    if model_name == 'ma1':
        return create_model_ma1(g)
    elif model_name == 'ma2':
        M = max(g.degree, key=lambda x: x[1])[1]
        return create_model_ma2(g, M)
    elif model_name == 'ma3':
        L = max(map(lambda x: x[2]['weight'], g.edges(data=True)))
        return create_model_ma3(g, L)
    elif model_name == 'f1':
        return create_model_f1(g)
    elif model_name == 'f2':
        return create_model_f2(g)
    else:
        return None

def solve(model_name, in_file_path, time_limit):
    g = read_graph_from_file(in_file_path)

    model = create_model(model_name, g)
    model.set_time_limit(time_limit)
    model.context.cplex_parameters.workmem = 6144
    model.context.cplex_parameters.threads = 1
    model.context.cplex_parameters.mip.limits.treememory = 6144

    solution = model.solve()
    time = model.solve_details.time
    
    status = model.solve_details.status
    gap = model.solve_details.gap
    num_nodes_processed = model.solve_details.nb_nodes_processed
    model.end()
    return status, time, solution.get_objective_value(), gap, num_nodes_processed

def solve_all(model_name, V):
    TIME_LIMIT = 1800
    DIR_PATH = '../instances/Our/'
    dir_path = os.path.abspath(DIR_PATH)

    part = 3
    out_file_path = f'../test/their_results/our/our_{V}_{model_name}_{part}.csv'
    with open(out_file_path, 'w') as f:
        f.write('status, time, cost, gap, num_nodes_processed\n')
        for i, file_path in enumerate(sorted(os.listdir(dir_path))):
            corrected_i = i
            if str(V) in file_path and corrected_i < part * 15 and corrected_i >= (part - 1) * 15:
                print(i, file_path)
                abs_path = os.path.join(dir_path, file_path)
                status, time, cost, gap, num_nodes = solve(model_name, abs_path, TIME_LIMIT)
                f.write(f'{status} {time}, {cost}, {gap}, {num_nodes}\n')

def main():
    solve_all('f2', 1000)

if __name__ == '__main__':
    main()