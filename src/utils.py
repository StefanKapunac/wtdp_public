import random
from sortedcontainers import SortedSet
import networkx as nx
from networkx.generators.random_graphs import gnp_random_graph

import gl

def read_line_of_ints(f):
    return [int(x) for x in f.readline().split()]

def read_graph_from_file(file_name):
    with open(file_name, 'r') as f:
        num_nodes, num_edges, max_node_weight, max_edge_weight = read_line_of_ints(f)
        g = nx.Graph()
        for i in range(num_nodes):
            id, weight = read_line_of_ints(f)
            g.add_node(id, weight=weight)
        for i in range(num_edges):
            id, u, v, weight = read_line_of_ints(f)
            g.add_edge(u, v, weight=weight)
        return g

def write_graph_to_file(g, max_node_weight, max_edge_weight, file_name):
    with open(file_name, 'w') as f:
        f.write(f'{len(g.nodes)} {len(g.edges)} {max_node_weight} {max_edge_weight}\n')
        for i, w in g.nodes(data=True):
            f.write(f'{i} {w["weight"]}\n')
        for idx, (i, j, w) in enumerate(g.edges(data=True)):
            f.write(f'{idx} {i} {j} {w["weight"]}\n')


def generate_random_graphs():
    ns = [250, 500, 1000]
    ps = [0.2, 0.5, 0.8]
    cs_ws = [((1,50), (1,10)), ((1,25), (1,25)), ((1,10), (1,50))]
    num_instances = 5
    for n in ns:
        for p in ps:
            for c,w in cs_ws:
                for idx in range(num_instances):
                    g = gnp_random_graph(n, p)
                    for i in g.nodes:
                        g.nodes[i]['weight'] = random.randint(c[0], c[1])
                    for i,j in g.edges:
                        g[i][j]['weight'] = random.randint(w[0], w[1])
                    file_name = f'instances/Our/Our-{n}-{p}-{c[1]}-{w[1]}-{idx}.wtdp'
                    write_graph_to_file(g, c[1], w[1], file_name)

def get_internal_and_external_edges(solution, g, sorted_external_edges):
    internal_edges = set([])
    external_edges = set([])
    for u in g.nodes:
        if u in solution:
            for v, w in sorted_external_edges[u]:
                internal_edges.add((u, v, w))
        else:
            v, w = sorted_external_edges[u][0]
            external_edges.add((u, v, w))
    return internal_edges, external_edges

def calc_initial_solution_cost(solution, g):
    gl.num_evals += 1

    sum_node_weights = 0
    internal_edge_weights = 0
    external_edge_weights = 0

    sorted_external_edges = [None] * len(g.nodes)

    num_incorrect = 0

    edge_uv_added = set()
    for u, u_weight in g.nodes(data=True):
        u_has_edge = False
        sorted_external_edges[u] = SortedSet([(v, g[u][v]['weight']) for v in solution if g.has_edge(u,v)],
                                                key=lambda x: x[1])
        if u in solution:
            sum_node_weights += u_weight['weight']
            for v in solution:
                if g.has_edge(u, v):
                    u_has_edge = True
                    u_hat = min(u, v)
                    v_hat = max(u, v)
                    if (u_hat, v_hat) not in edge_uv_added:
                        internal_edge_weights += g[u][v]['weight']
                        edge_uv_added.add((u_hat, v_hat))
        else:
            if len(sorted_external_edges[u]) > 0:
                u_has_edge = True
                external_edge_weights += sorted_external_edges[u][0][1]
        if not u_has_edge:
            num_incorrect += 1
            
    total_cost = sum_node_weights + internal_edge_weights + external_edge_weights
    return num_incorrect, total_cost, sorted_external_edges

def recalc_objective_function_node_added(solution, num_incorrect_nodes, cost, node, g, sorted_external_edges):
    # if i added a new node v:
    #   add v's weight
    #   subtract external edge weight
    #   add internal edges for v
    #   add updated external edge weights for nodes whose nearest neighbor is now v

    gl.num_evals += 1

    new_num_incorrect_nodes = num_incorrect_nodes

    node_weight = g.nodes(data=True)[node]['weight']
    new_cost = cost + node_weight
    if len(sorted_external_edges[node]) > 0: 
        new_cost -= sorted_external_edges[node][0][1] # if node had a neighbor in the solution, subtract external edge weight
     
    for _, w in sorted_external_edges[node]:
        new_cost += w

    for v in g[node]:
        edge = g[v][node]
        curr_weight = edge['weight']
        if len(sorted_external_edges[v]) == 0:
            # v didn't have a neighbor in the solution
            new_num_incorrect_nodes -= 1
            if v not in solution: # internal edges are already added, now add only external
                new_cost += curr_weight
        else:
            prev_min_weight = sorted_external_edges[v][0][1]
            if v not in solution and prev_min_weight > curr_weight: # node is v's new nearest neighbor from the solution
                new_cost = new_cost - prev_min_weight + curr_weight

    return new_num_incorrect_nodes, new_cost

def recalc_objective_function_node_removed(solution, num_incorrect_nodes, cost, node, g, sorted_external_edges):
    # if i removed a node v from the solution:
    #   subtract v's weight
    #   add external edge weight
    #   subtract internal edges for v
    #   add updated external edge weights for nodes whose nearest neighbor was v

    gl.num_evals += 1

    new_num_incorrect_nodes = num_incorrect_nodes

    node_weight = g.nodes(data=True)[node]['weight']
    new_cost = cost - node_weight
    if len(sorted_external_edges[node]) > 0:
        new_cost += sorted_external_edges[node][0][1] # if node had a neighbor in the solution, add external edge weight
    for _, w in sorted_external_edges[node]:
        new_cost -= w
    
    for v in g[node]:
        if len(sorted_external_edges[v]) == 1:
            # node was v's only neighbor in the solution
            new_num_incorrect_nodes += 1
            if v not in solution: # internal edges are already subtracted, now subtract only external
                new_cost -= g[v][node]['weight']
        else:
            prev_min = sorted_external_edges[v][0] # prev_min must exist here
            if v not in solution and prev_min[0] == node: # node is not v's nearest neighbor from the solution anymore
                new_cost = new_cost - prev_min[1] + sorted_external_edges[v][1][1]
            
    return new_num_incorrect_nodes, new_cost
    
# node is removed - remove it from the sorted external edges
def fix_external_edges_after_node_removed(node, g, sorted_external_edges):
    for v in g[node]:
        sorted_external_edges[v].remove((node, g[v][node]['weight']))

# node is added - add it to the sorted external edges
def fix_external_edges_after_node_added(node, g, sorted_external_edges):
    for v in g[node]:
        sorted_external_edges[v].add((node, g[v][node]['weight']))

def main():
    return

if __name__ == '__main__':
    main()