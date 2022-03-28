import networkx as nx
from itertools import chain, combinations

from utils import read_graph_from_file

# all subsets of length >= 2 (set length < 2 => automatically not a total dominating set)
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(2, len(s)+1))

# if subset is a total dominating set, returns its cost
# else, None
def calc_solution_cost(solution, g):
    sum_node_weights = 0
    internal_edge_weights = 0
    external_edge_weights = 0

    edge_uv_added = set()
    for u, u_weight in g.nodes(data=True):
        u_has_edge = False
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
            min_external_edge_weight = float('inf')
            for v in solution:
                if g.has_edge(u, v):
                    u_has_edge = True
                    if g[u][v]['weight'] < min_external_edge_weight:
                        min_external_edge_weight = g[u][v]['weight']
            external_edge_weights += min_external_edge_weight
        if not u_has_edge:
            return None

    total_cost = sum_node_weights + internal_edge_weights + external_edge_weights
    return total_cost


def main():
    g = read_graph_from_file('instances/Ma/MA-20-0.2-5-5-2.wtdp')

    all_subsets = list(powerset(g.nodes))
    print(f'Number of subsets of nodes: {len(all_subsets)}')

    min_cost = float('inf')
    solution = None
    for subset in all_subsets:
        cost = calc_solution_cost(subset, g)
        if cost is not None and cost < min_cost:
            min_cost = cost
            solution = subset
    print(solution)
    print(min_cost)

if __name__ == '__main__':
    main()