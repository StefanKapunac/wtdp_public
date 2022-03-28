import os
import networkx as nx
import graphistry

from utils import read_graph_from_file, write_graph_to_file

# node indices should be from 0 to n-1
def check(g):
    print(f'num nodes: {len(g.nodes)}')
    min_node = float('inf')
    max_node = float('-inf')
    for v in g.nodes:
        if v < min_node:
            min_node = v
        elif v > max_node:
            max_node = v
    print(f'min node: {min_node}')
    print(f'max node: {max_node}')

# add artificial weight of 1 to every node and every edge
def read_unweighted_graph_from_file(file_name):
    with open(file_name, 'r') as f:
        # skip header of csv
        is_csv = file_name[-4:] == '.csv'
        if is_csv:
            f.readline()
        
        node_ind_map = {}
        curr_node_ind = 0
        g = nx.Graph()
        for line in f:
            if is_csv:
                u, v = [int(x) for x in line.split(',')]
            else:
                u, v = [int(x) for x in line.split()]
            
            if u not in node_ind_map:
                node_ind_map[u] = curr_node_ind
                curr_node_ind += 1
            if v not in node_ind_map:
                node_ind_map[v] = curr_node_ind
                curr_node_ind += 1
            u = node_ind_map[u]
            v = node_ind_map[v]
            g.add_node(u, weight=1)
            g.add_node(v, weight=1)
            g.add_edge(u, v, weight=1)
    return g

def transform_all(folder_path, res_folder_path='../instances/Snap/weighted/'):
    for file_path in os.listdir(folder_path):
        abs_file_path = os.path.join(folder_path, file_path)
        g = read_unweighted_graph_from_file(abs_file_path)
        res_file_path = os.path.join(res_folder_path, file_path[:-4] + '.wtdp')
        write_graph_to_file(g, 1, 1, res_file_path)

def read_solution_from_file(file_path):
    with open(file_path, 'r') as f:
        solution_start, solution_end, internal_edges_start, internal_edges_end, external_edges_start, external_edges_end = False, False, False, False, False, False
        solution = set()
        internal_edges = set()
        external_edges = set()
        for line in f:
            line = line.strip()
            if not solution_start:
                if line == 'solution_nodes:':
                    solution_start = True
                    continue
            elif not internal_edges_start:
                if line == 'solution_internal_edges:':
                    solution_end = True
                    internal_edges_start = True
                    continue
            elif not external_edges_start:
                if line == 'solution_external_edges:':
                    internal_edges_end = True
                    external_edges_start = True
                    continue
            if solution_start and not solution_end:
                v = int(line)
                solution.add(v)
            elif internal_edges_start and not internal_edges_end:
                line = line.replace(' ', '')
                u, v, w = [int(x) for x in line[1:-1].split(',')]
                internal_edges.add((u, v, w))
            elif external_edges_start and not external_edges_end:
                line = line.replace(' ', '')
                u, v, w = [int(x) for x in line[1:-1].split(',')]
                external_edges.add((u, v, w))
    return solution, internal_edges, external_edges

def visualize_graph(g, solution):
    for v in g.nodes:
        if v in solution:
            g.nodes[v]['color'] = 0x0000FF00
        else:
            g.nodes[v]['color'] = 0xFFFFFF00
    graphistry.register(api=3, username='Stefan', password='wtdpwtdp')
    graphistry.graph(g) \
        .bind(source='src', destination='dst', node='nodeid', point_color='color') \
        .plot()

def main():
    g = read_graph_from_file('/home/pc/Desktop/ri/projekat/wtdp/instances/Snap/weighted/HU_edges.wtdp')
    solution, internal_edges, external_edges = read_solution_from_file('/home/pc/Desktop/ri/projekat/wtdp/test/results/snap/HU_edges.wtdp.out')
    print(len(solution))
    visualize_graph(g, solution)

if __name__ == '__main__':
    main()