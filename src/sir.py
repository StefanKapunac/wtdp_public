import random
from matplotlib import pyplot as plt
import csv

from utils import *
from real_utils import *

def sir_dynamics(g, solution, prob_ign_to_spr, prob_spr_to_sti, figures_folder_path, instance_name, alg_name, plot, draw_through_time):
    all_nodes = set(g.nodes)

    spreaders = solution
    stiflers = set()
    ignorants = all_nodes.difference(spreaders)

    num_spreaders_in_iteration = [len(spreaders)]
    num_stiflers_in_iteration = [len(stiflers)]
    num_ignorants_in_iteration = [len(ignorants)]

    if draw_through_time:
        visualize_sir_graph(g, spreaders, stiflers, ignorants)

    num_interactions_in_iteration = [0]

    t = 1
    while len(spreaders) != 0:
        ignorants_to_remove = set()
        spreaders_to_add = set()
        spreaders_to_remove = set()
        stiflers_to_add = set()
        num_interactions = 0
        for s in spreaders:
            for v in g[s]:
                if v in ignorants:
                    if random.random() < prob_ign_to_spr:
                        ignorants_to_remove.add(v)
                        spreaders_to_add.add(v)
                        num_interactions += 1
                else:
                    if random.random() < prob_spr_to_sti:
                        spreaders_to_remove.add(s)
                        stiflers_to_add.add(s)
                        num_interactions += 1
        spreaders = spreaders.union(spreaders_to_add).difference(spreaders_to_remove)
        ignorants = ignorants.difference(ignorants_to_remove)
        stiflers = stiflers.union(stiflers_to_add)

        t += 1
        num_spreaders_in_iteration.append(len(spreaders))
        num_stiflers_in_iteration.append(len(stiflers))
        num_ignorants_in_iteration.append(len(ignorants))

        if draw_through_time and t % 3 == 0:
            visualize_sir_graph(g, spreaders, stiflers, ignorants)

        num_interactions_in_iteration.append(num_interactions)

    print(t)
    print(num_spreaders_in_iteration)
    print(num_stiflers_in_iteration)
    print(num_ignorants_in_iteration)
    print(num_interactions_in_iteration)

    if plot:
        plt.plot(range(t), num_spreaders_in_iteration, label='spreaders')
        plt.plot(range(t), num_stiflers_in_iteration, label='stiflers')
        plt.plot(range(t), num_ignorants_in_iteration, label='ignorants')
        plt.legend()
        figure_name = f'{instance_name}_{alg_name}_{prob_ign_to_spr}_{prob_spr_to_sti}.png'
        # plt.title(f'{instance_name} {alg_name} \n dynamics with prob_ign_to_spr={prob_ign_to_spr}, prob_spr_to_sti={prob_spr_to_sti}')
        plt.xlabel('Iteration')
        plt.ylabel('Number of nodes')
        # plt.show()
        plt.savefig(os.path.join(figures_folder_path, figure_name))
        plt.close('all')

        plt.plot(range(t), num_interactions_in_iteration)
        # plt.title(f'{instance_name} {alg_name} \n Number of interactions through iterations')
        # plt.show()
        figure_name = f'{instance_name}_{alg_name}_interactions_{prob_ign_to_spr}_{prob_spr_to_sti}.png'
        plt.savefig(os.path.join(figures_folder_path, figure_name))
        plt.close('all')

    return t, len(spreaders), len(stiflers), len(ignorants)

def process_all(folder_path, ps):
    random.seed(12345)

    res_file_path = '../test/sir_dynamics/complete_all_results_raw.csv'

    with open(res_file_path, 'w') as f:
        writer = csv.writer(f)
        for prob_ign_to_spr in ps:
            for prob_spr_to_sti in ps:
                writer.writerow([f'{prob_ign_to_spr}, {prob_spr_to_sti}'])
                writer.writerow(['VNS', '', '', '', 'RANDOM'])
                writer.writerow(['', 'num_iterations', 'num_spreaders', 'num_stiflers', 'num_ignorants', 'num_iterations', 'num_spreaders', 'num_stiflers', 'num_ignorants'])
                
                all_file_paths = sorted(os.listdir(folder_path))
                for i in range(0, len(all_file_paths), 2):
                    instance_name = all_file_paths[i][:-5]
                    file_path = os.path.join(folder_path, all_file_paths[i])
                    solution_file_path = os.path.join(folder_path, all_file_paths[i + 1])
                    g = read_graph_from_file(file_path)
                    vns_solution, internal_edges, external_edges = read_solution_from_file(solution_file_path)
                    random_solution = set(random.sample(list(g.nodes), len(vns_solution)))

                    vns_t, vns_num_spreaders, vns_num_stiflers, vns_num_ignorants = sir_dynamics(g, vns_solution, prob_ign_to_spr, prob_spr_to_sti, '../test/sir_dynamics/figures/', instance_name, 'VNS', True, False)
                    random_t, random_num_spreaders, random_num_stiflers, random_num_ignorants = sir_dynamics(g, random_solution, prob_ign_to_spr, prob_spr_to_sti, '../test/sir_dynamics/figures/', instance_name, 'RANDOM', True, False)
                    writer.writerow([vns_t, vns_num_spreaders, vns_num_stiflers, vns_num_ignorants, random_t, random_num_spreaders, random_num_stiflers, random_num_ignorants])

def visualize_sir_graph(g, spreaders, stiflers, ignorants):
    for v in g.nodes:
        if v in spreaders:
            g.nodes[v]['color'] = 0x0000FF00
        elif v in stiflers:
            g.nodes[v]['color'] = 0xFF000000
        elif v in ignorants:
            g.nodes[v]['color'] = 0x00FF0000
        else:
            return None
    graphistry.register(api=3, username='Stefan', password='wtdpwtdp')
    graphistry.graph(g) \
        .bind(source='src', destination='dst', node='nodeid', point_color='color') \
        .plot()

def main():
    instance_name = 'musae_PTBR_edges'
    g = read_graph_from_file(f'../instances/Snap/raw-weighted/{instance_name}.wtdp')
    vns_solution, internal_edges, external_edges = read_solution_from_file(f'../instances/Snap/raw-weighted/{instance_name}.wtdp.out')
    random_solution = set(random.sample(list(g.nodes), len(vns_solution)))
    sir_dynamics(g, vns_solution, 0.1, 0.9, '../doc/plots/', instance_name, 'VNS', False, True)
    sir_dynamics(g, random_solution, 0.1, 0.9, '../doc/plots/', instance_name, 'RANDOM', False, True)
    
if __name__ == '__main__':
    main()