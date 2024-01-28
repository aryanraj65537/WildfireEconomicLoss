# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Graph partitioning with CQM solver."""

import itertools
from random import random
from collections import defaultdict
import sys

import networkx as nx
import numpy as np
import click
import matplotlib
from dimod import Binary, ConstrainedQuadraticModel, quicksum
from dwave.system import LeapHybridCQMSampler
import simulator
from simulator import MAP_HEIGHT, MAP_WIDTH, createNewForest

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

rishicodefunky = createNewForest()
rishisigma = list(itertools.islice({(i, j) for i in range(MAP_WIDTH) for j in range(MAP_HEIGHT)}, 120))
rishimonkey = simulator.main(60, rishicodefunky, [], rishicodefunky[0])
root_node = rishimonkey[3][(MAP_WIDTH//2, MAP_HEIGHT//2)]
print(rishimonkey[2])

def compute_distances_from_root(G, root_node):
    """Compute the shortest path lengths from root_node to all other nodes in the graph."""
    return nx.shortest_path_length(G, source=root_node)

def assign_edge_weights_based_on_node_weights(G, node_weights, root_distances):
    """Assign weights to edges based on the node weights and their distances from the root."""
    for edge in G.edges():
        node_a, node_b = edge

        # Determine which node is closer to the root
        if root_distances[node_a] < root_distances[node_b]:
            # Node A is closer to the root, set edge weight to the weight of node B
            G[node_a][node_b]['weight'] = node_weights[node_b]
        else:
            # Node B is closer or they are at the same distance, set edge weight to the weight of node A
            G[node_a][node_b]['weight'] = node_weights[node_a]

def build_graph(graph, nodes, degree, prob, p_in, p_out, new_edges, k_partition):
    """Builds graph from user specified parameters or use defaults.
    
    Args:
        See @click decorator before main.

    Returns:
        G (Graph): The graph to be partitioned
    """
    
    k = k_partition

    if k * nodes > 5000:
        raise ValueError("Problem size is too large.")
    elif nodes % k != 0:
        raise ValueError("Number of nodes must be divisible by k.")

    # Build graph using networkx
    if graph == 'partition':
        print("\nBuilding partition graph...")
        G = nx.random_partition_graph([int(nodes/k)]*k, p_in, p_out)

    elif graph == 'internet':
        print("\nReading in internet graph of size", nodes, "...")
        G = nx.random_internet_as_graph(nodes)

    elif graph == 'rand-reg':
        if degree >= nodes:
            raise ValueError("degree must be less than number of nodes")
        if degree * nodes % 2 == 1:
            raise ValueError("degree * nodes must be even")
        print("\nGenerating random regular graph...")
        G = nx.random_regular_graph(degree, nodes)

    elif graph == 'ER':
        print("\nGenerating Erdos-Renyi graph...")
        G = nx.erdos_renyi_graph(nodes, prob)

    elif graph == 'SF':
        if new_edges > nodes:
            raise ValueError("Number of edges must be less than number of nodes")
        print("\nGenerating Barabasi-Albert scale-free graph...")
        G = nx.barabasi_albert_graph(nodes, new_edges)

    else:
        # Should not be reachable, due to click argument validation
        raise ValueError(f"Unexpected graph type: {graph}")

    return G


# Visualize the input graph
def visualize_input_graph(G):
    """Visualize the graph to be partitioned.
    Args:
        G (Graph): Input graph to be partitioned
    
    Returns:
        None. Image saved as input_graph.png.
    """
    def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
        if not nx.is_tree(G):
            raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

        if root is None:
            if isinstance(G, nx.DiGraph):
                root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
            else:
                root = random.choice(list(G.nodes))

        def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):

            if pos is None:
                pos = {root:(xcenter,vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)  
            if len(children)!=0:
                dx = width/len(children) 
                nextx = xcenter - width/2 - dx/2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                        vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                        pos=pos, parent = root)
            return pos

                
        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    pos = hierarchy_pos(G, rishimonkey[3][(MAP_WIDTH//2, MAP_HEIGHT//2)])
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color='r', edgecolors='k')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), style='solid', edge_color='#808080')
    plt.draw()
    plt.savefig('input_graph.png')
    plt.close()


def build_cqm(G, k, root_node):
    """Build the CQM.
    Args:
        G (Graph): Input graph to be partitioned
        k (int): Number of partitions to be used
        root_node (int): Node chosen as the root
    
    Returns:
        cqm (ConstrainedQuadraticModel): The CQM for our problem
    """

    # Set up the partitions
    partitions = range(k)

    # Initialize the CQM object
    print("\nBuilding constrained quadratic model...")
    cqm = ConstrainedQuadraticModel()

    # Add binary variables, one for each node and each partition in the graph
    print("\nAdding variables....")
    v = [[Binary(f'v_{i},{p}') for p in partitions] for i in G.nodes]

    # One-hot constraint: each node is assigned to exactly one partition
    print("\nAdding one-hot constraints...")
    for i in G.nodes:
        cqm.add_discrete([f'v_{i},{p}' for p in partitions], label=f"one-hot-node-{i}")

    # Constraint: Partitions have equal size
    print("\nAdding partition size constraint...")
    for p in partitions:
        cqm.add_constraint(quicksum(v[i][p] for i in G.nodes) == G.number_of_nodes()/k, label=f'partition-size-{p}', weight=2)

    # Objective: minimize sum of weights of nodes in same partition as root + minimize edges between partitions
    print("\nAdding objective...")
    min_edges = []
    same_partition_as_root = []
    for i, j in G.edges:
        for p in partitions:
            min_edges.append(v[i][p] + v[j][p] - 2 * v[i][p] * v[j][p])
            if i == root_node or j == root_node:
                same_partition_as_root.append(G.nodes[i]['weight'] * v[i][p] + G.nodes[j]['weight'] * v[j][p] - 2 * G.nodes[i]['weight'] * v[i][p] * v[j][p])

    # Combine objectives with a balancing factor alpha
    alpha = 500000.0  # Adjust alpha to balance the importance of objectives
    #print(min_edges)
    #print(same_partition_as_root)
    cqm.set_objective(sum(min_edges) - (alpha * sum(same_partition_as_root)))
    #print("Objective function components:")
    #print(f"Min edges component (sum): {sum(min_edges)}")
    #print(f"Same partition as root component (sum): {alpha * sum(same_partition_as_root)}")
    #print(f"Total Objective (sum): {sum(min_edges) + alpha * sum(same_partition_as_root)}")
 
    # Initialize objective components
    min_interpartition_edges = []
    min_intrapartition_edges_root = []

    root_partition = -1  # Variable to keep track of the partition of the root node

    for i, j in G.edges:
        for p in partitions:
            is_i_in_p = v[i][p]
            is_j_in_p = v[j][p]
            edge_weight = G[i][j]['weight']
            
            # Objective 1: Minimize interpartition edge weights
            # Add edge weight if i and j are in different partitions
            min_interpartition_edges.append(edge_weight * (is_i_in_p + is_j_in_p - 2 * is_i_in_p * is_j_in_p))

            # Check if either i or j is the root node and if so, note the partition
            if i == root_node or j == root_node:
                root_partition = p
            
            # Objective 2: Minimize intrapartition edge weights for partition with the root node
            # Add edge weight if both i and j are in the same partition as the root
            if root_partition != -1:
                min_intrapartition_edges_root.append(edge_weight * is_i_in_p * is_j_in_p)

    # Combine objectives with balancing factors
    alpha1 = 1.0  # Adjust as needed to balance the importance of objectives
    alpha2 = 1.0  # Adjust as needed to balance the importance of objectives
    
    total_objective = alpha1 * quicksum(min_interpartition_edges) + alpha2 * quicksum(min_intrapartition_edges_root)
    
    cqm.set_objective(total_objective)

    return cqm


def run_cqm_and_collect_solutions(cqm, sampler):
    """Send the CQM to the sampler and return the best sample found.
    Args:
        cqm (ConstrainedQuadraticModel): The CQM for our problem
        sampler: The CQM sampler to be used. Must have sample_cqm function.
    
    Returns:
        dict: The first feasible solution found
    """

    # Initialize the solver
    print("\nSending to the solver...")
    
    # Solve the CQM problem using the solver
    sampleset = sampler.sample_cqm(cqm, label='Example - Graph Partitioning', time_limit = 10)

    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    # Return the first feasible solution
    if not len(feasible_sampleset):
        print("\nNo feasible solution found.\n")
        return sampleset.first.sample

    return feasible_sampleset.first.sample


def process_sample(sample, G, k, verbose=True):
    """Interpret the sample found in terms of our graph.
    Args:
        sample (dict): Sample to be used
        G (graph): Original input graph
        k (int): Number of partitions
        verbose (bool): Trigger to print output to command-line
    
    Returns:
        soln (list): List of partitions, indexed by node
        partitions (dict): Each item is partition: [nodes in partition]
    """

    partitions = defaultdict(list)
    soln = [-1]*G.number_of_nodes()
    partition_costs = defaultdict(float)
    root_partition_cost = 0.0
    root_partition_index = -1

    for node in G.nodes:
        for p in range(k):
            if sample[f'v_{node},{p}'] == 1:
                partitions[p].append(node)
                soln[node] = p
                # Summing the weights of nodes in each partition
                partition_costs[p] += G.nodes[node]['weight']
                if node == root_node:
                    root_partition_index = p

    # Count the nodes in each partition
    counts = np.zeros(k)
    for p in partitions:
        counts[p] += len(partitions[p])

    # Compute the number of links between different partitions
    sum_diff = 0
    for i, j in G.edges:
        if soln[i] != soln[j]:
            sum_diff += 1

    if verbose:
        print("Counts in each partition: ", counts)
        print("Number of links between partitions: ", sum_diff)
        print("Number of links within partitions:", len(G.edges)-sum_diff)
        print("Costs of each partition:")
        for p, cost in partition_costs.items():
            print(f" Partition {p}: Cost = {cost}")
        if root_partition_index != -1:
            root_partition_cost = partition_costs[root_partition_index]
            print(f"Cost of the partition containing the root (Partition {root_partition_index}): {root_partition_cost}")

    return soln, partitions

def visualize_results(G, partitions, soln, root_node):
    """Visualize the partition.
    Args:
        G (graph): Original input graph
        partitions (dict): Each item is partition: [nodes in partition]
        soln (list): List of partitions, indexed by node
        root_node (int): Node chosen as the root
    
    Returns:
        None. Output is saved as output_graph.png.
    """

    print("\nVisualizing output...")

    # Build hypergraph of partitions
    hypergraph = nx.Graph()
    hypergraph.add_nodes_from(partitions.keys())
    pos_h = nx.circular_layout(hypergraph, scale=2.)

    # Place nodes within partition
    pos_full = {}
    assignments = {node: soln[node] for node in range(len(soln))}
    for node, partition in assignments.items():
        pos_full[node] = pos_h[partition]

    pos_g = {}
    for _, nodes in partitions.items():
        subgraph = G.subgraph(nodes)
        pos_subgraph = nx.random_layout(subgraph)
        pos_g.update(pos_subgraph)

    # Combine hypergraph and partition graph positions
    pos = {}
    for node in G.nodes():
        pos[node] = pos_full[node] + pos_g[node]

    # Specify color for the root node
    node_colors = [15 if node == root_node else soln[node] for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=40, node_color=node_colors, edgecolors='k')

    # Draw good and bad edges in different colors
    bad_edges = [(u, v) for u, v in G.edges if soln[u] != soln[v]]
    good_edges = [(u,v) for u, v, in G.edges if soln[u] == soln[v]]

    nx.draw_networkx_edges(G, pos, edgelist=good_edges, style='solid', edge_color='#7f7f7f')
    nx.draw_networkx_edges(G, pos, edgelist=bad_edges, style='solid', edge_color='k')

    # Save the output image
    plt.draw()
    output_name = 'output_graph.png'
    plt.savefig(output_name)

    print("\tOutput stored in", output_name)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-g", "--graph", type=click.Choice(['partition', 'internet', 'rand-reg', 'ER', 'SF']),
              help="Graph to partition.", default='partition', show_default=True)
@click.option("-n", "--nodes", help="Set graph size for graph.", default=100, type=click.IntRange(1),
              show_default=True)
@click.option("-d", "--degree", help="Set node degree for random regular graph.", default=4,
              type=click.IntRange(1), show_default=True)
@click.option("-p", "--prob", help="Set graph edge probability for ER graph. Must be between 0 and 1.",
              type=click.FloatRange(0, 1), default=0.25, show_default=True)
@click.option("-i", "--p-in", help="Set probability of edges within groups for partition graph. Must be between 0 and 1.",
              type=click.FloatRange(0, 1), default=0.5, show_default=True)
@click.option("-o", "--p-out", help="Set probability of edges between groups for partition graph. Must be between 0 and 1.",
              type=click.FloatRange(0, 1), default=0.001, show_default=True)
@click.option("-e", "--new-edges", help="Set number of edges from new node to existing node in SF graph.",
              default=4, type=click.IntRange(1), show_default=True)
@click.option("-k", "--k-partition", help="Set number of partitions to divide graph into.", default=4,
              type=click.IntRange(2), show_default=True)
def main(graph, nodes, degree, prob, p_in, p_out, new_edges, k_partition):
    G = rishimonkey[2]
    
    # Randomly assign weights to each node
    node_weights = {node: random() for node in G.nodes}
    nx.set_node_attributes(G, node_weights, 'weight')

    # Select a root node randomly

    # Inside your main function or wherever appropriate
    root_distances = compute_distances_from_root(G, root_node)
    assign_edge_weights_based_on_node_weights(G, node_weights, root_distances)

    visualize_input_graph(G)

    cqm = build_cqm(G, k_partition, root_node)

    # Initialize the CQM solver
    print("\nOptimizing on LeapHybridCQMSampler...")
    sampler = LeapHybridCQMSampler(token='DEV-b80ff71a0ffa044bc0ca11d800f92727a84eaa8a')
    
    sample = run_cqm_and_collect_solutions(cqm, sampler)
    
    if sample is not None:
        soln, partitions = process_sample(sample, G, k_partition)
        print(soln)

        visualize_results(G, partitions, soln, root_node)


if __name__ ==   '__main__':
    # pylint: disable=no-value-for-parameter
    main()