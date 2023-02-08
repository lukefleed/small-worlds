#! /usr/bin/python3

import multiprocessing
import networkx as nx
import numpy as np
import argparse
import random
import time
from utils import *

def parallel_omega(G: nx.Graph, k: float, nrand: int, niter: int, n_processes: int, seed: int) -> float:
    """
    Computes the omega index for a given graph using parallelization.

    Parameters
    ----------
    function to compute the omega index of a graph in parallel. This is a much faster approach then the standard omega function. It parallelizes the computation of the random graphs and lattice networks.

    Parameters
    ----------
    `G`: nx.Graph
        The graph to compute the omega index

    `k`: float
        The percentage of nodes to sample from the graph.

    `niter`: int
        Approximate number of rewiring per edge to compute the equivalent random graph. Default is 12.

    `nrand`: int
        Number of random graphs generated to compute the maximal clustering coefficient (Cr) and average shortest path length (Lr). Default is 12

    `n_processes`: int
        Number of processes to use. Default is the number of cores of the machine.

    `seed`: int
        The seed to use to generate the random graphs. Default is 42.

    Returns
    -------
    `omega`: float
    """

    random.seed(seed)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len))

    if len(G) == 1:
        return 0

    if k is not None:
        G = random_sample(G, k)

    def worker(queue): # worker function to be used in parallel
        while True:
            task = queue.get()
            if task is None:
                break
            random_graph = nx.random_reference(G, niter, seed=seed)
            lattice_graph = nx.lattice_reference(G, niter, seed=seed)
            random_shortest_path = nx.average_shortest_path_length(random_graph)
            lattice_clustering = nx.average_clustering(lattice_graph)
            queue.put((random_shortest_path, lattice_clustering))

    manager = multiprocessing.Manager() # manager to share the queue
    queue = manager.Queue() # queue to share the results
    processes = [multiprocessing.Process(target=worker, args=(queue,)) for _ in range(n_processes)] # processes to be used
    for process in processes: # start the processes
        process.start()

    for _ in range(nrand): # put the tasks in the queue
        queue.put(1)

    for _ in range(n_processes): # put the stop signals in the queue
        queue.put(None)

    for process in processes: # wait for the processes to finish
        process.join()

    # collect the results
    shortest_paths = []
    clustering_coeffs = []
    while not queue.empty():
        random_shortest_path, lattice_clustering = queue.get() # get the results from the queue
        shortest_paths.append(random_shortest_path)
        clustering_coeffs.append(lattice_clustering)

    L = nx.average_shortest_path_length(G)
    C = nx.average_clustering(G)

    omega = (np.mean(shortest_paths) / L) - (C / np.mean(clustering_coeffs))
    return omega


graphs = ['checkins-foursquare', 'checkins-gowalla', 'checkins-brightkite', 'friends-foursquare', 'friends-gowalla', 'friends-brightkite']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("graph", help="Name of the graph to be used. Options are 'checkins-foursquare',  'checkins-gowalla', 'checkins-brightkite', 'friends-foursquare', 'friends-gowalla', 'friends-brightkite'")

    parser.add_argument("--k", help="Percentage of nodes to be sampled. Needs to be a float between 0 and 1. Default is 0.1", default=None, type=float)

    parser.add_argument("--nrand", help="Number of random graphs. Needs to be an integer. Default is 5", default=12, type=int)

    parser.add_argument("--niter", help="Approximate number of rewiring per edge to compute the equivalent random graph. Default is 12", default=12, type=int)

    parser.add_argument("--processes", help="Number of processes to be used. Needs to be an integer. Default is all available", default=multiprocessing.cpu_count(), type=int)

    parser.add_argument("--seed", help="Seed for the random number generator. Needs to be an integer. Default is 42", default=42, type=int)

    parser.add_help = True
    args = parser.parse_args()

    graphs = ['checkins-foursquare', 'checkins-gowalla', 'checkins-brightkite', 'friends-foursquare', 'friends-gowalla', 'friends-brightkite']

    if args.graph not in graphs:
        raise ValueError("Graph name is not valid. Options are 'checkins-foursquare',  'checkins-gowalla', 'checkins-brightkite', 'friends-foursquare', 'friends-gowalla', 'friends-brightkite'")

    if args.processes > multiprocessing.cpu_count():
        print("Number of processes is higher than available. Setting it to default value: all available")
        args.processes = multiprocessing.cpu_count()
    elif args.processes < 1:
        raise ValueError("Number of processes needs to be at least 1")

    name = args.graph.split('-')[1]
    if 'checkins' in args.graph:
        G = create_graph_from_checkins(name)
    elif 'friends' in args.graph:
        G = create_friendships_graph(name)
    G.name = str(args.graph) + " Checkins Graph"

    results = {}
    for graph in graphs:
        print("\nComputing omega for graph {} with {} nodes and {} edges".format(graph, len(G), G.number_of_edges()))
        print("Number of processes used: ", multiprocessing.cpu_count())

        start = time.time()
        omega = parallel_omega(G, k = float(args.k), nrand=int(args.nrand), niter=int(args.niter), n_processes=int(args.processes), seed=42)
        end = time.time()

        print("Omega: ", omega)
        print("Number of random graphs: ", args.nrand)
        print("Number of processes used: ", args.processes)
        print("Time: ", end - start)

        results[graph] = omega

    with open('results.tsv', 'w') as f:
        for key in results.keys():
            f.write("%s\t%s\n" % (key, results[key]))

## Variant if you want to run it on a server for all the graphs

# if __name__ == "__main__":
#     results = {}

#     # loop in reverse order
#     for graph in graphs[::-1]:
#         print("\nComputing omega for graph: ", graph)
#         print("Number of processes used: ", multiprocessing.cpu_count())

#         if 'checkins' in graph:
#             G = create_graph_from_checkins(graph.split('-')[1])
#         elif 'friends' in graph:
#             G = create_friendships_graph(graph.split('-')[1])

#         start = time.time()
#         omega = parallel_omega(G)
#         end = time.time()

#         print("Omega: ", omega)
#         print("Number of random graphs: ", 12)
#         print("Time: ", end - start)

#         results[graph] = omega

#     with open('results.tsv', 'w') as f:
#         for key in results.keys():
#             f.write("%s\t%s\n" % (key, results[key]))
