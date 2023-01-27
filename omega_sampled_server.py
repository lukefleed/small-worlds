#! /usr/bin/python3

import networkx as nx
from utils import *
import warnings
import time
import random
import argparse
warnings.filterwarnings("ignore")

def random_sample(graph, k):
    nodes = list(graph.nodes())
    n = int(k*len(nodes))
    nodes_sample = random.sample(nodes, n)

    G = graph.subgraph(nodes_sample)

    if not nx.is_connected(G):
        print("Graph is not connected. Taking the largest connected component")
        connected = max(nx.connected_components(G), key=len)
        G_connected = graph.subgraph(connected)

    print(nx.is_connected(G_connected))

    print("Number of nodes in the sampled graph: ", G.number_of_nodes())
    print("Number of edges in the sampled graph: ", G.number_of_edges())

    return G_connected

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", help="Name of the graph to be used. Options are 'checkins-foursquare',  'checkins-gowalla', 'checkins-brightkite', 'friends-foursquare', 'friends-gowalla', 'friends-brightkite'")
    parser.add_argument("k", help="Percentage of nodes to be sampled. Needs to be a float between 0 and 1")
    parser.add_argument("niter", help="Number of rewiring per edge. Needs to be an integer. Default is 5")
    parser.add_argument("nrand", help="Number of random graphs. Needs to be an integer. Default is 5")
    parser.add_help = True
    args = parser.parse_args()

    # if no input is given for niter and nrand, set them to default values
    if args.niter == None:
        print("No input for niter. Setting it to default value: 5")
        args.niter = 5

    if args.nrand == None:
        print("No input for nrand. Setting it to default value: 5")
        args.nrand = 5

    # the name of the graph is the first part of the input string
    name = args.graph.split('-')[1]
    if 'checkins' in args.graph:
        G = create_graph_from_checkins(name)
    elif 'friends' in args.graph:
        G = create_friendships_graph(name)
    G.name = str(args.graph) + " Checkins Graph"

    # sample the graph
    G_sample = random_sample(G, float(args.k))

    # compute omega
    start = time.time()
    print("\nComputing omega for graph: ", G.name)
    omega = nx.omega(G_sample, niter = int(args.niter), nrand = int(args.nrand))
    end = time.time()
    print("Omega coefficient for graph {}: {}".format(G.name, omega))
    print("Time taken: ", round(end-start,2))
