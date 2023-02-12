#! /usr/bin/python3

from queue import Empty
import multiprocessing
import networkx as nx
import numpy as np
import argparse
import random
import time
from utils import *

"""
This script computes the omega index for a given graph using parallelization. To see the implementation of the omega index, see the file omega.py in the same folder and check the function parallel_omega
"""

### PARSING ARGUMENTS ###

parser = argparse.ArgumentParser()

parser.add_argument("graph", help="Name of the graph to be used.", choices=['checkins-foursquare', 'checkins-gowalla', 'checkins-brightkite', 'friends-foursquare', 'friends-gowalla', 'friends-brightkite'])
parser.add_argument("--k", help="Percentage of nodes to be sampled. Needs to be a float between 0 and 1. Default is 0.", default=0, type=float)
parser.add_argument("--nrand", help="Number of random graphs. Needs to be an integer. Default is 12", default=12, type=int)
parser.add_argument("--niter", help="Approximate number of rewiring per edge to compute the equivalent random graph. Default is 12", default=12, type=int)
parser.add_argument("--processes", help="Number of processes to be used. Needs to be an integer. Default is the number of cores.", default=multiprocessing.cpu_count(), type=int)
parser.add_argument("--seed", help="Seed for the random number generator. Needs to be an integer. Default is 42", default=42, type=int)

parser.add_help = True
args = parser.parse_args()

# check if the number of processes is valid
if args.processes > multiprocessing.cpu_count():
    print("Number of processes is higher than available. Setting it to default value: all available")
    args.processes = multiprocessing.cpu_count()
elif args.processes < 1:
    raise ValueError("Number of processes needs to be at least 1")

# the name of the graph is the first part of the input string
name = args.graph.split('-')[1]
if 'checkins' in args.graph:
    G = create_graph_from_checkins(name) #function from utils.py, check it out there
elif 'friends' in args.graph:
    G = create_friendships_graph(name) #function from utils.py, check it out there
G.name = str(args.graph) + " Checkins Graph"

print("\nThe full graph {} has {} nodes and {} edges".format(args.graph, len(G), G.number_of_edges()))
print("Number of processes used: ", args.processes)

start = time.time()
# function from utils.py, check it out there (it's the parallel version of the omega index)
omega = parallel_omega(G, k = args.k, nrand=args.nrand, niter=args.niter, n_processes=args.processes, seed=42)
end = time.time()

print("\nOmega: ", omega)
print("Number of random graphs: ", args.nrand)
print("Time: ", round(end - start, 2), " seconds")
