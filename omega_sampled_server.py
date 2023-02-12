#! /usr/bin/python3

import time
import argparse
import networkx as nx
from utils import *

"""
Standard function to compute the omega index for a given graph. To see the implementation of the omega index, refer to the networkx documentation

https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.smallworld.omega.html#networkx.algorithms.smallworld.omega

This file has been created to be used with the server. It takes as input the name of the graph and the percentage of nodes to be sampled. It then computes the omega index for the sampled graph and returns the result. Run

```
./omega_sampled_server.py -h
```

to see the list of available graphs and the other parameters that can be passed as input.
"""

parser = argparse.ArgumentParser()

parser.add_argument("graph", help="Name of the graph to be used.", choices=['checkins-foursquare', 'checkins-gowalla', 'checkins-brightkite', 'friends-foursquare', 'friends-gowalla', 'friends-brightkite'])
parser.add_argument("--k", help="Percentage of nodes to be sampled. Needs to be a float between 0 and 1", default=0)
parser.add_argument("--niter", help="Number of rewiring per edge. Needs to be an integer. Default is 5", default=5)
parser.add_argument("--nrand", help="Number of random graphs. Needs to be an integer. Default is 5", default=5)

parser.add_help = True
args = parser.parse_args()

# the name of the graph is the first part of the input string
name = args.graph.split('-')[1]
if 'checkins' in args.graph:
    G = create_graph_from_checkins(name)
elif 'friends' in args.graph:
    G = create_friendships_graph(name)
G.name = str(args.graph) + " Checkins Graph"

# sample the graph
G_sample = random_sample(G, float(args.k)) # function from utils.py, check it out there

# compute omega
start = time.time()
print("\nComputing omega for graph: ", G.name)
omega = nx.omega(G_sample, niter = int(args.niter), nrand = int(args.nrand))
end = time.time()
print("\nOmega coefficient for graph {}: {}".format(G.name, omega))
print("Time taken: ", round(end-start,2), " seconds")
