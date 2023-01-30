"""
NOTEs:

- This file is note meant to be run, it's just a collection of functions that are used in the other files. It's just a way to keep the code clean and organized.

- Why do I use os.path.join and not the "/"? Because it's more portable, it works on every OS, while "/" works only on Linux and Mac. In windows you would have to change all the "/" with "\". With os.path.join you don't have to worry about it and, as always, f*** Microsoft.
"""

from multiprocessing import Pool
import itertools
import os
import random
import wget
import zipfile
import pandas as pd
import tqdm as tqdm
import networkx as nx
from typing import Literal
from itertools import combinations
import plotly.graph_objects as go
from collections import Counter
import numpy as np
import gdown
from networkx.utils import py_random_state
import shutil

# ------------------------------------------------------------------------#

def download_datasets():

    """
    Download the datasets from the web and unzip them. The datasets are downloaded from the SNAP website and from a Google Drive folder.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    The datasets are downloaded in the "data" folder. If the folder doesn't exist, it will be created. If the dataset is already downloaded, it will be skipped. The files are renamed to make them more readable.
    """


    dict = {
        "brightkite": ["https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz", "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"],
        "gowalla": ["https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz", "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"],
        "foursquare": ["https://drive.google.com/file/d/1PNk3zY8NjLcDiAbzjABzY5FiPAFHq6T8/view?usp=sharing"]
        }

    if not os.path.exists("data"):
        os.mkdir("data")
        print("Created data folder")

    for folder in dict.keys():
        if not os.path.exists(os.path.join("data", folder)):
            os.mkdir(os.path.join("data", folder))
            print("Created {} folder".format(folder))

    ## DOWNLOADING ##

    for folder in dict.keys():
        for url in dict[folder]:
            if folder == "foursquare":
                if not os.path.exists(os.path.join("data", folder, "foursquare_full.zip")):
                    output = os.path.join("data", folder, "foursquare_full.zip")
                    gdown.download(url, output, quiet=False, fuzzy=True)
                else :
                    print("{} already downloaded".format(url))
            else:
                if not os.path.exists(os.path.join("data", folder, url.split("/")[-1])):
                    print("Downloading {}...".format(url))
                    wget.download(url, os.path.join("data", folder))
                else :
                    print("{} already downloaded".format(url))

    ## UNZIPPING ##

    for folder in dict.keys():
        for file in os.listdir(os.path.join("data", folder)):
            if file.endswith(".gz"):
                print("Unzipping {}...".format(file))
                os.system("gunzip {}".format(os.path.join("data", folder, file)))
            elif file.endswith(".zip"):
                print("Unzipping {}...".format(file))
                os.system("unzip -o {} -d {}".format(os.path.join("data", folder, file), os.path.join("data", folder)))
                os.remove(os.path.join("data", folder, file))

    ## FOURSQUARE CLEANING ##

    for file in os.listdir(os.path.join("data", "foursquare", "dataset_WWW2019")):
        if file.endswith(".txt"):
            os.rename(os.path.join("data", "foursquare", "dataset_WWW2019", file), os.path.join("data", "foursquare", file))

    for file in ["dataset_WWW_friendship_old.txt", "dataset_WWW_readme.txt", "raw_Checkins_anonymized.txt"]:
        os.remove(os.path.join("data", "foursquare", file))

    shutil.rmtree(os.path.join("data", "foursquare", "dataset_WWW2019"))
    shutil.rmtree(os.path.join("data", "foursquare", "__MACOSX"))

    os.rename(os.path.join("data", "foursquare", "dataset_WWW_friendship_new.txt"), os.path.join("data", "foursquare", "foursquare_friends_edges.txt"))

    os.rename(os.path.join("data", "foursquare", "dataset_WWW_Checkins_anonymized.txt"), os.path.join("data", "foursquare", "foursquare_checkins.txt"))

    ## BRIGHTKITE CLEANING ##

    os.rename(os.path.join("data", "brightkite", "loc-brightkite_totalCheckins.txt"), os.path.join("data", "brightkite", "brightkite_checkins.txt"))

    os.rename(os.path.join("data", "brightkite", "loc-brightkite_edges.txt"), os.path.join("data", "brightkite", "brightkite_friends_edges.txt"))

    ## GOWALLA CLEANING ##

    os.rename(os.path.join("data", "gowalla", "loc-gowalla_totalCheckins.txt"), os.path.join("data", "gowalla", "gowalla_checkins.txt"))

    os.rename(os.path.join("data", "gowalla", "loc-gowalla_edges.txt"), os.path.join("data", "gowalla", "gowalla_friends_edges.txt"))

# ------------------------------------------------------------------------#

def create_graph_from_checkins(dataset: Literal['brightkite', 'gowalla', 'foursquareEU', 'foursquareIT'], create_file = True) -> nx.Graph:

    """
    Create a graph from the checkins of the dataset. The graph is undirected and the nodes are the users and the edges are the checkins in common.

    Parameters
    ----------
    `dataset` : Literal['brightkite', 'gowalla', 'foursquare']
        The dataset to use.
    `create_file` : bool, optional
        If True, the graph is saved in a file, by default True

    Returns
    -------
    `G` : networkx.Graph

    Raises
    ------
    ValueError
        If the dataset is not valid.

    """

    if dataset not in ['brightkite', 'gowalla', 'foursquare']:
        raise ValueError("Dataset not valid. Please choose between brightkite, gowalla, foursquare")


    file = os.path.join("data", dataset, dataset + "_checkins.txt")

    print("\nCreating the graph for the dataset {}...".format(dataset))

    df = pd.read_csv(file, sep="\t", header=None, names=["user_id", "venue_id"], engine='pyarrow')

    G = nx.Graph()
    venues_users = df.groupby("venue_id")["user_id"].apply(set)

    for users in tqdm.tqdm(venues_users):
        for user1, user2 in combinations(users, 2):
            G.add_edge(user1, user2)

    # path to the file where we want to save the graph
    edges_path = os.path.join("data", dataset , dataset + "_checkins_edges.tsv")

    print("Done! The graph has {} edges".format(G.number_of_edges()), " and {} nodes".format(G.number_of_nodes()))

    # delete from memory the dataframe
    del df

    if create_file:
        # save the graph in a file
        nx.write_edgelist(G, edges_path, data=True, delimiter="\t", encoding="utf-8")

    return G


# ------------------------------------------------------------------------#

def create_friendships_graph(dataset: Literal['brightkite', 'gowalla', 'foursquareEU', 'foursquareIT']) -> nx.Graph:

    """
    Create the graph of friendships for the dataset brightkite, gowalla or foursquare.
    The graph is saved in a file.

    Parameters
    ----------
    `dataset` : str
        The dataset for which we want to create the graph of friendships.

    Returns
    -------
    `G` : networkx.Graph
        The graph of friendships.

    Notes
    -----
    Since we are taking sub-samples of each check-ins dataset, we are also taking sub-samples of the friendship graph. A user is included in the friendship graph if he has at least one check-in in the sub-sample.
    """

    if dataset not in ["brightkite", "gowalla", "foursquare"]:
        raise ValueError("The dataset must be brightkite, gowalla or foursquare")


    file = os.path.join("data", dataset, dataset + "_friends_edges.txt")

    df_friends_all = pd.read_csv(file, sep="\t", header=None, names=["node1", "node2"], engine='pyarrow')
    unique_friends = set(df_friends_all["node1"].unique()).union(set(df_friends_all["node2"].unique()))

    df_checkins = pd.read_csv(os.path.join("data", dataset, dataset + "_checkins_edges.tsv"), sep="\t", header=None, names=["node1", "node2"])
    unique_checkins = set(df_checkins["node1"].unique()).union(set(df_checkins["node2"].unique()))

    unique_users = unique_friends.intersection(unique_checkins)

    df = df_friends_all[df_friends_all["node1"].isin(unique_users) & df_friends_all["node2"].isin(unique_users)]

    df.to_csv(os.path.join("data", dataset, dataset + "_friends_edges_filtered.tsv"), sep="\t", header=False, index=False)

    G = nx.from_pandas_edgelist(df, "node1", "node2", create_using=nx.Graph())
    del df_friends_all, df_checkins, df

    return G

# ------------------------------------------------------------------------#

def degree_distribution(G: nx.Graph, log: bool = True, save: bool = False) -> None:

        """
        This function takes in input a networkx graph object and plots the degree distribution of the graph.

        Parameters
        ----------
        `G` : networkx graph object
            The graph object

        `log` : bool, optional
            If True, the plot will be in log-log scale, by default True

        `save` : bool, optional
            If True, the plot will be saved in the folder "plots", by default False

        Returns
        -------
        None

        Notes
        -----
        Due to the characteristics of datasets, not using a log log scale will lead to a un-useful plot. Even if using a log scales alters the power-law distribution, it is still clearly visible and distinguishable from a poisson distribution (witch is what we are interested in in this case)

        """

        degrees = [G.degree(n) for n in G.nodes()]
        degreeCount = Counter(degrees)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(degreeCount.keys()), y=list(degreeCount.values()), name='Degree Distribution'))

        if log:
            fig.update_layout(
                title='Degree Distribution (log-log scale) of {}' .format(G.name),
                xaxis_title='Degree',
                yaxis_title='Number of Nodes',
                xaxis_type='log',
                yaxis_type='log',
                width=800,
                height=600,
                template='plotly_white'
            )

        else:
            fig.update_layout(
                title='Degree Distribution of {}' .format(G.name),
                xaxis_title='Degree',
                yaxis_title='Number of Nodes',
                width=800,
                height=600,
                template='plotly_white'
            )

        fig.show()

        if save:
            fig.write_image("plots/degree_distribution_{}.png".format(G.name))

# ------------------------------------------------------------------------#

def chunks(l, n):
    """
    Auxiliary function to divide a list of nodes `l` in `n` chunks

    Parameters
    ----------
    `l` : list
        List of nodes

    `n` : int
        Number of chunks

    """

    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x

# ------------------------------------------------------------------------#

def betweenness_centrality_parallel(G, processes=None, k =None) -> dict:
    """
    Compute the betweenness centrality for nodes in a graph using multiprocessing.

    Parameters
    ----------
    G : graph
        A networkx graph

    processes : int, optional
        The number of processes to use for computation.
        If `None`, then it sets processes = 1

    k : int, optional
        Percent of nodes to sample. If `None`, then all nodes are used.

    seed : int, optional
        Seed for random number generator (default=None).

    Returns
    -------
    dict

    Notes
    -----
    Do not use more then 6 process for big graphs, otherwise the memory will be full. Do it only if you have more at least 32 GB of RAM. For small graphs, you can use more processes.

    """

    # if process is None or 1, run the standard algorithm with one process
    if processes is None or processes == 1:
        print("\tRunning the networkx approximated algorithm with just one process")
        G_copy = G.copy()
        sample = int((k)*G_copy.number_of_nodes())
        print("\tNumber of nodes after removing {} % of nodes: {}" .format((k)*100, G_copy.number_of_nodes()))
        return np.mean(nx.betweenness_centrality(G, k=sample, seed=42).values())

    if processes > os.cpu_count():
        raise ValueError("The number of processes must be less than the number of cores in the system.")

    if k is not None:
        if (k < 0 or k > 1):
            raise ValueError("k must be between 0 and 1.")
        else:
            G_copy = G.copy()
            G_copy.remove_nodes_from(random.sample(G_copy.nodes(), int((k)*G_copy.number_of_nodes())))
            print("\tNumber of nodes after removing {}% of nodes: {}" .format((k)*100, G_copy.number_of_nodes()))
            print("\tNumber of edges after removing {}% of nodes: {}" .format((k)*100, G_copy.number_of_edges()))

    if k is None:
        G_copy = G.copy()

    p = Pool(processes=processes)
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(G_copy.nodes(), G_copy.order() // node_divisor))
    num_chunks = len(node_chunks)
    bt_sc = p.starmap(
        nx.betweenness_centrality_subset,
        zip(
            [G_copy] * num_chunks, # this returns a list of Gs
            node_chunks,
            [list(G_copy)] * num_chunks, # this returns a list of lists of nodes
            [True] * num_chunks,
            [None] * num_chunks,
        ),
    )

    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]

    return bt_c

# ------------------------------------------------------------------------#

def average_shortest_path(G: nx.Graph, k=None) -> float:

        """
        This function takes in input a networkx graph and returns the average shortest path length of the graph. This works also for disconnected graphs.

        Parameters
        ----------
        `G` : networkx graph
            The graph to compute the average shortest path length of.
        `k` : int
            percentage of nodes to remove from the graph. If k is None, the average shortest path length of each connected component is computed using all the nodes of the connected component.

        Returns
        -------
        float
            The average shortest path length of the graph.

        Raises
        ------
        ValueError
            If k is not between 0 and 1
        """

        if k is not None and (k < 0 or k > 1):
            raise ValueError("k must be between 0 and 1")
        elif k is None:
            G_copy = G.copy()
            connected_components = list(nx.connected_components(G))
        else:
            G_copy = G.copy()
            # remove the k% of nodes from G
            G_copy.remove_nodes_from(random.sample(G_copy.nodes(), int((k)*G_copy.number_of_nodes())))
            print("\tNumber of nodes after removing {}% of nodes: {}" .format((k)*100, G_copy.number_of_nodes()))
            print("\tNumber of edges after removing {}% of nodes: {}" .format((k)*100, G_copy.number_of_edges()))

        tmp = 0
        connected_components = list(nx.connected_components(G_copy))
        # remove all the connected components with less than 10 nodes
        connected_components = [c for c in connected_components if len(c) > 10]

        print("\tNumber of connected components with more then 10 nodes: {}" .format(len(connected_components)), "\r")
        for C in (G_copy.subgraph(c).copy() for c in connected_components):
            print("\tComputing average shortest path length of connected component with {} nodes and {} edges" .format(C.number_of_nodes(), C.number_of_edges()), "\r", end="")
            tmp += nx.average_shortest_path_length(C)

        return np.mean(tmp)

# ------------------------------------------------------------------------#

def average_clustering_coefficient(G: nx.Graph, k=None) -> float:

    """
    This function takes in input a networkx graph and returns the average clustering coefficient of the graph. This works also for disconnected graphs.

    Parameters
    ----------
    G : networkx graph
        The graph to compute the average clustering coefficient of.
    k : int
        percentage of nodes to remove from the graph. If k is None, the average clustering coefficient of each connected component is computed using all the nodes of the connected component.

    Returns
    -------
    float
        The average clustering coefficient of the graph.

    Raises
    ------
    ValueError
        If k is not between 0 and 1
    """

    if k is not None and (k < 0 or k > 1):
        raise ValueError("k must be between 0 and 1")

    elif k is None:
        return nx.average_clustering(G)

    else:
        G_copy = G.copy()
        G_copy.remove_nodes_from(random.sample(list(G_copy.nodes()), int((k)*G_copy.number_of_nodes())))
        print("\tNumber of nodes after removing {}% of nodes: {}" .format((k)*100, G_copy.number_of_nodes()))
        return nx.average_clustering(G_copy)


def generalized_average_clustering_coefficient(G: nx.Graph) -> float:

    """
    Generalized definition of the average clustering coefficient of a graph. It better applies to small world networks and it's way more efficient than the average_clustering_coefficient function with the standard definition of the clustering coefficient.

    Parameters
    ----------
    G : networkx graph
        The graph to compute the generalized average clustering coefficient of.

    Returns
    -------
    float
        The generalized average clustering coefficient of the graph.
    """

    C = 0
    for node in G.nodes():
        k = G.degree(node)
        C += (3*(k-1))/(2*(2*k - 1))

    return C/G.number_of_nodes()
# ------------------------------------------------------------------------#

def create_random_graphs(G: nx.Graph, model = None, save = True) -> nx.Graph:

    """Create a random graphs of the same model of the original graph G.

    Parameters
    ----------
    G : nx.Graph
        The original graph.
    model : str
        The model to use to generate the random graphs. It can be one of the following: "erdos", "watts_strogatz"
    save: bool
        If True, the random graph is saved in the folder data/random/model

    Returns
    -------
    G_random : nx.Graph

    """

    if model is None:
        model = "erdos_renyi"

    if model == "erdos_renyi":
        G_random = nx.erdos_renyi_graph(G.number_of_nodes(), nx.density(G))
        print("Creating a random graph with the Erdos-Renyi model {}" .format(G.name))
        print("Number of edges in the original graph: {}" .format(G.number_of_edges()))
        print("Number of edges in the random graph: {}" .format(G_random.number_of_edges()))
        G_random.name = G.name + " Erdos-Renyi"

        if save:
            # check if the folder exists, otherwise create it
            if not os.path.exists(os.path.join('data', 'random', 'erdos')):
                os.makedirs(os.path.join('data', 'random', 'erdos'))

            nx.write_gpickle(G_random, os.path.join('data', 'random', 'erdos', "erdos_" + str(G.number_of_nodes()) + "_" + str(G_random.number_of_edges()) + ".gpickle"))
            print("\tThe file graph has been saved in the folder data/random/erdos with the syntax erdos_n_nodes_n_edges.gpickle")

        return G_random

    elif model == "watts_strogatz":
        p = G.number_of_edges() / (G.number_of_nodes())
        avg_degree = int(np.mean([d for n, d in G.degree()]))
        G_random = nx.watts_strogatz_graph(G.number_of_nodes(), avg_degree, p)
        print("Number of edges in the original graph: {}" .format(G.number_of_edges()))
        print("Number of edges in the random graph: {}" .format(G_random.number_of_edges()))
        G_random.name = G.name + " Watts-Strogatz"

        if save:
            # check if the folder exists, otherwise create it
            if not os.path.exists(os.path.join('data', 'random', 'watts_strogatz')):
                os.makedirs(os.path.join('data', 'random', 'watts_strogatz'))

            nx.write_gpickle(G_random, os.path.join('data', 'random', 'watts_strogatz', "watts_strogatz_" + str(G.number_of_nodes()) + "_" + str(G_random.number_of_edges()) + ".gpickle"))
            print("\tThe file graph has been saved in the folder data/random/watts_strogatz with the syntax watts_strogatz_n_nodes_n_edges.gpickle")

        return G_random
