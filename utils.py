"""
NOTEs:

- This file is note meant to be run, it's just a collection of functions that are used in the other files. It's just a way to keep the code clean and organized.

- Why do I use os.path.join and not the "/"? Because it's more portable, it works on every OS, while "/" works only on Linux and Mac. In windows you would have to change all the "/" with "\". With os.path.join you don't have to worry about it and, as always, f*** Microsoft.
"""

import os
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


# ------------------------------------------------------------------------#

def download_datasets():

    urls = [
        ["https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz", "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"],
        ["https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz", "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"],
        ["https://drive.google.com/file/d/1PNk3zY8NjLcDiAbzjABzY5FiPAFHq6T8/view?usp=sharing"]
    ]

    folders = ["brightkite", "gowalla", "foursquare"]

    if not os.path.exists("data"):
        os.mkdir("data")

    for folder in folders:
        if not os.path.exists(os.path.join("data", folder)):
            os.mkdir(os.path.join("data", folder))

    # Download every url in their respective folder. For the last one, we have to use gdown, because it's a google drive link. If the file is already downloaded, skip the download

    for i in range(len(urls)):
        for url in urls[i]:
            if not os.path.exists(os.path.join("data", folders[i], url.split("/")[-1])):
                if i == 2:
                    output = os.path.join("data", folders[i], "something.zip")
                    gdown.download(url, output, quiet=False, fuzzy=True)
                else:
                    wget.download(url, os.path.join("data", folders[i]))

    # unzip all the files in the 3 folders. Then remove the .gz or .zip files

    for folder in folders:
        for file in os.listdir(os.path.join("data", folder)):
            print(folder, file)
            if file.endswith(".gz"):
                os.system("gunzip {}".format(os.path.join("data", folder, file)))
            elif file.endswith(".zip"):
                os.system("unzip {}".format(os.path.join("data", folder, file)))
                os.remove(os.path.join("data", folder, file))

    # take all the .txt files from data/foursquare/dataset_WWW2019 and move them to data/foursquare

    for file in os.listdir(os.path.join("data", "foursquare", "dataset_WWW2019")):
        if file.endswith(".txt"):
            os.rename(os.path.join("data", "foursquare", "dataset_WWW2019", file), os.path.join("data", "foursquare", file))

    # remove the dataset_WWW2019 folder, note that is not empty
    # os.rmdir(os.path.join("data", "foursquare", "dataset_WWW2019"))

    for file in ["dataset_WWW_friendship_old.txt", "dataset_WWW_readme.txt", "raw_Checkins_anonymized.txt", "raw_POIs.txt"]:
        os.remove(os.path.join("data", "foursquare", file))

    # Now we want to clean our data and rename the files.

    for file in os.listdir(os.path.join("data", "brightkite")):
        if file.endswith("_edges.txt"):
            os.rename(os.path.join("data", "brightkite", file), os.path.join("data", "brightkite", "brightkite_friends_edges.txt"))

    for file in os.listdir(os.path.join("data", "gowalla")):
        if file.endswith("_edges.txt"):
            os.rename(os.path.join("data", "gowalla", file), os.path.join("data", "gowalla", "gowalla_friends_edges.txt"))

    for file in os.listdir(os.path.join("data", "foursquare")):
        if file.endswith("dataset_WWW_friendship_new.txt"):
            os.rename(os.path.join("data", "foursquare", file), os.path.join("data", "foursquare", "foursquare_friends_edges.txt"))

    # Now we from the _totalCheckins.txt files we want to keep only the first and last column, which are the user ID and the venue ID. We also want to remove the header of the file.

    for file in os.listdir(os.path.join("data", "brightkite")):
        if file.endswith("_totalCheckins.txt"):
            df = pd.read_csv(os.path.join("data", "brightkite", file), sep="\t", header=None, names=["user_id", "check-in time", "latitude", "longitude", "venue_id"])
            df = df[["user_id", "venue_id"]]
            df.to_csv(os.path.join("data", "brightkite", "brightkite_checkins.txt"), sep="\t", header=False, index=False, errors="ignore", encoding="utf-8")
            os.remove(os.path.join("data", "brightkite", file))

    for file in os.listdir(os.path.join("data", "gowalla")):
        if file.endswith("_totalCheckins.txt"):
            df = pd.read_csv(os.path.join("data", "gowalla", file), sep="\t", header=None, names=["user_id", "check-in time", "latitude", "longitude", "venue_id"])
            df["check-in time"] = pd.to_datetime(df["check-in time"])
            df = df[df["check-in time"].dt.year == 2010]
            df = df.drop(["check-in time", "latitude", "longitude"], axis=1)
            df.to_csv(os.path.join("data", "gowalla", "gowalla_checkins.txt"), sep="\t", header=False, index=False, errors="ignore", encoding="utf-8")
            os.remove(os.path.join("data", "gowalla", file))

    for file in os.listdir(os.path.join("data", "foursquare")):
        if file.endswith("dataset_WWW_Checkins_anonymized.txt"):
            df = pd.read_csv(os.path.join("data", "foursquare", file), sep="\t", header=None)
            df = df[[0, 1]]
            df.to_csv(os.path.join("data", "foursquare", "foursquare_checkins.txt"), sep="\t", header=False, index=False, errors="ignore", encoding="utf-8")
            os.remove(os.path.join("data", "foursquare", file))

# ------------------------------------------------------------------------#


def create_graph_from_checkins(dataset: Literal['brightkite', 'gowalla', 'foursquareEU', 'foursquareIT'], create_file = True) -> nx.Graph:

    if dataset not in ['brightkite', 'gowalla', 'foursquareEU', 'foursquareIT']:
        raise ValueError("Dataset not valid. Please choose between brightkite, gowalla, foursquareEU, foursquareUS, foursquareIT")

    if dataset in ['brightkite', 'gowalla']:
        file = os.path.join("data", dataset, dataset + "_checkins.txt")

        print("\nCreating the graph for the dataset {}...".format(dataset))

        df = pd.read_csv(file, sep="\t", header=None, names=["user_id", "venue_id"])

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

    else:
        # path to the checkins file and the POIS file
        path_checkins = os.path.join("data", "foursquare", "foursquare_checkins.txt")
        path_POIS = os.path.join("data", "foursquare", "raw_POIs.txt")

        # dataframe with the checkins, we need only the user_id and the venue_id
        df_all = pd.read_csv(path_checkins, sep="\t", header=None, names=['user_id', 'venue_id', 'time', 'offset'])
        df_all = df_all[['user_id', 'venue_id']]

        # dataframe with the POIS, we need only the venue_id and the country code
        df_POIS = pd.read_csv(path_POIS, sep='\t', header=None, names=['venue_id', 'lat', 'lon', 'category', 'country code'])
        df_POIS = df_POIS[['venue_id', 'country code']]

        if dataset == "foursquareIT":
            venues_array = df_POIS[df_POIS['country code'] == 'IT']['venue_id'].values

        elif dataset == "foursquareEU":
            # list of the countries in the EU
            EU_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']

            venues_array = df_POIS[df_POIS['country code'].isin(EU_countries)]['venue_id'].values

        print("\nCreating the graph for the dataset {}...".format(dataset))

        # we create a dataframe with the checkins in the corresponding country
        df_country = df_all[df_all['venue_id'].isin(venues_array)]

        G = nx.Graph()
        venues_users = df_country.groupby("venue_id")["user_id"].apply(set)

        for users in tqdm.tqdm(venues_users):
            for user1, user2 in combinations(users, 2):
                G.add_edge(user1, user2)

        # path to the file where we want to save the graph
        edges_path = os.path.join("data", "foursquare", dataset + "_checkins_edges.tsv")

        print("Done! The graph has {} edges".format(G.number_of_edges()), " and {} nodes".format(G.number_of_nodes()))

        # delete from memory the dataframes
        del df_all, df_POIS, df_country

        if create_file:
            # save the graph in a file
            nx.write_edgelist(G, edges_path, data=True, delimiter="\t", encoding="utf-8")

        return G

# ------------------------------------------------------------------------#

def create_friendships_graph(dataset: Literal['brightkite', 'gowalla', 'foursquareEU', 'foursquareIT']) -> nx.Graph:

    """
    This function takes in input a tsv file with two columns, Each line in the file is an edge. The function returns an undirected networkx graph object.
    """

    if dataset not in ["brightkite", "gowalla", "foursquareEU", "foursquareIT"]:
        raise ValueError("The dataset must be brightkite, gowalla or foursquare")

    if dataset in ["foursquareEU", "foursquareIT"]:
        file = os.path.join("data", "foursquare", "foursquare_friends_edges.txt")

        # dataframe with the edges of the graph (friends)
        df_friends_all = pd.read_csv(file, sep="\t", header=None, names=["node1", "node2"])

        # set of the unique users in the graph (friends)
        unique_friends = set(df_friends_all["node1"].unique()).union(set(df_friends_all["node2"].unique()))

        # dataframe with the edges of the graph (checkins)
        df_checkins = pd.read_csv(os.path.join("data", "foursquare", dataset + "_checkins_edges.tsv"), sep="\t", header=None, names=["node1", "node2"])
        unique_checkins = set(df_checkins["node1"].unique()).union(set(df_checkins["node2"].unique()))

        # take the intersection of the two sets
        unique_users = unique_friends.intersection(unique_checkins)

        # create a dataframe with the edges of the graph
        df = df_friends_all[df_friends_all["node1"].isin(unique_users) & df_friends_all["node2"].isin(unique_users)]

        # create a tsv file with the edges of the graph that ends with _filtered.tsv
        df.to_csv(os.path.join("data", "foursquare", dataset + "_friends_edges_filtered.tsv"), sep="\t", header=False, index=False)

        # create the graph
        G = nx.from_pandas_edgelist(df, "node1", "node2", create_using=nx.Graph())

        return G

    elif dataset == "gowalla":
        file = os.path.join("data", dataset, dataset + "_friends_edges.txt")

        df_friends_all = pd.read_csv(file, sep="\t", header=None, names=["node1", "node2"])
        unique_friends = set(df_friends_all["node1"].unique()).union(set(df_friends_all["node2"].unique()))

        df_checkins = pd.read_csv(os.path.join("data", dataset, dataset + "_checkins_edges.tsv"), sep="\t", header=None, names=["node1", "node2"])
        unique_checkins = set(df_checkins["node1"].unique()).union(set(df_checkins["node2"].unique()))

        unique_users = unique_friends.intersection(unique_checkins)

        df = df_friends_all[df_friends_all["node1"].isin(unique_users) & df_friends_all["node2"].isin(unique_users)]

        df.to_csv(os.path.join("data", dataset, dataset + "_friends_edges_filtered.tsv"), sep="\t", header=False, index=False)

        G = nx.from_pandas_edgelist(df, "node1", "node2", create_using=nx.Graph())
        return G

    elif dataset == "brightkite":
        file = os.path.join("data", dataset, dataset + "_friends_edges.txt")
        df_friends_all = pd.read_csv(file, sep="\t", header=None, names=["node1", "node2"])

        G = nx.from_pandas_edgelist(df_friends_all, "node1", "node2", create_using=nx.Graph())
        return G

def degree_distribution(G: nx.Graph, log: bool = True, save: bool = False) -> None:

        """
        This function takes in input a networkx graph and as options:
        - log = True/False (default = True)
        - save = True/False (default = False)
        The functions plots, using the plotly library, the degree distribution of the graph. If log = True, the plot is in log-log scale. If save = True, the plot is saved in the folder "plots" with the name "degree_distribution_{}.png" where {} is the name of the graph in input.
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

def average_clustering(G: nx.Graph) -> float:

        """
        This function takes in input a networkx graph and returns the average clustering coefficient of the graph.
        """

        sum_clustering = 0
        for node in G.nodes():
            sum_clustering += nx.clustering(G, node)

        return sum_clustering / G.number_of_nodes()

# ------------------------------------------------------------------------#

def watts_strogatz_model(G: nx.Graph, p = 0.1) -> nx.Graph:

        """
        This function takes in input a networkx graph and a probability p and returns a new graph obtained by applying the Watts-Strogatz model to the input graph.

        It computes k as the average degree of the input graph.
        """

        k = int(round(np.mean(list(dict(G.degree()).values()))))

        G_new = nx.watts_strogatz_graph(G.number_of_nodes(), k, p)
        G_new = nx.Graph(G_new)
        G_new.name = "watts_strogatz_{}_{}_{}" .format(G.name, p, k)

        return G_new

def mean_shortest_path(G: nx.Graph) -> float:

        """
        This function takes in input a networkx graph and returns the average shortest path length of the graph. This works also for disconnected graphs.
        """

        tmp = 0
        connected_components = list(nx.connected_components(G))
        for C in (G.subgraph(c).copy() for c in connected_components):
            tmp += (nx.average_shortest_path_length(C, method='dijkstra'))

        return tmp/len(list(connected_components))
