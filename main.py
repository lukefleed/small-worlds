"""
NOTEs:

- This file is note meant to be run, it's just a collection of functions that are used in the other files. It's just a way to keep the code clean and organized.

- Why do I use os.path.join and not the "/"? Because it's more portable, it works on every OS, while "/" works only on Linux and Mac. If you want to use it on Windows, you have to change all the "/" with "\". With os.path.join you don't have to worry about it and, as always, f*** Microsoft.
"""

import os
import wget
import zipfile
import pandas as pd
import networkx as nx
from typing import Literal
from itertools import combinations

# ------------------------------------------------------------------------#

def download_datasets():

    urls = [
        ["https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz", "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"],

        ["https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz", "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"],

        ["http://www-public.it-sudparis.eu/~zhang_da/pub/dataset_tsmc2014.zip"]
    ]

    folders = ["brightkite", "gowalla", "foursquare"]

    # check if the data folder exists
    if not os.path.exists("data"):
        os.mkdir("data")

    # if they don't exist, create 3 subfolders in data called brightkite, gowalla and foursquare
    for folder in folders:
        if not os.path.exists(os.path.join("data", folder)):
            os.mkdir(os.path.join("data", folder))

    # download every url in urls[0] in the brightkite folder, and every url in urls[1] in the gowalla folder, and every url in urls[2] in the foursquare folder. If the file is already downloaded, skip the download

    for i in range(len(urls)):
        for url in urls[i]:
            if not os.path.exists(os.path.join("data", folders[i], url.split("/")[-1])):
                wget.download(url, os.path.join("data", folders[i]))

    # unzip the .gz files inside the brightkite and gowalla folders
    for file in os.listdir(os.path.join("data", "brightkite")):
        if file.endswith(".gz"):
            os.system("gunzip {}".format(os.path.join("data", "brightkite", file)))

    for file in os.listdir(os.path.join("data", "gowalla")):
        if file.endswith(".gz"):
            os.system("gunzip {}".format(os.path.join("data", "gowalla", file)))

    # extract the data of foursquare in a nice way, checking all the edge cases as a maniac. More details below

    """
    The code below it's very ugly to read, but it's effective. Basically, in every possible messy situation we have the files (maybe after testing) inside the foursquare folder, it will fix them and bring them as the program expects them to be.

    Firstly it checks if in the foursquare folder there is a folder called dataset_tsmc2014. If true, it checks if there are 3 files inside the foursquare folders, if yes, skip the process (everything is in order). If false, it moves all the files inside the dataset_tsmc2014 folder to the foursquare folder and delete the dataset_tsmc2014 folder (we don't want a nested folder)

    Then, if there is no dataset_tsmc2014 folder, it unzips the file. Then move all the .txt files inside the dataset_tsmc2014 folder in the foursquare folder. Then delete the dataset_tsmc2014 folder and the .zip file.
    """

    for file in os.listdir(os.path.join("data", "foursquare")):
        if file.endswith(".zip"):
            if os.path.exists(os.path.join("data", "foursquare", "dataset_tsmc2014")):
                if len(os.listdir(os.path.join("data", "foursquare"))) == 3:
                    pass
                else:
                    for file in os.listdir(os.path.join("data", "foursquare", "dataset_tsmc2014")):
                        os.rename(os.path.join("data", "foursquare", "dataset_tsmc2014", file), os.path.join("data", "foursquare", file))
                    os.rmdir(os.path.join("data", "foursquare", "dataset_tsmc2014"))
            else:
                with zipfile.ZipFile(os.path.join("data", "foursquare", file), 'r') as zip_ref:
                    zip_ref.extractall(os.path.join("data", "foursquare"))
                    os.remove(os.path.join("data", "foursquare", file))
                for file in os.listdir(os.path.join("data", "foursquare", "dataset_tsmc2014")):
                    os.rename(os.path.join("data", "foursquare", "dataset_tsmc2014", file), os.path.join("data", "foursquare", file))
                os.rmdir(os.path.join("data", "foursquare", "dataset_tsmc2014"))

    # Now we want to clean our data. Both for brightkite and gowalla, we want to rename _edges files as "brightkite_friends_edges.txt" and "gowalla_friends_edges.txt"

    for file in os.listdir(os.path.join("data", "brightkite")):
        if file.endswith("_edges.txt"):
            os.rename(os.path.join("data", "brightkite", file), os.path.join("data", "brightkite", "brightkite_friends_edges.txt"))

    for file in os.listdir(os.path.join("data", "gowalla")):
        if file.endswith("_edges.txt"):
            os.rename(os.path.join("data", "gowalla", file), os.path.join("data", "gowalla", "gowalla_friends_edges.txt"))

    # Now we from the _totalCheckins.txt files we want to keep only the first and last column, which are the user ID and the venue ID. We also want to remove the header of the file. Use pandas to do that. Then rename the files as "brightkite_checkins_edges.txt" and "gowalla_checkins_edges.txt

    for file in os.listdir(os.path.join("data", "brightkite")):
        if file.endswith("_totalCheckins.txt"):
            df = pd.read_csv(os.path.join("data", "brightkite", file), sep="\t", header=None)
            df = df[[0, 2]]
            df.to_csv(os.path.join("data", "brightkite", "brightkite_checkins.txt"), sep="\t", header=False, index=False, errors="ignore", encoding="utf-8")
            os.remove(os.path.join("data", "brightkite", file))

    for file in os.listdir(os.path.join("data", "gowalla")):
        if file.endswith("_totalCheckins.txt"):
            df = pd.read_csv(os.path.join("data", "gowalla", file), sep="\t", header=None)
            df = df[[0, 2]]
            df.to_csv(os.path.join("data", "gowalla", "gowalla_checkins.txt"), sep="\t", header=False, index=False, errors="ignore", encoding="utf-8")
            os.remove(os.path.join("data", "gowalla", file))

    # now for foursquare we want to keep only the first and second column, which are the user ID and the venue ID. We also want to remove the header of the file. Use pandas to do that. Do that for both _NYC.txt and _TKY.txt files. Then rename the files as "foursquare_checkins_edges_NYC.txt" and "foursquare_checkins_edges_TKY.txt

    for file in os.listdir(os.path.join("data", "foursquare")):
        if file.endswith("_NYC.txt"):
            df = pd.read_csv(os.path.join("data", "foursquare", file), sep="\t", header=None, encoding="utf-8", encoding_errors="ignore")
            df = df[[0, 1]]
            df.to_csv(os.path.join("data", "foursquare", "foursquare_checkins_NYC.txt"), sep="\t", header=False, index=False)
            os.remove(os.path.join("data", "foursquare", file))

        if file.endswith("_TKY.txt"):
            df = pd.read_csv(os.path.join("data", "foursquare", file), sep="\t", header=None, encoding="utf-8", encoding_errors="ignore")
            df = df[[0, 1]]
            df.to_csv(os.path.join("data", "foursquare", "foursquare_checkins_TKY.txt"), sep="\t", header=False, index=False)
            os.remove(os.path.join("data", "foursquare", file))

# ------------------------------------------------------------------------#

def create_checkins_graph_SLOW(dataset: Literal['brightkite', 'gowalla', 'foursquareNYC', 'foursquareTKY'])-> nx.Graph:

    """
    This function takes in input a tsv file, each line in the file is a check-in. The function returns an undirected networkx graph object.

    Firstly, we retrive the unique user ID: this are the nodes of our graph. We create a dictionary with the users ID as keys and the venues ID as values. Two users are connected if they have visited the same venue at least once. The weight of the edge is the number of common venues.
    """

    if dataset not in ['brightkite', 'gowalla',
     'foursquareNYC', 'foursquareTKY']:
        raise ValueError("Dataset not valid. Please choose between brightkite, gowalla, foursquareNYC, foursquareTKY")

    # based on the dataset, we have to read the file in a different way.
    if dataset == "foursquareNYC":
        file = os.path.join("data", "foursquare", "dataset_TSMC2014_NYC.txt")
        df = pd.read_csv(file, sep="\t", header=None, names=["UserID", "VenueID", "CategoryID", "CategoryName", "Latitude", "Longitude", "LocalTime" ,"UTCtime",], encoding="utf-8", encoding_errors="ignore")

    elif dataset == "foursquareTKY":
        file = os.path.join("data", "foursquare", "dataset_TSMC2014_TKY.txt")
        df = pd.read_csv(file, sep="\t", header=None, names=["UserID", "VenueID", "CategoryID", "CategoryName", "Latitude", "Longitude", "LocalTime" ,"UTCtime",], encoding="utf-8", encoding_errors="ignore")
    else:
        file = os.path.join("data", dataset, "loc-{}_totalCheckins.txt".format(dataset))
        df = pd.read_csv(file, sep="\t", header=None, names=["UserID", "CheckIn", "latitude", "longitude", "VenueID"], encoding="utf-8", encoding_errors="ignore")

    # get the unique users ID
    users = df["UserID"].unique()
    G = nx.Graph()
    G.add_nodes_from(users)
    print("Number of nodes added to the graph {}: {}".format(dataset, G.number_of_nodes()))

    users_venues = df.groupby("UserID")["VenueID"].apply(list).to_dict()

    for user1, user2 in combinations(users, 2):
        intersection = set(users_venues[user1]) & set(users_venues[user2])
        if len(intersection) > 0:
            G.add_edge(user1, user2, weight=len(intersection))

    print("Number of edges added to the graph {}: {}".format(dataset, G.number_of_edges()))

# ------------------------------------------------------------------------#

def friendships_graph(dataset: Literal['brightkite', 'gowalla']) -> nx.Graph:

    """
    This function takes in input a tsv file with two columns, Each line in the file is an edge. The function returns an undirected networkx graph object. It uses pandas to read the file since it's faster than the standard python open() function. If we don't want to use the standard python open() function, the following code works as well:

    G = nx.Graph()
    with open(file, "r") as f:
        for line in f:
            node1, node2 = line.split("\t")
            G.add_edge(node1, node2)

    """

    if dataset not in ["brightkite", "gowalla"]:
        raise ValueError("The dataset must be brightkite or gowalla")

    file = os.path.join("data", dataset, "{}_friends_edges.txt".format(dataset))
    df = pd.read_csv(file, sep="\t", header=None, names=["node1", "node2"])
    G = nx.from_pandas_edgelist(df, "node1", "node2", create_using=nx.Graph())

    return G

# ------------------------------------------------------------------------#

def checkins_graph_from_edges(dataset: Literal['brightkite', 'gowalla', 'foursquareNYC', 'foursquareTKY']) -> nx.Graph:

    """
    This function takes in input a tsv file with two columns, Each line in the file is an edge. The function returns an undirected networkx graph object. It uses pandas to read the file since it's faster than the standard python open() function. If we don't want to use the standard python open() function, the following code works as well:

    G = nx.Graph()
    with open(file, "r") as f:
        for line in f:
            node1, node2 = line.split("\t")
            G.add_edge(node1, node2)

    """

    if dataset not in ["brightkite", "gowalla", "foursquareNYC", "foursquareTKY"]:
        raise ValueError("The dataset must be brightkite, gowalla or foursquare")


    file = os.path.join("data", dataset, "{}_checkins_edges.tsv".format(dataset))

    # if dataset == "foursquareTKY":
    #     file = os.path.join("data", "foursquare", "foursquareNYC_checkins_graph.tsv")
    # elif dataset == "foursquareNYC":
    #     file = os.path.join("data", "foursquare", "foursquareTKY_checkins_graph.tsv")
    # else:
    #     file = os.path.join("data", dataset, "{}_checkins_graph.tsv".format(dataset))


    df = pd.read_csv(file, sep="\t", header=None, names=["node1", "node2"])
    G = nx.from_pandas_edgelist(df, "node1", "node2", create_using=nx.Graph())

    return G
