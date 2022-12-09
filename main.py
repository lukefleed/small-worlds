# /bin/env/python3

from itertools import combinations
import os
import wget
import zipfile
import networkx as nx
import pandas as pd
from typing import Literal


"""
NOTEs:

- This file is note meant to be run, it's just a collection of functions that are used in the other files. It's just a way to keep the code clean and organized.

- Why do I use os.path.join and not the "/"? Because it's more portable, it works on every OS, while "/" works only on Linux and Mac. If you want to use it on Windows, you have to change all the "/" with "\". With os.path.join you don't have to worry about it and, as always, f*** Microsoft.
"""


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

    # download every url in urls[0] in the brightkite folder, and every url in urls[1] in the gowalla folder, and every url in urls[2] in the foursquare folder. At ech iteration, it checks if the file already exists, if yes, it skips the download and prints a message. If no, it downloads the file and prints a message.

    for i in range(len(urls)):
        for url in urls[i]:
            # check if there are .txt files inside folder, if yes, skip the download
            if len([file for file in os.listdir(os.path.join("data", folders[i])) if file.endswith(".txt")]) > 0:
                print("The {} dataset is already downloaded and extracted as .txt file, if you want to download again the .gz file with this function, delete the .txt files in the folder".format(folders[i]))
                break
            # check if there are .gz files inside folder, if yes, skip the download
            elif len([file for file in os.listdir(os.path.join("data", folders[i])) if file.endswith(".gz")]) > 0:
                print("The {} dataset is already downloaded as .gz file, if you want to download again the .gz file with this function, delete the .gz files in the folder".format(folders[i]))
                break
            # if there are no .txt or .gz files, download the file
            else:
                print("Downloading {} dataset...".format(folders[i]))
                wget.download(url, os.path.join("data", folders[i]))
                print("Download completed of {} dataset".format(folders[i]))

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

    # if there are no .txt files inside the brightkite folder, unzip the .gz files
    if len([file for file in os.listdir(os.path.join("data", "brightkite")) if file.endswith(".txt")]) == 0:
        for file in os.listdir(os.path.join("data", "brightkite")):
            if file.endswith(".gz"):
                os.system("gunzip {}".format(os.path.join("data", "brightkite", file)))

    # if there are no .txt files inside the gowalla folder, unzip the .gz files
    if len([file for file in os.listdir(os.path.join("data", "gowalla")) if file.endswith(".txt")]) == 0:
        for file in os.listdir(os.path.join("data", "gowalla")):
            if file.endswith(".gz"):
                os.system("gunzip {}".format(os.path.join("data", "gowalla", file)))


def create_checkins_graph(dataset: Literal['brightkite', 'gowalla', 'foursquareNYC', 'foursquareTKY'])-> nx.Graph:

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

    file = os.path.join("data", dataset, "loc-{}_edges.txt".format(dataset))
    df = pd.read_csv(file, sep="\t", header=None, names=["node1", "node2"])
    G = nx.from_pandas_edgelist(df, "node1", "node2", create_using=nx.Graph())

    return G
