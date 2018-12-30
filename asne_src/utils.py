import json
import argparse
import pandas as pd
import networkx as nx
from texttable import Texttable

def parse_args():

    parser = argparse.ArgumentParser(description="Run ASNE.")

    parser.add_argument("--edge-path",
                        nargs="?",
                        default="./input/edges/chameleon_edges.csv",
                        help="Edge list path.")

    parser.add_argument("--features-path",
                        nargs="?",
                        default="./input/features/chameleon_features.json",
                        help="Features path.")

    parser.add_argument("--output-path",
                        nargs="?",
                        default="./output/chameleon_asne.csv",
                        help="Output path.")

    parser.add_argument("--node-embedding-dimensions",
                        type=int,
                        default=16,
                        help="Node embedding matrix dimensions. Default is 16.")

    parser.add_argument("--feature-embedding-dimensions",
                        type=int,
                        default=16,
                        help="Feature embedding matrix dimensions. Default is 16.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=64,
                        help="Batch size. Default is 64.")

    parser.add_argument("--alpha",
                        type=float,
                        default=1.0,
                        help="Mixing parameter. Default is 1.0.")

    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Number of epochs. Default is 10.")

    parser.add_argument("--negative-samples",
                        type=int,
                        default=5,
                        help="Number of negative samples. Default is 5.")

    return parser.parse_args()


def map_edges(graph):
    """
    Mapping the edges of the graph to a symmetric edge list.
    :param g: Networkx graph.
    :return edges: Symmetric edge list.
    """
    edges = [edge for edge in graph.edges()]
    edges_rev = [[edge[1],edge[0]] for edge in edges]
    edges = [[edge[0], edge[1]] for edge in edges]
    edges = edges + edges_rev
    return edges

def feature_reader(path):
    """
    Reading the features and transforming the keys.
    :param path: Path to the features.
    :return features: Feature dictionary.
    """
    features = json.load(open(path))
    features = {int(k): [int(x) for x in v] for k,v in features.items()}
    return features

def graph_reader(path):
    """
    Reading the edgelist.
    :param path: Edge list path.
    :return : NetworkX graph.
    """
    return nx.from_edgelist(pd.read_csv(path).values.tolist())

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    tab = Texttable() 
    tab.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(tab.draw())
