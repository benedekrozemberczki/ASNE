import json
import pandas as pd
import networkx as nx
import argparse
from texttable import Texttable

def parse_args():

    parser = argparse.ArgumentParser(description="Run ASNE.")

    parser.add_argument('--edge_path',
                        nargs='?',
                        default='./input/edges/chameleon_edges.csv',
                        help='Input data path')

    parser.add_argument('--features_path',
                        nargs='?',
                        default='./input/features/chameleon_features.json',
                        help='Input data path')

    parser.add_argument('--output_path',
                        nargs='?',
                        default='./output/chameleon_asne.csv',
                        help='Input data path')

    parser.add_argument('--node_embedding_dimensions',
                        type=int,
                        default=16,
                        help='Number of negative samples.')

    parser.add_argument('--feature_embedding_dimensions',
                        type=int,
                        default=16,
                        help='Number of negative samples.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Number of epochs.')

    parser.add_argument('--alpha',
                        type=float,
                        default=1.0,
                        help='Number of epochs.')

    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of epochs.')

    parser.add_argument('--negative_samples',
                        type=int,
                        default=5,
                        help='Number of negative samples.')

    return parser.parse_args()


def map_edges(g):
    edges = [edge for edge in g.edges()]
    edges_rev = map(lambda x: [x[1],x[0]], edges)
    edges = map(lambda x: [x[0],x[1]], edges)
    edges = edges + edges_rev
    return edges

def feature_reader(path):
    features = json.load(open(path))
    features = {int(k): map(lambda x: int(x), v) for k,v in features.iteritems()}
    return features

def graph_reader(path):
    return nx.from_edgelist(pd.read_csv(path).values.tolist())

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print t.draw()
