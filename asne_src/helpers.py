import json
import pandas as pd
import networkx as nx
import argparse

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

def parse_args():
    parser = argparse.ArgumentParser(description="Run SNE.")

    parser.add_argument('--edge_path',
                        nargs='?',
                        default='./data/chameleon_edges.csv',
                        help='Input data path')

    parser.add_argument('--features_path', nargs='?', default='./data/chameleon_features.json',
                        help='Input data path')

    parser.add_argument('--output_path', nargs='?', default='chameleon_asne.csv',
                        help='Input data path')

    parser.add_argument('--id_dim', type=int, default=16,
                        help='Dimension for id_part.')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of epochs.')

    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Number of epochs.')


    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')


    parser.add_argument('--n_neg_samples', type=int, default=5,
                        help='Number of negative samples.')
    parser.add_argument('--id_embedding_size', type=int, default=16,
                        help='Number of negative samples.')
    parser.add_argument('--attr_embedding_size', type=int, default=16,
                        help='Number of negative samples.')
    parser.add_argument('--random_seed', type=int, default=2019,
                        help='Number of negative samples.')

    parser.add_argument('--attr_dim', type=int, default=20,
                        help='Dimension for attr_part.')
    return parser.parse_args()
