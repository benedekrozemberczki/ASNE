"""Model runner."""

import os
from asne import ASNE
from utils import graph_reader, feature_reader, parse_args, tab_printer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def run_asne(args, graph, features):
    """
    Fitting an ASNE model and saving the embedding.
    :param args: Arguments object.
    :param graph: NetworkX graph.
    :param features: Features in a dictionary.
    """
    tab_printer(args)
    model = ASNE(args, graph, features)
    model.train()
    model.save_embedding()

if __name__ == "__main__":
    args = parse_args()
    graph = graph_reader(args.edge_path)
    features = feature_reader(args.features_path)
    run_asne(args, graph, features)
