from SNE import SNE
from helpers import graph_reader, feature_reader, parse_args

def run_SNE(args, graph, features):
    model = SNE(args, graph, features)
    model.train( )


if __name__ == '__main__':
    args = parse_args()
    graph = graph_reader(args.edge_path)
    features = feature_reader(args.features_path)
    run_SNE(args, graph, features)



