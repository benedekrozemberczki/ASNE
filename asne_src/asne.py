""" ASNE model implementation."""

import json
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import tensorflow as tf
from scipy import sparse
from utils import map_edges
from texttable import Texttable

class ASNE:
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the Wiki Chameleons.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node ID.
    """
    def __init__(self, args, graph, features):
        """
        Constructor of ASNE object.
        :param args: Arguments object.
        :param graph: Networkx graph.
        :param features: Dictionary of features.
        """
        self.args = args
        self.graph = graph
        self.features = features
        self.edges = map_edges(self.graph)
        self.nodes = self.graph.nodes()
        self.node_count = len(self.nodes)
        self.feature_count = max(map(lambda x: max(x+[0]), self.features.values())) + 1
        self._build_model()

    def _setup_variables(self):
        """
        Creating TensorFlow variables and  placeholders.
        """
        self.node_embedding = tf.random_uniform([self.node_count, self.args.node_embedding_dimensions], -1.0, 1.0)

        self.node_embedding = tf.Variable(self.node_embedding, dtype=tf.float32)

        self.feature_embedding = tf.random_uniform([self.feature_count, self.args.feature_embedding_dimensions], -1.0, 1.0)

        self.feature_embedding = tf.Variable(self.feature_embedding, dtype=tf.float32)

        self.combined_dimensions = self.args.node_embedding_dimensions + self.args.feature_embedding_dimensions

        self.noise_embedding = tf.Variable(tf.truncated_normal([self.node_count, self.combined_dimensions],
                                                               stddev=1.0/math.sqrt(self.combined_dimensions)),
                                                               dtype=tf.float32)

        self.noise_bias = tf.Variable(tf.zeros([self.node_count]),
                                      dtype=tf.float32)

        self.noise_bias = tf.Variable(tf.zeros([self.node_count]),
                                      dtype=tf.float32)

        self.left_nodes = tf.placeholder(tf.int32, shape=[None])

        self.node_features = tf.sparse_placeholder(tf.float32,
                                                   shape=[None, self.feature_count])

        self.right_nodes = tf.placeholder(tf.int32,
                                          shape=[None, 1])

    def _build_model(self):
        """
        Creating computation graph of ASNE.
        """
        self.graph = tf.Graph()

        with self.graph.as_default():

            self._setup_variables()

            self.node_embed = tf.nn.embedding_lookup(self.node_embedding, self.left_nodes, max_norm=1)

            self.feature_embed = tf.sparse_tensor_dense_matmul(self.node_features, self.feature_embedding)

            self.combined_embed = tf.cast(tf.concat([self.node_embed, self.args.alpha*self.feature_embed], 1), tf.float32)

            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=self.noise_embedding,
                                                                  biases=self.noise_bias,
                                                                  labels=self.right_nodes,
                                                                  inputs=self.combined_embed,
                                                                  num_sampled=self.args.negative_samples,
                                                                  num_classes=self.node_count))

            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def _generate_batch(self, i):
        """
        Creating a batch of node indices and features.
        :param i: Batch index
        :return feed_dict: Dictionary with numpy arras for indices and features.
        """
        left_nodes = np.array([e[0] for e in self.edges[self.args.batch_size*i:self.args.batch_size*(i+1)]])
        right_nodes = np.array([[e[1]] for e in self.edges[self.args.batch_size*i:self.args.batch_size*(i+1)]])

        node_indices = [index for index, edge in enumerate(self.edges[self.args.batch_size*i:self.args.batch_size*(i+1)]) for feature in self.features[edge[0]]]
        feature_indices = [feature for edge in self.edges[self.args.batch_size*i:self.args.batch_size*(i+1)] for feature in self.features[edge[0]]]

        values = np.ones(len(node_indices))

        features = sparse.coo_matrix((values, (node_indices, feature_indices)),
                                     shape=(self.args.batch_size, self.feature_count),
                                     dtype=np.float32)

        features = tf.SparseTensorValue(indices=np.array([features.row, features.col]).T,
                                        values=features.data,
                                        dense_shape=features.shape)

        feed_dict = {self.left_nodes: left_nodes,
                     self.node_features: features,
                     self.right_nodes: right_nodes}
        return feed_dict

    def _optimize(self, feed_dict):
        """
        Running weight optimization on a batch.
        :param feed_dict: Dictionary with inputs.
        """
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        self.costs = self.costs + [loss]

    def _epoch_start(self, epoch):
        """
        Printing the epoch number and setting up the cost list.
        :param epoch: Epoch number.
        """
        random.shuffle(self.edges)
        self.costs = []
        t = Texttable()
        t.add_rows([["Epoch: ", str(epoch+1)+"/"+str(self.args.epochs)+"."]])
        print(t.draw())

    def _epoch_end(self, epoch):
        """
        Printing the epoch average loss.
        :param epoch: Epoch number.
        """
        t = Texttable() 
        t.add_rows([["Average Loss: ", round(np.mean(self.costs), 4)]])
        print(t.draw())

    def train(self):
        """
        Training the ASNE model.
        """
        self.total_batch = int(len(self.edges) / self.args.batch_size)
        for epoch in range(self.args.epochs):
            self._epoch_start(epoch)
            for i in tqdm(range(self.total_batch)):
                feed_dict = self._generate_batch(i)
                self._optimize(feed_dict)
            self._epoch_end(epoch)

    def save_embedding(self):
        """
        Saving the embedding at the default path.
        """
        print("\nSaving the embedding.\n")
        embedding = self.sess.run(self.noise_embedding)
        ids = np.array(self.nodes).reshape(-1, 1)
        embedding = np.concatenate([ids, embedding], axis=1)
        columns = ["id"] + list(map(lambda x: "X_"+str(x), range(embedding.shape[1]-1)))
        embedding = pd.DataFrame(embedding, columns=columns)
        embedding.to_csv(self.args.output_path, index=None)
