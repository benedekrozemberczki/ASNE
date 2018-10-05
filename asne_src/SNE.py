import numpy as np
import tensorflow as tf
from tqdm import tqdm
import networkx as nx
import pandas as pd
import json
import random
from helpers import map_edges

class SNE:
    def __init__(self, args, graph, features):
        self.args = args
        self.graph = graph
        self.features = features
        self.edges = map_edges(self.graph)
        self.nodes = self.graph.nodes()
        self.node_N = len(self.nodes)
        self.attr_M = max(map(lambda x: max(x+[0]),self.features.values()))+1
        self._init_graph()


    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.train_data_id = tf.placeholder(tf.int32, shape=[None])
            self.train_data_attr = tf.placeholder(tf.float32, shape=[None, self.attr_M])
            self.train_labels = tf.placeholder(tf.int32, shape=[None, 1])

            network_weights = self._initialize_weights()
            self.weights = network_weights

            self.id_embed =  tf.nn.embedding_lookup(self.weights['in_embeddings'], self.train_data_id)
            self.attr_embed =  tf.matmul(self.train_data_attr, self.weights['attr_embeddings'])
            self.embed_layer = tf.concat(1, [self.id_embed, self.args.alpha * self.attr_embed])

            self.loss =  tf.reduce_mean(tf.nn.sampled_softmax_loss(self.weights['out_embeddings'], self.weights['biases'], self.embed_layer,
                                                  self.train_labels, self.args.n_neg_samples, self.node_N))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)

            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['in_embeddings'] = tf.Variable(tf.random_uniform([self.node_N, self.args.id_embedding_size], -1.0, 1.0))
        all_weights['attr_embeddings'] = tf.Variable(tf.random_uniform([self.attr_M,self.args.attr_embedding_size], -1.0, 1.0))
        all_weights['out_embeddings'] = tf.Variable(tf.truncated_normal([self.node_N, self.args.id_embedding_size + self.args.attr_embedding_size],
                                    stddev=1.0 / math.sqrt(self.args.id_embedding_size + self.args.attr_embedding_size)))
        all_weights['biases'] = tf.Variable(tf.zeros([self.node_N]))
        return all_weights

    def partial_fit(self, X):
        feed_dict = {self.train_data_id: X['batch_data_id'],
                     self.train_data_attr: X['batch_data_attr'],
                     self.train_labels: X['batch_data_label']}

        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def getEmbedding(self):
        embedding = self.sess.run(self.weights['out_embeddings'])
        ids = np.array(self.nodes).reshape(-1,1)
        self.out_embedding = np.concatenate([ids, embedding], axis = 1)
        print(self.out_embedding.shape)

    def generate_batch(self,i):
        batch_xs = {}
        data_id = np.array([edge[0] for edge in self.edges[self.args.batch_size*i:self.args.batch_size*(i+1)]])
        data_label = np.array([[edge[1]] for edge in self.edges[self.args.batch_size*i:self.args.batch_size*(i+1)]])
        attributes = np.zeros((self.args.batch_size, self.attr_M))
        for i, edge in enumerate(self.edges[self.args.batch_size*i:self.args.batch_size*(i+1)]):
             if len(self.features[edge[0]])>0:
                 attributes[i,self.features[edge[0]]]=1.0
        batch_xs['batch_data_id'] = data_id
        batch_xs['batch_data_attr'] =attributes
        batch_xs['batch_data_label'] = data_label
        return batch_xs
                 

    def train(self):

        total_batch = int(len(self.edges) / self.args.batch_size)
        for epoch in range(self.args.epoch ):
            print(epoch)
            random.shuffle(self.edges)
            costs = []
            for i in tqdm(range(total_batch)):
                batch_xs = self.generate_batch(i)
                cost = self.partial_fit(batch_xs)
                costs = costs +[cost]
        self.getEmbedding()
        self.save_embedding()

    def save_embedding(self):
        """
        Saving the embedding on disk.
        """
        print("\nSaving the embedding.\n")
        columns = ["id"] + map(lambda x: "X_"+str(x),range(self.out_embedding.shape[1]-1))
        self.out = pd.DataFrame(self.out_embedding , columns = columns)
        self.out.to_csv(self.args.output_path, index = None)


