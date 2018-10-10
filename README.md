ASNE
============================================
<p align="justify">
An implementation of "Attributed Social Network Embedding". ASNE is a graph embedding algorithm which learns an embedding of nodes and fuses the node representations with node attributes. The procedure places nodes in an abstract feature space where information aboutfrist order proximity is preserved and attributes of a node are also part of the representation. ASNE learns the joint feature-proximal representations using a probabilistic factorization model. In our implementation we assumed that the proximity matrix used in the approximation is sparse, hence the solution runtime can be linear in the number of edges. The model assumes that the node-feature matrix is sparse. Compared to other implementations this specific version has several advantages. Specifically:

1. Stores the feature matrix as a sparse dictionary.
2. Uses sparse matrix multiplication to speed up computations.</p>
<div style="text-align:center"><img src ="asne.jpeg" ,width=720/></div>

This repository provides an implementation for ASNE as described in the paper:
> Attributed Social Network Embedding.
> Lizi Liao, Xiangnan He, Hanwang Zhang, Tat-Seng Chua
> IEEE Transactions on Knowledge and Data Engineering, 2018.
> https://arxiv.org/abs/1705.04969

### Requirements

The codebase is implemented in Python 2.7. package versions used for development are just below.
```
networkx          1.11
tensorflow-gpu    1.3.0
tqdm              4.19.5
numpy             1.13.3
pandas            0.20.3
texttable         1.2.1
scipy             1.1.0
argparse          1.1.0
```
### Datasets

The code takes an input graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header. Nodes should be indexed starting with 0. Sample graphs for the `Wikipedia Chameleons` and `Wikipedia Giraffes` are included in the  `input/` directory. 

The feature matrix can be stored two ways:

If the feature matrix is a **sparse binary** one it is stored as a json. Nodes are keys of the json and features are the values. For each node feature column ids are stored as elements of a list. The feature matrix is structured as:

```javascript
{ 0: [0, 1, 38, 1968, 2000, 52727],
  1: [10000, 20, 3],
  2: [],
  ...
  n: [2018, 10000]}
```
If the feature matrix is **dense** it is assumed that it is stored as csv with comma separators. It has a header, the first column contains node identifiers and it is sorted by these identifers. It should look like this:

| **NODE ID**| **Feature 1** | **Feature 2** | **Feature 3** | **Feature 4** |
| --- | --- | --- | --- |--- |
| 0 | 3 |0 |1.37 |1 |
| 1 | 1 |1 |2.54 |-11 |
| 2 | 2 |0 |1.08 |-12 |
| 3 | 1 |1 |1.22 |-4 |
| ... | ... |... |... |... |
| n | 5 |0 |2.47 |21 |


### Options

Learning of the embedding is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options

```
  --edge-path    STR        Input graph path.           Default is `input/edges/chameleon_edges.csv`.
  --feature-path STR        Input Features path.        Default is `input/features/chameleon_features.json`.
  --output-path  STR        Embedding path.             Default is `output/chameleon_asne.csv`.
```

#### Model options

```
  --node_embedding_dimensions      INT        Number of node embeding dimensions.           Default is 16.
  --feature_embedding_dimensions   INT        Number of feature embeding dimensions.        Default is 16.
  --batch_size                     INT        Batch size for gradient descent.              Default is 64.
  --epochs                         INT        Number of training epochs.                    Default is 10.
  --alpha                          FLOAT      Matrix mixing parameter for embedding.        Default is 1.0.
  --negative_samples               INT        Number of negative samples.                   Default is 10.
```

### Examples

The following commands learn a graph embedding and write the embedding to disk. The node representations are ordered by the ID.

Creating an ASNE embedding of the default dataset with the default hyperparameter settings. Saving the embedding at the default path.

```
python src/main.py
```
Creating an ASNE embedding of the default dataset with 2x128 dimensions.

```
python src/main.py --node_embedding_dimensions 128  --feature_embedding_dimensions 128
```

Creating an ASNE embedding of the default dataset with asymmetric mixing.

```
python src/main.py --batch_size 512
```

Creating an embedding of an other dense structured dataset the `Wikipedia Giraffes`. Saving the output in a custom folder.

```
python src/main.py --edge-path input/edges/giraffe_edges.csv --feature-path input/features/giraffe_features.csv --output-path output/giraffe_fscnmf.csv --features dense
```
