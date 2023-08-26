# Domain-adaptive Graph Attention-supervised Network for Cross-network Edge Classification (DGASN)

This repository contains the author's implementation in Pytorch for the paper "Domain-adaptive Graph Attention-supervised Network for Cross-network Edge Classification".

# Environment Requirement

• python == 3.6.7

• pytorch == 1.10.2

• numpy == 1.19.2

• scipy == 1.5.1

• dgl == 0.8.2

• sklearn == 0.24.2

# Datasets

3 datasets are used in our paper.

Each ".mat" file stores a network dataset, where

the variable "network" represents an adjacency matrix,

the variable "attrb" represents a node attribute matrix,

the variable "group" represents a node label matrix.

We generate the edge labels based on the node label of two nodes on each edge. 

Specifically, a homophilous edge indicates that the two connected nodes share at least one common class-label. 

On the contrary, a heterophilous edge reflects that the two connected nodes have totally different class-labels.

# Code

"model.py" is the implementation of the DGASN model.

"DGASN_main.py" is an example case of the cross-network edge classification task from citationv1 to acmv9 networks.

# Plese cite our paper as:

Xiao Shen, Mengqiu Shao, Shirui Pan, Laurence T. Yang and Xi Zhou, "Domain-adaptive Graph Attention-supervised Network for Cross-network Edge Classification," IEEE Trans. Neural. Netw. Learn. Syst., 2023.

