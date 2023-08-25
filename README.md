# Domain-adaptive Graph Attention-supervised Network for Cross-network Edge Classification (DGASN)

This repository contains the author's implementation in Pytorch for the paper "Domain-adaptive Graph Attention-supervised Network for Cross-network Edge Classification".

# Environment Requirement

• python == 3.6.2

• pytorch == 1.13.1

• numpy == 1.16.2

• scipy == 1.2.1

• dgl == 0.21.1

# Datasets

5 datasets are used in our paper.

Each ".mat" file stores a network dataset, where

the variable "network" represents an adjacency matrix,

the variable "attrb" represents a node attribute matrix,

the variable "group" represents a node label matrix.

# Code

"model.py" is the implementation of the DGASN model.

"DGASN_main.py" is an example case of the cross-network node classification task from citationv1 to acmv9 networks.

# Plese cite our paper as:

Xiao Shen, Mengqiu Shao, Shirui Pan, Laurence T. Yang and Xi Zhou, "Domain-adaptive Graph Attention-supervised Network for Cross-network Edge Classification," IEEE Trans. Neural. Netw. Learn. Syst.

