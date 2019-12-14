# Parallelization of Graph Convolutional Network

This is the class project repo for CS 259 (High Performance Computing) Fall 2019 taught by Glenn Reinman.
The sequential version of the code comes from [here](https://github.com/cai-lw/parallel-gcn "parallel-gcn"). Reddit dataset can be downloaded [here](http://snap.stanford.edu/graphsage/reddit.zip "reddit dataset").
This project is a GPU acceleration of the Graph Convolutional Network.

Team members are: Zongze Li, Yuanhao Jia, Hengda Shi, Jintao Jiang

## Setup

### Convert reddit data format (networkx, numpy, scipy are required)

    python3 reddit_preprocess.py

### Run cora/citeseer/pubmed/reddit datasets on CPU

    make seq
    ./gcn-seq cora

### Run cora/citeseer/pubmed/reddit datasets on GPU

    make cuda
    ./gcn-cuda cora

### Clean up executable

    make clean
