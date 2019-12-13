# Parallelization of Graph Convolutional Network

This is the class project repo for CS 259 (High Performance Computing) Fall 2019 taught by Glenn Reinman.
The sequential version of the code comes from https://github.com/cai-lw/parallel-gcn.
This project is a GPU acceleration of the Graph Convolutional Network.

Team members are: Zongze Li, Yuanhao Jia, Hengda Shi, Jintao Jiang

## To run cora/citeseer/pubmed datasets on CPU:

    make seq
    ./gcn-seq cora

## To run cora/citeseer/pubmed datasets on GPU:

    make cuda
    ./gcn-cuda cora

## To clean up executable:

    make clean
