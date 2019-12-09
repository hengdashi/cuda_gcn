#Parallelization of Graph Convolutional Network
##CrossEntropy Module

Main modification is in the src/seq/module.cu CrossEntropy::forward

##To run cora dataset:

    make
    ./program cora
