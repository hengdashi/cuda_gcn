#ifndef CUDA_GCN_CUH
#define CUDA_GCN_CUH

#include "gcn.h"
#include "cuda_variable.cuh"
#include "cuda_module.cuh"

using std::vector;
using std::pair;

class CUDAGCN {
    vector<CUDAModule*> modules;
    vector<CUDAVariable> variables;
    CUDAVariable *input, *output;
    CUDASparseIndex *sp, *graph;
    int *truth;
    CUDAAdam *optimizer;
    float loss;
    void set_input();
    void set_truth(int current_split);
    float get_accuracy();
    float get_l2_penalty();
    pair<float, float> train_epoch();
    pair<float, float> eval(int current_split);
    GCNData *data;
public:
    GCNParams params;
    CUDAGCN(GCNParams params, GCNData *input_data);
    CUDAGCN() {}
    ~CUDAGCN();
    void run();
};

#endif
