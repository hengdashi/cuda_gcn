#ifndef CUDA_VARIABLE_CUH
#define CUDA_VARIABLE_CUH

#include "kernel.cuh"

struct CUDAVariable {
    float *data, *grad;
    bool requires_grad;
    CUDAVariable(int size, bool requires_grad=true);
    ~CUDAVariable();
};


#endif
