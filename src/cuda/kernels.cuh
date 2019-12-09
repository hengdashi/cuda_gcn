#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "variable.h"
#include "sparse.h"

__global__
void GraphSum_forward_kernel(float *d_in, float *d_out, int *d_indptr, int *d_indices, int dim, int numNodes);
void GraphSum_forward(Variable *in, Variable *out, SparseIndex *graph, int dim);

__global__
void GraphSum_backward_kernel(float *d_in, float *d_out, int *d_indptr, int *d_indices, int dim, int numNodes);
void GraphSum_backward(Variable *in, Variable *out, SparseIndex *graph, int dim);


__global__
void ReLU_forward_kernel(float *d_in, bool *d_mask, const long unsigned int datasize, bool training);
void ReLU_forward(Variable *in, bool *mask, bool training);

__global__
void ReLU_backward_kernel(float *d_in, bool *d_mask, long unsigned int datasize);
void ReLU_backward(Variable *in, bool *mask, bool training);

#endif
