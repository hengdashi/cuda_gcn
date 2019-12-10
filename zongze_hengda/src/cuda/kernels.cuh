#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "variable.h"
#include "sparse.h"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

__global__
void GraphSum_forward_kernel(float *d_in, float *d_out, int *d_indptr, int *d_indices, int dim, int numNodes);
void GraphSum_forward(Variable *in, Variable *out, SparseIndex *graph, int dim);

__global__
void GraphSum_backward_kernel(float *d_in, float *d_out, int *d_indptr, int *d_indices, int dim, int numNodes);
void GraphSum_backward(Variable *in, Variable *out, SparseIndex *graph, int dim);

__global__
void CrossEntropy_forward_kernel(float* logits_data, float* logits_grad, bool training, int num_classes, int* truth, int* count, float* thread_loss, int size);
void CrossEntropy_forward(Variable *logits, int *truth, float &total_loss, int &count, int num_classes, bool training);

__global__
void ReLU_forward_kernel(float *d_in, bool *d_mask, const long unsigned int datasize, bool training);
void ReLU_forward(Variable *in, bool *mask, bool training);

__global__
void ReLU_backward_kernel(float *d_in, bool *d_mask, long unsigned int datasize);
void ReLU_backward(Variable *in, bool *mask, bool training);

#endif
