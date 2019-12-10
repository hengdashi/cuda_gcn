#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "sparse.h"
#include "variable.h"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <curand_kernel.h>

// MatMul
__global__
void cuda_Matmul_forward_kernel(const float *a, const float *b, float *c, const uint m, const uint n, const uint p);

void cuda_Matmul_forward(Variable *a, Variable *b, Variable *c, int m, int n, int p);

__global__
void cuda_Matmul_backward_A_kernel(float *a_grad, const float *b, const float *c_grad, const uint m, const uint n, const uint p);

__global__
void cuda_Matmul_backward_B_kernel(float *b_grad, const float *a, const float *c_grad, const uint m, const uint n, const uint p);

void cuda_Matmul_backward(Variable *a, Variable *b, Variable *c, int m, int n, int p);


// Sparse Mat Mul
__global__
void cuda_SparseMatmul_forward_kernel(float *a_in, float *b_in, float *c_in, int *indptr, int *indices, int p);

void cuda_SparseMatmul_forward(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int p);

__global__
void cuda_SparseMatmul_backward_kernel(float *a_in, float *b_in, float *c_in, int *indptr, int *indices, int p);

void cuda_SparseMatmul_backward(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int p);


// GraphSum
__global__
void cuda_GraphSum_forward_kernel(float *d_in_data, float *d_out_data, int *d_indptr, int *d_indices, int dim, int numNodes);

void cuda_GraphSum_forward(Variable *in, Variable *out, SparseIndex *graph, int dim);

__global__
void cuda_GraphSum_backward_kernel(float *d_in_grad, float *d_out_grad, int *d_indptr, int *d_indices, int dim, int numNodes);

void cuda_GraphSum_backward(Variable *in, Variable *out, SparseIndex *graph, int dim);


// Cross Entropy
__global__ 
void cuda_CrossEntropy_forward_kernel(float* logits_data, float* logits_grad, bool training, int num_classes, int* truth, int* count, float* thread_loss, int size);

void cuda_CrossEntropy_forward(Variable *logits, int *truth, float &total_loss, int &count, int num_classes, bool training);


// ReLU
__global__
void cuda_ReLU_forward_kernel(float *d_in_data, bool *d_mask, const long unsigned int datasize, bool training);

void cuda_ReLU_forward(Variable *in, bool *mask, bool training);

__global__
void cuda_ReLU_backward_kernel(float *d_in_grad, bool *d_mask, long unsigned int datasize);

void cuda_ReLU_backward(Variable *in, bool *mask);


// Dropout
__global__
void cuda_Dropout_forward_kernel(float *in, int *mask, curandState *state, const uint size, const float p, const float scale, const bool useMask);

void cuda_Dropout_forward(Variable *in, int *mask, float p);

__global__
void cuda_Dropout_backward_kernel(float *in_grad, const int *mask, const uint size, const float scale);

void cuda_Dropout_backward(Variable *in, int *mask, float p);


// rand state
__global__
void cuda_init_rand_kernel(curandState *state);

void cuda_init_random_state(const uint size);
void cuda_free_random_state();

#endif
