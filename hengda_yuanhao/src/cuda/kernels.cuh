#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
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

__global__
void Matmul_forward_kernel(const float *a, const float *b, float *c, const uint m, const uint n, const uint p); 
void Matmul_forward(Variable *a, Variable *b, Variable *c, int m, int n, int p);

__global__
void cudaCalcAGrad(float *a_grad, const float *b, const float *c_grad, const uint m, const uint n, const uint p);
__global__
void cudaCalcBGrad(float *b_grad, const float *a, const float *c_grad, const uint m, const uint n, const uint p);
void Matmul_backward(Variable *a, Variable *b, Variable *c, int m, int n, int p);

__global__
void Dropout_forward_kernel(float *in, bool *mask, curandState *state, const uint size, const float p,const float scale, const bool useMask);
void Dropout_forward(Variable *in, bool *mask, float p);

__global__
void Dropout_backward_kernel(float *in_grad, const bool *mask, const uint size, const float scale);
void Dropout_backward(Variable *in, bool *mask, float p);

void cudaCallInitRandomState(const uint size);
void cudaCallFreeRandomState();

#endif
