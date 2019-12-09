#ifndef CUDA_ACC_H
#define CUDA_ACC_H

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

void cudaCallMatMulForward(
    const float *a,
    const float *b,
    float *c,
    const uint m, 
    const uint n, 
    const uint p);

void cudaCallMatMulBackward(
    const float *a,
    const float *b,
    float *a_grad,
    float *b_grad,
    const float *c_grad,
    const uint m,
    const uint n, 
    const uint p);

void cudaCallInitRandomState(const uint size);
void cudaCallFreeRandomState();

void cudaCallDropoutForward(
    float *in,
    bool *mask,
    const uint size,
    const float p,
    const bool useMask);

void cudaCallDropoutBackward(
    float *in_grad,
    const bool *mask,
    const uint size,
    const float p);

#endif