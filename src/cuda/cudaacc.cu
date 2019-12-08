#include "cudaacc.cuh"

#include <curand_kernel.h>

#define TILE_SIZE 32

__global__ void cudaMatMul(const float *a, const float *b, float *c, const uint m, const uint n, const uint p) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    int range = (n-1)/TILE_SIZE+1;
    float res = 0;

    for (int i = 0; i < range; i++) {
        if (row < m && i * TILE_SIZE + tx < n)
            tileA[ty][tx] = a[row * n + i * TILE_SIZE + tx];
        else
            tileA[ty][tx] = 0;
        if (col < p && i * TILE_SIZE + ty < n) 
            tileB[ty][tx] = b[(i * TILE_SIZE + ty) * p + col];
        else
            tileB[ty][tx] = 0;

        __syncthreads();
        for (int j = 0; j < TILE_SIZE; j++)
            res += tileA[ty][j] * tileB[j][tx];
        __syncthreads();
    }
    if (row < m && col < p)
        c[row * p + col] = res;

}

void cudaCallMatMulForward(
    const float *a,
    const float *b,
    float *c,
    const uint m, 
    const uint n, 
    const uint p) {
    
    dim3 block((p-1)/TILE_SIZE+1, (m-1)/TILE_SIZE+1, 1);
    dim3 thread_in_block(TILE_SIZE, TILE_SIZE, 1);
    cudaMatMul<<<block, thread_in_block>>>(a, b, c, m, n, p);
}

__global__ void cudaCalcAGrad(float *a_grad, const float *b, const float *c_grad, const uint m, const uint n, const uint p) {
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    __shared__ float tileCGrad[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    int range = (p-1)/TILE_SIZE+1;
    float res = 0;
    for (int i = 0; i < range; i++) {
        if (row < m && i * TILE_SIZE + tx < p)
            tileCGrad[ty][tx] = c_grad[row * p + i * TILE_SIZE + tx];
        else
            tileCGrad[ty][tx] = 0;
        if (col < n && i * TILE_SIZE + ty < p)
            tileB[ty][tx] = b[col * p + i * TILE_SIZE + ty];
        else
            tileB[ty][tx] = 0;
            
        __syncthreads();
        for (int j = 0; j < TILE_SIZE; j++)
            res += tileCGrad[ty][j] * tileB[j][tx];
        __syncthreads();
    }
    if (row < m && col < n)
        a_grad[row * n + col] = res;
}

__global__ void cudaCalcBGrad(float *b_grad, const float *a, const float *c_grad, const uint m, const uint n, const uint p) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileCGrad[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    int range = (m-1)/TILE_SIZE+1;
    float res = 0;
    for (int i = 0; i < range; i++) {
        if (row < n && i * TILE_SIZE + tx < m) 
            tileA[ty][tx] = a[(i * TILE_SIZE + tx) * n + row];
        else
            tileA[ty][tx] = 0;
        if (col < p && i * TILE_SIZE + ty < m) 
            tileCGrad[ty][tx] = c_grad[(i * TILE_SIZE + ty) * p + col];
        else
            tileCGrad[ty][tx] = 0;
        
        __syncthreads();
        for (int j = 0; j < TILE_SIZE; j++)
            res += tileA[ty][j] * tileCGrad[j][tx];
        __syncthreads();
    }
    if (row < n && col < p)
        b_grad[row * p + col] = res;
    
}

void cudaCallMatMulBackward(
    const float *a,
    const float *b,
    float *a_grad,
    float *b_grad,
    const float *c_grad,
    const uint m,
    const uint n,
    const uint p) {

    dim3 block_a((n-1)/TILE_SIZE+1, (m-1)/TILE_SIZE+1, 1);
    dim3 block_b((p-1)/TILE_SIZE+1, (n-1)/TILE_SIZE+1, 1);
    dim3 thread_in_block(TILE_SIZE, TILE_SIZE, 1);
    cudaCalcAGrad<<<block_a, thread_in_block>>>(a_grad, b, c_grad, m, n, p);
    cudaCalcBGrad<<<block_b, thread_in_block>>>(b_grad, a, c_grad, m, n, p);
}

__global__ void setupRandKernel(curandState *state) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, thread_index, 0, &state[thread_index]);
}

__global__ void cudaDropoutForward(float *in, bool *mask, curandState *state, const uint size, const float p) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    float x, scale = 1 / (1 - p); 
    bool keep;
    curandState localState = state[thread_index];
    if (thread_index < size) {
        x = curand_uniform(&localState);
        keep = x >= p;
        in[thread_index] *= keep ? scale : 0;
        if (mask) mask[thread_index] = keep;
    }
}

void cudaCallDropoutForward(
    const uint block_count,
    const uint per_block_thread_count,
    float *in,
    bool *mask,
    const uint size,
    const float p) {
    
    const uint thread_count = block_count * per_block_thread_count;
    curandState *devStates;
    cudaMalloc((void**) &devStates, thread_count * sizeof(curandState));
    setupRandKernel<<<block_count, per_block_thread_count>>>(devStates);
    cudaDropoutForward<<<block_count, per_block_thread_count>>>(in, mask, devStates, size, p);
    cudaFree(devStates);
}