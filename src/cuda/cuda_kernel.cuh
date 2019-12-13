#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <curand_kernel.h>

#define TILE_SIZE 32
#define MAX_THREAD_PER_BLOCK 1024

#define CUDA_CHECK(ans) { CUDA_ASSERT((ans), __FILE__, __LINE__); }

inline void CUDA_ASSERT(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA_ASSERT: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

extern curandState *devStates;

// MatMul
__global__
void cuda_Matmul_forward_kernel(const float *a, const float *b, float *c, const uint m, const uint n, const uint p);

__global__
void cuda_Matmul_backward_A_kernel(float *a_grad, const float *b, const float *c_grad, const uint m, const uint n, const uint p);

__global__
void cuda_Matmul_backward_B_kernel(float *b_grad, const float *a, const float *c_grad, const uint m, const uint n, const uint p);


// Sparse Mat Mul
__global__
void cuda_SparseMatmul_forward_kernel(float *a_in, float *b_in, float *c_in, int *indptr, int *indices, int p);

__global__
void cuda_SparseMatmul_backward_kernel(float *a_in, float *b_in, float *c_in, int *indptr, int *indices, int p);


// GraphSum
__global__
void cuda_GraphSum_forward_kernel(float *d_in_data, float *d_out_data, int *d_indptr, int *d_indices, int dim, int numNodes);

__global__
void cuda_GraphSum_backward_kernel(float *d_in_grad, float *d_out_grad, int *d_indptr, int *d_indices, int dim, int numNodes);


// Cross Entropy
__global__ 
void cuda_CrossEntropy_forward_A_kernel(float *logits_data, float *logits_grad, bool training, int num_classes, int* truth, int* count, float* thread_loss, int size);

__global__
void cuda_CrossEntropy_forward_B_kernel(float *logits_grad, int size, int count);


// ReLU
__global__
void cuda_ReLU_forward_kernel(float *d_in_data, bool *d_mask, const long unsigned int datasize, bool training);

__global__
void cuda_ReLU_backward_kernel(float *d_in_grad, bool *d_mask, long unsigned int datasize);


// Dropout
__global__
void cuda_Dropout_forward_kernel(float *in, int *mask, curandState *state, const uint size, const float p, const float scale, const bool useMask);

__global__
void cuda_Dropout_backward_kernel(float *in_grad, const int *mask, const uint size, const float scale);


// rand state
__global__
void cuda_init_rand_kernel(curandState *state);

void cuda_init_random_state(const uint size);
void cuda_free_random_state();


// adam
__global__
void cuda_Adam_step_kernel(float* grad, float* data, float* m, float* v, bool decay, float weight_decay, float beta1, float beta2, float eps, float step_size, int varsize);


__global__
void cuda_set_truth_kernel(int *truth, int *data_split, int *data_label, int current_split, int size);

#endif
