#include "cuda_kernel.cuh"

curandState *devStates;

// matrix mult
__global__
void cuda_Matmul_forward_kernel(const float *a, const float *b, float *c, const uint m, const uint n, const uint p) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    int range = (n-1) / TILE_SIZE + 1;
    float res = 0;

    #pragma unroll
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
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; j++)
            res += tileA[ty][j] * tileB[j][tx];
        __syncthreads();
    }
    if (row < m && col < p)
        c[row * p + col] = res;
}

__global__
void cuda_Matmul_backward_A_kernel(float *a_grad, const float *b, const float *c_grad, const uint m, const uint n, const uint p) {
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    __shared__ float tileCGrad[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    int range = (p-1) / TILE_SIZE + 1;
    float res = 0;
    #pragma unroll
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

        #pragma unroll
        for (int j = 0; j < TILE_SIZE; j++)
            res += tileCGrad[ty][j] * tileB[j][tx];
        __syncthreads();
    }
    if (row < m && col < n)
        a_grad[row * n + col] = res;
}

__global__
void cuda_Matmul_backward_B_kernel(float *b_grad, const float *a, const float *c_grad, const uint m, const uint n, const uint p) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileCGrad[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    int range = (m-1)/TILE_SIZE+1;
    float res = 0;

    #pragma unroll
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

        #pragma unroll
        for (int j = 0; j < TILE_SIZE; j++)
            res += tileA[ty][j] * tileCGrad[j][tx];
        __syncthreads();
    }
    if (row < n && col < p)
        b_grad[row * p + col] = res;
}


// sparse matmul
__global__
void cuda_SparseMatmul_forward_kernel(float *a_in, float *b_in, float *c_in, int *indptr, int *indices, int p) {
    int i = blockIdx.x;
    int k = threadIdx.x;

    #pragma unroll
    for (int jj = indptr[i]; jj < indptr[i + 1]; jj++) {
        int j = indices[jj];
        c_in[i * p + k] += a_in[jj] * b_in[j * p + k];
    }
}

__global__
void cuda_SparseMatmul_backward_kernel(float *a_in, float *b_in, float *c_in, int *indptr, int *indices, int p) {
    int i = blockIdx.x;
    int k = threadIdx.x;

    #pragma unroll
    for (int jj = indptr[i]; jj < indptr[i + 1]; jj++){
        int j = indices[jj];
        b_in[j * p + k] += c_in[i * p + k] * a_in[jj];
    }
}


// graph sum
__global__
void cuda_GraphSum_forward_kernel(float *d_in_data, float *d_out_data, int *d_indptr, int *d_indices, int dim, int numNodes) {
    int src = blockIdx.x;
    int j = threadIdx.x;

    int ptr_src_0 = d_indptr[src];
    int ptr_stc_1 = d_indptr[src + 1];

    #pragma unroll
    for (int i = ptr_src_0; i < ptr_stc_1; i++) {
        int dst = d_indices[i];
        float coef = 1.0 / sqrtf(
                (ptr_stc_1 - ptr_src_0) * (d_indptr[dst + 1] - d_indptr[dst])
        );
        // This only works for undirected graphs. Should be out[dst] += coef * in[src]]
        d_out_data[src * dim + j] += coef * d_in_data[dst * dim + j];
    }
}

__global__
void cuda_GraphSum_backward_kernel(float *d_in_grad, float *d_out_grad, int *d_indptr, int *d_indices, int dim, int numNodes) {
    int src = blockIdx.x;
    int j = threadIdx.x;

    int ptr_src_0 = d_indptr[src];
    int ptr_stc_1 = d_indptr[src + 1];

    #pragma unroll
    for (int i = ptr_src_0; i < ptr_stc_1; i++) {
        int dst = d_indices[i];
        float coef = 1.0 / sqrtf(
                (ptr_stc_1 - ptr_src_0) * (d_indptr[dst + 1] - d_indptr[dst])
        );
        // This only works for undirected graphs. Should be out[dst] += coef * in[src]
        d_in_grad[src * dim + j] += coef * d_out_grad[dst * dim + j];
    }
}


// cross entropy
__global__ 
void cuda_CrossEntropy_forward_A_kernel(float* logits_data, float* logits_grad, bool training, int num_classes, int* truth, int* count, float* thread_loss, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    if (truth[i] < 0) {
        count[i] = 0;
        return;
    }
    float *logit = &logits_data[i * num_classes];
    float max_logit = -1e30, sum_exp = 0;
    #pragma unroll
    for (int j = 0; j < num_classes; j++)
        max_logit = fmax(max_logit, logit[j]);
    #pragma unroll
    for (int j = 0; j < num_classes; j++) {
        logit[j] -= max_logit;
        sum_exp += expf(logit[j]);
    }
    if (training) {
        #pragma unroll
        for (int j = 0; j < num_classes; j++) {
            float prob = expf(logit[j]) / sum_exp;
            logits_grad[i * num_classes + j] = prob;
        }
        logits_grad[i * num_classes + truth[i]] -= 1.0;
    }
    count[i] = 1;
    thread_loss[i] = logf(sum_exp) - logit[truth[i]];
}

__global__
void cuda_CrossEntropy_forward_B_kernel(float *logits_grad, int size, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) logits_grad[i] /= count;
}


// ReLU
__global__
void cuda_ReLU_forward_kernel(float *d_in_data, bool *d_mask, const long unsigned int datasize, bool training) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= datasize) return;

    bool keep = d_in_data[i] > 0;
    if (training) d_mask[i] = keep;
    if (!keep) d_in_data[i] = 0;
}

__global__
void cuda_ReLU_backward_kernel(float *d_in_grad, bool *d_mask, long unsigned int datasize) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= datasize) return;
    if (!d_mask[i]) d_in_grad[i] = 0;
}


// Dropout
__global__
void cuda_Dropout_forward_kernel(float *in, int *mask, curandState *state, const uint size, const float p, const float scale, const bool useMask) {
    float x;
    bool keep;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        x = curand_uniform(&state[id % MAX_THREAD_PER_BLOCK]);
        keep = x >= p;
        in[id] *= keep ? scale : 0;
        if (useMask) mask[id] = keep;
    }
}

__global__
void cuda_Dropout_backward_kernel(float *in_grad, const int *mask, const uint size, const float scale) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) in_grad[id] *= mask[id] ? scale : 0;
}


// rand state
__global__
void cuda_init_rand_kernel(curandState *state) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, id, 0, &state[id]);
}

void cuda_init_random_state(const uint size) {
    // malloc
    CUDA_CHECK(cudaMalloc((void**) &devStates, size * sizeof(curandState)));

    dim3 block((size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);

    // kernel
    cuda_init_rand_kernel<<<block,thread_in_block>>>(devStates);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_free_random_state() {
    // free
    CUDA_CHECK(cudaFree(devStates));
}


// adam
__global__
void cuda_Adam_step_kernel(float* grad, float* data, float* m, float* v, bool decay, float weight_decay, float beta1, float beta2, float eps, float step_size, int varsize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= varsize) return;

    float g = grad[i];
    if (decay) g += weight_decay * data[i];
    m[i] = beta1 * m[i] + (1.0 - beta1) * g;
    v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
    data[i] -= step_size * m[i] / (sqrtf(v[i]) + eps);
}

__global__
void cuda_set_truth_kernel(int *truth, int *data_split, int *data_label, int current_split, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
        truth[id] = data_split[id] == current_split ? data_label[id] : -1;
}

__global__
void cuda_Variable_glorot_kernel(float *data, curandState *state, int size, float scale) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
        data[id] = (curand_uniform(&state[id % MAX_THREAD_PER_BLOCK]) - 0.5) * scale;
}
