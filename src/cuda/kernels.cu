#include "kernels.cuh"

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

void cuda_Matmul_forward(Variable *a, Variable *b, Variable *c, int m, int n, int p) {
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**) &d_a, m * n * sizeof(float));
    cudaMemcpy(d_a, a->data.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_b, n * p * sizeof(float));
    cudaMemcpy(d_b, b->data.data(), n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_c, m * p * sizeof(float));

    dim3 block((p-1) / TILE_SIZE + 1, (m-1) / TILE_SIZE + 1, 1);
    dim3 thread_in_block(TILE_SIZE, TILE_SIZE, 1);
    cuda_Matmul_forward_kernel<<<block, thread_in_block>>>(d_a, d_b, d_c, m, n, p);
    cudaDeviceSynchronize();

    cudaMemcpy(c->data.data(), d_c, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
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

__global__
void cuda_Matmul_backward_B_kernel(float *b_grad, const float *a, const float *c_grad, const uint m, const uint n, const uint p) {
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

void cuda_Matmul_backward(Variable *a, Variable *b, Variable *c, int m, int n, int p) {
    float *d_a, *d_b, *d_a_g, *d_b_g, *d_c_g;

    cudaMalloc((void**) &d_a, m * n * sizeof(float));
    cudaMemcpy(d_a, a->data.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_b, n * p * sizeof(float));
    cudaMemcpy(d_b, b->data.data(), n * p * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_a_g, m * n * sizeof(float));
    cudaMalloc((void**) &d_b_g, n * p * sizeof(float));
    cudaMalloc((void**) &d_c_g, m * p * sizeof(float));
    cudaMemcpy(d_c_g, c->grad.data(), m * p * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_a((n-1)/TILE_SIZE+1, (m-1)/TILE_SIZE+1, 1);
    dim3 block_b((p-1)/TILE_SIZE+1, (n-1)/TILE_SIZE+1, 1);
    dim3 thread_in_block(TILE_SIZE, TILE_SIZE, 1);
    cuda_Matmul_backward_A_kernel<<<block_a, thread_in_block>>>(d_a_g, d_b, d_c_g, m, n, p);
    cudaDeviceSynchronize();
    cuda_Matmul_backward_B_kernel<<<block_b, thread_in_block>>>(d_b_g, d_a, d_c_g, m, n, p);
    cudaDeviceSynchronize();

    cudaMemcpy(a->grad.data(), d_a_g, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b->grad.data(), d_b_g, n * p * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_a_g);
    cudaFree(d_b_g);
    cudaFree(d_c_g);
}


// sparse matmul
__global__
void cuda_SparseMatmul_forward_kernel(float *a_in, float *b_in, float *c_in, int *indptr, int *indices, int p) {
    int i = blockIdx.x;
    int k = blockIdx.y * MAX_THREAD_PER_BLOCK + threadIdx.x;
    
    for (int jj = indptr[i]; jj < indptr[i + 1]; jj++){
        int j = indices[jj];
        c_in[i * p + k] += a_in[jj] * b_in[j * p + k];
    }
}

void cuda_SparseMatmul_forward(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int p) {
    float *a_in, *b_in, *c_in;
    int *d_indptr, *d_indices;

    cudaMalloc((void**) &a_in, a->data.size() * sizeof(float));
    cudaMalloc((void**) &b_in, b->data.size() * sizeof(float));
    cudaMalloc((void**) &c_in, c->data.size() * sizeof(float));
    cudaMalloc(&d_indptr, sp->indptr.size() * sizeof(int));
    cudaMalloc(&d_indices, sp->indices.size() * sizeof(int));

    cudaMemcpy(a_in, a->data.data(), a->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_in, b->data.data(), b->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_in, c->data.data(), c->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indptr, sp->indptr.data(), sp->indptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, sp->indices.data(), sp->indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    if(sp->indptr.size() <= 1) return;

    dim3 gridsize(sp->indptr.size() - 1, 1);
    dim3 blocksize(p);

    if(p > MAX_THREAD_PER_BLOCK) {
        blocksize.x = MAX_THREAD_PER_BLOCK;
        gridsize.y = ceil((float)p / (float)blocksize.x);
    }

    cuda_SparseMatmul_forward_kernel<<<gridsize, blocksize>>>(a_in, b_in, c_in, d_indptr, d_indices, p);

    cudaMemcpy(c->data.data(), c_in, c->data.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a_in);
    cudaFree(b_in);
    cudaFree(c_in);
    cudaFree(d_indptr);
    cudaFree(d_indices);
}

__global__
void cuda_SparseMatmul_backward_kernel(float *a_in, float *b_in, float *c_in, int *indptr, int *indices, int p) {
    int i = blockIdx.x;
    int k = blockIdx.y * MAX_THREAD_PER_BLOCK + threadIdx.x;
    
    for (int jj = indptr[i]; jj < indptr[i + 1]; jj++){
        int j = indices[jj];
        b_in[j * p + k] += c_in[i * p + k] * a_in[jj];
    }
}

void cuda_SparseMatmul_backward(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int p) {
    float *a_in, *b_in, *c_in;
    int *d_indptr, *d_indices;

    cudaMalloc((void**) &a_in, a->data.size() * sizeof(float));
    cudaMalloc((void**) &b_in, b->grad.size() * sizeof(float));
    cudaMalloc((void**) &c_in, c->grad.size() * sizeof(float));
    cudaMalloc((void**) &d_indptr, sp->indptr.size() * sizeof(int));
    cudaMalloc((void**) &d_indices, sp->indices.size() * sizeof(int));

    cudaMemcpy(a_in, a->data.data(), a->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_in, b->grad.data(), b->grad.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_in, c->grad.data(), c->grad.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indptr, sp->indptr.data(), sp->indptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, sp->indices.data(), sp->indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    if(sp->indptr.size() <= 1) return;

    dim3 gridsize(sp->indptr.size() - 1, 1);
    dim3 blocksize(p);

    if(p > MAX_THREAD_PER_BLOCK) {
        blocksize.x = MAX_THREAD_PER_BLOCK;
        gridsize.y = ceil((double)p / (double) blocksize.x);
    }

    cuda_SparseMatmul_backward_kernel<<<gridsize, blocksize>>>(a_in, b_in, c_in, d_indptr, d_indices, p);

    cudaMemcpy(b->grad.data(), b_in, b->grad.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a_in);
    cudaFree(b_in);
    cudaFree(c_in);
    cudaFree(d_indptr);
    cudaFree(d_indices);
}


// graph sum
__global__
void cuda_GraphSum_forward_kernel(float *d_in_data, float *d_out_data, int *d_indptr, int *d_indices, int dim, int numNodes) {
    int src = blockIdx.x;
    int j = blockIdx.y * MAX_THREAD_PER_BLOCK + threadIdx.x;

    for (int i = d_indptr[src]; i < d_indptr[src + 1]; i++) {
        int dst = d_indices[i];
        float coef = 1.0 / sqrtf(
                (d_indptr[src + 1] - d_indptr[src]) * (d_indptr[dst + 1] - d_indptr[dst])
        );
        // This only works for undirected graphs. Should be out[dst] += coef * in[src]
        d_out_data[src * dim + j] += coef * d_in_data[dst * dim + j];
    }
    // }
}

void cuda_GraphSum_forward(Variable *in, Variable *out, SparseIndex *graph, int dim) {
    float *d_in_data, *d_out_data;
    int *d_indptr, *d_indices;

    // allocate memory
    cudaMalloc((void**) &d_in_data, in->data.size() * sizeof(float));
    cudaMalloc((void**) &d_out_data, out->data.size() * sizeof(float));
    cudaMalloc((void**) &d_indptr, graph->indptr.size() * sizeof(int));
    cudaMalloc((void**) &d_indices, graph->indices.size() * sizeof(int));

    // copy memory from host to device
    cudaMemcpy(d_in_data, in->data.data(), in->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indptr, graph->indptr.data(), graph->indptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, graph->indices.data(), graph->indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    // kernel
    const int numNodes = graph->indptr.size() - 1;
    dim3 numBlocks(numNodes, 1);
    dim3 threadsPerBlock(dim, 1);

    if(dim > MAX_THREAD_PER_BLOCK) {
        numBlocks.x = MAX_THREAD_PER_BLOCK;
        threadsPerBlock.y = ceil((float)dim / (float)numBlocks.x);
    }

    cuda_GraphSum_forward_kernel<<<numBlocks, threadsPerBlock>>>(d_in_data, d_out_data, d_indptr, d_indices, dim, numNodes);
    cudaDeviceSynchronize();

    // copy result back to out
    cudaMemcpy(out->data.data(), d_out_data, out->data.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_in_data);
    cudaFree(d_out_data);
    cudaFree(d_indptr);
    cudaFree(d_indices);
}

__global__
void cuda_GraphSum_backward_kernel(float *d_in_grad, float *d_out_grad, int *d_indptr, int *d_indices, int dim, int numNodes) {
    int src = blockIdx.x;
    int j = blockIdx.y * MAX_THREAD_PER_BLOCK + threadIdx.x;

    // for (int src = 0; src < numNodes; ++src) {
    for (int i = d_indptr[src]; i < d_indptr[src + 1]; i++) {
        int dst = d_indices[i];
        float coef = 1.0 / sqrtf(
                (d_indptr[src + 1] - d_indptr[src]) * (d_indptr[dst + 1] - d_indptr[dst])
        );
        // This only works for undirected graphs. Should be out[dst] += coef * in[src]
        d_in_grad[src * dim + j] += coef * d_out_grad[dst * dim + j];
    }
    // }
}

void cuda_GraphSum_backward(Variable *in, Variable *out, SparseIndex *graph, int dim) {
    float *d_in_grad, *d_out_grad;
    int *d_indptr, *d_indices;

    // allocate memory
    cudaMalloc((void**) &d_in_grad, in->grad.size() * sizeof(float));
    cudaMalloc((void**) &d_out_grad, out->grad.size() * sizeof(float));
    cudaMalloc((void**) &d_indptr, graph->indptr.size() * sizeof(int));
    cudaMalloc((void**) &d_indices, graph->indices.size() * sizeof(int));

    // copy memory from host to device
    cudaMemcpy(d_out_grad, out->grad.data(), out->grad.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indptr, graph->indptr.data(), graph->indptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, graph->indices.data(), graph->indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    // kernel
    const int numNodes = graph->indptr.size() - 1;
    dim3 numBlocks(numNodes, 1);
    dim3 threadsPerBlock(dim, 1);

    if(dim > MAX_THREAD_PER_BLOCK) {
        numBlocks.x = MAX_THREAD_PER_BLOCK;
        threadsPerBlock.y = ceil((float)dim / (float)numBlocks.x);
    }

    cuda_GraphSum_backward_kernel<<<numBlocks, threadsPerBlock>>>(d_in_grad, d_out_grad, d_indptr, d_indices, dim, numNodes);
    cudaDeviceSynchronize();

    // copy result back to out
    cudaMemcpy(in->grad.data(), d_in_grad, in->grad.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_in_grad);
    cudaFree(d_out_grad);
    cudaFree(d_indptr);
    cudaFree(d_indices);
}


// cross entropy
__global__ 
void cuda_CrossEntropy_forward_kernel(float* logits_data, float* logits_grad, bool training, int num_classes, int* truth, int* count, float* thread_loss, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;
    if (truth[i] < 0) {
        count[i] = 0;
        return;
    }
    float *logit = &logits_data[i * num_classes];
    float max_logit = -1e30, sum_exp = 0;
    for (int j = 0; j < num_classes; j++)
        max_logit = fmax(max_logit, logit[j]);
    for (int j = 0; j < num_classes; j++) {
        logit[j] -= max_logit;
        sum_exp += expf(logit[j]);
    }
    if (training) {
        for (int j = 0; j < num_classes; j++) {
            float prob = expf(logit[j]) / sum_exp;
            logits_grad[i * num_classes + j] = prob;
        }
        logits_grad[i * num_classes + truth[i]] -= 1.0;
    }
    count[i] = 1;
    thread_loss[i] = logf(sum_exp) - logit[truth[i]];
}

void cuda_CrossEntropy_forward(Variable *logits, int *truth, float &total_loss, int &count, int num_classes, bool training) {
    // grid + block size
    int grid = 32;
    int block = (logits->data.size()/num_classes+32) / 32;
    // printf("block size: %d\n", block);	
    // printf("logits data size: %ld\n", logits->data.size());	
    // printf("num_classes: %d\n", num_classes);

    // data structures in GPU:
    float* d_logits_data, *d_loss, *d_logits_grad;
    int* d_truth, *d_count;
    int logits_data_size = (int)(logits->data.size())*sizeof(float);
    int logits_grad_size = (int)(logits->grad.size())*sizeof(float);
    int loss_size = (int)(logits->data.size()/num_classes)*sizeof(float);
    int truth_size = (int)(logits->data.size()/num_classes)*sizeof(int);

    // host function variables
    float *logits_data = logits->data.data();
    float *logits_grad = logits->grad.data();

    // cudaMalloc
    cudaMalloc(&d_logits_data, logits_data_size);
    cudaMalloc(&d_logits_grad, logits_grad_size);
    cudaMalloc(&d_loss, loss_size);
    cudaMalloc(&d_truth, truth_size);
    cudaMalloc(&d_count, truth_size);

    // copy data to GPU memory
    cudaMemcpy(d_logits_data, logits_data, logits_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_logits_grad, logits_grad, logits_grad_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_truth, truth, truth_size, cudaMemcpyHostToDevice);

    // run kernel function
    cuda_CrossEntropy_forward_kernel<<< grid, block >>>(d_logits_data, d_logits_grad, training, num_classes, d_truth, d_count, d_loss, logits->data.size());
    cudaDeviceSynchronize();

    // updates logits->data and logits->grad in host function
    cudaMemcpy(&(logits->data[0]), d_logits_data, logits_data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&(logits->grad[0]), d_logits_grad, logits_grad_size, cudaMemcpyDeviceToHost);

    // accumulate and add count and total_loss variables by thrust::
    thrust::device_ptr<int> count_ptr = thrust::device_pointer_cast(d_count);
    count = thrust::reduce(count_ptr, count_ptr+(logits->data.size()/num_classes), (int)0, thrust::plus<int>());
    thrust::device_ptr<float> loss_ptr = thrust::device_pointer_cast(d_loss);
    total_loss = thrust::reduce(loss_ptr, loss_ptr+(logits->data.size()/num_classes), (float)0.0, thrust::plus<float>());
    cudaDeviceSynchronize();

    // free memory
    cudaFree(d_logits_data);
    cudaFree(d_logits_grad);
    cudaFree(d_loss);
    cudaFree(d_truth);
    cudaFree(d_count);
}


// ReLU
__global__
void cuda_ReLU_forward_kernel(float *d_in_data, bool *d_mask, const long unsigned int datasize, bool training) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= datasize) return;

    bool keep = d_in_data[i] > 0;
    if (training) d_mask[i] = keep;
    if (!keep) d_in_data[i] = 0;
}

void cuda_ReLU_forward(Variable *in, bool *mask, bool training) {
    float *d_in_data;
    bool *d_mask;
    const long unsigned int datasize = in->data.size();

    cudaMalloc(&d_in_data, datasize * sizeof(float));
    cudaMalloc(&d_mask, datasize * sizeof(bool));

    cudaMemcpy(d_in_data, in->data.data(), datasize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, datasize * sizeof(bool), cudaMemcpyHostToDevice);
    // printf("ReLU data size %lu\n", in->data.size());

    const int bsize = 128;
    dim3 numBlocks(bsize, 1);
    dim3 threadsPerBlock(ceil(float(datasize) / bsize), 1);
    cuda_ReLU_forward_kernel<<<numBlocks, threadsPerBlock>>>(d_in_data, d_mask, datasize, training);
    cudaDeviceSynchronize();

    cudaMemcpy(in->data.data(), d_in_data, datasize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mask, d_mask, datasize * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_in_data);
    cudaFree(d_mask);
}

__global__
void cuda_ReLU_backward_kernel(float *d_in_grad, bool *d_mask, long unsigned int datasize) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i >= datasize) return;

    if (!d_mask[i]) d_in_grad[i] = 0;
}

void cuda_ReLU_backward(Variable *in, bool *mask) {
    float *d_in_grad;
    bool *d_mask;
    const long unsigned int datasize = in->data.size();

    cudaMalloc(&d_in_grad, datasize * sizeof(float));
    cudaMalloc(&d_mask, datasize * sizeof(bool));

    cudaMemcpy(d_in_grad, in->grad.data(), datasize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, datasize * sizeof(bool), cudaMemcpyHostToDevice);

    const int bsize = 128;
    dim3 numBlocks(bsize, 1);
    dim3 threadsPerBlock(ceil(float(datasize) / bsize), 1);
    cuda_ReLU_backward_kernel<<<numBlocks, threadsPerBlock>>>(d_in_grad, d_mask, datasize);
    cudaDeviceSynchronize();

    cudaMemcpy(in->grad.data(), d_in_grad, datasize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in_grad);
    cudaFree(d_mask);
}


// Dropout
__global__
void cuda_Dropout_forward_kernel(float *in, int *mask, curandState *state, const uint size, const float p, const float scale, const bool useMask) {
    float x;
    bool keep;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        x = curand_uniform(&state[id]);
        keep = x >= p;
        in[id] *= keep ? scale : 0;
        if (useMask) mask[id] = keep;
    }
}

void cuda_Dropout_forward(Variable *in, int *mask, float p) {
    int size = in->data.size();
    float *d_in;
    int *d_mask;

    cudaMalloc((void**) &d_in, size * sizeof(float));
    cudaMemcpy(d_in, in->data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    if (mask) cudaMalloc((void**) &d_mask, size * sizeof(int));

    float scale = 1 / (1 - p);
    dim3 block((size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    cuda_Dropout_forward_kernel<<<block, thread_in_block>>>(d_in, d_mask, devStates, size, p, scale, (mask != nullptr));
    cudaDeviceSynchronize();

    cudaMemcpy(in->data.data(), d_in, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (mask) cudaMemcpy(mask, d_mask, size * sizeof(int), cudaMemcpyDeviceToHost);
}

__global__
void cuda_Dropout_backward_kernel(float *in_grad, const int *mask, const uint size, const float scale) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
        in_grad[id] *= mask[id] ? scale : 0;
}

void cuda_Dropout_backward(Variable *in, int *mask, float p) {
    uint size = in->data.size();
    float *d_in_g;
    int *d_mask;

    cudaMalloc((void**) &d_in_g, size * sizeof(float));
    cudaMemcpy(d_in_g, in->grad.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_mask, size * sizeof(int));
    cudaMemcpy(d_mask, mask, size * sizeof(int), cudaMemcpyHostToDevice);

    float scale = 1 / (1 - p);
    dim3 block((size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    cuda_Dropout_backward_kernel<<<block, thread_in_block>>>(d_in_g, d_mask, size, scale);
    cudaDeviceSynchronize();

    cudaMemcpy(in->grad.data(), d_in_g, size * sizeof(float), cudaMemcpyDeviceToHost);
}


// rand state
__global__
void cuda_init_rand_kernel(curandState *state) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, id, 0, &state[id]);
}

void cuda_init_random_state(const uint size) {
    cudaMalloc((void**) &devStates, size * sizeof(curandState));
    dim3 block((size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    cuda_init_rand_kernel<<<block,thread_in_block>>>(devStates);
    cudaDeviceSynchronize();
}

void cuda_free_random_state() {
    cudaFree(devStates);
}
