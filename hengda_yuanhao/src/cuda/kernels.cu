#include "kernels.cuh"

#define TILE_SIZE 32
#define MAX_THREAD_PER_BLOCK 1024

curandState *devStates;

__global__
void GraphSum_forward_kernel(float *d_in, float *d_out, int *d_indptr, int *d_indices, int dim, int numNodes) {
    // printf("graphsum forward loop count: %lu\n", nodecount);
    uint src = (blockIdx.x * blockDim.x) + threadIdx.x;
    // printf("src: %u\n", src);
    if (src >= numNodes)
        return;
    // printf("src: %d, i: %d, size: %d\n", src, d_indptr[src], d_indptr[src + 1]);
    // for (int src = 0; src < numNodes; ++src) {
    for (int i = d_indptr[src]; i < d_indptr[src + 1]; i++) {
        int dst = d_indices[i];
        float coef = 1.0 / sqrtf(
                (d_indptr[src + 1] - d_indptr[src]) * (d_indptr[dst + 1] - d_indptr[dst])
        );
        // printf("dim: %d\n", dim);
        for (int j = 0; j < dim; j++) {
            // This only works for undirected graphs. Should be out[dst] += coef * in[src]
            d_out[src * dim + j] += coef * d_in[dst * dim + j];
        }
    }
    // }
}

void GraphSum_forward(Variable *in, Variable *out, SparseIndex *graph, int dim) {
    float *d_in;
    float *d_out;
    int *d_indptr;
    int *d_indices;

    // allocate memory
    cudaMalloc(&d_in, in->data.size() * sizeof(float));
    cudaMalloc(&d_out, out->data.size() * sizeof(float));
    cudaMalloc(&d_indptr, graph->indptr.size() * sizeof(int));
    cudaMalloc(&d_indices, graph->indices.size() * sizeof(int));

    // copy memory from host to device
    cudaMemcpy(d_in, in->data.data(), in->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indptr, graph->indptr.data(), graph->indptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, graph->indices.data(), graph->indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    // kernel
    // printf("size of indptr: %lu\n", graph->indptr.size());
    const int numNodes = graph->indptr.size() - 1;
    const int bsize = 32;
    dim3 numBlocks(bsize, 1);
    dim3 threadsPerBlock(ceil(float(numNodes)/bsize), 1);
    // dim3 numBlocks(1, 1);
    // dim3 threadsPerBlock(1, 1);
    GraphSum_forward_kernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, d_indptr, d_indices, dim, numNodes);
    cudaDeviceSynchronize();

    // copy result back to out
    cudaMemcpy(out->data.data(), d_out, out->data.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_indptr);
    cudaFree(d_indices);
}

__global__
void GraphSum_backward_kernel(float *d_in, float *d_out, int *d_indptr, int *d_indices, int dim, int numNodes) {
    // printf("graphsum backward loop count: %lu\n", nodecount);
    uint src = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (src >= numNodes) {
        return;
    }
    // for (int src = 0; src < numNodes; ++src) {
    for (int i = d_indptr[src]; i < d_indptr[src + 1]; i++) {
        int dst = d_indices[i];
        float coef = 1.0 / sqrtf(
                (d_indptr[src + 1] - d_indptr[src]) * (d_indptr[dst + 1] - d_indptr[dst])
        );
        for (int j = 0; j < dim; j++) {
            // This only works for undirected graphs. Should be out[dst] += coef * in[src]
            d_in[src * dim + j] += coef * d_out[dst * dim + j];
        }
    }
    // }
}

void GraphSum_backward(Variable *in, Variable *out, SparseIndex *graph, int dim) {
    float *d_in;
    float *d_out;
    int *d_indptr;
    int *d_indices;

    // allocate memory
    cudaMalloc(&d_in, in->grad.size() * sizeof(float));
    cudaMalloc(&d_out, out->grad.size() * sizeof(float));
    cudaMalloc(&d_indptr, graph->indptr.size() * sizeof(int));
    cudaMalloc(&d_indices, graph->indices.size() * sizeof(int));

    // copy memory from host to device
    cudaMemcpy(d_out, out->grad.data(), out->grad.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indptr, graph->indptr.data(), graph->indptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, graph->indices.data(), graph->indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    // kernel
    const int numNodes = graph->indptr.size() - 1;
    const int bsize = 32;
    dim3 numBlocks(bsize, 1);
    dim3 threadsPerBlock(ceil(float(numNodes) / bsize), 1);
    GraphSum_backward_kernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, d_indptr, d_indices, dim, numNodes);
    cudaDeviceSynchronize();

    // copy result back to out
    cudaMemcpy(in->grad.data(), d_in, in->grad.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_indptr);
    cudaFree(d_indices);
}

__global__
void ReLU_forward_kernel(float *d_in, bool *d_mask, const long unsigned int datasize, bool training) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= datasize) {
      return;
    }

    bool keep = d_in[i] > 0;
    if (training) d_mask[i] = keep;
    if (!keep) d_in[i] = 0;
}

void ReLU_forward(Variable *in, bool *mask, bool training) {
    float *d_in;
    bool *d_mask; 
    const long unsigned int datasize = in->data.size();

    cudaMalloc(&d_in, datasize * sizeof(float));
    cudaMalloc(&d_mask, datasize * sizeof(bool));

    cudaMemcpy(d_in, in->data.data(), datasize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, datasize * sizeof(bool), cudaMemcpyHostToDevice);
    // printf("ReLU data size %lu\n", in->data.size());

    const int bsize = 128;
    dim3 numBlocks(bsize, 1);
    dim3 threadsPerBlock(ceil(float(datasize) / bsize), 1);
    ReLU_forward_kernel<<<numBlocks, threadsPerBlock>>>(d_in, d_mask, datasize, training);
    cudaDeviceSynchronize();

    cudaMemcpy(in->data.data(), d_in, datasize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mask, d_mask, datasize * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_mask);
}

__global__
void ReLU_backward_kernel(float *d_in, bool *d_mask, long unsigned int datasize) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i >= datasize) {
      return;
    }

    if (!d_mask[i]) d_in[i] = 0;
}

void ReLU_backward(Variable *in, bool *mask, bool training) {
    float *d_in;
    bool *d_mask;
    const long unsigned int datasize = in->data.size();

    cudaMalloc(&d_in, datasize * sizeof(float));
    cudaMalloc(&d_mask, datasize * sizeof(bool));

    cudaMemcpy(d_in, in->grad.data(), datasize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, datasize * sizeof(bool), cudaMemcpyHostToDevice);

    const int bsize = 128;
    dim3 numBlocks(bsize, 1);
    dim3 threadsPerBlock(ceil(float(datasize) / bsize), 1);
    ReLU_backward_kernel<<<numBlocks, threadsPerBlock>>>(d_in, d_mask, datasize);
    cudaDeviceSynchronize();

    cudaMemcpy(in->grad.data(), d_in, datasize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_mask);
}

__global__
void Matmul_forward_kernel(const float *a, const float *b, float *c, const uint m, const uint n, const uint p) {
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

void Matmul_forward(Variable *a, Variable *b, Variable *c, int m, int n, int p) {
    float *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**) &dev_a, m * n * sizeof(float));
    cudaMemcpy(dev_a, a->data.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &dev_b, n * p * sizeof(float));
    cudaMemcpy(dev_b, b->data.data(), n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &dev_c, m * p * sizeof(float));

    dim3 block((p-1)/TILE_SIZE+1, (m-1)/TILE_SIZE+1, 1);
    dim3 thread_in_block(TILE_SIZE, TILE_SIZE, 1);
    Matmul_forward_kernel<<<block, thread_in_block>>>(dev_a, dev_b, dev_c, m, n, p);

    cudaMemcpy(c->data.data(), dev_c, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_a);
    cudaFree(dev_c);
}

__global__
void cudaCalcAGrad(float *a_grad, const float *b, const float *c_grad, const uint m, const uint n, const uint p) {
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

__global__
void cudaCalcBGrad(float *b_grad, const float *a, const float *c_grad, const uint m, const uint n, const uint p) {
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

void Matmul_backward(Variable *a, Variable *b, Variable *c, int m, int n, int p) {
    float *dev_a, *dev_b, *dev_a_g, *dev_b_g, *dev_c_g;
    cudaMalloc((void**) &dev_a, m * n * sizeof(float));
    cudaMemcpy(dev_a, a->data.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &dev_b, n * p * sizeof(float));
    cudaMemcpy(dev_b, b->data.data(), n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &dev_a_g, m * n * sizeof(float));
    cudaMalloc((void**) &dev_b_g, n * p * sizeof(float));
    cudaMalloc((void**) &dev_c_g, m * p * sizeof(float));
    cudaMemcpy(dev_c_g, c->grad.data(), m * p * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_a((n-1)/TILE_SIZE+1, (m-1)/TILE_SIZE+1, 1);
    dim3 block_b((p-1)/TILE_SIZE+1, (n-1)/TILE_SIZE+1, 1);
    dim3 thread_in_block(TILE_SIZE, TILE_SIZE, 1);
    cudaCalcAGrad<<<block_a, thread_in_block>>>(dev_a_g, dev_b, dev_c_g, m, n, p);
    cudaCalcBGrad<<<block_b, thread_in_block>>>(dev_b_g, dev_a, dev_c_g, m, n, p);

    cudaMemcpy(a->grad.data(), dev_a_g, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b->grad.data(), dev_b_g, n * p * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_a_g);
    cudaFree(dev_b_g);
    cudaFree(dev_c_g);
}

__global__
void Dropout_forward_kernel(float *in, bool *mask, curandState *state, const uint size, const float p,const float scale, const bool useMask) {
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

void Dropout_forward(Variable *in, bool *mask, float p) {
    int size = in->data.size();
    float *dev_in;
    cudaMalloc((void**) &dev_in, size * sizeof(float));
    cudaMemcpy(dev_in, in->data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    bool *dev_mask;
    if (mask) cudaMalloc((void**) &dev_mask, size * sizeof(bool));

    float scale = 1 / (1 - p);
    dim3 block((size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    Dropout_forward_kernel<<<block, thread_in_block>>>(dev_in, dev_mask, devStates, size, p, scale, (mask != nullptr));

    cudaMemcpy(in->data.data(), dev_in, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (mask) cudaMemcpy(mask, dev_mask, size * sizeof(bool), cudaMemcpyDeviceToHost);
}

__global__
void Dropout_backward_kernel(float *in_grad, const bool *mask, const uint size, const float scale) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
        in_grad[id] *= mask[id] ? scale : 0;
}

void Dropout_backward(Variable *in, bool *mask, float p) {
    uint size = in->data.size();
    float *dev_in_g;
    bool *dev_mask;

    cudaMalloc((void**) &dev_in_g, size * sizeof(float));
    cudaMemcpy(dev_in_g, in->grad.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &dev_mask, size * sizeof(bool));
    cudaMemcpy(dev_mask, mask, size * sizeof(bool), cudaMemcpyHostToDevice);

    float scale = 1 / (1 - p);
    dim3 block((size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    Dropout_backward_kernel<<<block, thread_in_block>>>(dev_in_g, dev_mask, size, scale);

    cudaMemcpy(in->grad.data(), dev_in_g, size * sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void setupRandKernel(curandState *state) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, id, 0, &state[id]);
}

void cudaCallInitRandomState(const uint size) {
    cudaMalloc((void**) &devStates, size * sizeof(curandState));
    dim3 block((size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    setupRandKernel<<<block,thread_in_block>>>(devStates);
}

void cudaCallFreeRandomState() {
    cudaFree(devStates);
}
