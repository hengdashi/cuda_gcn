#include "kernels.cuh"

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
