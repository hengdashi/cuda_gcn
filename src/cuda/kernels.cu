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
void CrossEntropy_forward_kernel(float* logits_data, float* logits_grad, bool training, int num_classes, int* truth, int* count, float* thread_loss, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size)  return;
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

void CrossEntropy_forward(Variable *logits, int *truth, float &total_loss, int &count, int num_classes, bool training) {
    // host function variables
    float *logits_data = logits->data.data();
    float *logits_grad = logits->grad.data();

    // grid + block size
    int grid = 256;
    int block = (logits->data.size()/num_classes+256) / 256;

    // data structures in GPU:
    float* d_logits_data, *d_loss, *d_logits_grad;
    int* d_truth, *d_count;
    int logits_data_size = (int)(logits->data.size())*sizeof(float);
    int logits_grad_size = (int)(logits->grad.size())*sizeof(float);
    int loss_size = (int)(logits->data.size()/num_classes)*sizeof(float);
    int truth_size = (int)(logits->data.size()/num_classes)*sizeof(int);

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
    CrossEntropy_forward_kernel <<< grid, block >>>(d_logits_data, d_logits_grad, training, num_classes, d_truth, d_count, d_loss, logits->data.size());
    cudaDeviceSynchronize();

    // updates logits->data and logits->grad in host function
    cudaMemcpy(&(logits->data[0]), d_logits_data, logits_data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&(logits->grad[0]), d_logits_grad, logits_grad_size, cudaMemcpyDeviceToHost);

    // accumulate and add count and total_loss variables by thrust::
    thrust::device_ptr<int> count_ptr = thrust::device_pointer_cast(d_count);
    count = thrust::reduce(count_ptr, count_ptr+(logits->data.size()/num_classes), (int)0, thrust::plus<int>());
    thrust::device_ptr<float> loss_ptr = thrust::device_pointer_cast(d_loss);
    total_loss = thrust::reduce(loss_ptr, loss_ptr+(logits->data.size()/num_classes), (float)0.0, thrust::plus<float>());

    // free memory
    cudaFree(d_logits_data);
    cudaFree(d_loss);
    cudaFree(d_logits_grad);
    cudaFree(d_truth);
    cudaFree(d_count);
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
