#include "module.h"
#include "rand.h"
#include "timer.h"
#include <cstdlib>
#include <cmath>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <iostream>

Matmul::Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p) :
        a(a), b(b), c(c), m(m), n(n), p(p) {}

void Matmul::forward(bool training) {
    timer_start(TMR_MATMUL_FW);
    c->zero();

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < p; k++)
                c->data[i * p + k] += a->data[i * n + j] * b->data[j * p + k];
    timer_stop(TMR_MATMUL_FW);
}

void Matmul::backward() {
    timer_start(TMR_MATMUL_BW);
    a->zero_grad();
    b->zero_grad();

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float tmp = 0;
            for (int k = 0; k < p; k++) {
                tmp += c->grad[i * p + k] * b->data[j * p + k];
                b->grad[j * p + k] += c->grad[i * p + k] * a->data[i * n + j];
            }
		    a->grad[i * n + j] = tmp;
        }
    }
    timer_stop(TMR_MATMUL_BW);
}

SparseMatmul::SparseMatmul(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int m, int n, int p) :
        a(a), b(b), c(c), sp(sp), m(m), n(n), p(p) {}

__global__ void sparse_matmul_kernel(float* a_in, float* b_in, float* c_in, float* indptr, float* indices, int p){
    int blockx = blockIdx.x;
    int threadx = blockIdx.y * 1024 + threadIdx.x;
    
    for (int jj = indptr[blockx]; jj < indptr[blockx + 1]; jj++){
        int j = indices[jj];
        c_in->data[blockx * p + threadx] += a_in->data[jj] * b_in->data[j * p + threadx];
    }
}

void SparseMatmul::forward(bool training) {
    timer_start(TMR_SPMATMUL_FW);
    c->zero();

    float* a_in;
    float* b_in;
    float* c_in;
    int* d_indptr;
    int* d_indices;
    cudaMalloc((void**) &a_in, a->data.size() * sizeof(float));
    cudaMalloc((void**) &b_in, b->data.size() * sizeof(float));
    cudaMalloc((void**) &c_in, c->data.size() * sizeof(float));
    cudaMalloc(&d_indptr, sp->indptr.size() * sizeof(int));
    cudaMalloc(&d_indices, sp->indices.size() * sizeof(int));
    cudaMemcpy(a_in, a->data, a->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_in, b->data, b->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_in, c->data, c->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indptr, sp->indptr.data(), sp->indptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, sp->indices.data(), sp->indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    if(sp->indptr.size() <= 1) return;

    dim3 gridsize(sp->indptr.size() - 1, 1);
    dim3 blocksize(p);

    if(p > 1024){
        blocksize.x = 1024;
        gridsize.y = ceil((double)p / (double) blocksize.x);
    }

    sparse_matmul_kernel<<<gridsize, blocksize>>>(a_in, b_in, c_in, d_indptr, d_indices, p);

    cudaMemcpy(c, c_in, c->data.size() * sizeof(float), cudaMemcpyDeviceToHost);

    /*
    for (int i = 0; i < sp->indptr.size() - 1; i++) {
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
            int j = sp->indices[jj];
            for (int k = 0; k < p; k++)
                c->data[i * p + k] += a->data[jj] * b->data[j * p + k];
        }
    }*/

    cudaFree(a_in);
    cudaFree(b_in);
    cudaFree(c_in);
    cudaFree(d_indptr);
    cudaFree(d_indices);

    timer_stop(TMR_SPMATMUL_FW);
}

__global__ void sparse_matmul_back_kernel(float* a_in, float* b_in, float* c_in, float* indptr, float* indices, int p){
    int blockx = blockIdx.x;
    int threadx = blockIdx.y * 1024 + threadIdx.x;
    
    for (int jj = indptr[blockx]; jj < indptr[blockx + 1]; jj++){
        int j = indices[jj];
        b_in->grad[j * p + threadx] += c_in->grad[blockx * p + threadx] * a_in->grad[jj];
    }
}

void SparseMatmul::backward() {
    timer_start(TMR_SPMATMUL_BW);
    b->zero_grad();
    int row = 0;

    float* a_in;
    float* b_in;
    float* c_in;
    int* d_indptr;
    int* d_indices;
    cudaMalloc((void**) &a_in, a->data.size() * sizeof(float));
    cudaMalloc((void**) &b_in, b->data.size() * sizeof(float));
    cudaMalloc((void**) &c_in, c->data.size() * sizeof(float));
    cudaMalloc(&d_indptr, sp->indptr.size() * sizeof(int));
    cudaMalloc(&d_indices, sp->indices.size() * sizeof(int));
    cudaMemcpy(a_in, a->data, a->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_in, b->data, b->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_in, c->data, c->data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indptr, sp->indptr.data(), sp->indptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, sp->indices.data(), sp->indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    if(sp->indptr.size() <= 1) return;

    dim3 gridsize(sp->indptr.size() - 1, 1);
    dim3 blocksize(p);

    if(p > 1024){
        blocksize.x = 1024;
        gridsize.y = ceil((double)p / (double) blocksize.x);
    }

    sparse_matmul_back_kernel<<<gridsize, blocksize>>>(a_in, b_in, c_in, d_indptr, d_indices, p);

    cudaMemcpy(c, c_in, c->data.size() * sizeof(float), cudaMemcpyDeviceToHost);

    /*
    for (int i = 0; i < sp->indptr.size() - 1; i++) {
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
            int j = sp->indices[jj];
            for (int k = 0; k < p; k++)
                b->grad[j * p + k] += c->grad[i * p + k] * a->data[jj];
        }
    }*/

    cudaFree(a_in);
    cudaFree(b_in);
    cudaFree(c_in);
    cudaFree(d_indptr);
    cudaFree(d_indices);

    timer_stop(TMR_SPMATMUL_BW);
}

GraphSum::GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim) :
        in(in), out(out), graph(graph), dim(dim) {}

void GraphSum::forward(bool training) {
    timer_start(TMR_GRAPHSUM_FW);
    out->zero();

    for (int src = 0; src < graph->indptr.size() - 1; src++) {
        for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
            int dst = graph->indices[i];
            float coef = 1.0 / sqrtf(
                    (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
            );
            for (int j = 0; j < dim; j++)
                // This only works for undirected graphs. Should be out[dst] += coef * in[src]
                out->data[src * dim + j] += coef * in->data[dst * dim + j];
        }
    }
    timer_stop(TMR_GRAPHSUM_FW);
}

void GraphSum::backward() {
    timer_start(TMR_GRAPHSUM_BW);
    in->zero_grad();

    for (int src = 0; src < graph->indptr.size() - 1; src++) {
        for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
            int dst = graph->indices[i];
            float coef = 1.0 / sqrtf(
                    (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
            );
            for (int j = 0; j < dim; j++)
                in->grad[src * dim + j] += coef * out->grad[dst * dim + j];
        }
    }
    timer_stop(TMR_GRAPHSUM_BW);
}

CrossEntropyLoss::CrossEntropyLoss(Variable *logits, int *truth, float *loss, int num_classes) :
        logits(logits), truth(truth), loss(loss), num_classes(num_classes) {}

void CrossEntropyLoss::forward(bool training) {
    timer_start(TMR_LOSS_FW);
    float total_loss = 0;
    int count = 0;
    if (training) logits->zero_grad();

    for (int i = 0; i < logits->data.size() / num_classes; i++) {
        if (truth[i] < 0) continue;
        count++;
        float *logit = &logits->data[i * num_classes];
        float max_logit = -1e30, sum_exp = 0;

        for (int j = 0; j < num_classes; j++)
            max_logit = fmax(max_logit, logit[j]);
        for (int j = 0; j < num_classes; j++) {
            logit[j] -= max_logit;
            sum_exp += expf(logit[j]);
        }
        total_loss += logf(sum_exp) - logit[truth[i]];

        if (training) {
            for (int j = 0; j < num_classes; j++) {
                float prob = expf(logit[j]) / sum_exp;
                logits->grad[i * num_classes + j] = prob;
            }
            logits->grad[i * num_classes + truth[i]] -= 1.0;
        }
    }

    *loss = total_loss / count;
    if (training) {
        for (int i = 0; i < logits->grad.size(); i++)
            logits->grad[i] /= count;
    }
    timer_stop(TMR_LOSS_FW);
}

void CrossEntropyLoss::backward() {
}

ReLU::ReLU(Variable *in) {
    this->in = in;
    mask = new bool[in->data.size()];
}

ReLU::~ReLU() {
    delete[] mask;
}

void ReLU::forward(bool training) {
    timer_start(TMR_RELU_FW);

    for (int i = 0; i < in->data.size(); i++) {
        bool keep = in->data[i] > 0;
        if (training) mask[i] = keep;
        if (!keep) in->data[i] = 0;
    }
    timer_stop(TMR_RELU_FW);
}

void ReLU::backward() {
    timer_start(TMR_RELU_BW);

    for (int i = 0; i < in->data.size(); i++)
        if (!mask[i]) in->grad[i] = 0;
    timer_stop(TMR_RELU_BW);
}

Dropout::Dropout(Variable *in, float p) {
    this->in = in;
    this->p = p;
    if (!in->grad.empty()) mask = new int[in->data.size()];
    else mask = nullptr;
}

Dropout::~Dropout() {
    if (mask) delete[] mask;
}

void Dropout::forward(bool training) {
    if (!training) return;
    timer_start(TMR_DROPOUT_FW);
    const int threshold = int(p * MY_RAND_MAX);
    float scale = 1 / (1 - p);

    for (int i = 0; i < in->data.size(); i++) {
        bool keep = (int)RAND() >= threshold;
        in->data[i] *= keep ? scale : 0;
        if (mask) mask[i] = keep;
    }
    timer_stop(TMR_DROPOUT_FW);
}

void Dropout::backward() {
    if (!mask) return;
    timer_start(TMR_DROPOUT_BW);
    float scale = 1 / (1 - p);

    for (int i = 0; i < in->data.size(); i++)
        in->grad[i] *= mask[i] ? scale : 0;
    timer_stop(TMR_DROPOUT_BW);
}