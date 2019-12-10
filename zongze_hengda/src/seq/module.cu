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

void SparseMatmul::forward(bool training) {
    timer_start(TMR_SPMATMUL_FW);
    c->zero();

    for (int i = 0; i < sp->indptr.size() - 1; i++) {
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
            int j = sp->indices[jj];
            for (int k = 0; k < p; k++)
                c->data[i * p + k] += a->data[jj] * b->data[j * p + k];
        }
    }
    timer_stop(TMR_SPMATMUL_FW);
}

void SparseMatmul::backward() {
    timer_start(TMR_SPMATMUL_BW);
    b->zero_grad();
    int row = 0;

    for (int i = 0; i < sp->indptr.size() - 1; i++) {
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
            int j = sp->indices[jj];
            for (int k = 0; k < p; k++)
                b->grad[j * p + k] += c->grad[i * p + k] * a->data[jj];
        }
    }
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

// kernel function of CrossEntropy
__global__ void crossEntropyLossForward(float* logits_data, float* logits_grad, bool training, int num_classes, int* truth, int* count, float* thread_loss, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size)	return;
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

void CrossEntropyLoss::forward(bool training) {
    float total_loss = 0;
    int count = 0;
    if (training) logits->zero_grad();


    /** *
      * below are GPU accelevated version
      *
      */

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
    crossEntropyLossForward <<< grid, block >>>(d_logits_data, d_logits_grad, training, num_classes, d_truth, d_count, d_loss, logits->data.size());
    
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

    /** *
      * below are CPU version of CrossEntropy
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
    */
    
    *loss = total_loss / count;
    if (training) {
        for (int i = 0; i < logits->grad.size(); i++)
            logits->grad[i] /= count;
    }
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
