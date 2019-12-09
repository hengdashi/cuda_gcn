#include "module.h"
#include "rand.h"
#include "timer.h"
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include "cudaacc.cuh"

Matmul::Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p) :
        a(a), b(b), c(c), m(m), n(n), p(p) {}

void Matmul::forward(bool training) {
    timer_start(TMR_MATMUL_FW);
    c->zero();

    // for (int i = 0; i < m; i++)
    //     for (int j = 0; j < n; j++)
    //         for (int k = 0; k < p; k++)
    //             c->data[i * p + k] += a->data[i * n + j] * b->data[j * p + k];
    
    float *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**) &dev_a, m * n * sizeof(float));
    cudaMemcpy(dev_a, a->data.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &dev_b, n * p * sizeof(float));
    cudaMemcpy(dev_b, b->data.data(), n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &dev_c, m * p * sizeof(float));
    
    cudaCallMatMulForward(dev_a, dev_b, dev_c, m, n, p);

    cudaMemcpy(c->data.data(), dev_c, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_a);
    cudaFree(dev_c);


    timer_stop(TMR_MATMUL_FW);
}

void Matmul::backward() {
    timer_start(TMR_MATMUL_BW);
    a->zero_grad();
    b->zero_grad();

    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         float tmp = 0;
    //         for (int k = 0; k < p; k++) {
    //             tmp += c->grad[i * p + k] * b->data[j * p + k];
    //             b->grad[j * p + k] += c->grad[i * p + k] * a->data[i * n + j];
    //         }
	// 	    a->grad[i * n + j] = tmp;
    //     }
    // }

    float *dev_a, *dev_b, *dev_a_g, *dev_b_g, *dev_c_g;
    cudaMalloc((void**) &dev_a, m * n * sizeof(float));
    cudaMemcpy(dev_a, a->data.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &dev_b, n * p * sizeof(float));
    cudaMemcpy(dev_b, b->data.data(), n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &dev_a_g, m * n * sizeof(float));
    cudaMalloc((void**) &dev_b_g, n * p * sizeof(float));
    cudaMalloc((void**) &dev_c_g, m * p * sizeof(float));
    cudaMemcpy(dev_c_g, c->grad.data(), m * p * sizeof(float), cudaMemcpyHostToDevice);

    cudaCallMatMulBackward(dev_a, dev_b, dev_a_g, dev_b_g, dev_c_g, m, n, p);

    cudaMemcpy(a->grad.data(), dev_a_g, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b->grad.data(), dev_b_g, n * p * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_a_g);
    cudaFree(dev_b_g);
    cudaFree(dev_c_g);


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
    if (!in->grad.empty()) mask = new bool[in->data.size()];
    else mask = nullptr;
}

Dropout::~Dropout() {
    if (mask) delete[] mask;
}

void Dropout::forward(bool training) {
    if (!training) return;
    timer_start(TMR_DROPOUT_FW);

    // const int threshold = int(p * MY_RAND_MAX);
    // float scale = 1 / (1 - p);
    // for (int i = 0; i < in->data.size(); i++) {
    //     bool keep = (int)RAND() >= threshold;
    //     in->data[i] *= keep ? scale : 0;
    //     if (mask) mask[i] = keep;
    // }

    int size = in->data.size();
    float *dev_in;
    cudaMalloc((void**) &dev_in, size * sizeof(float));
    cudaMemcpy(dev_in, in->data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    bool *dev_mask;
    if (mask) cudaMalloc((void**) &dev_mask, size * sizeof(bool));

    cudaCallDropoutForward(dev_in, dev_mask, size, p, (mask != nullptr));

    cudaMemcpy(in->data.data(), dev_in, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (mask) cudaMemcpy(mask, dev_mask, size * sizeof(bool), cudaMemcpyDeviceToHost);

    timer_stop(TMR_DROPOUT_FW);
}

void Dropout::backward() {
    if (!mask) return;
    timer_start(TMR_DROPOUT_BW);
    
    // float scale = 1 / (1 - p);
    // for (int i = 0; i < in->data.size(); i++)
    //     in->grad[i] *= mask[i] ? scale : 0;

    uint size = in->data.size();
    float *dev_in_g;
    bool *dev_mask;

    cudaMalloc((void**) &dev_in_g, size * sizeof(float));
    cudaMemcpy(dev_in_g, in->grad.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &dev_mask, size * sizeof(bool));
    cudaMemcpy(dev_mask, mask, size * sizeof(bool), cudaMemcpyHostToDevice);

    cudaCallDropoutBackward(dev_in_g, dev_mask, size, p);

    cudaMemcpy(in->grad.data(), dev_in_g, size * sizeof(float), cudaMemcpyDeviceToHost);

    timer_stop(TMR_DROPOUT_BW);
}
