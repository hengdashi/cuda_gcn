#ifndef CUDA_MODULE_H
#define CUDA_MODULE_H

#include "kernel.cuh"
#include "cuda_variable.cuh"

class CUDAModule {
public:
    virtual void forward(bool) = 0;
    virtual void backward() = 0;
    virtual ~Module() {};
};

class CUDAMatmul: public CUDAModule {
    CUDAVariable *a, *b, *c;
    int m, n, p;
public:
    Matmul(CUDAVariable *a, CUDAVariable *b, CUDAVariable *c, int m, int n, int p);
    ~Matmul() {}
    void forward(bool);
    void backward();
};

class CUDASparseMatmul: public CUDAModule {
    CUDAVariable *a, *b, *c;
    SparseIndex *sp;
    int m, n, p;
public:
    SparseMatmul(CUDAVariable *a, CUDAVariable *b, CUDAVariable *c, SparseIndex *sp, int m, int n, int p);
    ~SparseMatmul() {}
    void forward(bool);
    void backward();
};

class CUDAGraphSum: public CUDAModule {
    CUDAVariable *in, *out;
    SparseIndex *graph;
    int dim;
public:
    GraphSum(CUDAVariable *in, CUDAVariable *out, SparseIndex *graph, int dim);
    ~GraphSum() {}
    void forward(bool);
    void backward();
};

class CUDACrossEntropyLoss: public CUDAModule {
    CUDAVariable *logits;
    int *truth;
    float *loss;
    int num_classes;
public:
    CrossEntropyLoss(CUDAVariable *logits, int *truth, float *loss, int num_classes);
    ~CrossEntropyLoss() {}
    void forward(bool);
    void backward();
};

class CUDAReLU: public CUDAModule {
    CUDAVariable *in;
    bool *mask;
public:
    ReLU(CUDAVariable *in);
    ~ReLU();
    void forward(bool);
    void backward();
};

class CUDADropout: public CUDAModule {
    CUDAVariable *in;
    int *mask;
    float p;
public:
    Dropout(CUDAVariable *in, float p);
    ~Dropout();
    void forward(bool);
    void backward();
};

#endif