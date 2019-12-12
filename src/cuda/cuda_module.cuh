#ifndef CUDA_MODULE_CUH
#define CUDA_MODULE_CUH

#include "optim.h"
#include "kernel.cuh"
#include "cuda_variable.cuh"

using std::vector;
using std::pair;

class CUDAModule {
public:
    virtual void forward(bool) = 0;
    virtual void backward() = 0;
    virtual ~CUDAModule() {};
};

class CUDAMatmul: public CUDAModule {
    CUDAVariable *a, *b, *c;
    int m, n, p;
public:
    CUDAMatmul(CUDAVariable *a, CUDAVariable *b, CUDAVariable *c, int m, int n, int p);
    ~CUDAMatmul() {}
    void forward(bool);
    void backward();
};

class CUDASparseMatmul: public CUDAModule {
    CUDAVariable *a, *b, *c;
    CUDASparseIndex *sp;
    int m, n, p;
public:
    CUDASparseMatmul(CUDAVariable *a, CUDAVariable *b, CUDAVariable *c, CUDASparseIndex *sp, int m, int n, int p);
    ~CUDASparseMatmul() {}
    void forward(bool);
    void backward();
};

class CUDAGraphSum: public CUDAModule {
    CUDAVariable *in, *out;
    CUDASparseIndex *graph;
    int dim;
public:
    CUDAGraphSum(CUDAVariable *in, CUDAVariable *out, CUDASparseIndex *graph, int dim);
    ~CUDAGraphSum() {}
    void forward(bool);
    void backward();
};

class CUDACrossEntropyLoss: public CUDAModule {
    CUDAVariable *logits;
    int *truth;
    float *loss;
    int num_classes;
public:
    CUDACrossEntropyLoss(CUDAVariable *logits, int *truth, float *loss, int num_classes);
    ~CUDACrossEntropyLoss() {}
    void forward(bool);
    void backward();
};

class CUDAReLU: public CUDAModule {
    CUDAVariable *in;
    bool *mask;
public:
    CUDAReLU(CUDAVariable *in);
    ~CUDAReLU();
    void forward(bool);
    void backward();
};

class CUDADropout: public CUDAModule {
    CUDAVariable *in;
    int *mask;
    float p;
public:
    CUDADropout(CUDAVariable *in, float p);
    ~CUDADropout();
    void forward(bool);
    void backward();
};

class CUDAAdamVariable {
public:
    float *data, *grad, *m, *v;
    int size;
    bool decay;

    CUDAAdamVariable(CUDAVariable*, bool);
    ~CUDAAdamVariable();
};


class CUDAAdam {
    AdamParams params;
    int step_count;
    vector<CUDAAdamVariable> vars;
public:
    CUDAAdam() {}
    CUDAAdam(vector<pair<CUDAVariable*, bool>> vars, AdamParams params);
    void step();
};

#endif