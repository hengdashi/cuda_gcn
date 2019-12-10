#include "variable.h"
#include "rand.h"
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <algorithm>

Variable::Variable(int size, bool requires_grad, bool thread_local_grad):
    data(size), grad(requires_grad ? size : 0) {}

void Variable::glorot(int in_size, int out_size) {
    float range = sqrtf(6.0f / (in_size + out_size));
    for(int i = 0; i < data.size(); i++)
        data[i] = (float(RAND()) / MY_RAND_MAX - 0.5) * range * 2;
}

void Variable::zero() {
    for(int i = 0; i < data.size(); i++)
        data[i] = 0;
}

void Variable::zero_grad() {
    for(int i = 0; i < grad.size(); i++)
        grad[i] = 0;
}

void Variable::print(int col) {
    int count = 0;
    for(float x: data) {
        printf("%.4f ", x);
        count++;
        if(count % col == 0) printf("\n");
    }
}

float Variable::grad_norm() {
    float norm = 0;
    for(float x: grad) norm += x * x;
    return sqrtf(norm);
}
