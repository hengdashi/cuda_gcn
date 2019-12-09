#include "optim.h"
#include <cmath>
#include <cstdlib>

AdamParams AdamParams::get_default() {
    return {0.001, 0.9, 0.999, 1e-8, 0.0};
}

AdamVariable::AdamVariable(Variable *var, bool decay):
    data(&var->data), grad(&var->grad), m(var->data.size(), 0.0), v(var->data.size(), 0.0), decay(decay) {}

int AdamVariable::size() {
    return data->size();
}

Adam::Adam(std::vector<std::pair<Variable*, bool>> vars, AdamParams params){
    step_count = 0;
    this->params = params;
    for (auto v: vars)
        this->vars.emplace_back(v.first, v.second);
}

__global__ void step_kernel(float* grad, float* data, float* m, float* v, bool decay, float weight_decay, float beta1, float beta2, float eps){
    int idx = blockIdx.x * 1024 + threadIdx.x;

    float g = grad[idx];
    if (decay) g += weight_decay * data[idx];
    m[idx] = beta1 * m[idx] + (1.0 - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0 - beta2) * g * g;
    data[idx] -= step_size * m[idx] / (sqrtf(v[idx]) + eps);
}

void Adam::step(){
    step_count++;
    float step_size = params.lr * sqrtf(1 - powf(params.beta2, step_count)) / (1 - powf(params.beta1, step_count));

    for (auto &var: vars) {
        float* d_grad;
        float* d_data;
        float* d_m;
        float* d_v;
        cudaMalloc((void**) &d_grad, (*var.grad).size() * sizeof(float));
        cudaMalloc((void**) &d_data, (*var.data).size() * sizeof(float));
        cudaMalloc((void**) &d_m, var.m.size() * sizeof(float));
        cudaMalloc((void**) &d_v, var.v.size() * sizeof(float));
        cudaMemcpy(d_grad, *var.grad, (*var.grad).size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_data, *var.data, (*var.data).size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m, var.m, var.m.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, var.v, var.v.size() * sizeof(float), cudaMemcpyHostToDevice);

        dim3 gridsize(1);
        dim3 blocksize(var.size());

        if(var.size() > 1024){
            blocksize.x = 1024;
            gridsize.x = ceil((double)var.size() / (double) blocksize.x);
        }

        step_kernel<<<gridsize, blocksize>>>(d_grad, d_data, d_m, d_v, var.decay, params.weight_decay, params.beta1, params.beta2, params.eps);

        cudaMemcpy(var.m, d_m, var.m.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(var.v, d_v, var.v.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(*var.data, d_data, var.data.size() * sizeof(float), cudaMemcpyDeviceToHost);

        /*
        for (int i = 0; i < var.size(); i++) {
            float grad = (*var.grad)[i];
            if (var.decay) grad += params.weight_decay * (*var.data)[i];
            var.m[i] = params.beta1 * var.m[i] + (1.0 - params.beta1) * grad;
            var.v[i] = params.beta2 * var.v[i] + (1.0 - params.beta2) * grad * grad;
            (*var.data)[i] -= step_size * var.m[i] / (sqrtf(var.v[i]) + params.eps);
        }*/

        cudaFree(d_grad);
        cudaFree(d_data);
        cudaFree(d_m);
        cudaFree(d_v);
    }
}




















