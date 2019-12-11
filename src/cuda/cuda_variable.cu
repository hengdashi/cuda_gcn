#include "cuda_variable.cuh"

CUDAVariable::CUDAVariable(int size, bool requires_grad) {
    this->requires_grad = requires_grad;
    CUDA_CHECK(cudaMalloc((void**) &data, size * sizeof(float)));
    if (requires_grad) {
        CUDA_CHECK(cudaMalloc((void**) &grad, size * sizeof(float)));
    }
}

CUDAVariable::~CUDAVariable() {
    CUDA_CHECK(cudaFree(data));
    if (requires_grad) CUDA_CHECK(cudaFree(grad));
}
