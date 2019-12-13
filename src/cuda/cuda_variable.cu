#include "cuda_variable.cuh"

CUDAVariable::CUDAVariable(int size, bool requires_grad) {
    this->requires_grad = requires_grad;
    this->size = size;
    CUDA_CHECK(cudaMalloc((void**) &data, size * sizeof(float)));
    if (requires_grad) {
        CUDA_CHECK(cudaMalloc((void**) &grad, size * sizeof(float)));
    }
}

CUDAVariable::~CUDAVariable() {
    CUDA_CHECK(cudaFree(data));
    if (requires_grad) CUDA_CHECK(cudaFree(grad));
}

__global__ void cuda_Variable_glorot_kernel(float *data, curandState *state, int size, float scale) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
        data[id] = (curand_uniform(&state[id % MAX_THREAD_PER_BLOCK]) - 0.5) * scale;
}

void CUDAVariable::glorot(int in_size, int out_size) {
    float range = sqrtf(6.0f / (in_size + out_size)), scale = range * 2;

    dim3 block((size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    cuda_Variable_glorot_kernel<<<block, thread_in_block>>>(data, devStates, size, scale);
}

void CUDAVariable::zero() {
    CUDA_CHECK(cudaMemset(data, 0, size * sizeof(float)));
}

void CUDAVariable::zero_grad() {
    CUDA_CHECK(cudaMemset(grad, 0, size * sizeof(float)));
}

void CUDAVariable::print(int col) {
    float cpu_data[size];
    CUDA_CHECK(cudaMemcpy(cpu_data, data, size * sizeof(float), cudaMemcpyDeviceToHost));
    int count = 0;
    printf("---------DATA----------\n");
    for (int i = 0; i < size; ++i) {
        printf("%.4f ", cpu_data[i]);
        count++;
        if (count % col == 0) printf("\n");
    }
    printf("\n");
}

float CUDAVariable::grad_norm() {
    float norm = 0;
    float *cpu_grad = new float[size];
    CUDA_CHECK(cudaMemcpy(cpu_grad, grad, size * sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i < size; ++i)
        norm += cpu_grad[i] * cpu_grad[i];
    delete[] cpu_grad;
    return sqrtf(norm);
}

CUDASparseIndex::CUDASparseIndex(const SparseIndex &sp) {
    indices_size = sp.indices.size();
    indptr_size = sp.indptr.size();

    CUDA_CHECK(cudaMalloc((void**) &indices, indices_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(indices, sp.indices.data(), indices_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**) &indptr, indptr_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(indptr, sp.indptr.data(), indptr_size * sizeof(int), cudaMemcpyHostToDevice));
}

CUDASparseIndex::~CUDASparseIndex() {
    if (indices != nullptr) cudaFree(indices);
    if (indptr != nullptr) cudaFree(indptr);
}