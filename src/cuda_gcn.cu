#include "cuda_gcn.cuh"
#include "timer.h"
#include <algorithm>
#include <thrust/transform.h>

using std::max;
using std::max_element;

CUDAGCN::CUDAGCN(GCNParams params, GCNData *input_data) {

    cuda_init_random_state(MAX_THREAD_PER_BLOCK);

    this->params = params;
    data = input_data;
    sp = new CUDASparseIndex(data->feature_index);
    graph = new CUDASparseIndex(data->graph);
    modules.reserve(8);
    variables.reserve(8);

    // dropout
    variables.emplace_back(data->feature_index.indices.size(), false);
    input = &variables.back();
    modules.push_back(new CUDADropout(input, params.dropout));
    
    // sparse matmul
    variables.emplace_back(params.num_nodes * params.hidden_dim);
    CUDAVariable *layer1_var1 = &variables.back();
    variables.emplace_back(params.input_dim * params.hidden_dim, true);
    CUDAVariable *layer1_weight = &variables.back();
    layer1_weight->glorot(params.input_dim, params.hidden_dim);
    modules.push_back(new CUDASparseMatmul(input, layer1_weight, layer1_var1, sp, params.num_nodes, params.input_dim, params.hidden_dim));
    
    // graph sum
    variables.emplace_back(params.num_nodes * params.hidden_dim);
    CUDAVariable *layer1_var2 = &variables.back();
    modules.push_back(new CUDAGraphSum(layer1_var1, layer1_var2, graph, params.hidden_dim));

    // ReLU
    modules.push_back(new CUDAReLU(layer1_var2));

    // dropout
    modules.push_back(new CUDADropout(layer1_var2, params.dropout));

    // dense matmul
    variables.emplace_back(params.num_nodes * params.output_dim);
    CUDAVariable *layer2_var1 = &variables.back();
    variables.emplace_back(params.hidden_dim * params.output_dim, true);
    CUDAVariable *layer2_weight = &variables.back();
    layer2_weight->glorot(params.hidden_dim, params.output_dim);
    modules.push_back(new CUDAMatmul(layer1_var2, layer2_weight, layer2_var1, params.num_nodes, params.hidden_dim, params.output_dim));

    // graph sum
    variables.emplace_back(params.num_nodes * params.output_dim);
    output = &variables.back();
    modules.push_back(new CUDAGraphSum(layer2_var1, output, graph, params.output_dim));

    // cross entropy loss
    CUDA_CHECK(cudaMalloc((void**) &truth, params.num_nodes * sizeof(int)));
    modules.push_back(new CUDACrossEntropyLoss(output, truth, &loss, params.output_dim));

    // optimizer
    AdamParams adam_params = AdamParams::get_default();
    adam_params.lr = params.learning_rate;
    adam_params.weight_decay = params.weight_decay;
    optimizer = new CUDAAdam({{layer1_weight, true}, {layer2_weight, false}}, adam_params);

    // other variable
    CUDA_CHECK(cudaMalloc((void**) &d_l2_penalty, variables[2].size * sizeof(float)));
}

CUDAGCN::~CUDAGCN() {
    cuda_free_random_state();
    for (auto &m : modules) delete m;
    delete sp;
    delete graph;
    delete optimizer;
    CUDA_CHECK(cudaFree(truth));
    CUDA_CHECK(cudaFree(d_l2_penalty));
}

void CUDAGCN::set_input() {
    CUDA_CHECK(cudaMemcpy(input->data, data->feature_value.data(), input->size * sizeof(float), cudaMemcpyHostToDevice));
}

void CUDAGCN::set_truth(int current_split) {
    int *d_data_split, *d_data_label;
    CUDA_CHECK(cudaMalloc((void**) &d_data_split, params.num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &d_data_label, params.num_nodes * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data_split, data->split.data(), params.num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data_label, data->label.data(), params.num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    dim3 block((params.num_nodes-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    cuda_set_truth_kernel<<<block, thread_in_block>>>(truth, d_data_split, d_data_label, current_split, params.num_nodes);
    CUDA_CHECK(cudaFree(d_data_split));
    CUDA_CHECK(cudaFree(d_data_label));
}

// TODO: reduction (using thrust?)
float CUDAGCN::get_accuracy() {
    int *cpu_truth = new int[params.num_nodes];
    float *cpu_output = new float[output->size];
    CUDA_CHECK(cudaMemcpy(cpu_truth, truth, params.num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu_output, output->data, output->size * sizeof(float), cudaMemcpyDeviceToHost));

    int wrong = 0, total = 0;
    for(int i = 0; i < params.num_nodes; i++) {
        if(cpu_truth[i] < 0) continue;
        total++;
        float truth_logit = cpu_output[i * params.output_dim + cpu_truth[i]];
        for(int j = 0; j < params.output_dim; j++)
            if (cpu_output[i * params.output_dim + j] > truth_logit) {
                wrong++;
                break;
            }
    }
    delete[] cpu_truth;
    delete[] cpu_output;
    return float(total - wrong) / total;
}

struct square_functor{
    square_functor() {}
    __host__ __device__ float operator()(const float &x) const {
        return x * x;
    }
};
float CUDAGCN::get_l2_penalty() {
    int size = variables[2].size;
    thrust::device_ptr<float> l2_ptr(d_l2_penalty), var2_ptr(variables[2].data);
    thrust::transform(var2_ptr, var2_ptr + size, l2_ptr, square_functor());
    float l2 = thrust::reduce(l2_ptr, l2_ptr + size, (float)0.0, thrust::plus<float>());
    return params.weight_decay * l2 / 2;
}

pair<float, float> CUDAGCN::train_epoch() {
    set_input();
    set_truth(1);
    for (auto m: modules)
        m->forward(true);
    float train_loss = loss + get_l2_penalty();
    float train_acc = get_accuracy();
    for (int i = modules.size() - 1; i >= 0; i--)
        modules[i]->backward();
    optimizer->step();
    return {train_loss, train_acc};
}

pair<float, float> CUDAGCN::eval(int current_split) {
    set_input();
    set_truth(current_split);
    for (auto m: modules)
        m->forward(false);
    float test_loss = loss + get_l2_penalty();
    float test_acc = get_accuracy();
    return {test_loss, test_acc};
}

void CUDAGCN::run() {
    int epoch = 1;

    std::vector<float> loss_history;
    for(; epoch <= params.epochs; epoch++) {
        float train_loss, train_acc, val_loss, val_acc;
        timer_start(TMR_TRAIN);
        std::tie(train_loss, train_acc) = train_epoch();
        std::tie(val_loss, val_acc) = eval(2);
        printf("epoch=%d train_loss=%.5f train_acc=%.5f val_loss=%.5f val_acc=%.5f time=%.5f\n",
            epoch, train_loss, train_acc, val_loss, val_acc, timer_stop(TMR_TRAIN));
        loss_history.push_back(val_loss);
        if(params.early_stopping > 0 && epoch >= params.early_stopping) {
            float recent_loss = 0.0;
            for(int i = epoch - params.early_stopping; i < epoch; i++)
                recent_loss += loss_history[i];
            if (val_loss > recent_loss / params.early_stopping) {
                printf("Early stopping...\n");
                break;
            }
        }
    }
    printf("total training time=%.5f\n", timer_total(TMR_TRAIN));

    float test_loss, test_acc;
    timer_start(TMR_TEST);
    std::tie(test_loss, test_acc) = eval(3);
    printf("test_loss=%.5f test_acc=%.5f time=%.5f\n", test_loss, test_acc, timer_stop(TMR_TEST));
}
