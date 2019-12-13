#include <vector>
#include "module.h"
#include "optim.h"
#include "variable.h"
#include "parser.h"
#include <iostream>
#include <string>

#include "cuda_gcn.cuh"

using namespace std;

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    if (argc < 2) {
        #ifdef __NVCC__
        string program = "gcn-cuda";
        #else
        string program = "gcn-seq";
        #endif

        cout << program << " graph_name [num_nodes input_dim hidden_dim "
        "output_dim dropout learning_rate, weight_decay epochs early_stopping]" << endl;
        return EXIT_FAILURE;
    }

    GCNParams params = GCNParams::get_default();
    GCNData data;
    std::string input_name(argv[1]);
    Parser parser(&params, &data, input_name);
    if (!parser.parse()) {
        std::cerr << "Cannot read input: " << input_name << std::endl;
        exit(EXIT_FAILURE);
    }

    #ifdef __NVCC__
    std::cout << "RUNNING ON GPU" << std::endl;
    #else
    std::cout << "RUNNING ON CPU" << std::endl;
    #endif

    // GCN gcn(params, &data);
    // gcn.run();
    CUDAGCN cuda_gcn(params, &data);
    cuda_gcn.run();
    
    return EXIT_SUCCESS;
}
