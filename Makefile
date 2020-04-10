SRC_DIR = src
COMMON_DIR = $(SRC_DIR)/common
SEQ_DIR = $(SRC_DIR)/seq
CUDA_DIR = $(SRC_DIR)/cuda

COMMONFLAGS = -O3 -std=c++11 -I$(SRC_DIR) -I$(COMMON_DIR) -I$(SEQ_DIR) -I$(CUDA_DIR)
CXXFLAGS = -Wall -Wno-sign-compare -Wno-unused-variable -Wno-unknown-pragmas

COMMONOBJS = $(COMMON_DIR)/parser.o \
						 $(COMMON_DIR)/timer.o
SEQOBJS = $(SEQ_DIR)/gcn.o \
					$(SEQ_DIR)/module.o \
					$(SEQ_DIR)/optim.o \
					$(SEQ_DIR)/rand.o \
					$(SEQ_DIR)/sparse.o \
					$(SEQ_DIR)/variable.o
CUDAOBJS = $(CUDA_DIR)/cuda_gcn.o \
					 $(CUDA_DIR)/cuda_kernel.o \
					 $(CUDA_DIR)/cuda_module.o \
					 $(CUDA_DIR)/cuda_variable.o

EXECUTABLES = seq_gcn cuda_gcn

all: clean seq cuda

seq: $(SRC_DIR)/seqmain.o $(COMMONOBJS) $(SEQOBJS)
	$(CXX) $(COMMONFLAGS) $(CXXFLAGS) $^ -o $@_gcn

cuda: $(SRC_DIR)/cudamain.o $(COMMONOBJS) $(SEQOBJS) $(CUDAOBJS)
	nvcc $(COMMONFLAGS) $^ -o $@_gcn

$(SRC_DIR)/seqmain.o: $(SRC_DIR)/main.cpp
	$(CXX) -c $(COMMONFLAGS) $(CXXFLAGS) $< -o $@

$(SRC_DIR)/cudamain.o: $(SRC_DIR)/main.cpp
	nvcc -dc $(COMMONFLAGS) $< -o $@

$(COMMON_DIR)/%.o: $(COMMON_DIR)/%.cpp
	$(CXX) -c $(COMMONFLAGS) $(CXXFLAGS) $< -o $@

$(SEQ_DIR)/%.o: $(SEQ_DIR)/%.cpp
	$(CXX) -c $(COMMONFLAGS) $(CXXFLAGS) $< -o $@

$(CUDA_DIR)/%.o: $(CUDA_DIR)/%.cu
	nvcc -dc $(COMMONFLAGS) $< -o $@

clean:
	$(RM) $(EXECUTABLES) $(SRC_DIR)/*.o $(COMMONOBJS) $(SEQOBJS) $(CUDAOBJS)
