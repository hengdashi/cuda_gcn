CXX=gcc
CXXFLAGS= -O3 -std=c++14 -Wall -Wno-sign-compare -Wno-unused-variable -Wno-unknown-pragmas
LDFLAGS=-lm -lstdc++
CXXFILES=src/seq/gcn.cpp src/seq/optim.cpp src/seq/module.cpp src/seq/variable.cpp src/seq/parser.cpp src/seq/rand.cpp src/seq/timer.cpp
HFILES=src/seq/gcn.h src/seq/optim.h src/seq/module.h src/seq/variable.h src/seq/sparse.h src/seq/parser.h src/seq/rand.h src/seq/timer.h 

CUDAFILES=src/cuda/gcn.cpp src/cuda/optim.cpp src/cuda/module.cpp src/cuda/variable.cpp src/cuda/parser.cpp src/cuda/rand.cpp src/cuda/timer.cpp src/cuda/cudaacc.cu
CUDAHFILES=src/cuda/gcn.h src/cuda/optim.h src/cuda/module.h src/cuda/variable.h src/cuda/sparse.h src/cuda/parser.h src/cuda/rand.h src/cuda/timer.h src/cuda/cudaacc.cuh
CUDAFLAGS=-lcurand

seq: src/seq/main.cpp $(CXXFILES) $(HFILES)
	$(CXX) $(CXXFLAGS) -o gcn-seq $(CXXFILES) src/seq/main.cpp $(LDFLAGS)

cuda: src/cuda/main.cpp $(CUDAFILES) $(HFILES)
	nvcc -O3 -std=c++14 $(CUDAFLAGS) -o gcn-cuda $(CUDAFILES) src/cuda/main.cpp $(LDFLAGS)