CXX=gcc
CXXFLAGS= -O3 -std=c++11
GCCFLAGS= -Wall -Wno-sign-compare -Wno-unused-variable -Wno-unknown-pragmas
LDFLAGS=-lm -lstdc++

prefix=src/
CXXFILES=$(prefix)*.cpp
HFILES=$(prefix)*.h

CUDAFILES=$(prefix)*.cu
CUDAHFILES=$(prefix)*.cuh

seq: $(prefix)main.cpp $(CXXFILES) $(HFILES)
	$(CXX) $(CXXFLAGS) -o gcn-seq $(CXXFILES) $(LDFLAGS)

cuda: $(prefix)main.cpp $(CXXFILES) $(CUDAFILES) $(HFILES) $(CUDAHFILES)
	nvcc $(CXXFLAGS) -o gcn-cuda $(CXXFILES) $(CUDAFILES) $(LDFLAGS)

clean:
	$(RM) gcn-seq gcn-cuda
