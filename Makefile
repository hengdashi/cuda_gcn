CXX=gcc
CXXFLAGS= -O3 -std=c++11 -Wall -Wno-sign-compare -Wno-unused-variable -Wno-unknown-pragmas
LDFLAGS=-lm -lstdc++
CXXFILES=src/seq/gcn.cpp src/seq/optim.cpp src/seq/variable.cpp src/seq/parser.cpp src/seq/rand.cpp src/seq/timer.cpp
HFILES=src/seq/gcn.h src/seq/optim.h src/seq/module.h src/seq/variable.h src/seq/sparse.h src/seq/parser.h src/seq/rand.h src/seq/timer.h 


# seq: src/seq/main.cpp $(CXXFILES) $(HFILES)
# 	$(CXX) $(CXXFLAGS) -o gcn-seq $(CXXFILES) src/seq/main.cpp $(LDFLAGS)



all: clean program 

program: module.o
	$(CXX) $(CXXFLAGS) -o program src/seq/main.cpp module.o $(CXXFILES) -L/usr/local/cuda/lib64 -lcuda -lcudart $(LDFLAGS)

module.o:
	nvcc -c src/seq/module.cu 

.PHONY:clean
clean: 
	rm -f *.o program 
