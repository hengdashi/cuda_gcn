CXX=gcc
CXXFLAGS= -O3 -std=c++11
GCCFLAGS= -Wall -Wno-sign-compare -Wno-unused-variable -Wno-unknown-pragmas
LDFLAGS=-lm -lstdc++
CXXFILES=src/seq/gcn.cpp src/seq/optim.cpp src/seq/module.cpp src/seq/variable.cpp src/seq/parser.cpp src/seq/rand.cpp src/seq/timer.cpp
HFILES=src/seq/gcn.h src/seq/optim.h src/seq/module.h src/seq/variable.h src/seq/sparse.h src/seq/parser.h src/seq/rand.h src/seq/timer.h 

HENGDAFILES=src/hengda/gcn.cpp src/hengda/optim.cpp src/hengda/module.cu src/hengda/variable.cpp src/hengda/parser.cpp src/hengda/rand.cpp src/hengda/timer.cpp
HENGDAHFILES=src/hengda/gcn.h src/hengda/optim.h src/hengda/module.h src/hengda/variable.h src/hengda/sparse.h src/hengda/parser.h src/hengda/rand.h src/hengda/timer.h

seq: src/seq/main.cpp $(CXXFILES) $(HFILES)
	$(CXX) $(CXXFLAGS) -o gcn-seq $(CXXFILES) src/seq/main.cpp $(LDFLAGS)

hengda: src/hengda/main.cpp $(HENGDAFILES) $(HENGDAHFILES)
	nvcc $(CXXFLAGS) -o gcn-hengda $(HENGDAFILES) src/hengda/main.cpp $(LDFLAGS)
