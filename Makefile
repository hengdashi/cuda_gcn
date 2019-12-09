CXX=gcc
CXXFLAGS= -O3 -std=c++11
GCCFLAGS= -Wall -Wno-sign-compare -Wno-unused-variable -Wno-unknown-pragmas
LDFLAGS=-lm -lstdc++

prefix=src/seq/
CXXFILES=$(prefix)*.cpp
HFILES=$(prefix)*.h

myprefix=src/hengda/
HENGDAFILES=$(myprefix)*.cpp $(myprefix)*.cu
HENGDAHFILES=$(myprefix)*.h $(myprefix)*.cuh

seq: $(prefix)main.cpp $(CXXFILES) $(HFILES)
	$(CXX) $(CXXFLAGS) -o gcn-seq $(CXXFILES) $(LDFLAGS)

hengda: $(myprefix)main.cpp $(HENGDAFILES) $(HENGDAHFILES)
	nvcc $(CXXFLAGS) -o gcn-hengda $(HENGDAFILES) $(LDFLAGS)

clean:
	$(RM) gcn-seq gcn-hengda
