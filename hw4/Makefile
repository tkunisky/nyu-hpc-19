CXX = g++

CXXFLAGS = -fopenmp -std=c++11 -O3 -march=native

CUDA_INCDIR = -I $(CUDA_HOME)/include -I $(CUDA_HOME)/samples/common/inc
CUDA_LIBS = -lblas -L${CUDA_HOME}/lib64 -lcudart -lcublas

NVCC = nvcc
NVCCFLAGS = -std=c++11
NVCCFLAGS += -Xcompiler "-fopenmp"

TARGETS = $(basename $(wildcard *.cu))

all : $(TARGETS)

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

%:%.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
