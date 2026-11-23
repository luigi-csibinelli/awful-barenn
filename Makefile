.RECIPEPREFIX = >
CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS ?= -std=c++17 -O3 -g -fopenmp
# --fmad=false: sem fused multiply-add, cada conta arredonda como na CPU e a
# saída da GPU fica idêntica bit a bit à da CPU.
NVFLAGS  ?= -std=c++17 -O3 --fmad=false

all: barenn

barenn: barenn.cpp barenn.hpp
> $(CXX) $(CXXFLAGS) -o $@ barenn.cpp

barenn_cuda: barenn.cu barenn.hpp
> $(NVCC) $(NVFLAGS) -o $@ barenn.cu

clean:
> rm -f barenn barenn_cuda

.PHONY: all clean
