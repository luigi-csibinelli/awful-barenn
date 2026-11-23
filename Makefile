.RECIPEPREFIX = >
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -g -fopenmp
TARGET   := barenn

$(TARGET): barenn.cpp
> $(CXX) $(CXXFLAGS) -o $@ $<

clean:
> rm -f $(TARGET)

.PHONY: clean
