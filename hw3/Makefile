CXX = g++

# Add extra version of flags to work with Apple clang:
CXXFLAGS = -fopenmp -std=c++11 -O3 -march=native
# CXXFLAGS = -Xpreprocessor -fopenmp -lomp -std=c++11 -O3 -march=native

# turning off auto-vectorization since this can make hand-vectorized code slower
CXXFLAGS += -fno-tree-vectorize

TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
