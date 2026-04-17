CXX      ?= g++
CXXSTD   = -std=c++17
WARN     = -Wall -Wextra
ARCH     = -march=native

SSEFLAGS = -msse2

DEBUG_FLAGS   = -O0 -g $(ARCH) $(SSEFLAGS) $(WARN) $(CXXSTD)
RELEASE_FLAGS = -O3 -DNDEBUG $(ARCH) $(SSEFLAGS) $(WARN) $(CXXSTD) -fno-tree-vectorize \
                -funroll-loops

SRC = main.cpp

all: lab5_debug lab5_release

lab5_debug: $(SRC)
	$(CXX) $(DEBUG_FLAGS) $(SRC) -o $@

lab5_release: $(SRC)
	$(CXX) $(RELEASE_FLAGS) $(SRC) -o $@

lab5_release_autovec: $(SRC)
	$(CXX) -O3 -DNDEBUG $(ARCH) $(SSEFLAGS) $(WARN) $(CXXSTD) -funroll-loops $(SRC) -o $@

run: all
	./lab5_debug
	./lab5_release

run-autovec: lab5_release_autovec
	./lab5_release_autovec

clean:
	rm -f lab5_debug lab5_release lab5_release_autovec

.PHONY: all run run-autovec clean
