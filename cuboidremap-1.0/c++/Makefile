# Makefile for C++ cuboid remapping implementation

# Uncomment to use GCC
CXX = g++

# Uncomment to use Intel compiler
#CXX = icpc

# Uncomment to turn on aggressive optimizations
OPTFLAGS = -O3

CPPFLAGS =
CXXFLAGS = -Wall $(OPTFLAGS)
LDFLAGS =


all: remap

remap: cuboid.o remap.o
	$(CXX) -o remap $^ $(LDFLAGS)

clean:
	rm -f remap *.o
