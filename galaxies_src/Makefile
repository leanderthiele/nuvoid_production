CXX = g++
#CXX = icpc

CXXFLAGS = -std=c++17 -O3 -ffast-math -march=native -ggdb3 -fPIC -Wall -Wextra -Wpedantic -Wno-vla -DHAVE_INLINE

LFLAGS = -lgsl -lgslcblas -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -Wl,-rpath /home/lthiele/bigfile/shared -L/home/lthiele/bigfile/shared/lib -lbigfile
#-lfftw3f -lfftw3

all: pyglx

galaxies.o: galaxies.cpp
	$(CXX) -c $(CXXFLAGS) -o galaxies.o galaxies.cpp

populate.o: populate.cpp
	$(CXX) -c $(CXXFLAGS) -o populate.o populate.cpp

read_cat.o: read_cat.cpp
	$(CXX) -c $(CXXFLAGS) -o read_cat.o read_cat.cpp

read_hdf5.o: read_hdf5.cpp
	$(CXX) -c $(CXXFLAGS) -o read_hdf5.o read_hdf5.cpp

read_bigfile.o: read_bigfile.cpp
	$(CXX) -c $(CXXFLAGS) -I/home/lthiele/bigfile/src -o read_bigfile.o read_bigfile.cpp

read_fof.o: read_fof.cpp
	$(CXX) -c $(CXXFLAGS) -o read_fof.o read_fof.cpp

mesh.o: mesh.cpp
	$(CXX) -c $(CXXFLAGS) -o mesh.o mesh.cpp

mark.o: mark.cpp
	$(CXX) -c $(CXXFLAGS) -o mark.o mark.cpp

pk.o: pk.cpp
	$(CXX) -c $(CXXFLAGS) -o pk.o pk.cpp

power.o: power.cpp
	$(CXX) -c $(CXXFLAGS) -o power.o power.cpp

pyapi.o: pyapi.cpp
	$(CXX) -c $(CXXFLAGS) -fconstexpr-depth=8192 -ftemplate-depth=4096 $(shell python3 -m pybind11 --includes) -o pyapi.o pyapi.cpp -fopenmp

libgalaxies.a: galaxies.o populate.o read_cat.o read_bigfile.o read_fof.o mesh.o mark.o pk.o power.o
	ar -r -o libgalaxies.a galaxies.o populate.o read_cat.o read_bigfile.o read_fof.o mesh.o mark.o pk.o power.o

test: libgalaxies.a test.cpp
	$(CXX) $(CXXFLAGS) -o test test.cpp -L. -lgalaxies $(LFLAGS)

wisdom: create_wisdom.cpp
	$(CXX) $(CXXFLAGS) -o create_wisdom create_wisdom.cpp -lfftw3f -lfftw3

pyglx: libgalaxies.a pyapi.o
	$(CXX) -shared -o pyglx$(shell python3-config --extension-suffix) pyapi.o libgalaxies.a $(LFLAGS) -fopenmp

.PHONY: clean
clean:
	rm -f *.o *.a *.so test create_wisdom
