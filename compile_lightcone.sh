g++ --std=c++17 -Wall -Wextra -O3 -ggdb \
  -I./pymangle -I./cuboidremap-1.0/c++ -I./healpix_lite \
  -DHAVE_INLINE \
  -o lightcone lightcone.cpp cuboidremap-1.0/c++/cuboid.cpp \
  -L./pymangle/pymangle -L./healpix_lite \
  -lm -lgsl -lgslcblas -lcmangle -lhealpixlite -fopenmp
