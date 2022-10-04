for f in *.cc; do
  if [[ $f == _* ]]; then continue; fi
  g++ --std=c++17 -Wall -Wextra -fPIC -O3 -c -o ${f%.*}.o $f
done

gcc -shared -o libhealpixlite.so *.o -lm
