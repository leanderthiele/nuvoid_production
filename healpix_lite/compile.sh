for f in *.cc; do
  if [[ $f == _* ]]; then continue; fi
  echo $f
  g++ --std=c++17 -Wall -Wextra -O3 -c -o ${f%.*}.o $f
done

ar -r -o libhealpixlite.a *.o
