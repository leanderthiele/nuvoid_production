for f in *.c; do
  if [[ $f == _* ]]; then continue; fi
  gcc --std=gnu99 -Wall -Wextra -fPIC -O3 -c -o ${f%.*}.o $f
done

gcc -shared -o libcmangle.so *.o -lm
