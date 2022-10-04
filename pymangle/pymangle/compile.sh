for f in *.c; do
  if [[ $f == _* ]]; then continue; fi
  echo $f
  gcc --std=gnu99 -Wall -Wextra -O3 -c -o ${f%.*}.o $f
done

ar -r -o libcmangle.a *.o
