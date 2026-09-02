#!/bin/zsh
# tessera vs Accelerate. Discards a cold warmup run of each (clock ramp on this
# machine costs ~15% on the first measurement) and reports the best thereafter.
cd "$(dirname "$0")/.."
K=asm/mmkc_128_128_c2.tsa
best(){ local b=0 v; for i in 1 2 3; do v=$("$@" | head -1 | awk '{print $5}'); (( v > b )) && b=$v; done; print $b }
printf "%-24s %11s %11s %8s\n" "shape" "tessera" "Accel(1t)" "ratio"
for shape in "1024 1024 128" "1024 1024 256" "1024 1024 512" "1024 1024 1024" "2048 2048 512" "2048 2048 1024"; do
  set -A s ${=shape}; M=$s[1]; N=$s[2]; KK=$s[3]
  ./tessera -M $M -N $N -K $KK -r 3 --jit $K >/dev/null                       # warmup
  VECLIB_MAXIMUM_THREADS=1 ./bench/blas_ref -M $M -N $N -K $KK -r 3 >/dev/null
  t=$(best ./tessera -M $M -N $N -K $KK -r 7 --jit $K)
  a=$(best env VECLIB_MAXIMUM_THREADS=1 ./bench/blas_ref -M $M -N $N -K $KK -r 7)
  printf "%-24s %11s %11s %7.2fx\n" "M=$M N=$N K=$KK" $t $a $(print "$t/$a" | bc -l)
done
