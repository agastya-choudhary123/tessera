CC      = clang
# Apple M4 implements SME2 but NOT non-streaming SVE. Compiling ordinary C with
# +sme2 lets the autovectorizer emit SVE, which SIGILLs outside streaming mode.
# So general code is built plain; SME only ever appears inside streaming-mode
# inline asm or in words the JIT emits itself.
CFLAGS  = -O2 -Wall -Wextra -Wno-unused-parameter -mcpu=apple-m4 -Isrc
LDFLAGS = -lm -lpthread
SRC     = src/asm.c src/vm.c src/verify.c src/dma.c src/jit.c src/main.c
OBJ     = $(SRC:.c=.o)

tessera: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)
%.o: %.c src/tessera.h
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	rm -f $(OBJ) tessera bench/*.o bench/smepeak bench/smcost bench/memhier bench/blas_ref
distclean: clean
	rm -f $(KERNELS)
.PHONY: clean

# ---- benchmarks -------------------------------------------------------
# smepeak/smcost need SME in hand-written asm, so their kernel files (and only
# those) are built with +sme2; see the note at the top of this file.
bench/smepeak: bench/smepeak.c bench/smepeak_kernels.c
	$(CC) -O2 -march=armv9.2-a+sme2 -c bench/smepeak_kernels.c -o bench/smepeak_kernels.o
	$(CC) -O2 -mcpu=apple-m4 bench/smepeak.c bench/smepeak_kernels.o -o $@
bench/smcost: bench/smcost.c bench/smcost_kernels.c
	$(CC) -O2 -march=armv9.2-a+sme2 -c bench/smcost_kernels.c -o bench/smcost_kernels.o
	$(CC) -O2 -mcpu=apple-m4 bench/smcost.c bench/smcost_kernels.o -o $@
bench/memhier: bench/memhier.c
	$(CC) -O2 -mcpu=apple-m4 $< -o $@
bench/blas_ref: bench/blas_ref.c
	$(CC) -O2 -mcpu=apple-m4 -DACCELERATE_NEW_LAPACK $< -framework Accelerate -o $@
benches: bench/smepeak bench/smcost bench/memhier bench/blas_ref

# ---- kernels ----------------------------------------------------------
# Tile shape is compile-time state, so kernels are generated per shape.
KERNELS = asm/mmkc_128_128_c2.tsa asm/mm_k128_j32.tsa asm/mm_k128_j64.tsa \
          asm/mm_k128_j128.tsa bench/kloop_u4.tsa
kernels: $(KERNELS)
asm/mmkc_128_128_c2.tsa:
	python3 tools/gen_matmul_kc.py --kc 128 -J 128 --cbuf 2 --unroll 4 -o $@
asm/mm_k128_j%.tsa:
	python3 tools/gen_matmul32.py -K 128 -J $* --unroll 4 -o $@
bench/kloop_u%.tsa:
	python3 tools/gen_kloop.py --unroll $* -o $@
test: tessera kernels
	./tests/run_tests.sh
.PHONY: benches test kernels
