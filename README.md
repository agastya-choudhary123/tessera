tessera
-------

tessera is a bytecode VM for tensor programs. It defines a small ISA with
explicit scratchpad moves and no hidden cache, and ships an assembler, a static
verifier, a reference interpreter, and a JIT that compiles hot regions to
native AArch64 + SME2.

The ISA is the point. Accelerators look the way they do, with software-managed
scratchpads, tagged asynchronous DMA and layout transforms in the movement
engine, because caches do not scale to the bandwidth a matrix unit demands.
tessera builds that argument end to end on an Apple M4 and measures it.

```
tessera assembly  ->  assembler  ->  verifier  ->  interpreter (oracle)
                                              \->  JIT -> AArch64 + SME2
```

### Documentation quick links

* [Building](#building)
* [Usage](#usage)
* [Benchmarks](#benchmarks)
* [isa/ISA.md](isa/ISA.md) — the ISA specification
* [NOTES.md](NOTES.md) — design notes, and how this compares to Trainium

### Requirements

Apple silicon with SME2 (M4 or later) and the macOS command line tools.

M4 implements SME2 but not non-streaming SVE, so ordinary C must not be
compiled with `+sme2`: the autovectorizer will emit SVE that SIGILLs outside
streaming mode. The Makefile scopes that flag to the two files that need it.

### Building

```
$ make                # assembler, verifier, interpreter, JIT
$ make test           # 15 tests: verifier rejection, interp/JIT equivalence
$ make benches        # machine characterisation and the BLAS baseline
```

### Usage

```
$ python3 tools/gen_matmul_kc.py --kc 128 -J 128 --cbuf 2 -o asm/mm.tsa
$ ./tessera -M 1024 -N 1024 -K 1024 --check --jit -r 7 asm/mm.tsa

$ ./tessera --disasm asm/mm.tsa | head             # bytecode listing
$ TESSERA_JIT_DUMP=/tmp/jit.bin ./tessera --jit …  # raw native code, for objdump
```

`--check` verifies every result against a double-precision reference. Relative
Frobenius error is ~2.1e-7, which is f32 rounding.

### Benchmarks

Measured on an Apple M4 (10 cores: 4 P + 6 E, 16 GB unified), best-of-N,
QoS-pinned to a P-core. `bench/compare.sh` discards a cold warmup run of each
side, since clock ramp costs about 15% on the first measurement, then takes the
best of three best-of-seven runs.

One P-core's matrix unit does 2008 GFLOP/s. SME2's `fmopa` is a 16x16 f32 outer
product, 512 flops in one instruction, retiring in 0.255 ns. That is 28x the
same core's NEON `fmla` peak of ~71 GFLOP/s.

The memory hierarchy cannot feed it except at the top (`bench/memhier`):

| footprint | bandwidth | flops/byte needed to sustain 2008 GFLOP/s |
|---|---|---|
| <= 128 KB (L1D) | 327–382 GB/s | 5.3–6.1 |
| <= 4 MB (L2) | ~139 GB/s | 14.4 |
| <= 24 MB (SLC) | 78–131 GB/s | 15–26 |
| DRAM | 34–70 GB/s | 29–59 |

Only the 128 KB tier is within reach of a well-blocked kernel, and that is
exactly the tier where a hardware cache picks residency for you by LRU. For a
tiled matmul that is the wrong policy: the panel you want pinned for the next
thousand outer products is, to an LRU cache, the line you touched longest ago.
Hence a scratchpad you address explicitly.

With data movement removed (`bench/kloop_u4.tsa`, outer products over resident
scratchpad data) the compiled loop sustains **1859 GFLOP/s** against the 2008
GFLOP/s ceiling, so code generation is not the bottleneck anywhere below that.
Against the reference interpreter on the same kernel and shape, the JIT is
25.6x (729 vs 28.4 GFLOP/s at 256³).

Holding a wider B block resident lets each packed A panel be reused across more
output tiles. Nothing changes but one number in the schedule (M=N=1024, K=128):

| resident B width `Jc` | A-panel reuse | DRAM traffic | throughput |
|---|---|---|---|
| 32 | 1x | 22.0 MB | 265 GFLOP/s |
| 64 | 2x | 13.4 MB | 531 GFLOP/s |
| 128 | 4x | 9.0 MB | 818 GFLOP/s |

Against Apple's Accelerate, which reaches 1430 GFLOP/s single-threaded and only
1707 with 8 threads. That 1.19x says the SME unit is shared, not replicated per
core, so single-threaded Accelerate is the fair comparison for tessera's one
compute thread:

| shape | tessera | Accelerate (1 thread) | ratio |
|---|---|---|---|
| 1024³, K=128 | 735 | 1226 | 0.60x |
| 1024³, K=256 | 877 | 1487 | 0.59x |
| 1024³, K=512 | 866 | 1649 | 0.53x |
| 1024³, K=1024 | 855 | 1711 | 0.50x |
| 2048², K=512 | 782 | 1653 | 0.47x |
| 2048², K=1024 | 787 | 1619 | 0.49x |

Roughly half of a mature hand-tuned vendor library, from a bytecode VM whose
kernels are emitted by a 150-line Python scheduler.

### The verifier

The ISA's guarantee is worth something only if it is checked. `ts_verify`
proves that no compute instruction names DRAM (V1), that every scratchpad
access is in bounds (V2), and that no scratchpad byte is read with a DMA still
in flight (V3).

V3 is an interval dataflow iterated to a fixpoint, so it catches loop-carried
hazards. `tests/bad_loop_carried.tsa` is a loop whose body reads correctly top
to bottom, with the `ldv` before the `dma.in`, but whose read races the
previous iteration's transfer across the back edge:

```
$ ./tessera tests/bad_loop_carried.tsa
V3 pc=5: reads SPM bytes with a DMA still in flight (missing wait): ldv v0, spm[0]
verification FAILED
```

That is the bug class that makes hand-written double-buffered accelerator
kernels miserable, because it is timing-dependent: correct on a slow DMA
engine, corrupt on a fast one. A tile shape that overflows the scratchpad is
likewise a compile-time error rather than silent thrashing.

### Limitations

The gap to Accelerate is data movement, not code generation. The scratchpad is
128 KB, and at the best configuration the kernel uses all 131072 bytes of it.
Trainium's on-chip SRAM is two orders of magnitude larger, and that one number
explains most of the difference. [NOTES.md](NOTES.md) works through it.

Single compute thread, plus DMA workers. f32 only. Matmul is the only kernel
family with a scheduler in `tools/`.

### Layout

```
src/tessera.h    ISA definition: opcodes, address spaces, machine model
src/asm.c        assembler and disassembler
src/verify.c     V1/V2/V3 static verification, tag dataflow to a fixpoint
src/vm.c         reference interpreter, the oracle the JIT is checked against
src/dma.c        movement engine: 4 SPSC queues, pack16 and acc transforms
src/arm64.h      AArch64 + SME2 encoders, every constant derived from clang
src/jit.c        bytecode -> native, MAP_JIT, streaming-region selection
tools/           kernel schedulers that emit tessera assembly
isa/ISA.md       the ISA specification
bench/           machine characterisation and the Accelerate baseline
tests/           correctness suite, and the programs the verifier must reject
```
