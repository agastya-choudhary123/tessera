# tessera

A bytecode VM for tensor programs: a small ISA with **explicit scratchpad moves
and no hidden cache**, an assembler, a static verifier, a reference
interpreter, and a JIT that compiles hot regions to native AArch64 + SME2.

The ISA is the project. Accelerators look the way they do — software-managed
scratchpads, tagged asynchronous DMA, layout transforms in the movement engine
— because caches do not scale to the bandwidth a matrix unit demands. This is
that argument, built and measured end to end on an Apple M4.

```
tessera assembly  ->  assembler  ->  verifier  ->  interpreter (oracle)
                                              \->  JIT -> AArch64 + SME2
```

## Why this machine makes the argument well

Two measurements on an M4 (10 cores: 4 P + 6 E, 16 GB unified, `bench/`):

**One P-core's matrix unit does 2008 GFLOP/s.** SME2's `fmopa` is a 16x16 f32
outer product — 512 flops in one instruction, retiring in 0.255 ns, essentially
one per cycle. That is **28x** the same core's NEON `fmla` peak of ~71 GFLOP/s.

**The memory hierarchy cannot feed it, except at the top.** `bench/memhier`
sweeps a streaming read over increasing footprints:

| footprint | bandwidth | flops/byte needed to sustain 2008 GFLOP/s |
|---|---|---|
| <= 128 KB (L1D) | **327–382 GB/s** | 5.3–6.1 |
| <= 4 MB (L2) | ~139 GB/s | 14.4 |
| <= 24 MB (SLC) | 78–131 GB/s | 15–26 |
| DRAM | 34–70 GB/s | 29–59 |

(The L1 tier has real run-to-run spread; L2 is stable at 138–142. The part
that matters is the cliff between them, which is ~2.7x every time.)

Only the 128 KB tier is within reach of a well-blocked kernel. And that tier is
exactly the one where a hardware cache picks residency *for* you, by LRU, which
is the wrong policy for a tiled matmul: the panel you want pinned for the next
thousand outer products is, to an LRU cache, simply the line you touched
longest ago. Hence a scratchpad you address explicitly.

**On this machine the model is not a metaphor.** SME's ZA register file *is* an
architecturally visible 4 KB accumulator scratchpad with no cache behind it,
which you must explicitly load and store. `za0..za3` in this ISA are those
tiles, one-to-one.

## Results

All figures measured on this M4, best-of-N, QoS-pinned to a P-core. Correctness
is checked against a double-precision reference at every shape
(`--check`); relative Frobenius error is ~2.1e-7, i.e. f32 rounding.

**The JIT's inner loop reaches 93% of the hardware ceiling.** With data
movement removed (`bench/kloop_u4.tsa`, outer products over resident
scratchpad data), the compiled loop sustains **1859 GFLOP/s** against the 2008
GFLOP/s `fmopa` peak. Code generation is not the bottleneck anywhere below that.

**JIT vs. the reference interpreter, same kernel and shape: 25.6x**
(729 vs 28.4 GFLOP/s at 256³).

**Explicit residency control is worth 3.1x, and the ISA is what lets you spend
it.** Holding a wider B block resident in the scratchpad lets each packed A
panel be reused across more output tiles. Nothing changes but one number in the
schedule; M=N=1024, K=128:

| resident B width `Jc` | A-panel reuse | DRAM traffic | throughput |
|---|---|---|---|
| 32 | 1x | 22.0 MB | 265 GFLOP/s |
| 64 | 2x | 13.4 MB | 531 GFLOP/s |
| 128 | 4x | 9.0 MB | **818 GFLOP/s** |

**Against Apple's own Accelerate.** Accelerate's `sgemm` reaches 1430 GFLOP/s
single-threaded and only 1707 with 8 threads — a 1.19x gain that says the SME
unit is *shared*, not replicated per core. Since tessera runs one compute
thread (plus DMA workers), single-threaded Accelerate is the fair comparison:

Reproduce with `bench/compare.sh`, which discards a cold warmup run of each
(clock ramp costs ~15% on the first measurement) and takes the best of three
best-of-seven runs thereafter:

| shape | tessera | Accelerate (1 thread) | ratio |
|---|---|---|---|
| 1024³, K=128 | 735 | 1226 | 0.60x |
| 1024³, K=256 | 877 | 1487 | 0.59x |
| 1024³, K=512 | 866 | 1649 | 0.53x |
| 1024³, K=1024 | 855 | 1711 | 0.50x |
| 2048², K=512 | 782 | 1653 | 0.47x |
| 2048², K=1024 | 787 | 1619 | 0.49x |

Roughly half of a mature, hand-tuned vendor library, from a bytecode VM whose
kernels are emitted by a 150-line Python scheduler. The remaining gap is data
movement, not code generation — see below.

## How Trainium and this differ, honestly

This is not a claim to beat Trainium; a Trn1 chip is a different class of
hardware. The claim is about the *programming model*, which is the same one,
and about which parts of it Apple silicon makes better or worse.

**Better on Apple: movement is not a transfer.** Unified memory means `dram` in
this ISA is the same physical memory the compute unit reads. A `dma.in` is a
copy within one address space — there is no PCIe hop, no host/device split, no
separate allocation, no `cudaMemcpy`-shaped API. The VM's DRAM pointers *are*
the caller's tensors; `bench/blas_ref` and tessera are handed the identical
arrays. On a discrete accelerator, the same program pays a round trip over the
interconnect before the first outer product retires. The DMA "engine" here is a
bank of threads (`src/dma.c`) reading the same memory the compute thread does.

**Better on Apple: the scratchpad is real and so is the accumulator.** ZA is
architecturally visible state with no cache behind it, which is precisely the
resource this ISA is designed around.

**Worse on Apple, and it is the dominant limit: the scratchpad is 128 KB, not
24 MB.** Trainium's on-chip SRAM is two orders of magnitude larger, and that
single number explains essentially the whole gap in the table above. The
scratchpad has to hold a resident B block, four packed A panels and C staging
at once; at the best configuration the kernel uses **131072 of 131072 bytes**,
exactly 100% of the budget. Every blocking parameter is pinned against that
wall:

- ZA holds four 16x16 f32 tiles, so one k-sweep accumulates at most a 32x32
  output block: 8 flops/byte, which against ~60 GB/s of DRAM caps the kernel
  near 480 GFLOP/s no matter how good the code is.
- The escape is a second level of blocking in the scratchpad (the `Jc` table
  above), and how far you can push it is set directly by the 128 KB.
- Beyond that, k-chunking is required: the reduction is split into `Kc`-deep
  chunks whose partial sums are reduced into C by the DMA engine on writeback
  (`acc` mode). That lifts the limit on K entirely — K=1024 does not fit
  otherwise — at the cost of read-modify-writing C once per chunk, which is
  64% of all DRAM traffic at K=1024.

That is the honest shape of the result: the model transfers to Apple silicon
cleanly and the compute side lands at 93% of peak, but a laptop's L1-sized
scratchpad is what stands between 860 GFLOP/s and the multiplier's 2008.

## Two things the measurements changed about the design

**The JIT compiles loop nests, not traces, because streaming mode is
expensive.** Entering and leaving streaming SVE costs ~12 ns (`bench/smcost`);
an inner k-loop at K=64 is ~24 ns of arithmetic. A conventional innermost-trace
JIT would have spent a third of the kernel toggling `PSTATE.SM`. So the JIT
grows the region to the outermost compilable nest and enters streaming mode
once — which in turn forced DMA issue to be emitted *inline* as scalar stores
and a release-store doorbell, since calling the C runtime would have required
leaving streaming mode. Details in `isa/ISA.md` §5.

**One DMA engine is not enough.** With a single queue the engine ran ~80% busy
at 30–45 GB/s while the matrix unit waited on it: the kernel was
movement-bound, not compute-bound. A transposing strided gather is
latency-bound, so the only fix is concurrency. `src/dma.c` is now a bank of
four single-producer/single-consumer queues, one worker each, with tags
statically bound to queues — the same reason real accelerators ship 8–16 DMA
engines rather than one fast one. The two packed A panels of a block carry
*different* tags so they land on different queues and pack concurrently.

## The verifier earns its place

The ISA's guarantee is only worth something if it is checked. `ts_verify`
proves no compute instruction names DRAM (V1), every scratchpad access is in
bounds (V2), and no scratchpad byte is read with a DMA still in flight (V3).

V3 is an interval dataflow iterated to a fixpoint, so it catches loop-carried
hazards. `tests/bad_loop_carried.tsa` is a loop whose body looks correct read
top to bottom — the `ldv` precedes the `dma.in` — but whose read races the
previous iteration's transfer across the back edge:

```
$ ./tessera tests/bad_loop_carried.tsa
V3 pc=5: reads SPM bytes with a DMA still in flight (missing wait): ldv v0, spm[0]
verification FAILED
```

That is the bug class that makes hand-written double-buffered accelerator
kernels miserable: it is timing-dependent, so it is correct on a slow DMA
engine and corrupt on a fast one. Here it cannot reach the machine. Likewise a
tile shape that overflows the scratchpad is a compile-time error rather than
silent thrashing — which is the entire argument for an explicit scratchpad,
made mechanical.

## Build and run

```sh
make                      # the VM: assembler, verifier, interpreter, JIT
make test                 # 15 tests: verifier rejection + interp/JIT correctness
make benches              # the machine characterisation and the BLAS baseline

# generate a kernel, verify it, check it, run it
python3 tools/gen_matmul_kc.py --kc 128 -J 128 --cbuf 2 -o asm/mm.tsa
./tessera -M 1024 -N 1024 -K 1024 --check --jit -r 7 asm/mm.tsa

./tessera --disasm asm/mm.tsa | head          # bytecode listing
TESSERA_JIT_DUMP=/tmp/jit.bin ./tessera --jit ...   # raw native code, for objdump
```

## Layout

```
src/tessera.h    ISA definition: opcodes, address spaces, machine model
src/asm.c        assembler: .tsa text -> bytecode, and the disassembler
src/verify.c     V1/V2/V3 static verification, tag dataflow to a fixpoint
src/vm.c         reference interpreter (the oracle the JIT is checked against)
src/dma.c        the movement engine: 4 SPSC queues, pack16 and acc transforms
src/arm64.h      AArch64 + SME2 encoders, every constant derived from clang
src/jit.c        bytecode -> native, MAP_JIT, streaming-mode region selection
tools/           kernel schedulers that emit tessera assembly
isa/ISA.md       the ISA specification
bench/           machine characterisation + Accelerate baseline
tests/           correctness suite and the programs the verifier must reject
```

## Requirements

Apple silicon with SME2 (M4 or later) and macOS with Xcode command line tools.
Note that M4 implements SME2 but **not** non-streaming SVE, so ordinary C must
not be compiled with `+sme2` — the autovectorizer will emit SVE that SIGILLs
outside streaming mode. The Makefile scopes that flag to the two files that
need it.
