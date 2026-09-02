# tessera — design notes

Longer-form notes that used to live in the README.

## How Trainium and this differ

This is not a claim to beat Trainium; a Trn1 chip is a different class of
hardware. The claim is about the programming model, which is the same one, and
about which parts of it Apple silicon makes better or worse.

Better on Apple: movement is not a transfer. Unified memory means `dram` in
this ISA is the same physical memory the compute unit reads. A `dma.in` is a
copy within one address space. There is no PCIe hop, no host/device split, no
separate allocation, no `cudaMemcpy`-shaped API. The VM's DRAM pointers are the
caller's tensors; `bench/blas_ref` and tessera are handed the identical arrays.
On a discrete accelerator the same program pays a round trip over the
interconnect before the first outer product retires. The DMA engine here is a
bank of threads (`src/dma.c`) reading the same memory the compute thread does.

Better on Apple: the scratchpad is real, and so is the accumulator. SME's ZA
register file is architecturally visible state with no cache behind it, which
is precisely the resource this ISA is designed around.

Worse on Apple, and it dominates: the scratchpad is 128 KB, not 24 MB.
Trainium's on-chip SRAM is two orders of magnitude larger, and that single
number explains essentially the whole gap against Accelerate. The scratchpad
has to hold a resident B block, four packed A panels and C staging at once; at
the best configuration the kernel uses 131072 of 131072 bytes, exactly 100% of
the budget. Every blocking parameter is pinned against that wall:

- ZA holds four 16x16 f32 tiles, so one k-sweep accumulates at most a 32x32
  output block. That is 8 flops/byte, which against ~60 GB/s of DRAM caps the
  kernel near 480 GFLOP/s no matter how good the code is.
- The escape is a second level of blocking in the scratchpad (the `Jc` sweep in
  the README), and how far it can be pushed is set directly by the 128 KB.
- Beyond that, k-chunking is required: the reduction is split into `Kc`-deep
  chunks whose partial sums are reduced into C by the DMA engine on writeback
  (`acc` mode). That lifts the limit on K entirely, since K=1024 does not fit
  otherwise, at the cost of read-modify-writing C once per chunk. At K=1024
  that is 64% of all DRAM traffic.

So the model transfers to Apple silicon cleanly and the compute side lands at
93% of peak, but a laptop's L1-sized scratchpad is what stands between
860 GFLOP/s and the multiplier's 2008.

## Two things the measurements changed about the design

The JIT compiles loop nests, not traces, because streaming mode is expensive.
Entering and leaving streaming SVE costs ~12 ns (`bench/smcost`), and an inner
k-loop at K=64 is ~24 ns of arithmetic. A conventional innermost-trace JIT
would have spent a third of the kernel toggling `PSTATE.SM`. So the JIT grows
the region to the outermost compilable nest and enters streaming mode once,
which in turn forced DMA issue to be emitted inline as scalar stores and a
release-store doorbell, since calling the C runtime would have required leaving
streaming mode. Details in `isa/ISA.md` §5.

One DMA engine is not enough. With a single queue the engine ran ~80% busy at
30-45 GB/s while the matrix unit waited on it: the kernel was movement-bound,
not compute-bound. A transposing strided gather is latency-bound, so the only
fix is concurrency. `src/dma.c` is now a bank of four single-producer,
single-consumer queues with one worker each and tags statically bound to
queues. This is the same reason real accelerators ship 8-16 DMA engines rather
than one fast one. The two packed A panels of a block carry different tags so
they land on different queues and pack concurrently.
