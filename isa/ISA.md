# The tessera ISA

A bytecode ISA for tensor programs in which **memory movement is an
instruction, never a side effect**.

The design rule, from which everything else follows:

> No compute instruction may name a DRAM operand. Data reaches the compute
> units only because some instruction moved it there, and only after the
> program waited for that move to retire.

There is therefore no cache, no implicit fill, no prefetcher and no LRU policy
deciding what is resident. The program decides, and `ts_verify` proves the
decision was legal before a single instruction executes.

---

## 1. Machine model

| space | size | addressed by | who moves data in and out |
|---|---|---|---|
| `dram` | unified memory | 64-bit pointer in a scalar register | `dma.in` / `dma.out` only |
| `spm` | 128 KB scratchpad | byte offset, optional index register | `dma.*`, `ldv/stv`, `ldza/stza` |
| `vec` | 32 x 64 B (`v0..v31`) | register name | `ldv` / `stv` |
| `za` | 4 x 16x16 f32 tiles | tile name | `ldza` / `stza` / `zerza` |
| scalar | 16 x 64-bit (`r0..r15`) | register name | `movi`, `arg`, ALU |

`spm` is 128 KB because that is the measured size of the tier on an Apple M4
P-core that can actually feed the matrix unit: 306 GB/s up to 128 KB, falling
to 132 GB/s at 192 KB (`bench/memhier`). The scratchpad is sized to the
hardware, not the other way round.

`za` is not a software abstraction. It is the SME ZA register file, an
architecturally visible 4 KB accumulator array with no cache behind it, which
must be explicitly loaded and stored. The ISA's central idea is already real
silicon on this machine.

## 2. Encoding

Fixed 16 bytes, deliberately roomy, in the spirit of the wide encodings real
accelerators use. Decoding is a load and a switch.

```c
typedef struct {
    uint8_t  op, a, b, c;   /* register / tile / tag operands */
    uint16_t d, e;          /* shape fields, modes            */
    int32_t  imm0, imm1;
} ts_insn;
```

## 3. Instructions

### Movement: DRAM <-> scratchpad (asynchronous, tagged)

```
dma.in   spm[off], rBASE, rSTRIDE, rows, row_bytes, tN [, pack16]
dma.out  spm[off], rBASE, rSTRIDE, rows, row_bytes, tN [, acc]
wait     <tag or bitmask>
```

A transfer is a 2D strided descriptor pushed to one of `TS_DMA_NQ` queues and
retired asynchronously; `wait` blocks on per-tag completion counters. A tag is
statically bound to a queue (`queue = tag % nq`), so the JIT resolves the queue
address at compile time and issuing costs a handful of scalar stores — the
software form of a doorbell write.

Two modes do real work in the movement engine, as DMA engines on real
accelerators do:

- **`pack16`** transposes a 16 x K f32 tile into the K x 16 panel layout the
  outer-product unit consumes, during the copy. Layout transformation is free
  because it rides on a transfer that had to happen anyway.
- **`acc`** makes writeback a reduction (`dst += src`, f32). A k-chunked matmul
  produces partial sums; doing that reduction in the movement engine keeps it
  off the compute critical path.

### Movement: scratchpad <-> vec <-> za (synchronous)

```
ldv  vD, spm[off + rI]        stv  spm[off + rI], vS
ldza zaD, spm[off], stride    stza spm[off], zaS, stride
zerza zaD
```

### Compute — scratchpad, vector and accumulator operands only

```
mopa zaD, vA, vB     ; ZA[i][j] += vA[i] * vB[j]   16x16 f32 rank-1 update
vadd / vmul / vfma / vmax / vzero / vdup
```

`mopa` is one `fmopa` — 512 flops in one instruction, 0.255 ns on an M4 P-core.

### Scalar and control

```
movi  arg  addi  add  sub  mul  muli  shli  shri  cmplt  bnz  bz  jmp  halt
```

`arg rD, n` binds the n-th kernel argument (a DRAM base address or a
dimension). Loop bounds are runtime values; **tile shapes are compile-time**,
because a DMA descriptor is hardware state, not an argument. That is how every
real accelerator compiler works.

## 4. The verifier

`ts_verify` runs before execution and proves three properties.

- **V1 — no hidden operand.** No compute instruction names a DRAM operand.
  Structural, from the opcode table.
- **V2 — capacity.** Every scratchpad access lies inside 128 KB. A tile shape
  that does not fit is a *compile-time error*. On a cache-based machine the
  same program "works" and silently thrashes.
- **V3 — tag discipline.** No scratchpad byte is read while a DMA writing it is
  in flight (RAW), and no DMA overwrites a region still being read (WAR). This
  is an interval dataflow over the CFG, iterated to a fixpoint, so that
  *loop-carried* hazards are caught — the single-buffer-where-you-needed-two
  bug that is the whole difficulty of hand-writing pipelined accelerator code,
  and which is invisible to any straight-line inspection of the loop body.

`tests/bad_*.tsa` are four programs that are each rejected for one of these
reasons, including one whose hazard exists only across the loop back edge.

## 5. What the JIT does

`src/jit.c` lowers bytecode to AArch64 + SME2 written into `MAP_JIT` pages
(`pthread_jit_write_protect_np`, `sys_icache_invalidate`).

The mapping is deliberately 1:1 where it matters: `v0..v31` become `z0..z31`,
`za0..za3` become `ZA0.S..ZA3.S`. Compilation introduces **no** vector or
accumulator traffic of its own.

Region selection is driven by a measurement rather than by the textbook. The
usual choice, compiling the innermost hot loop, is wrong here: entering and
leaving streaming SVE mode costs ~12 ns (`bench/smcost`) while an inner k-loop
at K=64 is only ~24 ns of arithmetic, so a per-trace `smstart`/`smstop` would
spend a third of the kernel toggling `PSTATE.SM`. The JIT therefore grows the
region to the outermost compilable loop nest and pays for streaming mode once
per kernel invocation.

That forces the other unusual decision in the file: DMA issue happens *inside*
the region, so it must not leave streaming mode either. Calling the C runtime
would require `smstop` (the callee uses NEON, illegal while streaming), so the
JIT emits the descriptor-ring push inline as scalar stores and a release-store
to the head index. Streaming mode restricts SIMD state, not scalar state, so
this is legal — and free.
