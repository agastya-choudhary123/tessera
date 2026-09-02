#!/usr/bin/env python3
"""Emit tessera assembly for a double-buffered, two-level-blocked f32 matmul.

The shape of this schedule is forced by two hard limits on the machine, and
naming them is the point of the exercise:

  1. ZA holds four 16x16 f32 tiles, so the largest output block one k-sweep can
     accumulate is 32x32. That caps arithmetic intensity at 8 flops/byte, which
     against ~60 GB/s of DRAM caps the whole kernel near 480 GFLOP/s -- even
     though the multiplier itself will do ~2000.

  2. The escape is the scratchpad, not the register file. Hold a *wide* B block
     (Jc columns) resident in the 128 KB scratchpad and sweep each packed A
     panel across all Jc/32 output tiles in it. A is then read from DRAM once
     per (row-block, B-block) instead of once per output tile, and intensity
     rises to ~Jc/2 flops/byte. That reuse is an explicit residency decision --
     precisely the decision a hardware LRU cache will not let you express.

Schedule:
  for j in steps of Jc:            DMA B[:, j:j+Jc] -> resident (tag t6)
    for i in steps of 32:          double-buffered, software-pipelined
      prefetch A row-block i+32 into the other buffer (tags t0/t1 or t2/t3)
      wait for this buffer's two packed panels
      for jj in 0..Jc/32-1:        unrolled; reuses the SAME A panel
        zero ZA, k-sweep of outer products, store the 32x32 tile to staging
      DMA the whole 32xJc staging block back to C (tags t4/t5)

The two packed A panels of a block carry different tags so they land on
different DMA queues and pack concurrently. The out-of-range prefetch on the
last iteration is clamped to row-block 0 by multiplying the index by the 0/1
comparison result: no branch, no out-of-bounds read, result discarded.
"""
import sys, argparse

SPM = 128*1024

def emit(K, Jc, U):
    assert Jc % 32 == 0, "Jc must be a multiple of the 32-wide output tile"
    NT   = Jc//32                 # output tiles per A panel
    KB4  = K*4                    # bytes per row of a K-deep A panel
    KP   = K*64                   # one packed 16-wide A panel
    BSZ  = K*Jc*4                 # resident B block
    CSZ  = 32*Jc*4                # C staging block
    A0   = [BSZ,          BSZ+2*KP]
    A1   = [BSZ+KP,       BSZ+3*KP]
    CST  = [BSZ+4*KP,     BSZ+4*KP+CSZ]
    used = CST[1]+CSZ
    if used > SPM:
        sys.exit(f"K={K} Jc={Jc} needs {used}B > {SPM}B scratchpad")
    L=[];A=L.append
    A(f"; two-level-blocked matmul: K={K} Jc={Jc} tiles/panel={NT} unroll={U}")
    A(f"; scratchpad  B={BSZ}  A(4 panels)={4*KP}  C(2x)={2*CSZ}  total={used}/{SPM}B")
    A(f"; A-panel reuse = {NT}x  ->  ~{Jc//2} flops/byte of A traffic")
    A(".args 6")
    for r,(nm,ix) in enumerate([("A",0),("B",1),("C",2),("M",3),("N",4),("K",5)]):
        A(f"  arg r{r}, {ix}        ; {nm}")
    A("  muli r8, r5, 4   ; lda")
    A("  muli r9, r4, 4   ; ldb = ldc")
    A("  movi r7, 0       ; j")
    A("j_loop:")
    A("  shli r14, r7, 2")
    A("  add  r14, r14, r1")
    A(f"  dma.in spm[0], r14, r9, {K}, {Jc*4}, t6      ; B[:, j:j+{Jc}] -> resident")
    A("  movi r15, 0")
    A("  mul  r14, r15, r8")
    A("  add  r14, r14, r0")
    A(f"  dma.in spm[{A0[0]}], r14, r8, 16, {KB4}, t0, pack16")
    A("  muli r15, r8, 16")
    A("  add  r15, r14, r15")
    A(f"  dma.in spm[{A1[0]}], r15, r8, 16, {KB4}, t1, pack16")
    A("  wait 64          ; t6: B must land before any outer product")
    A("  movi r6, 0       ; i")
    A("i_loop:")

    def half(buf, atags, ctag):
        nxt=1-buf; nt=(2,3) if nxt==1 else (0,1)
        A(f"  ; ---- half {buf}: compute from A buffer {buf}, prefetch into {nxt} ----")
        A("  addi r15, r6, 32")
        A("  cmplt r14, r15, r3")
        A("  mul  r15, r15, r14      ; clamp the tail prefetch to row-block 0")
        A("  mul  r14, r15, r8")
        A("  add  r14, r14, r0")
        A(f"  dma.in spm[{A0[nxt]}], r14, r8, 16, {KB4}, t{nt[0]}, pack16")
        A("  muli r15, r8, 16")
        A("  add  r15, r14, r15")
        A(f"  dma.in spm[{A1[nxt]}], r15, r8, 16, {KB4}, t{nt[1]}, pack16")
        A(f"  wait {(1<<atags[0])|(1<<atags[1])}           ; both A panels of buffer {buf} landed")
        A(f"  wait {1<<ctag}          ; C staging {buf} free")
        for jj in range(NT):
            A(f"  ; -- output tile {jj} of {NT}, reusing A buffer {buf} --")
            for t in range(4): A(f"  zerza za{t}")
            A(f"  movi r10, {A0[buf]}")
            A(f"  movi r11, {A1[buf]}")
            A(f"  movi r12, {jj*128}")
            A(f"  movi r13, {A0[buf]+KP}")
            A(f"k_{buf}_{jj}:")
            for u in range(U):
                v=4*u
                A(f"  ldv  v{v},   spm[{u*64}+r10]")
                A(f"  ldv  v{v+1}, spm[{u*64}+r11]")
                A(f"  ldv  v{v+2}, spm[{u*Jc*4}+r12]")
                A(f"  ldv  v{v+3}, spm[{u*Jc*4+64}+r12]")
            for u in range(U):
                v=4*u
                A(f"  mopa za0, v{v},   v{v+2}")
                A(f"  mopa za1, v{v},   v{v+3}")
                A(f"  mopa za2, v{v+1}, v{v+2}")
                A(f"  mopa za3, v{v+1}, v{v+3}")
            A(f"  addi r10, r10, {64*U}")
            A(f"  addi r11, r11, {64*U}")
            A(f"  addi r12, r12, {Jc*4*U}")
            A("  cmplt r15, r10, r13")
            A(f"  bnz  r15, k_{buf}_{jj}")
            c=CST[buf]+jj*128
            A(f"  stza spm[{c}], za0, {Jc*4}")
            A(f"  stza spm[{c+64}], za1, {Jc*4}")
            A(f"  stza spm[{c+16*Jc*4}], za2, {Jc*4}")
            A(f"  stza spm[{c+16*Jc*4+64}], za3, {Jc*4}")
        A("  mul  r14, r6, r9")
        A("  add  r14, r14, r2")
        A("  shli r15, r7, 2")
        A("  add  r14, r14, r15")
        A(f"  dma.out spm[{CST[buf]}], r14, r9, 32, {Jc*4}, t{ctag}")
        A("  addi r6, r6, 32")

    half(0,(0,1),4); half(1,(2,3),5)
    A("  cmplt r15, r6, r3")
    A("  bnz  r15, i_loop")
    A("  wait 48          ; drain both C writebacks (t4|t5)")
    A(f"  addi r7, r7, {Jc}")
    A("  cmplt r15, r7, r4")
    A("  bnz  r15, j_loop")
    A("  halt")
    return "\n".join(L)+"\n"

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("-K",type=int,default=128)
    ap.add_argument("-J",type=int,default=64,help="resident B block width")
    ap.add_argument("--unroll",type=int,default=4)
    ap.add_argument("-o",default="-")
    a=ap.parse_args()
    s=emit(a.K,a.J,a.unroll)
    sys.stdout.write(s) if a.o=="-" else open(a.o,"w").write(s)
