#!/usr/bin/env python3
"""Emit tessera assembly for a k-chunked, two-level-blocked f32 matmul.

Why this exists. The non-chunked schedule (gen_matmul32.py) accumulates the
whole K reduction in ZA, so the A panel and the resident B block must both fit
in the 128 KB scratchpad at once. That bounds K*(128 + 4*Jc) <= 131072, so a
large K forces a narrow B block, which collapses A-panel reuse and leaves the
kernel DRAM-bound. Measured: K=256 forces Jc=32 and 480 GFLOP/s, against 880
at K=128/Jc=128.

k-chunking removes that coupling. The reduction is split into chunks of Kc;
each chunk produces a partial 32xJc tile, and the partials are summed into C by
the DMA engine on writeback (mode `acc`). Kc and Jc are then chosen
independently of the problem's K, so a wide B block is affordable at any depth.

Loop order  j -> kc -> i, which is what makes the residency work out:
  - B[kc:kc+Kc, j:j+Jc] is DMA'd once and stays resident across the whole i
    sweep (read exactly once from DRAM per (j, kc)).
  - the packed A panel is reused across all Jc/32 output tiles in that block.
  - C is touched once per (i, j, kc) as a read-modify-write, done in the
    movement engine so the reduction never lands on the compute critical path.

The kc = 0 iteration is peeled so it can overwrite C rather than accumulate
into it, which removes the need to pre-zero C.
"""
import sys, argparse
SPM = 128*1024

def emit(Kc, Jc, U, CB=2):
    NT   = Jc//32
    KP   = Kc*64
    BSZ  = Kc*Jc*4
    CSZ  = 32*Jc*4
    A0   = [BSZ,        BSZ+2*KP]
    A1   = [BSZ+KP,     BSZ+3*KP]
    CST  = [BSZ+4*KP + b*CSZ for b in range(CB)]   # C staging, CB-way
    used = CST[-1]+CSZ
    if used > SPM: sys.exit(f"Kc={Kc} Jc={Jc} needs {used}B > {SPM}B scratchpad")
    L=[];A=L.append
    A(f"; k-chunked two-level-blocked matmul: Kc={Kc} Jc={Jc} tiles/panel={NT} unroll={U}")
    A(f"; scratchpad  B={BSZ}  A(4 panels)={4*KP}  C({CB}x)={CB*CSZ}  total={used}/{SPM}B")
    A(".args 6")
    for r,nm in enumerate(["A","B","C","M","N","K"]): A(f"  arg r{r}, {r}        ; {nm}")
    A("  muli r8, r5, 4   ; lda = K*4  (also the kc byte-offset bound)")
    A("  muli r9, r4, 4   ; ldb = ldc")
    A("  movi r7, 0       ; j")
    A("j_loop:")
    A("  movi r5, 0       ; kc byte offset (r5's K value is dead after lda)")

    def kc_block(acc, label):
        A(f"  ; ========== k-chunk block ({'accumulate' if acc else 'overwrite'}) ==========")
        A("  shri r14, r5, 2         ; kc (elements)")
        A("  mul  r14, r14, r9")
        A("  add  r14, r14, r1")
        A("  shli r15, r7, 2")
        A("  add  r14, r14, r15")
        A(f"  dma.in spm[0], r14, r9, {Kc}, {Jc*4}, t6     ; B[kc:kc+{Kc}, j:j+{Jc}] resident")
        A("  movi r15, 0")
        A("  mul  r14, r15, r8")
        A("  add  r14, r14, r0")
        A("  add  r14, r14, r5")
        A(f"  dma.in spm[{A0[0]}], r14, r8, 16, {Kc*4}, t0, pack16")
        A("  muli r15, r8, 16")
        A("  add  r15, r14, r15")
        A(f"  dma.in spm[{A1[0]}], r15, r8, 16, {Kc*4}, t1, pack16")
        A("  wait 64")
        A("  movi r6, 0       ; i")
        A(f"i_loop_{label}:")
        for buf,atags in ((0,(0,1)),(1,(2,3))):
            nxt=1-buf; nt=(2,3) if nxt==1 else (0,1)
            cb = buf % CB; ctag = 4+cb
            A(f"  ; ---- half {buf} ----")
            A("  addi r15, r6, 32")
            A("  cmplt r14, r15, r3")
            A("  mul  r15, r15, r14      ; clamp tail prefetch to row-block 0")
            A("  mul  r14, r15, r8")
            A("  add  r14, r14, r0")
            A("  add  r14, r14, r5")
            A(f"  dma.in spm[{A0[nxt]}], r14, r8, 16, {Kc*4}, t{nt[0]}, pack16")
            A("  muli r15, r8, 16")
            A("  add  r15, r14, r15")
            A(f"  dma.in spm[{A1[nxt]}], r15, r8, 16, {Kc*4}, t{nt[1]}, pack16")
            A(f"  wait {(1<<atags[0])|(1<<atags[1])}")
            A(f"  wait {1<<ctag}          ; C staging {cb} free")
            for jj in range(NT):
                for t in range(4): A(f"  zerza za{t}")
                A(f"  movi r10, {A0[buf]}")
                A(f"  movi r11, {A1[buf]}")
                A(f"  movi r12, {jj*128}")
                A(f"  movi r13, {A0[buf]+KP}")
                A(f"k_{label}_{buf}_{jj}:")
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
                A(f"  bnz  r15, k_{label}_{buf}_{jj}")
                c=CST[cb]+jj*128
                A(f"  stza spm[{c}], za0, {Jc*4}")
                A(f"  stza spm[{c+64}], za1, {Jc*4}")
                A(f"  stza spm[{c+16*Jc*4}], za2, {Jc*4}")
                A(f"  stza spm[{c+16*Jc*4+64}], za3, {Jc*4}")
            A("  mul  r14, r6, r9")
            A("  add  r14, r14, r2")
            A("  shli r15, r7, 2")
            A("  add  r14, r14, r15")
            A(f"  dma.out spm[{CST[cb]}], r14, r9, 32, {Jc*4}, t{ctag}{', acc' if acc else ''}")
            A("  addi r6, r6, 32")
        A("  cmplt r15, r6, r3")
        A(f"  bnz  r15, i_loop_{label}")
        A(f"  wait {sum(1<<(4+b) for b in range(CB))}          ; drain C before the next k-chunk touches it")
        A(f"  addi r5, r5, {Kc*4}")

    kc_block(False, "first")
    A("  cmplt r15, r5, r8")
    A("  bz   r15, j_done")
    A("kc_loop:")
    kc_block(True, "rest")
    A("  cmplt r15, r5, r8")
    A("  bnz  r15, kc_loop")
    A("j_done:")
    A(f"  addi r7, r7, {Jc}")
    A("  cmplt r15, r7, r4")
    A("  bnz  r15, j_loop")
    A("  halt")
    return "\n".join(L)+"\n"

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--kc",type=int,default=128); ap.add_argument("-J",type=int,default=128)
    ap.add_argument("--unroll",type=int,default=4); ap.add_argument("--cbuf",type=int,default=2); ap.add_argument("-o",default="-")
    a=ap.parse_args(); s=emit(a.kc,a.J,a.unroll,a.cbuf)
    sys.stdout.write(s) if a.o=="-" else open(a.o,"w").write(s)
