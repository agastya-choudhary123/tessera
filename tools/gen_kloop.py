#!/usr/bin/env python3
"""Isolate the JIT's inner loop: no DMA at all, just outer products over data
already resident in the scratchpad. Measures how close the compiled k-loop gets
to the 1351 GFLOP/s fmopa ceiling, with data movement removed from the picture.

Run as:  tessera -M <reps> -N 512 -K 512 --jit bench/kloop.tsa
so that main.c's 2*M*N*K accounting equals the real flop count
(reps * 256 k-steps * 4 mopa * 512 flops).
"""
import sys, argparse
ap=argparse.ArgumentParser(); ap.add_argument("--unroll",type=int,default=4)
ap.add_argument("-o",default="-"); a=ap.parse_args()
U=a.unroll; K=256
A0,A1,B=0,16384,32768; C=98304
L=[];E=L.append
E(f"; inner-loop isolation, unroll={U}, K={K}, no DMA")
E(".args 6")
E("  arg r3, 3        ; reps")
E("  movi r6, 0")
E("outer:")
for t in range(4): E(f"  zerza za{t}")
E(f"  movi r10, {A0}")
E(f"  movi r11, {A1}")
E(f"  movi r12, {B}")
E(f"  movi r13, {A0+K*64}")
E("k:")
for u in range(U):
    v=4*u
    E(f"  ldv  v{v},   spm[{u*64}+r10]")
    E(f"  ldv  v{v+1}, spm[{u*64}+r11]")
    E(f"  ldv  v{v+2}, spm[{u*128}+r12]")
    E(f"  ldv  v{v+3}, spm[{u*128+64}+r12]")
for u in range(U):
    v=4*u
    E(f"  mopa za0, v{v},   v{v+2}")
    E(f"  mopa za1, v{v},   v{v+3}")
    E(f"  mopa za2, v{v+1}, v{v+2}")
    E(f"  mopa za3, v{v+1}, v{v+3}")
E(f"  addi r10, r10, {64*U}")
E(f"  addi r11, r11, {64*U}")
E(f"  addi r12, r12, {128*U}")
E("  cmplt r15, r10, r13")
E("  bnz  r15, k")
E(f"  stza spm[{C}], za0, 128")
E("  addi r6, r6, 1")
E("  cmplt r15, r6, r3")
E("  bnz  r15, outer")
E("  halt")
s="\n".join(L)+"\n"
sys.stdout.write(s) if a.o=="-" else open(a.o,"w").write(s)
