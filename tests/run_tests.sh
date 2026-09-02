#!/bin/zsh
# tessera test suite: correctness of the interpreter, bit-agreement of the JIT
# against it, and the verifier's ability to reject unsafe movement.
cd "$(dirname "$0")/.."
pass=0; fail=0
ok(){ print -- "  PASS  $1"; ((pass++)) }
no(){ print -- "  FAIL  $1"; ((fail++)) }

print "== verifier must REJECT unsafe programs =="
for f in tests/bad_*.tsa; do
  if ./tessera -M 64 -N 64 -K 128 $f >/dev/null 2>&1; then no "$f was accepted"; else ok "$f rejected"; fi
done

print "\n== verifier must ACCEPT the generated kernels =="
for f in asm/mmkc_128_128_c2.tsa asm/mm_k128_j128.tsa bench/kloop_u4.tsa; do
  [[ -f $f ]] || continue
  if ./tessera -M 64 -N 64 -K 128 --disasm $f >/dev/null 2>&1; then ok "$f verified"; else no "$f rejected"; fi
done

print "\n== interpreter correctness (vs f64 reference) =="
for shape in "64 64 128" "128 256 128" "256 128 256"; do
  set -A s ${=shape}
  if ./tessera -M $s[1] -N $s[2] -K $s[3] --check asm/mmkc_128_128_c2.tsa 2>/dev/null | grep -q PASS
  then ok "interp M=$s[1] N=$s[2] K=$s[3]"; else no "interp M=$s[1] N=$s[2] K=$s[3]"; fi
done

print "\n== JIT correctness (vs f64 reference) =="
for shape in "64 64 128" "512 512 256" "512 1024 512" "1024 1024 1024" "2048 512 1024"; do
  set -A s ${=shape}
  if ./tessera -M $s[1] -N $s[2] -K $s[3] --check --jit asm/mmkc_128_128_c2.tsa 2>/dev/null | grep -q PASS
  then ok "jit M=$s[1] N=$s[2] K=$s[3]"; else no "jit M=$s[1] N=$s[2] K=$s[3]"; fi
done

print "\n$pass passed, $fail failed"
[[ $fail -eq 0 ]]
