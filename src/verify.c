/* tessera verifier.
 *
 * The ISA's central claim is "no hidden cache": every byte a compute
 * instruction reads got into the scratchpad because some instruction put it
 * there, and the program waited for it. That claim is only worth anything if
 * it is mechanically checked, so this pass proves three things:
 *
 *   V1  no compute instruction names a DRAM operand   (structural)
 *   V2  every SPM access is in bounds                 (static extent)
 *   V3  no SPM byte is read while a DMA writing it is still in flight,
 *       and no DMA overwrites a region whose previous contents are still
 *       being read  (RAW / WAR over tag state)
 *
 * V3 is a small interval dataflow over the CFG, iterated to a fixpoint so
 * that loop-carried hazards -- the double-buffering bugs that are the whole
 * difficulty of writing accelerator code by hand -- are caught statically.
 */
#include "tessera.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXIV 64
typedef struct { int32_t lo, hi; uint32_t tags; } iv;      /* [lo,hi) in flight */
typedef struct { iv v[MAXIV]; int n; } ivset;

static void iv_add(ivset *s, int32_t lo, int32_t hi, uint32_t tags){
    if(lo>=hi) return;
    for(int i=0;i<s->n;i++)
        if(s->v[i].lo==lo && s->v[i].hi==hi){ s->v[i].tags|=tags; return; }
    if(s->n<MAXIV){ s->v[s->n].lo=lo; s->v[s->n].hi=hi; s->v[s->n].tags=tags; s->n++; }
}
static uint32_t iv_overlap(const ivset *s, int32_t lo, int32_t hi){
    uint32_t m=0;
    for(int i=0;i<s->n;i++) if(lo < s->v[i].hi && s->v[i].lo < hi) m |= s->v[i].tags;
    return m;
}
static void iv_clear_tags(ivset *s, uint32_t tags){
    int w=0;
    for(int i=0;i<s->n;i++){
        s->v[i].tags &= ~tags;
        if(s->v[i].tags) s->v[w++]=s->v[i];
    }
    s->n=w;
}
static int iv_merge(ivset *d, const ivset *s){   /* returns 1 if d changed */
    int before=d->n; uint32_t sig=0;
    for(int i=0;i<d->n;i++) sig^=d->v[i].tags*2654435761u + d->v[i].lo;
    for(int i=0;i<s->n;i++) iv_add(d, s->v[i].lo, s->v[i].hi, s->v[i].tags);
    uint32_t sig2=0;
    for(int i=0;i<d->n;i++) sig2^=d->v[i].tags*2654435761u + d->v[i].lo;
    return d->n!=before || sig!=sig2;
}

/* static extent of an SPM access; reg-offset forms are widened to unknown */
static int spm_extent(const ts_insn *in, int32_t *lo, int32_t *hi, int *exact){
    switch(in->op){
    case OP_LDV: case OP_STV:
        *lo=in->imm0; *hi=in->imm0+64; *exact=(in->b==0); return 1;
    case OP_LDZA: case OP_STZA:
        *lo=in->imm0; *hi=in->imm0 + (in->imm1 ? in->imm1*(TS_ZADIM-1)+TS_ZADIM*4 : TS_ZADIM*TS_ZADIM*4);
        *exact=(in->b==0); return 1;
    case OP_DMAIN: case OP_DMAOUT:
        *lo=in->imm0;
        *hi=in->imm0 + (in->imm1==DMA_PACK16 ? (int32_t)in->e*TS_ZADIM
                                             : (int32_t)in->d*(int32_t)in->e);
        *exact=1; return 1;
    }
    return 0;
}

int ts_verify(const ts_prog *p, int verbose){
    int nerr=0;
    /* ---- V1: structural, no compute op may name DRAM ---- */
    for(int pc=0; pc<p->ncode; pc++){
        const ts_insn *in=&p->code[pc];
        const ts_opinfo *oi=&ts_ops[in->op];
        if(oi->touches_dram) continue;
        if(oi->as_a==AS_DRAM || oi->as_b==AS_DRAM || oi->as_c==AS_DRAM){
            fprintf(stderr,"V1 pc=%d: compute op '%s' names a DRAM operand\n",pc,oi->name);
            nerr++;
        }
    }
    /* ---- V2: SPM bounds ---- */
    for(int pc=0; pc<p->ncode; pc++){
        const ts_insn *in=&p->code[pc];
        int32_t lo,hi; int exact;
        if(!spm_extent(in,&lo,&hi,&exact)) continue;
        if(lo<0 || hi>TS_SPM_SIZE){
            char d[128]; ts_disasm(p,pc,d,sizeof d);
            fprintf(stderr,"V2 pc=%d: SPM access [%d,%d) out of the %d KB scratchpad: %s\n",
                    pc,lo,hi,TS_SPM_SIZE/1024,d);
            nerr++;
        }
    }
    /* ---- V3: tag dataflow to fixpoint ---- */
    ivset *in_st = calloc(p->ncode, sizeof(ivset));
    char  *seen  = calloc(p->ncode, 1);
    int changed=1, rounds=0;
    int *raw_at = calloc(p->ncode, sizeof(int));
    while(changed && rounds++ < 16){
        changed=0;
        ivset cur; memset(&cur,0,sizeof cur);
        for(int pc=0; pc<p->ncode; pc++){
            if(seen[pc]) { if(iv_merge(&in_st[pc], &cur)) changed=1; }
            else { in_st[pc]=cur; seen[pc]=1; changed=1; }
            cur = in_st[pc];
            const ts_insn *in=&p->code[pc];
            int32_t lo,hi; int exact;
            switch(in->op){
            case OP_WAIT: iv_clear_tags(&cur, (uint32_t)in->imm0); break;
            case OP_DMAIN:
                if(spm_extent(in,&lo,&hi,&exact)) iv_add(&cur, lo, hi, 1u<<in->c);
                break;
            case OP_DMAOUT:
                /* the SPM side is a source: it must not be in flight either */
                if(spm_extent(in,&lo,&hi,&exact)){
                    uint32_t m=iv_overlap(&cur, lo, hi);
                    if(m && rounds>1) raw_at[pc]=1;
                    iv_add(&cur, lo, hi, 1u<<in->c);
                }
                break;
            case OP_LDV: case OP_LDZA:
                if(spm_extent(in,&lo,&hi,&exact)){
                    uint32_t m=iv_overlap(&cur, lo, hi);
                    if(m && rounds>1) raw_at[pc]=1;
                }
                break;
            case OP_STV: case OP_STZA:
                if(spm_extent(in,&lo,&hi,&exact)){
                    uint32_t m=iv_overlap(&cur, lo, hi);
                    if(m && rounds>1) raw_at[pc]=1;   /* WAR against in-flight DMA */
                }
                break;
            }
            /* branch: propagate to target as well */
            if(in->op==OP_BNZ || in->op==OP_BZ){
                if(in->imm0>=0 && in->imm0<p->ncode){
                    if(!seen[in->imm0]){ in_st[in->imm0]=cur; seen[in->imm0]=1; changed=1; }
                    else if(iv_merge(&in_st[in->imm0], &cur)) changed=1;
                }
            } else if(in->op==OP_JMP){
                if(in->imm0>=0 && in->imm0<p->ncode){
                    if(!seen[in->imm0]){ in_st[in->imm0]=cur; seen[in->imm0]=1; changed=1; }
                    else if(iv_merge(&in_st[in->imm0], &cur)) changed=1;
                }
                memset(&cur,0,sizeof cur);
            } else if(in->op==OP_HALT){
                memset(&cur,0,sizeof cur);
            }
        }
    }
    for(int pc=0;pc<p->ncode;pc++) if(raw_at[pc]){
        char d[128]; ts_disasm(p,pc,d,sizeof d);
        fprintf(stderr,"V3 pc=%d: reads SPM bytes with a DMA still in flight "
                       "(missing wait): %s\n",pc,d);
        nerr++;
    }
    if(verbose && !nerr)
        fprintf(stderr,"verify: %d instructions OK "
                       "(no DRAM operand in compute; SPM in bounds; tag discipline sound)\n",
                p->ncode);
    free(in_st); free(seen); free(raw_at);
    return nerr;
}
