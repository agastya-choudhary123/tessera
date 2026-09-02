/* tessera trace JIT: bytecode -> native AArch64 + SME2.
 *
 * Region selection. The obvious choice -- compile the innermost hot loop --
 * is wrong on this machine, and measurably so: entering and leaving streaming
 * SVE mode costs ~12 ns (bench/smcost), while an inner k-loop at K=64 is only
 * ~24 ns of arithmetic. A per-trace smstart/smstop would spend a third of the
 * kernel toggling PSTATE.SM. So the JIT grows the region outward to the
 * outermost loop nest that is entirely compilable, and pays for streaming mode
 * exactly once per kernel invocation.
 *
 * That has a knock-on consequence which shapes the rest of this file: DMA
 * issue happens *inside* the region, so it must not leave streaming mode
 * either. Calling the C runtime would require smstop (the callee uses NEON,
 * illegal while streaming) so instead the JIT emits the descriptor-ring push
 * inline, as plain scalar stores plus a release-store to the head index. That
 * is the software form of a doorbell write, and it is legal in streaming mode
 * because streaming mode restricts SIMD state, not scalar state.
 */
#define _GNU_SOURCE
#include "tessera.h"
#include "dma.h"
#include "arm64.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <sys/mman.h>
#include <pthread.h>
#include <libkern/OSCacheControl.h>

/* ---- register map ----------------------------------------------------
 * ts r0..r7  -> x1..x8      ts r8..r15 -> x19..x26
 * x0  = ts_vm*      x27 = scratchpad base     x28 = &ts_dma
 * x9,x10,x11,x13,x14,x15 scratch    x12 = ZA slice select (w12)
 * x16,x17,x18 are reserved by the platform ABI and never touched.
 * ts v0..v31 map 1:1 onto z0..z31; ts za0..za3 map 1:1 onto ZA0.S..ZA3.S,
 * so no vector or accumulator traffic is introduced by compilation at all. */
#define R(i)     ((i)<8 ? 1+(i) : 19+((i)-8))
#define X_VM     0
#define X_SPM    27
#define X_DMA    28
#define X_S0     9
#define X_S1     10
#define X_S2     11
#define X_SL     12
#define X_S3     13
#define X_S4     14
#define X_S5     15

typedef struct {
    uint32_t *buf;      /* instruction words                          */
    int       n, cap;
    int      *natoff;   /* bytecode pc -> word index                  */
    struct { int at; int target_pc; int cond; } *fix;
    int       nfix, capfix;
    int       halt_fix[64], nhalt;
} emitter;

static void E(emitter *e, uint32_t w){
    if(e->n==e->cap){ e->cap=e->cap?e->cap*2:1024; e->buf=realloc(e->buf,e->cap*4); }
    e->buf[e->n++]=w;
}
static void fixup(emitter *e,int target_pc,int cond){
    if(e->nfix==e->capfix){ e->capfix=e->capfix?e->capfix*2:64;
        e->fix=realloc(e->fix,e->capfix*sizeof *e->fix); }
    e->fix[e->nfix].at=e->n; e->fix[e->nfix].target_pc=target_pc; e->fix[e->nfix].cond=cond;
    e->nfix++; E(e,0);
}
/* materialise an arbitrary 64-bit constant */
static void emit_imm(emitter *e,int rd,int64_t v){
    uint64_t u=(uint64_t)v;
    E(e,A_MOVZ(rd,(uint32_t)(u&0xffff),0));
    if((u>>16)&0xffff) E(e,A_MOVK(rd,(uint32_t)((u>>16)&0xffff),1));
    if((u>>32)&0xffff) E(e,A_MOVK(rd,(uint32_t)((u>>32)&0xffff),2));
    if((u>>48)&0xffff) E(e,A_MOVK(rd,(uint32_t)((u>>48)&0xffff),3));
}
/* rd = spm_base + imm + (idx? r[idx-1] : 0) */
static void emit_spm_addr(emitter *e,int rd,int32_t imm,int idx){
    if(imm>=0 && imm<4096) E(e,A_ADDI(rd,X_SPM,(uint32_t)imm));
    else { emit_imm(e,rd,imm); E(e,A_ADD(rd,X_SPM,rd)); }
    if(idx) E(e,A_ADD(rd,rd,R(idx-1)));
}

/* ---- inline DMA descriptor push (the doorbell write) ------------------ */
#define OFF_ISSUED  ((uint32_t)offsetof(ts_dma_engine,issued))
#define OFF_DONE    ((uint32_t)offsetof(ts_dma_engine,done))
#define OFF_BYTES   ((uint32_t)offsetof(ts_dma_engine,bytes))
/* byte offset of the queue a tag is statically bound to */
#define OFF_Q(tag)  ((uint32_t)(offsetof(ts_dma_engine,q) + (size_t)TS_Q_OF(tag)*sizeof(ts_dma_q)))
#define Q_HEAD 0u
#define Q_TAIL 8u
#define Q_RING ((uint32_t)offsetof(ts_dma_q,ring))

static void emit_dma(emitter *e,const ts_insn *in,int is_in){
    uint32_t rows, row_bytes, dst_stride;
    if(is_in){
        rows       = (in->imm1==DMA_PACK16) ? TS_ZADIM : in->d;
        row_bytes  = in->e;
        dst_stride = in->e;
    } else { rows=in->d; row_bytes=in->e; dst_stride=0; }

    /* x15 = &queue[tag % NQ]; the queue is chosen at compile time, so a
     * transfer costs no more than it did when there was only one queue. */
    emit_imm(e,X_S5,OFF_Q(in->c));
    E(e,A_ADD(X_S5,X_DMA,X_S5));

    E(e,A_LDR(X_S0,X_S5,Q_HEAD));              /* single producer: relaxed  */
    int spin=e->n;
    E(e,A_ADDI(X_S1,X_S5,Q_TAIL));
    E(e,A_LDAR(X_S1,X_S1));
    E(e,A_SUB(X_S2,X_S0,X_S1));
    E(e,A_CMPI(X_S2,TS_DMA_RING));
    int br=e->n; E(e,0);
    E(e,A_YIELD());
    E(e,A_B((spin-e->n)*4));
    e->buf[br]=A_BCOND(CC_LO,(e->n-br)*4);
    /* x11 = &ring[head & (RING-1)] */
    E(e,A_UBFX(X_S2,X_S0,0,6));                /* RING == 64                */
    E(e,A_LSLI(X_S2,X_S2,6));                  /* * sizeof(descriptor)      */
    E(e,A_ADD(X_S2,X_S5,X_S2));
    E(e,A_ADDI(X_S2,X_S2,Q_RING));
    if(is_in){
        emit_spm_addr(e,X_S3,in->imm0,0);
        E(e,A_STR(X_S3,X_S2,0));
        E(e,A_STR(R(in->a),X_S2,8));
    } else {
        E(e,A_STR(R(in->a),X_S2,0));
        emit_spm_addr(e,X_S3,in->imm0,0);
        E(e,A_STR(X_S3,X_S2,8));
    }
    emit_imm(e,X_S3,(int64_t)((uint64_t)rows | ((uint64_t)row_bytes<<32)));
    E(e,A_STR(X_S3,X_S2,16));
    if(is_in){
        E(e,A_STR(R(in->b),X_S2,24));
        emit_imm(e,X_S3,dst_stride); E(e,A_STR(X_S3,X_S2,32));
    } else {
        emit_imm(e,X_S3,row_bytes);  E(e,A_STR(X_S3,X_S2,24));
        E(e,A_STR(R(in->b),X_S2,32));
    }
    emit_imm(e,X_S3,(int64_t)((uint64_t)(uint32_t)(is_in?in->imm1:
                                   (in->imm1==DMA_ACC?DMA_ACC:DMA_PLAIN))
                              | ((uint64_t)(uint32_t)in->c<<32)));
    E(e,A_STR(X_S3,X_S2,40));
    /* issued[tag]++ and bytes += ... : producer-owned, so plain RMW is sound */
    E(e,A_LDR(X_S3,X_DMA,OFF_ISSUED+8u*in->c));
    E(e,A_ADDI(X_S3,X_S3,1));
    E(e,A_STR(X_S3,X_DMA,OFF_ISSUED+8u*in->c));
    E(e,A_LDR(X_S3,X_DMA,OFF_BYTES));
    emit_imm(e,X_S4,(int64_t)rows*row_bytes);
    E(e,A_ADD(X_S3,X_S3,X_S4));
    E(e,A_STR(X_S3,X_DMA,OFF_BYTES));
    /* doorbell: release-store the new head */
    E(e,A_ADDI(X_S0,X_S0,1));
    E(e,A_ADDI(X_S1,X_S5,Q_HEAD));
    E(e,A_STLR(X_S0,X_S1));
}

static void emit_wait(emitter *e,uint32_t mask){
    for(int t=0;t<TS_NTAG;t++){
        if(!(mask&(1u<<t))) continue;
        E(e,A_LDR(X_S0,X_DMA,OFF_ISSUED+8u*t));      /* want */
        int spin=e->n;
        E(e,A_ADDI(X_S1,X_DMA,OFF_DONE+8u*t));
        E(e,A_LDAR(X_S1,X_S1));
        E(e,A_CMP(X_S1,X_S0));
        int br=e->n; E(e,0);                          /* b.hs -> done */
        E(e,A_YIELD());
        E(e,A_B((spin-e->n)*4));
        e->buf[br]=A_BCOND(CC_HS,(e->n-br)*4);
    }
}

/* ZA tile <-> scratchpad, one 16-lane row at a time (slice index = w12) */
static void emit_za_mem(emitter *e,const ts_insn *in,int store){
    int32_t stride = in->imm1 ? in->imm1 : TS_ZADIM*4;
    emit_spm_addr(e,X_S4,in->imm0,in->b);
    for(int row=0;row<TS_ZADIM;row++){
        E(e,A_MOVZW(X_SL,(uint32_t)row));
        E(e,store ? A_ST1W_ZA(in->a,0,0,X_S4) : A_LD1W_ZA(in->a,0,0,X_S4));
        if(row!=TS_ZADIM-1){
            if(stride<4096) E(e,A_ADDI(X_S4,X_S4,(uint32_t)stride));
            else { emit_imm(e,X_S5,stride); E(e,A_ADD(X_S4,X_S4,X_S5)); }
        }
    }
}

/* ---- the compiler ----------------------------------------------------- */
typedef struct { void *code; size_t size; void (*fn)(ts_vm*); } ts_jit_fn;
static ts_jit_fn g_fn;

static int compile(const ts_prog *p, emitter *e){
    e->natoff=calloc(p->ncode,sizeof(int));
    /* prologue: save callee-saved GPRs, then the callee-saved halves of
     * v8..v15 -- smstart zeroes every Z register, and the low 64 bits of
     * v8-v15 belong to our caller. Saving them BEFORE entering streaming
     * mode matters; a NEON store is not legal once PSTATE.SM is set. */
    E(e,A_STP_PRE(29,30,31,-16));
    E(e,A_STP_PRE(19,20,31,-16));
    E(e,A_STP_PRE(21,22,31,-16));
    E(e,A_STP_PRE(23,24,31,-16));
    E(e,A_STP_PRE(25,26,31,-16));
    E(e,A_STP_PRE(27,28,31,-16));
    E(e,A_SUBI(31,31,64));
    for(int d=8;d<16;d++) E(e,A_STR_D(d,31,(uint32_t)(d-8)*8));

    E(e,A_LDR(X_SPM,X_VM,(uint32_t)offsetof(ts_vm,spm)));
    emit_imm(e,X_DMA,(int64_t)(intptr_t)&ts_dma);
    for(int i=0;i<TS_NREG;i++)
        E(e,A_LDR(R(i),X_VM,(uint32_t)(offsetof(ts_vm,r)+8u*i)));
    E(e,A_SMSTART());
    E(e,A_PTRUE_B(0));

    for(int pc=0; pc<p->ncode; pc++){
        e->natoff[pc]=e->n;
        const ts_insn *in=&p->code[pc];
        switch(in->op){
        case OP_HALT:
            e->halt_fix[e->nhalt++]=e->n; E(e,0); break;
        case OP_MOVI: emit_imm(e,R(in->a),in->imm0); break;
        case OP_ARG:
            E(e,A_LDR(X_S0,X_VM,(uint32_t)offsetof(ts_vm,args)));
            E(e,A_LDR(R(in->a),X_S0,(uint32_t)(8*in->imm0))); break;
        case OP_ADDI:
            if(in->imm0>=0 && in->imm0<4096) E(e,A_ADDI(R(in->a),R(in->b),(uint32_t)in->imm0));
            else if(in->imm0<0 && -in->imm0<4096) E(e,A_SUBI(R(in->a),R(in->b),(uint32_t)(-in->imm0)));
            else { emit_imm(e,X_S0,in->imm0); E(e,A_ADD(R(in->a),R(in->b),X_S0)); } break;
        case OP_ADD:   E(e,A_ADD(R(in->a),R(in->b),R(in->c))); break;
        case OP_SUB:   E(e,A_SUB(R(in->a),R(in->b),R(in->c))); break;
        case OP_MUL:   E(e,A_MUL(R(in->a),R(in->b),R(in->c))); break;
        case OP_MULI:  emit_imm(e,X_S0,in->imm0); E(e,A_MUL(R(in->a),R(in->b),X_S0)); break;
        case OP_SHLI:  E(e,A_LSLI(R(in->a),R(in->b),in->imm0)); break;
        case OP_SHRI:  E(e,A_LSRI(R(in->a),R(in->b),in->imm0)); break;
        case OP_CMPLT: E(e,A_CMP(R(in->b),R(in->c))); E(e,A_CSET(R(in->a),CC_LT)); break;
        case OP_BNZ:   E(e,A_CMPI(R(in->a),0)); fixup(e,in->imm0,CC_NE); break;
        case OP_BZ:    E(e,A_CMPI(R(in->a),0)); fixup(e,in->imm0,CC_EQ); break;
        case OP_JMP:   fixup(e,in->imm0,-1); break;

        case OP_DMAIN:  emit_dma(e,in,1); break;
        case OP_DMAOUT: emit_dma(e,in,0); break;
        case OP_WAIT:   emit_wait(e,(uint32_t)in->imm0); break;

        case OP_LDV: case OP_STV: {
            /* Fast path: SVE ldr/str take a VL-scaled 9-bit signed displacement,
             * so a scratchpad access with a small aligned offset costs one
             * address add and nothing else. This is where most of the JIT's
             * advantage over the interpreter's addressing actually comes from. */
            int ld = (in->op==OP_LDV);
            if(in->imm0 % 64 == 0 && A_ZIMM_OK(in->imm0/64)){
                int base;
                if(in->b){ E(e,A_ADD(X_S0,X_SPM,R(in->b-1))); base=X_S0; }
                else base=X_SPM;
                E(e, ld ? A_LDR_Z_I(in->a,base,in->imm0/64)
                        : A_STR_Z_I(in->a,base,in->imm0/64));
            } else {
                emit_spm_addr(e,X_S0,in->imm0,in->b);
                E(e, ld ? A_LDR_Z(in->a,X_S0) : A_STR_Z(in->a,X_S0));
            }
        } break;
        case OP_LDZA: emit_za_mem(e,in,0); break;
        case OP_STZA: emit_za_mem(e,in,1); break;
        case OP_ZERZA: E(e,A_ZERO_ZA_S(in->a)); break;
        case OP_MOPA:  E(e,A_FMOPA_S(in->a,in->b,in->c,0,0)); break;
        case OP_VADD:  E(e,A_FADD_S(in->a,in->b,in->c)); break;
        case OP_VMUL:  E(e,A_FMUL_S(in->a,in->b,in->c)); break;
        case OP_VFMA:  E(e,A_FMLA_S(in->a,in->b,in->c,0)); break;
        case OP_VMAX:
            if(in->a!=in->b) E(e,A_MOV_Z(in->a,in->b));
            E(e,A_FMAX_S(in->a,in->c,0)); break;
        case OP_VZERO: E(e,A_DUP_S_IMM0(in->a)); break;
        case OP_VDUP:  /* broadcast a f32 bit pattern via the scratchpad tail */
            emit_imm(e,X_S0,(uint32_t)in->imm0);
            E(e,A_ADDI(X_S1,X_SPM,0));
            emit_imm(e,X_S2,TS_SPM_SIZE-64);
            E(e,A_ADD(X_S1,X_S1,X_S2));
            E(e,A_STR(X_S0,X_S1,0));
            E(e,A_LDR_Z(in->a,X_S1));
            fprintf(stderr,"jit: vdup lowering is scalar-broadcast-free; unsupported\n");
            return -1;
        default:
            fprintf(stderr,"jit: no lowering for opcode %d at pc=%d\n",in->op,pc);
            return -1;
        }
    }
    /* epilogue */
    int epi=e->n;
    for(int i=0;i<e->nhalt;i++) e->buf[e->halt_fix[i]]=A_B((epi-e->halt_fix[i])*4);
    E(e,A_SMSTOP());
    for(int i=0;i<TS_NREG;i++)
        E(e,A_STR(R(i),X_VM,(uint32_t)(offsetof(ts_vm,r)+8u*i)));
    for(int d=8;d<16;d++) E(e,A_LDR_D(d,31,(uint32_t)(d-8)*8));
    E(e,A_ADDI(31,31,64));
    E(e,A_LDP_POST(27,28,31,16));
    E(e,A_LDP_POST(25,26,31,16));
    E(e,A_LDP_POST(23,24,31,16));
    E(e,A_LDP_POST(21,22,31,16));
    E(e,A_LDP_POST(19,20,31,16));
    E(e,A_LDP_POST(29,30,31,16));
    E(e,A_RET());
    /* branch fixups */
    for(int i=0;i<e->nfix;i++){
        int at=e->fix[i].at, tgt=e->natoff[e->fix[i].target_pc];
        e->buf[at] = e->fix[i].cond<0 ? A_B((tgt-at)*4) : A_BCOND(e->fix[i].cond,(tgt-at)*4);
    }
    return 0;
}

int ts_jit_available(void){ return 1; }

static const ts_prog *g_compiled_for;

int ts_jit_run(ts_vm *vm, const ts_prog *p, int verbose){
    if(g_compiled_for != p){
        emitter e; memset(&e,0,sizeof e);
        if(compile(p,&e)){ free(e.buf); free(e.natoff); free(e.fix); return -1; }
        size_t sz=(size_t)e.n*4, pages=(sz+16383)&~(size_t)16383;
        void *m=mmap(NULL,pages,PROT_READ|PROT_WRITE|PROT_EXEC,
                     MAP_PRIVATE|MAP_ANON|MAP_JIT,-1,0);
        if(m==MAP_FAILED){ perror("mmap MAP_JIT"); return -1; }
        pthread_jit_write_protect_np(0);          /* W^X: writable side      */
        memcpy(m,e.buf,sz);
        pthread_jit_write_protect_np(1);          /* W^X: executable side    */
        sys_icache_invalidate(m,sz);
        if(g_fn.code) munmap(g_fn.code,g_fn.size);
        g_fn.code=m; g_fn.size=pages; g_fn.fn=(void(*)(ts_vm*))m;
        g_compiled_for=p;
        if(verbose) fprintf(stderr,"jit: compiled %d bytecode insns -> %d native words (%zu B)\n",
                            p->ncode,e.n,sz);
        const char *dump=getenv("TESSERA_JIT_DUMP");
        if(dump){ FILE*f=fopen(dump,"wb"); if(f){ fwrite(e.buf,4,e.n,f); fclose(f);
            fprintf(stderr,"jit: raw code written to %s (%d words)\n",dump,e.n); } }
        free(e.buf); free(e.natoff); free(e.fix);
    }
    g_fn.fn(vm);
    return 0;
}
