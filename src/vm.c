/* tessera reference interpreter.
 *
 * Deliberately simple and obviously correct: it is the oracle the JIT is
 * checked against. Every SPM access is bounds-checked here even though the
 * verifier already proved the static ones, because the oracle should not
 * trust the pass under test.
 */
#define _GNU_SOURCE
#include "tessera.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

void     ts_dma_start(void);
void     ts_dma_stop(void);
void     ts_dma_issue(void*,const void*,uint32_t,uint32_t,int64_t,int64_t,int,int);
uint64_t ts_dma_wait(uint32_t);

ts_vm *ts_vm_new(void){
    ts_vm *vm=calloc(1,sizeof *vm);
    /* the scratchpad is a real, page-aligned, permanently-resident buffer.
     * 16K-aligned so it maps cleanly onto Apple's 16 KB pages. */
    vm->spm = mmap(NULL, TS_SPM_SIZE, PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANON, -1, 0);
    if(vm->spm==MAP_FAILED){ free(vm); return NULL; }
    memset(vm->spm, 0, TS_SPM_SIZE);   /* fault it in once, up front */
    return vm;
}
void ts_vm_free(ts_vm *vm){ if(!vm) return; munmap(vm->spm, TS_SPM_SIZE); free(vm); }

#define BOUND(off,len) do{ \
    if((off)<0 || (long)(off)+(long)(len) > TS_SPM_SIZE){ \
        fprintf(stderr,"vm: SPM out of range at pc=%d: [%ld,%ld)\n",pc,(long)(off),(long)(off)+(long)(len)); \
        return -1; } }while(0)

int ts_run(ts_vm *vm, const ts_prog *p){
    int pc=0;
    for(int i=0;i<vm->nargs && i<TS_NREG;i++) ;
    while(pc>=0 && pc<p->ncode){
        const ts_insn *in=&p->code[pc];
        vm->n_insn++;
        int next=pc+1;
        switch(in->op){
        case OP_HALT: return 0;
        case OP_MOVI:  vm->r[in->a]=in->imm0; break;
        case OP_ARG:
            if(in->imm0<0 || in->imm0>=vm->nargs){
                fprintf(stderr,"vm: arg %d out of range (nargs=%d)\n",in->imm0,vm->nargs); return -1; }
            vm->r[in->a]=(int64_t)(intptr_t)vm->args[in->imm0]; break;
        case OP_ADDI:  vm->r[in->a]=vm->r[in->b]+in->imm0; break;
        case OP_ADD:   vm->r[in->a]=vm->r[in->b]+vm->r[in->c]; break;
        case OP_SUB:   vm->r[in->a]=vm->r[in->b]-vm->r[in->c]; break;
        case OP_MUL:   vm->r[in->a]=vm->r[in->b]*vm->r[in->c]; break;
        case OP_MULI:  vm->r[in->a]=vm->r[in->b]*in->imm0; break;
        case OP_SHLI:  vm->r[in->a]=vm->r[in->b]<<in->imm0; break;
        case OP_SHRI:  vm->r[in->a]=(int64_t)((uint64_t)vm->r[in->b]>>in->imm0); break;
        case OP_CMPLT: vm->r[in->a]=(vm->r[in->b]<vm->r[in->c]); break;
        case OP_BNZ:   if(vm->r[in->a]) next=in->imm0; break;
        case OP_BZ:    if(!vm->r[in->a]) next=in->imm0; break;
        case OP_JMP:   next=in->imm0; break;

        case OP_DMAIN: {
            uint32_t rows = (in->imm1==DMA_PACK16) ? TS_ZADIM : in->d;
            int32_t  dstb = (in->imm1==DMA_PACK16) ? (int32_t)in->e*TS_ZADIM
                                                   : (int32_t)in->d*(int32_t)in->e;
            BOUND(in->imm0, dstb);
            ts_dma_issue(vm->spm+in->imm0, (const void*)(intptr_t)vm->r[in->a],
                         rows, in->e, vm->r[in->b], in->e, in->imm1, in->c);
            vm->n_dma++; vm->dma_bytes += (uint64_t)rows*in->e;
        } break;
        case OP_DMAOUT: {
            BOUND(in->imm0, (long)in->d*in->e);
            ts_dma_issue((void*)(intptr_t)vm->r[in->a], vm->spm+in->imm0,
                         in->d, in->e, in->e, vm->r[in->b],
                         in->imm1==DMA_ACC?DMA_ACC:DMA_PLAIN, in->c);
            vm->n_dma++; vm->dma_bytes += (uint64_t)in->d*in->e;
        } break;
        case OP_WAIT: vm->n_wait_stall += ts_dma_wait((uint32_t)in->imm0); break;

        case OP_LDV: { long o=in->imm0+(in->b?vm->r[in->b-1]:0); BOUND(o,64);
                       memcpy(vm->v[in->a], vm->spm+o, 64); } break;
        case OP_STV: { long o=in->imm0+(in->b?vm->r[in->b-1]:0); BOUND(o,64);
                       memcpy(vm->spm+o, vm->v[in->a], 64); } break;
        case OP_LDZA:{ long o=in->imm0+(in->b?vm->r[in->b-1]:0); long st=in->imm1?in->imm1:TS_ZADIM*4;
                       BOUND(o, st*(TS_ZADIM-1)+TS_ZADIM*4);
                       for(int r=0;r<TS_ZADIM;r++) memcpy(vm->za[in->a][r], vm->spm+o+r*st, TS_ZADIM*4);
                     } break;
        case OP_STZA:{ long o=in->imm0+(in->b?vm->r[in->b-1]:0); long st=in->imm1?in->imm1:TS_ZADIM*4;
                       BOUND(o, st*(TS_ZADIM-1)+TS_ZADIM*4);
                       for(int r=0;r<TS_ZADIM;r++) memcpy(vm->spm+o+r*st, vm->za[in->a][r], TS_ZADIM*4);
                     } break;
        case OP_ZERZA: memset(vm->za[in->a],0,sizeof vm->za[0]); break;

        case OP_MOPA: {   /* ZA[i][j] += va[i] * vb[j] -- rank-1 update */
            const float *x=vm->v[in->b], *y=vm->v[in->c];
            float (*z)[TS_ZADIM]=vm->za[in->a];
            for(int i=0;i<TS_ZADIM;i++){ float xi=x[i];
                for(int j=0;j<TS_ZADIM;j++) z[i][j]+=xi*y[j]; }
            vm->n_mopa++;
        } break;
        case OP_VADD: for(int l=0;l<TS_VLANES;l++) vm->v[in->a][l]=vm->v[in->b][l]+vm->v[in->c][l]; break;
        case OP_VMUL: for(int l=0;l<TS_VLANES;l++) vm->v[in->a][l]=vm->v[in->b][l]*vm->v[in->c][l]; break;
        case OP_VFMA: for(int l=0;l<TS_VLANES;l++) vm->v[in->a][l]+=vm->v[in->b][l]*vm->v[in->c][l]; break;
        case OP_VMAX: for(int l=0;l<TS_VLANES;l++){ float x=vm->v[in->b][l],y=vm->v[in->c][l];
                          vm->v[in->a][l]= x>y?x:y; } break;
        case OP_VZERO: memset(vm->v[in->a],0,64); break;
        case OP_VDUP: { float f; uint32_t b=(uint32_t)in->imm0; memcpy(&f,&b,4);
                        for(int l=0;l<TS_VLANES;l++) vm->v[in->a][l]=f; } break;
        default:
            fprintf(stderr,"vm: unimplemented opcode %d at pc=%d\n",in->op,pc); return -1;
        }
        pc=next;
    }
    return 0;
}
