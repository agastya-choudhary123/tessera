#define _GNU_SOURCE
#include "tessera.h"
#include "dma.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sys/qos.h>

void ts_dma_start(void); void ts_dma_stop(void);
uint64_t ts_dma_bytes(void); uint64_t ts_dma_busy_ns(void); uint64_t ts_dma_ndesc(void); void ts_dma_reset_stats(void);
int  ts_jit_available(void);
int  ts_jit_run(ts_vm *vm, const ts_prog *p, int verbose);

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

static void ref_matmul(const float*A,const float*B,float*C,int M,int N,int K){
    for(int i=0;i<M;i++) for(int j=0;j<N;j++){ double s=0;
        for(int k=0;k<K;k++) s+=(double)A[(long)i*K+k]*B[(long)k*N+j];
        C[(long)i*N+j]=(float)s; }
}
/* relative Frobenius norm -- the right metric for a matmul, since individual
 * dot products can land arbitrarily close to zero and blow up a pointwise
 * relative error even when every bit of the computation is correct. */
static double relfro(const float*x,const float*y,long n,double *maxabs){
    double num=0, den=0, ma=0;
    for(long i=0;i<n;i++){ double d=x[i]-y[i]; num+=d*d; den+=(double)y[i]*y[i];
        if(fabs(d)>ma) ma=fabs(d); }
    *maxabs=ma; return sqrt(num)/(sqrt(den)+1e-30);
}

int main(int argc,char**argv){
    const char *path=NULL; int M=64,N=64,K=64; int reps=1; int do_jit=0, verbose=0, check=0, dis=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-M")) M=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-N")) N=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-K")) K=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-r")) reps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--jit")) do_jit=1;
        else if(!strcmp(argv[i],"--check")) check=1;
        else if(!strcmp(argv[i],"--disasm")) dis=1;
        else if(!strcmp(argv[i],"-v")) verbose=1;
        else path=argv[i];
    }
    if(!path){ fprintf(stderr,"usage: tessera [-M m -N n -K k -r reps] [--jit] [--check] [--disasm] [-v] prog.tsa\n"); return 2; }

    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    char err[256]={0};
    ts_prog *p=ts_assemble_file(path,err,sizeof err);
    if(!p){ fprintf(stderr,"assemble: %s\n",err); return 1; }
    if(verbose) fprintf(stderr,"assembled %d instructions from %s\n",p->ncode,path);
    if(dis){ char b[160];
        for(int i=0;i<p->ncode;i++){ ts_disasm(p,i,b,sizeof b);
            printf("%4d  %-12s %s\n",i,p->labels[i]?p->labels[i]:"",b); } }
    if(ts_verify(p, verbose)){ fprintf(stderr,"verification FAILED\n"); return 1; }

    if(M%16||N%16){ fprintf(stderr,"M and N must be multiples of 16\n"); return 1; }

    float *A=aligned_alloc(16384,(size_t)M*K*4);
    float *B=aligned_alloc(16384,(size_t)K*N*4);
    float *C=aligned_alloc(16384,(size_t)M*N*4);
    float *R=check?malloc((size_t)M*N*4):NULL;
    srandom(1234);
    for(long i=0;i<(long)M*K;i++) A[i]=(float)((random()%2001)-1000)/1000.0f;
    for(long i=0;i<(long)K*N;i++) B[i]=(float)((random()%2001)-1000)/1000.0f;
    memset(C,0,(size_t)M*N*4);

    void *args[6]={A,B,C,(void*)(intptr_t)M,(void*)(intptr_t)N,(void*)(intptr_t)K};

    ts_dma_start();
    ts_vm *vm=ts_vm_new();
    vm->args=args; vm->nargs=6;

    int rc;
    const char *mode = do_jit?"jit":"interp";
    if(do_jit && !ts_jit_available()){ fprintf(stderr,"jit unavailable on this build\n"); return 1; }

    /* warm + correctness */
    rc = do_jit ? ts_jit_run(vm,p,verbose) : ts_run(vm,p);
    if(rc){ fprintf(stderr,"%s: execution failed\n",mode); return 1; }

    if(check){
        ref_matmul(A,B,R,M,N,K);
        double ma, e=relfro(C,R,(long)M*N,&ma);
        printf("check  : rel Frobenius err = %.3e   max abs err = %.3e   %s\n",
               e, ma, e<1e-5?"PASS":"FAIL");
        if(!(e<1e-5)) return 1;
    }

    double best=1e30; double busy_at_best=0; uint64_t bytes_at_best=0, ndesc_at_best=0;
    for(int r=0;r<reps;r++){
        memset(C,0,(size_t)M*N*4);
        ts_dma_reset_stats();
        double t0=now();
        rc = do_jit ? ts_jit_run(vm,p,0) : ts_run(vm,p);
        double dt=now()-t0;
        if(rc) return 1;
        if(dt<best){ best=dt; busy_at_best=ts_dma_busy_ns()*1e-9;
                     bytes_at_best=ts_dma_bytes(); ndesc_at_best=ts_dma_ndesc(); }
    }
    double flops=2.0*M*N*K;
    printf("%-6s : %6.3f ms   %8.2f GFLOP/s   M=%d N=%d K=%d\n", mode, best*1e3, flops/best/1e9, M,N,K);
    printf("  dma  : %4.1f/%d queues busy   %6.2f GB/s aggregate   %llu descriptors   %.1f MB\n",
           busy_at_best/best, TS_DMA_NQ, bytes_at_best/best/1e9,
           (unsigned long long)ndesc_at_best, bytes_at_best/1e6);
    if(verbose)
        fprintf(stderr,"  insn=%llu mopa=%llu dma=%llu dma_bytes=%llu wait_spins=%llu\n",
            (unsigned long long)vm->n_insn,(unsigned long long)vm->n_mopa,
            (unsigned long long)vm->n_dma,(unsigned long long)vm->dma_bytes,
            (unsigned long long)vm->n_wait_stall);

    ts_vm_free(vm); ts_dma_stop(); ts_prog_free(p);
    free(A);free(B);free(C);free(R);
    return 0;
}
