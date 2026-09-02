/* Baseline: Apple Accelerate sgemm, same shapes, same machine, same timing
 * method (best of N). Accelerate on M4 uses the AMX/SME coprocessor through
 * Apple's own hand-tuned kernels, so this is the strongest baseline available
 * here -- not a strawman. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sys/qos.h>
#include <Accelerate/Accelerate.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(int argc,char**argv){
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
    int M=1024,N=1024,K=128,reps=5;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-M"))M=atoi(argv[++i]); else if(!strcmp(argv[i],"-N"))N=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-K"))K=atoi(argv[++i]); else if(!strcmp(argv[i],"-r"))reps=atoi(argv[++i]);
    }
    float*A=aligned_alloc(16384,(size_t)M*K*4),*B=aligned_alloc(16384,(size_t)K*N*4),*C=aligned_alloc(16384,(size_t)M*N*4);
    srandom(1234);
    for(long i=0;i<(long)M*K;i++)A[i]=(float)((random()%2001)-1000)/1000.0f;
    for(long i=0;i<(long)K*N;i++)B[i]=(float)((random()%2001)-1000)/1000.0f;
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.0f,A,K,B,N,0.0f,C,N);
    double best=1e30;
    for(int r=0;r<reps;r++){ double t0=now();
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.0f,A,K,B,N,0.0f,C,N);
        double dt=now()-t0; if(dt<best)best=dt; }
    printf("accel  : %6.3f ms   %8.2f GFLOP/s   M=%d N=%d K=%d\n",best*1e3,2.0*M*N*K/best/1e9,M,N,K);
    return 0;
}
