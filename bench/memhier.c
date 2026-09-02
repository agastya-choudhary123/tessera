#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sys/qos.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
int main(){
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  size_t sizes[] = {16*1024, 32*1024, 64*1024, 128*1024, 192*1024, 256*1024, 512*1024,
                    1<<20, 2<<20, 3<<20, 4<<20, 6<<20, 8<<20, 12<<20, 16<<20, 24<<20,
                    32<<20, 64<<20, 128<<20, 256<<20, 512<<20};
  printf("%12s  %10s  %s\n","footprint","GB/s","level");
  for(unsigned i=0;i<sizeof(sizes)/sizeof(*sizes);i++){
    size_t n=sizes[i]; char*b=aligned_alloc(16384,n); memset(b,1,n);
    long reps = (long)((2ULL<<30)/n); if(reps<3) reps=3;
    // streaming read with 8 independent ldp chains
    double t0=now(); volatile unsigned long sink=0;
    for(long r=0;r<reps;r++){
      unsigned long a0=0,a1=0,a2=0,a3=0;
      for(size_t o=0;o<n;o+=256){
        unsigned long *p=(unsigned long*)(b+o);
        a0+=p[0]+p[4]; a1+=p[8]+p[12]; a2+=p[16]+p[20]; a3+=p[24]+p[28];
      }
      sink+=a0+a1+a2+a3;
    }
    double dt=now()-t0; (void)sink;
    double gbs = (double)n*reps/dt/1e9;
    const char*lvl = n<=128*1024?"L1D":(n<=4u<<20?"L2":(n<=24u<<20?"SLC":"DRAM"));
    printf("%9zu KB  %10.1f  %s\n", n/1024, gbs, lvl);
    free(b);
  }
  return 0; }
