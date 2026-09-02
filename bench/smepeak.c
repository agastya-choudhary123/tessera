#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <sys/qos.h>
double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
void sme_peak(long); void neon_peak(long);
int main(){
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  long N=20000000; double t0,t1;
  sme_peak(100000); neon_peak(100000);
  t0=now(); sme_peak(N); t1=now();
  double s=8.0*512*N/(t1-t0)/1e9;
  printf("SME  fmopa : %7.1f GFLOP/s   %.3f ns/fmopa\n", s, (t1-t0)/N/8*1e9);
  t0=now(); neon_peak(N); t1=now();
  double n=8.0*8*N/(t1-t0)/1e9;
  printf("NEON fmla  : %7.1f GFLOP/s   %.3f ns/fmla\n", n, (t1-t0)/N/8*1e9);
  printf("SME / NEON : %.1fx\n", s/n);
  return 0; }
