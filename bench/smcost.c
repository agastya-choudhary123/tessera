#include <stdio.h>
#include <time.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
void smtoggle(long n);
void smtoggle_za(long n);
void nulll(long n);
int main(){ long N=2000000; double t;
  smtoggle(1000); nulll(1000); smtoggle_za(1000);
  t=now(); nulll(N);        double b=now()-t;
  t=now(); smtoggle(N);     double a=now()-t;
  t=now(); smtoggle_za(N);  double c=now()-t;
  printf("empty loop        : %.2f ns/iter\n", b/N*1e9);
  printf("smstart/smstop sm : %.2f ns/iter (%.2f ns net)\n", a/N*1e9, (a-b)/N*1e9);
  printf("smstart/smstop +za: %.2f ns/iter (%.2f ns net)\n", c/N*1e9, (c-b)/N*1e9);
  return 0; }
