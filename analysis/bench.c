/* Host throughput: ABS (float, q16) vs Irwin-Hall(12) vs Box-Muller, same xorshift32 RNG. */
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "../src/sharma_abs.h"
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static inline float u01(uint32_t*s){return (sharma_abs_xorshift32_r(s)>>8)*(1.0f/16777216.0f);}
int main(void){
  const int T=20000000; volatile double sink=0; uint32_t s=1; double t0; float acc;
  t0=now(); acc=0; for(int i=0;i<T;i++) acc+=compute_eta_r(&s,-10,10,12); sink+=acc;
  printf("ABS float n=12   %.1f ns/sample\n",(now()-t0)/T*1e9);
  t0=now(); acc=0; for(int i=0;i<T;i++) acc+=compute_eta_q16_r(&s,-10,10,12); sink+=acc;
  printf("ABS q16   n=12   %.1f ns/sample\n",(now()-t0)/T*1e9);
  t0=now(); acc=0; for(int i=0;i<T;i++){float z=-6;for(int k=0;k<12;k++)z+=u01(&s);acc+=z;} sink+=acc;
  printf("Irwin-Hall 12    %.1f ns/sample\n",(now()-t0)/T*1e9);
  t0=now(); acc=0; for(int i=0;i<T;i+=2){float u=u01(&s)+1e-7f,v=u01(&s);float r=sqrtf(-2*logf(u));acc+=r*cosf(6.2831853f*v)+r*sinf(6.2831853f*v);} sink+=acc;
  printf("Box-Muller       %.1f ns/sample\n",(now()-t0)/T*1e9);
  return (int)(sink*0);
}
