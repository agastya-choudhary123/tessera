void sme_peak(long iters){
  __asm__ volatile("smstart\n ptrue p0.b\n"
    "1:\n"
    "fmopa za0.s, p0/m, p0/m, z0.s, z1.s\n fmopa za1.s, p0/m, p0/m, z2.s, z3.s\n"
    "fmopa za2.s, p0/m, p0/m, z4.s, z5.s\n fmopa za3.s, p0/m, p0/m, z6.s, z7.s\n"
    "fmopa za0.s, p0/m, p0/m, z8.s, z9.s\n fmopa za1.s, p0/m, p0/m, z10.s, z11.s\n"
    "fmopa za2.s, p0/m, p0/m, z12.s, z13.s\n fmopa za3.s, p0/m, p0/m, z14.s, z15.s\n"
    "subs %0, %0, #1\n bne 1b\n smstop\n" : "+r"(iters) :: "p0","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15","v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31","memory"); }
void neon_peak(long iters){
  __asm__ volatile("1:\n"
    "fmla v0.4s, v16.4s, v24.4s\n fmla v1.4s, v17.4s, v25.4s\n"
    "fmla v2.4s, v18.4s, v26.4s\n fmla v3.4s, v19.4s, v27.4s\n"
    "fmla v4.4s, v20.4s, v28.4s\n fmla v5.4s, v21.4s, v29.4s\n"
    "fmla v6.4s, v22.4s, v30.4s\n fmla v7.4s, v23.4s, v31.4s\n"
    "subs %0, %0, #1\n bne 1b\n" : "+r"(iters) :: "memory"); }
