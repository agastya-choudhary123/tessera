#define CL "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15","v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31"
void smtoggle(long n){ __asm__ volatile("1:\n smstart sm\n smstop sm\n subs %0,%0,#1\n bne 1b\n":"+r"(n)::CL,"memory"); }
void smtoggle_za(long n){ __asm__ volatile("1:\n smstart\n smstop\n subs %0,%0,#1\n bne 1b\n":"+r"(n)::CL,"memory"); }
void nulll(long n){ __asm__ volatile("1:\n subs %0,%0,#1\n bne 1b\n":"+r"(n)::"memory"); }
