/* tessera assembler: .tsa text -> ts_prog bytecode */
#define _GNU_SOURCE
#include "tessera.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

const ts_opinfo ts_ops[OP__COUNT] = {
 [OP_HALT]  = {"halt",  AS_NONE,  AS_NONE,  AS_NONE, 0},
 [OP_MOVI]  = {"movi",  AS_SCALAR,AS_NONE,  AS_NONE, 0},
 [OP_ARG]   = {"arg",   AS_SCALAR,AS_NONE,  AS_NONE, 0},
 [OP_ADDI]  = {"addi",  AS_SCALAR,AS_SCALAR,AS_NONE, 0},
 [OP_ADD]   = {"add",   AS_SCALAR,AS_SCALAR,AS_SCALAR,0},
 [OP_SUB]   = {"sub",   AS_SCALAR,AS_SCALAR,AS_SCALAR,0},
 [OP_MUL]   = {"mul",   AS_SCALAR,AS_SCALAR,AS_SCALAR,0},
 [OP_MULI]  = {"muli",  AS_SCALAR,AS_SCALAR,AS_NONE, 0},
 [OP_SHLI]  = {"shli",  AS_SCALAR,AS_SCALAR,AS_NONE, 0},
 [OP_SHRI]  = {"shri",  AS_SCALAR,AS_SCALAR,AS_NONE, 0},
 [OP_CMPLT] = {"cmplt", AS_SCALAR,AS_SCALAR,AS_SCALAR,0},
 [OP_BNZ]   = {"bnz",   AS_SCALAR,AS_NONE,  AS_NONE, 0},
 [OP_BZ]    = {"bz",    AS_SCALAR,AS_NONE,  AS_NONE, 0},
 [OP_JMP]   = {"jmp",   AS_NONE,  AS_NONE,  AS_NONE, 0},
 [OP_DMAIN] = {"dma.in", AS_DRAM, AS_SCALAR,AS_TAG,  1},
 [OP_DMAOUT]= {"dma.out",AS_DRAM, AS_SCALAR,AS_TAG,  1},
 [OP_WAIT]  = {"wait",  AS_NONE,  AS_NONE,  AS_NONE, 0},
 [OP_LDV]   = {"ldv",   AS_VEC,   AS_SCALAR,AS_NONE, 0},
 [OP_STV]   = {"stv",   AS_VEC,   AS_SCALAR,AS_NONE, 0},
 [OP_LDZA]  = {"ldza",  AS_ZA,    AS_NONE,  AS_NONE, 0},
 [OP_STZA]  = {"stza",  AS_ZA,    AS_NONE,  AS_NONE, 0},
 [OP_ZERZA] = {"zerza", AS_ZA,    AS_NONE,  AS_NONE, 0},
 [OP_MOPA]  = {"mopa",  AS_ZA,    AS_VEC,   AS_VEC,  0},
 [OP_VADD]  = {"vadd",  AS_VEC,   AS_VEC,   AS_VEC,  0},
 [OP_VMUL]  = {"vmul",  AS_VEC,   AS_VEC,   AS_VEC,  0},
 [OP_VFMA]  = {"vfma",  AS_VEC,   AS_VEC,   AS_VEC,  0},
 [OP_VMAX]  = {"vmax",  AS_VEC,   AS_VEC,   AS_VEC,  0},
 [OP_VZERO] = {"vzero", AS_VEC,   AS_NONE,  AS_NONE, 0},
 [OP_VDUP]  = {"vdup",  AS_VEC,   AS_NONE,  AS_NONE, 0},
};

/* ---------------- tokenizer ---------------- */
#define MAXTOK 16
typedef struct { char *t[MAXTOK]; int n; int line; } toks;

static int split(char *s, toks *o, int line){
    o->n=0; o->line=line;
    char *p=s;
    while(*p){
        while(*p && (isspace((unsigned char)*p) || *p==',')) p++;
        if(!*p || *p==';' || *p=='#') break;
        char *b=p;
        while(*p && !isspace((unsigned char)*p) && *p!=',' && *p!=';' && *p!='#') p++;
        char save=*p; *p='\0';                 /* terminate and keep it that way */
        if(o->n<MAXTOK) o->t[o->n++]=b;
        if(save) p++;
    }
    return o->n;
}

/* ---------------- label table ---------------- */
typedef struct { char name[64]; int pc; } lab;
typedef struct { lab *v; int n, cap; } labtab;
static void lab_add(labtab *L, const char*nm, int pc){
    if(L->n==L->cap){ L->cap=L->cap?L->cap*2:64; L->v=realloc(L->v,L->cap*sizeof(lab)); }
    snprintf(L->v[L->n].name,64,"%s",nm); L->v[L->n].pc=pc; L->n++;
}
static int lab_find(labtab *L, const char*nm){
    for(int i=0;i<L->n;i++) if(!strcmp(L->v[i].name,nm)) return L->v[i].pc;
    return -1;
}

/* ---------------- operand parsing ---------------- */
static int p_reg(const char*s, char pfx, int max, const char *what, char*err, size_t el){
    if(!s || tolower((unsigned char)s[0])!=pfx){ snprintf(err,el,"expected %s, got '%s'",what,s?s:"(nothing)"); return -1; }
    char *e; long v=strtol(s+1,&e,10);
    if(*e || v<0 || v>=max){ snprintf(err,el,"bad %s '%s' (0..%d)",what,s,max-1); return -1; }
    return (int)v;
}
static int p_za(const char*s, char*err,size_t el){
    if(!s || strncasecmp(s,"za",2)){ snprintf(err,el,"expected za tile, got '%s'",s?s:""); return -1; }
    char *e; long v=strtol(s+2,&e,10);
    if(*e||v<0||v>=TS_NZA){ snprintf(err,el,"bad za tile '%s'",s); return -1; }
    return (int)v;
}
/* integer immediate; supports 0x, negative, and 'K'/'M' suffix, and f32 via 0f<float> */
static int p_imm(const char*s, int32_t *out, char*err,size_t el){
    if(!s){ snprintf(err,el,"expected immediate"); return -1; }
    if(s[0]=='0'&&(s[1]=='f'||s[1]=='F')){ float f=strtof(s+2,NULL); uint32_t b; memcpy(&b,&f,4); *out=(int32_t)b; return 0; }
    char *e; long v=strtol(s,&e,0);
    if(*e=='K'||*e=='k'){ v*=1024; e++; } else if(*e=='M'||*e=='m'){ v*=1024*1024; e++; }
    if(*e){ snprintf(err,el,"bad immediate '%s'",s); return -1; }
    *out=(int32_t)v; return 0;
}
/* SPM operand written as spm[<imm>] or spm[<imm>+rN] */
static int p_spm(const char*s, int32_t *off, int *reg, char*err,size_t el){
    *reg=0;
    if(!s||strncasecmp(s,"spm[",4)){ snprintf(err,el,"expected spm[...], got '%s'",s?s:""); return -1; }
    char buf[128]; snprintf(buf,sizeof buf,"%s",s+4);
    char *cb=strchr(buf,']'); if(!cb){ snprintf(err,el,"unclosed spm[ in '%s'",s); return -1; }
    *cb='\0';
    char *plus=strchr(buf,'+');
    if(plus){ *plus='\0';
        int r=p_reg(plus+1,'r',TS_NREG,"scalar reg",err,el); if(r<0) return -1;
        *reg=r+1; }   /* +1 so that 0 unambiguously means "no index register" */
    return p_imm(buf,off,err,el);
}

ts_prog *ts_assemble_file(const char *path, char *err, size_t el){
    FILE *f=fopen(path,"rb");
    if(!f){ snprintf(err,el,"cannot open %s",path); return NULL; }
    /* two passes: collect labels, then emit */
    char line[512]; labtab L={0};
    int pc=0, lineno=0;
    while(fgets(line,sizeof line,f)){
        lineno++; toks tk; char tmp[512]; snprintf(tmp,sizeof tmp,"%s",line);
        if(!split(tmp,&tk,lineno)) continue;
        char *t0=tk.t[0]; size_t n=strlen(t0);
        if(n && t0[n-1]==':'){ t0[n-1]='\0'; lab_add(&L,t0,pc);
            if(tk.n==1) continue;
            memmove(tk.t,tk.t+1,(tk.n-1)*sizeof(char*)); tk.n--; }
        if(tk.n && tk.t[0][0]=='.') continue;   /* directive */
        pc++;
    }
    rewind(f);
    ts_prog *P=calloc(1,sizeof *P);
    P->code=calloc(pc>0?pc:1,sizeof(ts_insn));
    P->labels=calloc(pc>0?pc:1,sizeof(char*));
    P->ncode=0; lineno=0;
    int fail=0;
    while(fgets(line,sizeof line,f) && !fail){
        lineno++; toks tk; char tmp[512]; snprintf(tmp,sizeof tmp,"%s",line);
        if(!split(tmp,&tk,lineno)) continue;
        char *lbl=NULL; char *t0=tk.t[0]; size_t n=strlen(t0);
        if(n && t0[n-1]==':'){ t0[n-1]='\0'; lbl=t0;
            if(tk.n==1) continue;
            memmove(tk.t,tk.t+1,(tk.n-1)*sizeof(char*)); tk.n--; }
        if(!tk.n) continue;
        if(tk.t[0][0]=='.'){
            if(!strcasecmp(tk.t[0],".args") && tk.n>1) P->nargs=atoi(tk.t[1]);
            continue;
        }
        ts_insn in; memset(&in,0,sizeof in);
        const char *m=tk.t[0];
        char **A=tk.t+1; int na=tk.n-1;
        int op=-1;
        for(int i=0;i<OP__COUNT;i++) if(ts_ops[i].name && !strcasecmp(ts_ops[i].name,m)){ op=i; break; }
        if(op<0){ snprintf(err,el,"line %d: unknown mnemonic '%s'",lineno,m); fail=1; break; }
        in.op=op;
        char e2[192]; e2[0]=0;
        #define NEED(k) do{ if(na<(k)){ snprintf(err,el,"line %d: %s needs %d operands, got %d",lineno,m,k,na); fail=1; goto done; } }while(0)
        #define CHK(x)  do{ if((x)<0){ snprintf(err,el,"line %d: %s: %s",lineno,m,e2); fail=1; goto done; } }while(0)
        switch(op){
        case OP_HALT: break;
        case OP_MOVI: case OP_ARG:
            NEED(2); { int r=p_reg(A[0],'r',TS_NREG,"scalar reg",e2,sizeof e2); CHK(r); in.a=r;
                       CHK(p_imm(A[1],&in.imm0,e2,sizeof e2)); } break;
        case OP_ADDI: case OP_MULI: case OP_SHLI: case OP_SHRI:
            NEED(3); { int r=p_reg(A[0],'r',TS_NREG,"scalar reg",e2,sizeof e2); CHK(r); in.a=r;
                       int s=p_reg(A[1],'r',TS_NREG,"scalar reg",e2,sizeof e2); CHK(s); in.b=s;
                       CHK(p_imm(A[2],&in.imm0,e2,sizeof e2)); } break;
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_CMPLT:
            NEED(3); { int x,y,z; x=p_reg(A[0],'r',TS_NREG,"scalar reg",e2,sizeof e2); CHK(x);
                       y=p_reg(A[1],'r',TS_NREG,"scalar reg",e2,sizeof e2); CHK(y);
                       z=p_reg(A[2],'r',TS_NREG,"scalar reg",e2,sizeof e2); CHK(z);
                       in.a=x; in.b=y; in.c=z; } break;
        case OP_BNZ: case OP_BZ:
            NEED(2); { int r=p_reg(A[0],'r',TS_NREG,"scalar reg",e2,sizeof e2); CHK(r); in.a=r;
                       int tgt=lab_find(&L,A[1]);
                       if(tgt<0){ snprintf(err,el,"line %d: unknown label '%s'",lineno,A[1]); fail=1; goto done; }
                       in.imm0=tgt; } break;
        case OP_JMP:
            NEED(1); { int tgt=lab_find(&L,A[0]);
                       if(tgt<0){ snprintf(err,el,"line %d: unknown label '%s'",lineno,A[0]); fail=1; goto done; }
                       in.imm0=tgt; } break;
        /* dma.in  spm[off], rBASE, rSTRIDE, rows, row_bytes, tagN [, mode] */
        case OP_DMAIN: case OP_DMAOUT: {
            NEED(6); int sreg; CHK(p_spm(A[0],&in.imm0,&sreg,e2,sizeof e2));
            if(sreg){ snprintf(err,el,"line %d: dma spm offset must be a constant",lineno); fail=1; goto done; }
            int base=p_reg(A[1],'r',TS_NREG,"dram base reg",e2,sizeof e2); CHK(base); in.a=base;
            int str =p_reg(A[2],'r',TS_NREG,"stride reg",e2,sizeof e2);     CHK(str);  in.b=str;
            int32_t rows,rb; CHK(p_imm(A[3],&rows,e2,sizeof e2)); CHK(p_imm(A[4],&rb,e2,sizeof e2));
            in.d=(uint16_t)rows; in.e=(uint16_t)rb;
            int tg=p_reg(A[5],'t',TS_NTAG,"tag",e2,sizeof e2); CHK(tg); in.c=tg;
            in.imm1=DMA_PLAIN;
            if(na>=7){ if(!strcasecmp(A[6],"pack16")) in.imm1=DMA_PACK16;
                       else if(!strcasecmp(A[6],"plain")) in.imm1=DMA_PLAIN;
                       else if(!strcasecmp(A[6],"acc")) in.imm1=DMA_ACC;
                       else { snprintf(err,el,"line %d: unknown dma mode '%s'",lineno,A[6]); fail=1; goto done; } }
        } break;
        case OP_WAIT: NEED(1); {
            /* accept t0 or a bitmask immediate */
            if(tolower((unsigned char)A[0][0])=='t' && isdigit((unsigned char)A[0][1])){
                int tg=p_reg(A[0],'t',TS_NTAG,"tag",e2,sizeof e2); CHK(tg); in.imm0=1<<tg;
            } else CHK(p_imm(A[0],&in.imm0,e2,sizeof e2));
        } break;
        case OP_LDV: case OP_STV: {
            NEED(2); int v,sreg;
            if(op==OP_LDV){ v=p_reg(A[0],'v',TS_NVEC,"vec reg",e2,sizeof e2); CHK(v); in.a=v;
                            CHK(p_spm(A[1],&in.imm0,&sreg,e2,sizeof e2)); in.b=sreg; }
            else          { CHK(p_spm(A[0],&in.imm0,&sreg,e2,sizeof e2)); in.b=sreg;
                            v=p_reg(A[1],'v',TS_NVEC,"vec reg",e2,sizeof e2); CHK(v); in.a=v; }
        } break;
        case OP_LDZA: case OP_STZA: {
            NEED(3); int z,sreg;
            if(op==OP_LDZA){ z=p_za(A[0],e2,sizeof e2); CHK(z); in.a=z;
                             CHK(p_spm(A[1],&in.imm0,&sreg,e2,sizeof e2)); in.b=sreg; }
            else           { CHK(p_spm(A[0],&in.imm0,&sreg,e2,sizeof e2)); in.b=sreg;
                             z=p_za(A[1],e2,sizeof e2); CHK(z); in.a=z; }
            CHK(p_imm(A[2],&in.imm1,e2,sizeof e2));   /* row stride bytes */
        } break;
        case OP_ZERZA: NEED(1); { int z=p_za(A[0],e2,sizeof e2); CHK(z); in.a=z; } break;
        case OP_MOPA: NEED(3); {
            int z=p_za(A[0],e2,sizeof e2); CHK(z); in.a=z;
            int x=p_reg(A[1],'v',TS_NVEC,"vec reg",e2,sizeof e2); CHK(x); in.b=x;
            int y=p_reg(A[2],'v',TS_NVEC,"vec reg",e2,sizeof e2); CHK(y); in.c=y; } break;
        case OP_VADD: case OP_VMUL: case OP_VFMA: case OP_VMAX: NEED(3); {
            int x=p_reg(A[0],'v',TS_NVEC,"vec reg",e2,sizeof e2); CHK(x); in.a=x;
            int y=p_reg(A[1],'v',TS_NVEC,"vec reg",e2,sizeof e2); CHK(y); in.b=y;
            int z=p_reg(A[2],'v',TS_NVEC,"vec reg",e2,sizeof e2); CHK(z); in.c=z; } break;
        case OP_VZERO: NEED(1); { int x=p_reg(A[0],'v',TS_NVEC,"vec reg",e2,sizeof e2); CHK(x); in.a=x; } break;
        case OP_VDUP: NEED(2); { int x=p_reg(A[0],'v',TS_NVEC,"vec reg",e2,sizeof e2); CHK(x); in.a=x;
                                 CHK(p_imm(A[1],&in.imm0,e2,sizeof e2)); } break;
        default: snprintf(err,el,"line %d: unhandled op %s",lineno,m); fail=1; goto done;
        }
        P->labels[P->ncode] = lbl ? strdup(lbl) : NULL;
        P->code[P->ncode++] = in;
    }
done:
    fclose(f); free(L.v);
    if(fail){ ts_prog_free(P); return NULL; }
    return P;
}

void ts_prog_free(ts_prog *p){
    if(!p) return;
    if(p->labels){ for(int i=0;i<p->ncode;i++) free(p->labels[i]); free(p->labels); }
    free(p->code); free(p);
}

void ts_disasm(const ts_prog *p, int pc, char *buf, size_t n){
    const ts_insn *i=&p->code[pc];
    const char *m = ts_ops[i->op].name ? ts_ops[i->op].name : "???";
    switch(i->op){
    case OP_MOVI: case OP_ARG: snprintf(buf,n,"%-7s r%d, %d",m,i->a,i->imm0); break;
    case OP_ADDI: case OP_MULI: case OP_SHLI: case OP_SHRI: snprintf(buf,n,"%-7s r%d, r%d, %d",m,i->a,i->b,i->imm0); break;
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_CMPLT: snprintf(buf,n,"%-7s r%d, r%d, r%d",m,i->a,i->b,i->c); break;
    case OP_BNZ: case OP_BZ: snprintf(buf,n,"%-7s r%d, %d",m,i->a,i->imm0); break;
    case OP_JMP: snprintf(buf,n,"%-7s %d",m,i->imm0); break;
    case OP_DMAIN: case OP_DMAOUT:
        snprintf(buf,n,"%-7s spm[%d], r%d, r%d, %u, %u, t%d%s",m,i->imm0,i->a,i->b,i->d,i->e,i->c,
                 i->imm1==DMA_PACK16?", pack16":(i->imm1==DMA_ACC?", acc":"")); break;
    case OP_WAIT: snprintf(buf,n,"%-7s 0x%x",m,i->imm0); break;
    case OP_LDV: if(i->b) snprintf(buf,n,"%-7s v%d, spm[%d+r%d]",m,i->a,i->imm0,i->b-1); else snprintf(buf,n,"%-7s v%d, spm[%d]",m,i->a,i->imm0); break;
    case OP_STV: if(i->b) snprintf(buf,n,"%-7s spm[%d+r%d], v%d",m,i->imm0,i->b-1,i->a); else snprintf(buf,n,"%-7s spm[%d], v%d",m,i->imm0,i->a); break;
    case OP_LDZA: if(i->b) snprintf(buf,n,"%-7s za%d, spm[%d+r%d], %d",m,i->a,i->imm0,i->b-1,i->imm1); else snprintf(buf,n,"%-7s za%d, spm[%d], %d",m,i->a,i->imm0,i->imm1); break;
    case OP_STZA: if(i->b) snprintf(buf,n,"%-7s spm[%d+r%d], za%d, %d",m,i->imm0,i->b-1,i->a,i->imm1); else snprintf(buf,n,"%-7s spm[%d], za%d, %d",m,i->imm0,i->a,i->imm1); break;
    case OP_ZERZA: snprintf(buf,n,"%-7s za%d",m,i->a); break;
    case OP_MOPA: snprintf(buf,n,"%-7s za%d, v%d, v%d",m,i->a,i->b,i->c); break;
    case OP_VADD: case OP_VMUL: case OP_VFMA: case OP_VMAX:
        snprintf(buf,n,"%-7s v%d, v%d, v%d",m,i->a,i->b,i->c); break;
    case OP_VZERO: snprintf(buf,n,"%-7s v%d",m,i->a); break;
    case OP_VDUP: snprintf(buf,n,"%-7s v%d, 0x%08x",m,i->a,(unsigned)i->imm0); break;
    default: snprintf(buf,n,"%s",m);
    }
}
