/* Minimal AArch64 + SME2 instruction encoder for the tessera JIT.
 *
 * Every base word and field position below was derived by assembling the
 * instruction with clang and diffing single-field variants; tests/enc_test
 * re-checks all of them against the system assembler, so a wrong constant
 * here fails loudly instead of silently corrupting a kernel.
 */
#ifndef TS_ARM64_H
#define TS_ARM64_H
#include <stdint.h>

/* condition codes */
enum { CC_EQ=0, CC_NE=1, CC_HS=2, CC_LO=3, CC_MI=4, CC_PL=5,
       CC_HI=8, CC_LS=9, CC_GE=10, CC_LT=11, CC_GT=12, CC_LE=13 };

/* ---- scalar ---------------------------------------------------------- */
static inline uint32_t A_MOVZ(int rd,uint32_t imm16,int hw){ return 0xd2800000u|((uint32_t)hw<<21)|(imm16<<5)|rd; }
static inline uint32_t A_MOVK(int rd,uint32_t imm16,int hw){ return 0xf2800000u|((uint32_t)hw<<21)|(imm16<<5)|rd; }
static inline uint32_t A_MOVZW(int rd,uint32_t imm16){ return 0x52800000u|(imm16<<5)|rd; }        /* 32-bit */
static inline uint32_t A_ADD (int rd,int rn,int rm){ return 0x8b000000u|(rm<<16)|(rn<<5)|rd; }
static inline uint32_t A_SUB (int rd,int rn,int rm){ return 0xcb000000u|(rm<<16)|(rn<<5)|rd; }
static inline uint32_t A_MUL (int rd,int rn,int rm){ return 0x9b007c00u|(rm<<16)|(rn<<5)|rd; }
static inline uint32_t A_ADDI(int rd,int rn,uint32_t i12){ return 0x91000000u|(i12<<10)|(rn<<5)|rd; }
static inline uint32_t A_SUBI(int rd,int rn,uint32_t i12){ return 0xd1000000u|(i12<<10)|(rn<<5)|rd; }
static inline uint32_t A_ADD_LSL(int rd,int rn,int rm,int sh){ return 0x8b000000u|(rm<<16)|((uint32_t)sh<<10)|(rn<<5)|rd; }
static inline uint32_t A_LSLI(int rd,int rn,int sh){ uint32_t immr=(64-sh)&63, imms=63-sh;
       return 0xd3400000u|(immr<<16)|(imms<<10)|(rn<<5)|rd; }
static inline uint32_t A_LSRI(int rd,int rn,int sh){ return 0xd3400000u|((uint32_t)sh<<16)|(63u<<10)|(rn<<5)|rd; }
static inline uint32_t A_UBFX(int rd,int rn,int lsb,int width){ uint32_t immr=lsb, imms=lsb+width-1;
       return 0xd3400000u|(immr<<16)|(imms<<10)|(rn<<5)|rd; }
static inline uint32_t A_MOV (int rd,int rm){ return 0xaa0003e0u|(rm<<16)|rd; }
static inline uint32_t A_LDR (int rt,int rn,uint32_t off){ return 0xf9400000u|((off/8)<<10)|(rn<<5)|rt; }
static inline uint32_t A_STR (int rt,int rn,uint32_t off){ return 0xf9000000u|((off/8)<<10)|(rn<<5)|rt; }
static inline uint32_t A_LDRR(int rt,int rn,int rm){ return 0xf8606800u|(rm<<16)|(rn<<5)|rt; }
static inline uint32_t A_LDAR(int rt,int rn){ return 0xc8dffc00u|(rn<<5)|rt; }
static inline uint32_t A_STLR(int rt,int rn){ return 0xc89ffc00u|(rn<<5)|rt; }
static inline uint32_t A_CMP (int rn,int rm){ return 0xeb00001fu|(rm<<16)|(rn<<5); }
static inline uint32_t A_CMPI(int rn,uint32_t i12){ return 0xf100001fu|(i12<<10)|(rn<<5); }
static inline uint32_t A_CSET(int rd,int cc){ return 0x9a9f07e0u|((uint32_t)(cc^1)<<12)|rd; }
static inline uint32_t A_BCOND(int cc,int32_t off){ return 0x54000000u|(((uint32_t)(off/4)&0x7ffffu)<<5)|(uint32_t)cc; }
static inline uint32_t A_B(int32_t off){ return 0x14000000u|((uint32_t)(off/4)&0x03ffffffu); }
static inline uint32_t A_CBNZ(int rt,int32_t off){ return 0xb5000000u|(((uint32_t)(off/4)&0x7ffffu)<<5)|rt; }
static inline uint32_t A_RET(void){ return 0xd65f03c0u; }
static inline uint32_t A_NOP(void){ return 0xd503201fu; }
static inline uint32_t A_YIELD(void){ return 0xd503203fu; }
static inline uint32_t A_STP_PRE(int t1,int t2,int rn,int off){ return 0xa9800000u|(((uint32_t)((off/8)&0x7f))<<15)|(t2<<10)|(rn<<5)|t1; }
static inline uint32_t A_LDP_POST(int t1,int t2,int rn,int off){ return 0xa8c00000u|(((uint32_t)((off/8)&0x7f))<<15)|(t2<<10)|(rn<<5)|t1; }
static inline uint32_t A_STR_D(int dt,int rn,uint32_t off){ return 0xfd000000u|((off/8)<<10)|(rn<<5)|dt; }
static inline uint32_t A_LDR_D(int dt,int rn,uint32_t off){ return 0xfd400000u|((off/8)<<10)|(rn<<5)|dt; }

/* ---- SME / streaming SVE --------------------------------------------- */
static inline uint32_t A_SMSTART(void){ return 0xd503477fu; }   /* sm + za  */
static inline uint32_t A_SMSTOP (void){ return 0xd503467fu; }
static inline uint32_t A_PTRUE_B(int pd){ return 0x2518e3e0u|pd; }
static inline uint32_t A_LDR_Z(int zt,int rn){ return 0x85804000u|(rn<<5)|zt; }
static inline uint32_t A_STR_Z(int zt,int rn){ return 0xe5804000u|(rn<<5)|zt; }
/* offset in units of VL (=64B here), 9-bit signed, split imm9h[21:16]/imm9l[12:10] */
#define A_ZIMM_OK(vl) ((vl)>=-256 && (vl)<=255)
static inline uint32_t A_ZIMM(int32_t vl){ uint32_t u=(uint32_t)vl & 0x1ffu;
       return ((u>>3)<<16) | ((u&7u)<<10); }
static inline uint32_t A_LDR_Z_I(int zt,int rn,int32_t vl){ return 0x85804000u|A_ZIMM(vl)|(rn<<5)|zt; }
static inline uint32_t A_STR_Z_I(int zt,int rn,int32_t vl){ return 0xe5804000u|A_ZIMM(vl)|(rn<<5)|zt; }
static inline uint32_t A_FMOPA_S(int za,int zn,int zm,int pn,int pm){
       return 0x80800000u|((uint32_t)zm<<16)|((uint32_t)pm<<13)|((uint32_t)pn<<10)|((uint32_t)zn<<5)|(uint32_t)za; }
static inline uint32_t A_ZERO_ZA_S(int tile){ return 0xc0080000u|(0x11u<<tile); }
static inline uint32_t A_ZERO_ZA_ALL(void){ return 0xc00800ffu; }
/* ZA tile slice <-> memory, slice index = w12 + imm(0..3) */
static inline uint32_t A_LD1W_ZA(int tile,int imm,int pg,int rn){
       return 0xe09f0000u|((uint32_t)pg<<10)|((uint32_t)rn<<5)|((uint32_t)tile<<2)|(uint32_t)imm; }
static inline uint32_t A_ST1W_ZA(int tile,int imm,int pg,int rn){
       return 0xe0bf0000u|((uint32_t)pg<<10)|((uint32_t)rn<<5)|((uint32_t)tile<<2)|(uint32_t)imm; }
/* streaming-SVE float ALU, .s */
static inline uint32_t A_FADD_S(int zd,int zn,int zm){ return 0x65800000u|((uint32_t)zm<<16)|((uint32_t)zn<<5)|zd; }
static inline uint32_t A_FMUL_S(int zd,int zn,int zm){ return 0x65800800u|((uint32_t)zm<<16)|((uint32_t)zn<<5)|zd; }
static inline uint32_t A_FMLA_S(int zda,int zn,int zm,int pg){ return 0x65a00000u|((uint32_t)zm<<16)|((uint32_t)pg<<13)|((uint32_t)zn<<5)|zda; }
static inline uint32_t A_FMAX_S(int zdn,int zm,int pg){ return 0x65868000u|((uint32_t)pg<<10)|((uint32_t)zm<<5)|zdn; }
static inline uint32_t A_DUP_S_IMM0(int zd){ return 0x25b8c000u|zd; }     /* dup z.s, #0 */
static inline uint32_t A_MOV_Z(int zd,int zn){ return 0x04603000u|((uint32_t)zn<<16)|((uint32_t)zn<<5)|zd; }
#endif
