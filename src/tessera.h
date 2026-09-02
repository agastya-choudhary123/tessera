/* tessera - a bytecode VM for tensor programs.
 *
 * ISA principle: memory movement is an INSTRUCTION, never a side effect.
 * Compute instructions may only name SPM / VEC / ZA operands. No compute
 * instruction can name DRAM. There is therefore no hidden cache, no implicit
 * fill, and no LRU policy deciding what is resident -- the program decides,
 * and the verifier proves it decided legally.
 */
#ifndef TESSERA_H
#define TESSERA_H
#include <stdint.h>
#include <stddef.h>

/* ---- machine model -------------------------------------------------- */
/* SPM is sized to the measured L1D working set of an Apple M4 P-core
 * (128 KB sustained 306 GB/s; 192 KB falls off a cliff to 132 GB/s). */
#define TS_SPM_SIZE   (128*1024)
#define TS_NREG       16          /* scalar regs r0..r15                 */
#define TS_NVEC       32          /* vector regs v0..v31, 64B each (SVL) */
#define TS_VLANES     16          /* f32 lanes per vector (512b SVL)     */
#define TS_NZA        4           /* f32 ZA tiles za0..za3, 16x16 each   */
#define TS_ZADIM      16
#define TS_NTAG       8           /* DMA tags for async completion       */

/* ---- instruction encoding ------------------------------------------- */
/* Fixed 16 bytes. Wide and trivially decodable, in the spirit of the
 * VLIW-ish encodings real accelerators use. */
typedef struct {
    uint8_t  op;
    uint8_t  a, b, c;     /* register / tile / tag operands */
    uint16_t d, e;        /* small immediates, modes        */
    int32_t  imm0;
    int32_t  imm1;
} ts_insn;

enum {
    /* control / scalar */
    OP_HALT = 0,
    OP_MOVI,      /* r[a] = imm0                                    */
    OP_ARG,       /* r[a] = kernel_args[imm0]  (a DRAM base addr)   */
    OP_ADDI,      /* r[a] = r[b] + imm0                             */
    OP_ADD,       /* r[a] = r[b] + r[c]                             */
    OP_SUB,       /* r[a] = r[b] - r[c]                             */
    OP_MUL,       /* r[a] = r[b] * r[c]                             */
    OP_MULI,      /* r[a] = r[b] * imm0                             */
    OP_SHLI,      /* r[a] = r[b] << imm0                            */
    OP_SHRI,      /* r[a] = r[b] >> imm0  (logical)                  */
    OP_CMPLT,     /* r[a] = (r[b] <  r[c])                          */
    OP_BNZ,       /* if (r[a]) pc = imm0                            */
    OP_BZ,        /* if (!r[a]) pc = imm0                           */
    OP_JMP,       /* pc = imm0                                      */

    /* movement: DRAM <-> SPM. asynchronous, tagged. */
    OP_DMAIN,     /* SPM[imm0] <- DRAM[r[a]], rows=d, row_bytes=e,
                     src row stride = r[b] bytes, tag=c, mode=imm1  */
    OP_DMAOUT,    /* DRAM[r[a]] <- SPM[imm0], same shape fields     */
    OP_WAIT,      /* block until tags in bitmask imm0 have retired  */

    /* movement: SPM <-> VEC <-> ZA. synchronous, single-cycle class. */
    OP_LDV,       /* v[a] <- SPM[imm0 + r[b]]        (64B)          */
    OP_STV,       /* SPM[imm0 + r[b]] <- v[a]                       */
    OP_LDZA,      /* za[a] <- SPM[imm0], row stride imm1            */
    OP_STZA,      /* SPM[imm0] <- za[a], row stride imm1            */
    OP_ZERZA,     /* za[a] = 0                                      */

    /* compute. SPM/VEC/ZA operands only -- never DRAM. */
    OP_MOPA,      /* za[a] += outer(v[b], v[c])   16x16 f32 rank-1  */
    OP_VADD,      /* v[a] = v[b] + v[c]                             */
    OP_VMUL,      /* v[a] = v[b] * v[c]                             */
    OP_VFMA,      /* v[a] += v[b] * v[c]                            */
    OP_VMAX,      /* v[a] = max(v[b], v[c])                         */
    OP_VZERO,     /* v[a] = 0                                       */
    OP_VDUP,      /* v[a] = broadcast f32 bitpattern imm0           */

    OP__COUNT
};

/* address space of an operand, for the verifier */
enum { AS_NONE=0, AS_SCALAR, AS_SPM, AS_VEC, AS_ZA, AS_DRAM, AS_TAG, AS_IMM };

typedef struct {
    const char *name;
    uint8_t     as_a, as_b, as_c;   /* address space of each reg field   */
    uint8_t     touches_dram;       /* 1 only for DMA instructions       */
} ts_opinfo;

extern const ts_opinfo ts_ops[OP__COUNT];

/* DMA modes */
enum {
    DMA_PLAIN = 0,   /* straight 2D strided copy                          */
    DMA_PACK16,      /* transpose-pack f32 into 16-row panels: the layout
                        the SME outer-product unit actually wants. Real
                        accelerators do this in the DMA engine; so do we. */
    DMA_ACC,         /* writeback with accumulate: dst[i] += src[i], f32.
                        A k-chunked matmul produces partial sums, and doing
                        the reduction in the movement engine keeps it off the
                        critical path -- real DMA engines reduce on writeback
                        for the same reason. */
};

/* ---- program ---------------------------------------------------------- */
typedef struct {
    ts_insn  *code;
    int       ncode;
    char    **labels;      /* debug: label at each pc, or NULL           */
    int       nargs;       /* how many kernel_args the program reads     */
} ts_prog;

/* ---- machine state ---------------------------------------------------- */
typedef struct {
    uint8_t  *spm;                       /* TS_SPM_SIZE, 16K-page aligned */
    int64_t   r[TS_NREG];
    float     v[TS_NVEC][TS_VLANES];
    float     za[TS_NZA][TS_ZADIM][TS_ZADIM];
    void    **args;
    int       nargs;
    /* counters */
    uint64_t  n_insn, n_mopa, n_dma, dma_bytes, n_wait_stall;
} ts_vm;

/* assembler */
ts_prog *ts_assemble_file(const char *path, char *err, size_t errlen);
void     ts_prog_free(ts_prog *p);
void     ts_disasm(const ts_prog *p, int pc, char *buf, size_t n);

/* verifier: returns 0 ok, else count of violations (printed to stderr) */
int      ts_verify(const ts_prog *p, int verbose);

/* interpreter */
int      ts_run(ts_vm *vm, const ts_prog *p);
ts_vm   *ts_vm_new(void);
void     ts_vm_free(ts_vm *vm);

#endif
