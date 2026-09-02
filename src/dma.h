/* tessera DMA engine: a bank of independent descriptor queues.
 *
 * Measurement that forced this shape: with a single queue the engine ran ~80%
 * busy at 30-45 GB/s while the SME unit sat idle waiting on it -- the machine
 * was movement-bound, not compute-bound. That is the same reason real
 * accelerators ship 8-16 DMA engines rather than one fast one: a transposing,
 * strided gather is latency-bound, so the only way to raise throughput is to
 * run several of them at once.
 *
 * Each queue is strictly single-producer/single-consumer, so no CAS is needed
 * anywhere. A tag is statically bound to a queue (queue = tag % nq), which
 * means the JIT resolves the queue address at compile time and a transfer
 * costs the same handful of scalar stores it did with one queue.
 */
#ifndef TS_DMA_H
#define TS_DMA_H
#include <stdint.h>
#include <stdatomic.h>
#include "tessera.h"

#define TS_DMA_NQ       4     /* queues == worker threads                   */
#define TS_DMA_RING     64    /* descriptors per queue (power of two)       */
#define TS_DMA_DESCSZ   64    /* power of two: slot address is a shift      */

typedef struct {
    void       *dst;
    const void *src;
    uint32_t    rows, row_bytes;
    int64_t     src_stride, dst_stride;
    int32_t     mode, tag;
    uint64_t    _pad[2];
} ts_dma_desc;
_Static_assert(sizeof(ts_dma_desc)==TS_DMA_DESCSZ, "descriptor must stay 64B");

typedef struct {
    _Atomic uint64_t head;                  /* producer index               */
    _Atomic uint64_t tail;                  /* consumer index               */
    ts_dma_desc      ring[TS_DMA_RING];
} ts_dma_q;

typedef struct {
    _Atomic uint64_t issued[TS_NTAG];       /* per-tag, producer-owned      */
    _Atomic uint64_t done[TS_NTAG];         /* per-tag, consumer-owned      */
    _Atomic uint64_t bytes, busy_ns, ndesc, spins;
    _Atomic int      stop;
    int              running;
    ts_dma_q         q[TS_DMA_NQ];
} ts_dma_engine;

extern ts_dma_engine ts_dma;
#define TS_Q_OF(tag) ((tag) % TS_DMA_NQ)
#endif
