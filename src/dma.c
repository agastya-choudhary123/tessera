#define _GNU_SOURCE
#include "tessera.h"
#include "dma.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sys/qos.h>
#include <arm_neon.h>

ts_dma_engine ts_dma;
static pthread_t g_worker[TS_DMA_NQ];

/* 16x16 f32 transpose, the inner block of the pack16 layout transform */
static inline void transpose16x16_f32(float *dst, const float *src,
                                      long src_stride_f, long dst_stride_f){
    for(int ii=0; ii<16; ii+=4){
        for(int jj=0; jj<16; jj+=4){
            float32x4_t r0=vld1q_f32(src+(ii+0)*src_stride_f+jj);
            float32x4_t r1=vld1q_f32(src+(ii+1)*src_stride_f+jj);
            float32x4_t r2=vld1q_f32(src+(ii+2)*src_stride_f+jj);
            float32x4_t r3=vld1q_f32(src+(ii+3)*src_stride_f+jj);
            float32x4x2_t a=vtrnq_f32(r0,r1), b=vtrnq_f32(r2,r3);
            vst1q_f32(dst+(jj+0)*dst_stride_f+ii, vcombine_f32(vget_low_f32 (a.val[0]), vget_low_f32 (b.val[0])));
            vst1q_f32(dst+(jj+1)*dst_stride_f+ii, vcombine_f32(vget_low_f32 (a.val[1]), vget_low_f32 (b.val[1])));
            vst1q_f32(dst+(jj+2)*dst_stride_f+ii, vcombine_f32(vget_high_f32(a.val[0]), vget_high_f32(b.val[0])));
            vst1q_f32(dst+(jj+3)*dst_stride_f+ii, vcombine_f32(vget_high_f32(a.val[1]), vget_high_f32(b.val[1])));
        }
    }
}

static void dma_exec(const ts_dma_desc *d){
    if(d->mode == DMA_PACK16){
        long K = d->row_bytes/4;
        const float *s=(const float*)d->src; float *t=(float*)d->dst;
        long ss=d->src_stride/4, k=0;
        for(; k+16<=K; k+=16) transpose16x16_f32(t+k*16, s+k, ss, 16);
        for(; k<K; k++) for(int i=0;i<16;i++) t[k*16+i]=s[(long)i*ss+k];
    } else if(d->mode == DMA_ACC){
        for(uint32_t r=0;r<d->rows;r++){
            float *t=(float*)((char*)d->dst + (long)r*d->dst_stride);
            const float *s=(const float*)((const char*)d->src + (long)r*d->src_stride);
            uint32_t n=d->row_bytes/4, i=0;
            for(; i+4<=n; i+=4) vst1q_f32(t+i, vaddq_f32(vld1q_f32(t+i), vld1q_f32(s+i)));
            for(; i<n; i++) t[i]+=s[i];
        }
    } else {
        char *t=(char*)d->dst; const char *s=(const char*)d->src;
        for(uint32_t r=0;r<d->rows;r++)
            memcpy(t + (long)r*d->dst_stride, s + (long)r*d->src_stride, d->row_bytes);
    }
}

static void *dma_thread(void *arg){
    ts_dma_q *q = &ts_dma.q[(intptr_t)arg];
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    for(;;){
        uint64_t t = atomic_load_explicit(&q->tail, memory_order_relaxed);
        uint64_t h = atomic_load_explicit(&q->head, memory_order_acquire);
        if(t == h){
            if(atomic_load_explicit(&ts_dma.stop, memory_order_relaxed)) return NULL;
            __asm__ volatile("yield");
            continue;
        }
        ts_dma_desc *d = &q->ring[t % TS_DMA_RING];
        uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        dma_exec(d);
        atomic_fetch_add_explicit(&ts_dma.busy_ns,
            clock_gettime_nsec_np(CLOCK_UPTIME_RAW)-t0, memory_order_relaxed);
        atomic_fetch_add_explicit(&ts_dma.ndesc, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&ts_dma.done[d->tag], 1, memory_order_release);
        atomic_store_explicit(&q->tail, t+1, memory_order_release);
    }
}

void ts_dma_start(void){
    if(ts_dma.running) return;
    memset(&ts_dma, 0, sizeof ts_dma);
    for(intptr_t i=0;i<TS_DMA_NQ;i++){
        int rc=pthread_create(&g_worker[i], NULL, dma_thread, (void*)i);
        if(rc){ fprintf(stderr,"dma: pthread_create failed: %d\n",rc); abort(); }
    }
    ts_dma.running = 1;
}
void ts_dma_stop(void){
    if(!ts_dma.running) return;
    atomic_store(&ts_dma.stop, 1);
    for(int i=0;i<TS_DMA_NQ;i++) pthread_join(g_worker[i], NULL);
    ts_dma.running = 0;
}

void ts_dma_issue(void *dst, const void *src, uint32_t rows, uint32_t row_bytes,
                  int64_t src_stride, int64_t dst_stride, int mode, int tag){
    ts_dma_q *q = &ts_dma.q[TS_Q_OF(tag)];
    uint64_t h = atomic_load_explicit(&q->head, memory_order_relaxed);
    while(h - atomic_load_explicit(&q->tail, memory_order_acquire) >= TS_DMA_RING)
        __asm__ volatile("yield");
    ts_dma_desc *d=&q->ring[h % TS_DMA_RING];
    d->dst=dst; d->src=src; d->rows=rows; d->row_bytes=row_bytes;
    d->src_stride=src_stride; d->dst_stride=dst_stride; d->mode=mode; d->tag=tag;
    atomic_fetch_add_explicit(&ts_dma.issued[tag], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&ts_dma.bytes, (uint64_t)rows*row_bytes, memory_order_relaxed);
    atomic_store_explicit(&q->head, h+1, memory_order_release);
}

uint64_t ts_dma_wait(uint32_t tagmask){
    uint64_t spins=0;
    for(int t=0;t<TS_NTAG;t++){
        if(!(tagmask & (1u<<t))) continue;
        uint64_t want = atomic_load_explicit(&ts_dma.issued[t], memory_order_relaxed);
        while(atomic_load_explicit(&ts_dma.done[t], memory_order_acquire) < want){
            spins++; __asm__ volatile("yield");
        }
    }
    atomic_thread_fence(memory_order_acquire);
    return spins;
}
uint64_t ts_dma_bytes(void){ return atomic_load(&ts_dma.bytes); }
uint64_t ts_dma_busy_ns(void){ return atomic_load(&ts_dma.busy_ns); }
uint64_t ts_dma_ndesc(void){ return atomic_load(&ts_dma.ndesc); }
void     ts_dma_reset_stats(void){ atomic_store(&ts_dma.busy_ns,0);
    atomic_store(&ts_dma.bytes,0); atomic_store(&ts_dma.ndesc,0); }

/* dump queue state -- turns a would-be deadlock into a diagnosable event */
void ts_dma_dump(FILE *f){
    fprintf(f,"dma: %d queues\n", TS_DMA_NQ);
    for(int i=0;i<TS_DMA_NQ;i++)
        fprintf(f,"  q%d head=%llu tail=%llu\n", i,
            (unsigned long long)atomic_load(&ts_dma.q[i].head),
            (unsigned long long)atomic_load(&ts_dma.q[i].tail));
    for(int t=0;t<TS_NTAG;t++){
        uint64_t is=atomic_load(&ts_dma.issued[t]), dn=atomic_load(&ts_dma.done[t]);
        if(is||dn) fprintf(f,"  t%d issued=%llu done=%llu%s\n", t,
            (unsigned long long)is,(unsigned long long)dn, is>dn?"  <-- outstanding":"");
    }
}
