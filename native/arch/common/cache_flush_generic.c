// SPDX-License-Identifier: MIT
#include "cross_cpu_arch.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CROSS_CPU_EVICT_BYTES (32u * 1024u * 1024u)
#define CROSS_CPU_EVICT_FALLBACK_BYTES (4u * 1024u * 1024u)

static pthread_once_t g_evict_once = PTHREAD_ONCE_INIT;
static uint64_t *g_evict_buf = NULL;
static size_t g_evict_len = 0;

static void init_evict_buffer(void) {
    size_t bytes = CROSS_CPU_EVICT_BYTES;
    uint64_t *buf = (uint64_t *)malloc(bytes);
    if (buf == NULL) {
        bytes = CROSS_CPU_EVICT_FALLBACK_BYTES;
        buf = (uint64_t *)malloc(bytes);
    }
    if (buf == NULL) {
        return;
    }
    size_t words = bytes / sizeof(uint64_t);
    for (size_t i = 0; i < words; i++) {
        buf[i] = ((uint64_t)i * 0x9e3779b97f4a7c15ULL) ^ 0xa5a5a5a55a5a5a5aULL;
    }
    g_evict_buf = buf;
    g_evict_len = words;
}

/*
 * Portable cache-eviction fallback for architectures without a user-space
 * cache line flush instruction. Performs a contention-free read sweep across
 * an oversized buffer (stride 64 bytes) bracketed by sequentially consistent
 * fences to evict dirty/clean lines from CPU caches into main memory.
 */
void cross_cpu_cache_flush_generic(const void *ptr, size_t len) {
    (void)ptr;
    (void)len;

    pthread_once(&g_evict_once, init_evict_buffer);
    atomic_thread_fence(memory_order_seq_cst);

    if (g_evict_buf == NULL || g_evict_len == 0) {
        return;
    }

    uint64_t acc = 0;
    /* Stride 8 x uint64_t = 64 bytes per cache line */
    for (size_t i = 0; i < g_evict_len; i += 8u) {
        acc ^= g_evict_buf[i];
    }

    volatile uint64_t sink = acc;
    (void)sink;
    atomic_thread_fence(memory_order_seq_cst);
}
