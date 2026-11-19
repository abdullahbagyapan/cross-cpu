// SPDX-License-Identifier: MIT
#include "cross_cpu_arch.h"

#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

/* Compute exact A^steps and C * sum_{j=0..steps-1} A^j (mod 2^64) in O(log steps) */
static void lcg_power_u64(uint64_t a, uint64_t c, uint32_t steps,
                          uint64_t *out_mul, uint64_t *out_add) {
    uint64_t cur_m = a;
    uint64_t cur_a = c;
    uint64_t acc_m = 1ULL;
    uint64_t acc_a = 0ULL;
    uint32_t n = steps;
    while (n > 0u) {
        if (n & 1u) {
            acc_m = acc_m * cur_m;
            acc_a = acc_a * cur_m + cur_a;
        }
        cur_a = (cur_m + 1ULL) * cur_a;
        cur_m = cur_m * cur_m;
        n >>= 1u;
    }
    *out_mul = acc_m;
    *out_add = acc_a;
}

static int32_t scalar_compute_int64(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint64_t mul = 6364136223846793005ULL;
    const uint64_t inc = 1442695040888963407ULL;

    uint64_t ref_m = 1ULL;
    uint64_t ref_a = 0ULL;
    lcg_power_u64(mul, inc, inner, &ref_m, &ref_a);

    uint64_t seed = ctx->seed;
    uint64_t ops = 0;

    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        uint64_t s0 = splitmix64(seed + (b * 4ULL) + 0ULL);
        uint64_t s1 = splitmix64(seed + (b * 4ULL) + 1ULL);
        uint64_t s2 = splitmix64(seed + (b * 4ULL) + 2ULL);
        uint64_t s3 = splitmix64(seed + (b * 4ULL) + 3ULL);

        volatile uint64_t init0 = s0;
        volatile uint64_t init1 = s1;
        volatile uint64_t init2 = s2;
        volatile uint64_t init3 = s3;

        uint64_t x0 = init0;
        uint64_t x1 = init1;
        uint64_t x2 = init2;
        uint64_t x3 = init3;

        for (uint32_t i = 0; i < inner; i++) {
            x0 = x0 * mul + inc;
            x1 = x1 * mul + inc;
            x2 = x2 * mul + inc;
            x3 = x3 * mul + inc;
        }

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            x0 ^= 0x1ULL;
        }

        const uint64_t e0 = s0 * ref_m + ref_a;
        const uint64_t e1 = s1 * ref_m + ref_a;
        const uint64_t e2 = s2 * ref_m + ref_a;
        const uint64_t e3 = s3 * ref_m + ref_a;

        if (x0 != e0 || x1 != e1 || x2 != e2 || x3 != e3) {
            ctx->errors++;
            ctx->expected_sample = (double)e0;
            ctx->observed_sample = (double)x0;
            snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                     "scalar int64 bit mismatch on batch %llu (expected 0x%016llx, got 0x%016llx)",
                     (unsigned long long)b, (unsigned long long)e0, (unsigned long long)x0);
            ctx->ops_done = ops;
            return CROSS_CPU_STATUS_COMPUTE_ERROR;
        }
        /* 4 chains * 2 ops (mul + add) * inner */
        ops += (uint64_t)inner * 8ULL;
    }

    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t scalar_compute_int32(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint32_t mul = 1664525u;
    const uint32_t inc = 1013904223u;

    uint64_t ref_m64 = 1ULL;
    uint64_t ref_a64 = 0ULL;
    lcg_power_u64((uint64_t)mul, (uint64_t)inc, inner, &ref_m64, &ref_a64);
    const uint32_t ref_m = (uint32_t)ref_m64;
    const uint32_t ref_a = (uint32_t)ref_a64;

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        uint64_t h0 = splitmix64(ctx->seed + (b * 2ULL));
        uint64_t h1 = splitmix64(ctx->seed + (b * 2ULL) + 1ULL);

        uint32_t s0 = (uint32_t)h0;
        uint32_t s1 = (uint32_t)(h0 >> 32);
        uint32_t s2 = (uint32_t)h1;
        uint32_t s3 = (uint32_t)(h1 >> 32);

        volatile uint32_t v0 = s0, v1 = s1, v2 = s2, v3 = s3;
        uint32_t x0 = v0, x1 = v1, x2 = v2, x3 = v3;

        for (uint32_t i = 0; i < inner; i++) {
            x0 = x0 * mul + inc;
            x1 = x1 * mul + inc;
            x2 = x2 * mul + inc;
            x3 = x3 * mul + inc;
        }

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            x1 ^= 0x80u;
        }

        const uint32_t e0 = s0 * ref_m + ref_a;
        const uint32_t e1 = s1 * ref_m + ref_a;
        const uint32_t e2 = s2 * ref_m + ref_a;
        const uint32_t e3 = s3 * ref_m + ref_a;

        if (x0 != e0 || x1 != e1 || x2 != e2 || x3 != e3) {
            ctx->errors++;
            ctx->expected_sample = (double)e1;
            ctx->observed_sample = (double)x1;
            snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                     "scalar int32 bit mismatch on batch %llu", (unsigned long long)b);
            ctx->ops_done = ops;
            return CROSS_CPU_STATUS_COMPUTE_ERROR;
        }
        ops += (uint64_t)inner * 8ULL;
    }

    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t scalar_compute_int16(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint16_t mul = 25173u;
    const uint16_t inc = 13849u;

    uint64_t ref_m64 = 1ULL;
    uint64_t ref_a64 = 0ULL;
    lcg_power_u64((uint64_t)mul, (uint64_t)inc, inner, &ref_m64, &ref_a64);
    const uint16_t ref_m = (uint16_t)ref_m64;
    const uint16_t ref_a = (uint16_t)ref_a64;

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        uint64_t h = splitmix64(ctx->seed + b);
        uint16_t s0 = (uint16_t)(h);
        uint16_t s1 = (uint16_t)(h >> 16);
        uint16_t s2 = (uint16_t)(h >> 32);
        uint16_t s3 = (uint16_t)(h >> 48);

        volatile uint16_t v0 = s0, v1 = s1, v2 = s2, v3 = s3;
        uint16_t x0 = v0, x1 = v1, x2 = v2, x3 = v3;

        for (uint32_t i = 0; i < inner; i++) {
            x0 = (uint16_t)(x0 * mul + inc);
            x1 = (uint16_t)(x1 * mul + inc);
            x2 = (uint16_t)(x2 * mul + inc);
            x3 = (uint16_t)(x3 * mul + inc);
        }

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            x2 ^= 0x04u;
        }

        const uint16_t e0 = (uint16_t)(s0 * ref_m + ref_a);
        const uint16_t e1 = (uint16_t)(s1 * ref_m + ref_a);
        const uint16_t e2 = (uint16_t)(s2 * ref_m + ref_a);
        const uint16_t e3 = (uint16_t)(s3 * ref_m + ref_a);

        if (x0 != e0 || x1 != e1 || x2 != e2 || x3 != e3) {
            ctx->errors++;
            ctx->expected_sample = (double)e2;
            ctx->observed_sample = (double)x2;
            snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                     "scalar int16 bit mismatch on batch %llu", (unsigned long long)b);
            ctx->ops_done = ops;
            return CROSS_CPU_STATUS_COMPUTE_ERROR;
        }
        ops += (uint64_t)inner * 8ULL;
    }

    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t scalar_compute_int8(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint8_t mul = 141u;
    const uint8_t inc = 53u;

    uint64_t ref_m64 = 1ULL;
    uint64_t ref_a64 = 0ULL;
    lcg_power_u64((uint64_t)mul, (uint64_t)inc, inner, &ref_m64, &ref_a64);
    const uint8_t ref_m = (uint8_t)ref_m64;
    const uint8_t ref_a = (uint8_t)ref_a64;

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        uint64_t h = splitmix64(ctx->seed + b);
        uint8_t s0 = (uint8_t)(h);
        uint8_t s1 = (uint8_t)(h >> 8);
        uint8_t s2 = (uint8_t)(h >> 16);
        uint8_t s3 = (uint8_t)(h >> 24);

        volatile uint8_t v0 = s0, v1 = s1, v2 = s2, v3 = s3;
        uint8_t x0 = v0, x1 = v1, x2 = v2, x3 = v3;

        for (uint32_t i = 0; i < inner; i++) {
            x0 = (uint8_t)(x0 * mul + inc);
            x1 = (uint8_t)(x1 * mul + inc);
            x2 = (uint8_t)(x2 * mul + inc);
            x3 = (uint8_t)(x3 * mul + inc);
        }

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            x3 ^= 0x01u;
        }

        const uint8_t e0 = (uint8_t)(s0 * ref_m + ref_a);
        const uint8_t e1 = (uint8_t)(s1 * ref_m + ref_a);
        const uint8_t e2 = (uint8_t)(s2 * ref_m + ref_a);
        const uint8_t e3 = (uint8_t)(s3 * ref_m + ref_a);

        if (x0 != e0 || x1 != e1 || x2 != e2 || x3 != e3) {
            ctx->errors++;
            ctx->expected_sample = (double)e3;
            ctx->observed_sample = (double)x3;
            snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                     "scalar int8 bit mismatch on batch %llu", (unsigned long long)b);
            ctx->ops_done = ops;
            return CROSS_CPU_STATUS_COMPUTE_ERROR;
        }
        ops += (uint64_t)inner * 8ULL;
    }

    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

/*
 * Bounded contractive FMA recurrence:
 *   y_{k+1} = a * y_k + b, with 0 < a < 1.
 * Paired chains run (+y0, +b) and (-y0, -b) so exact IEEE-754 symmetry yields
 * y_pos + y_neg == 0.0, while analytical fixed-point bounds verify the value.
 */
static int32_t scalar_compute_fp64(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 128u;
    const double a = 0.75;
    const double b0 = 0.3125;
    const double b1 = 0.4375;

    /* After 128 steps of y = 0.75*y + b, 0.75^128 < 1e-16 -> y == 4*b to within 4 ULP */
    const double ref0 = b0 / (1.0 - a);
    const double ref1 = b1 / (1.0 - a);

    uint64_t ops = 0;
    for (uint64_t iter = 0; iter < ctx->batch_iters; iter++) {
        const double delta = (double)((ctx->seed + iter) & 0xFFu) * 0.00390625;
        volatile double vp0 = 0.5 + delta;
        volatile double vn0 = -(0.5 + delta);
        volatile double vp1 = 1.0 + delta;
        volatile double vn1 = -(1.0 + delta);

        double yp0 = vp0, yn0 = vn0;
        double yp1 = vp1, yn1 = vn1;

        for (uint32_t i = 0; i < inner; i++) {
            yp0 = fma(a, yp0, b0);
            yn0 = fma(a, yn0, -b0);
            yp1 = fma(a, yp1, b1);
            yn1 = fma(a, yn1, -b1);
        }

        if (ctx->inject_fault && ctx->thread_idx == 0u && iter == 0ULL) {
            yp0 += 1.0e-6;
        }

        if (!isfinite(yp0) || !isfinite(yn0) || !isfinite(yp1) || !isfinite(yn1) ||
            fabs(yp0 + yn0) > 1.0e-12 || fabs(yp1 + yn1) > 1.0e-12 ||
            fabs(yp0 - ref0) > 1.0e-12 || fabs(yp1 - ref1) > 1.0e-12) {
            ctx->errors++;
            ctx->expected_sample = ref0;
            ctx->observed_sample = yp0;
            snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                     "scalar fp64 verification mismatch (expected %.12g, got %.12g, sym=%.3e)",
                     ref0, yp0, yp0 + yn0);
            ctx->ops_done = ops;
            return CROSS_CPU_STATUS_COMPUTE_ERROR;
        }
        /* 4 FMA chains * 2 FLOPs * inner */
        ops += (uint64_t)inner * 8ULL;
    }

    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t scalar_compute_fp32(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 128u;
    const float a = 0.75f;
    const float b0 = 0.3125f;
    const float b1 = 0.4375f;

    const float ref0 = b0 / (1.0f - a);
    const float ref1 = b1 / (1.0f - a);

    uint64_t ops = 0;
    for (uint64_t iter = 0; iter < ctx->batch_iters; iter++) {
        const float delta = (float)((ctx->seed + iter) & 0xFFu) * 0.00390625f;
        volatile float vp0 = 0.5f + delta;
        volatile float vn0 = -(0.5f + delta);
        volatile float vp1 = 1.0f + delta;
        volatile float vn1 = -(1.0f + delta);

        float yp0 = vp0, yn0 = vn0;
        float yp1 = vp1, yn1 = vn1;

        for (uint32_t i = 0; i < inner; i++) {
            yp0 = fmaf(a, yp0, b0);
            yn0 = fmaf(a, yn0, -b0);
            yp1 = fmaf(a, yp1, b1);
            yn1 = fmaf(a, yn1, -b1);
        }

        if (ctx->inject_fault && ctx->thread_idx == 0u && iter == 0ULL) {
            yp0 += 1.0e-3f;
        }

        if (!isfinite(yp0) || !isfinite(yn0) || !isfinite(yp1) || !isfinite(yn1) ||
            fabsf(yp0 + yn0) > 1.0e-5f || fabsf(yp1 + yn1) > 1.0e-5f ||
            fabsf(yp0 - ref0) > 1.0e-5f || fabsf(yp1 - ref1) > 1.0e-5f) {
            ctx->errors++;
            ctx->expected_sample = (double)ref0;
            ctx->observed_sample = (double)yp0;
            snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                     "scalar fp32 verification mismatch (expected %.8g, got %.8g, sym=%.3e)",
                     (double)ref0, (double)yp0, (double)(yp0 + yn0));
            ctx->ops_done = ops;
            return CROSS_CPU_STATUS_COMPUTE_ERROR;
        }
        ops += (uint64_t)inner * 8ULL;
    }

    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t scalar_cache_or_bandwidth(cross_cpu_worker_ctx_t *ctx) {
    if (ctx->thread_buf == NULL || ctx->thread_buf_bytes < sizeof(uint64_t)) {
        snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                 "Working buffer not allocated for memory/cache workload");
        return CROSS_CPU_STATUS_INIT_ERROR;
    }

    uint64_t *buf = (uint64_t *)ctx->thread_buf;
    const size_t words = ctx->thread_buf_bytes / sizeof(uint64_t);
    uint64_t bytes = 0;

    for (uint64_t pass = 0; pass < ctx->batch_iters; pass++) {
        const uint64_t salt = splitmix64(ctx->seed + pass);

        for (size_t i = 0; i < words; i++) {
            buf[i] = salt ^ ((uint64_t)i * 0x9e3779b97f4a7c15ULL);
        }

        if (ctx->inject_fault && ctx->thread_idx == 0u && pass == 0ULL && words > 0u) {
            buf[words / 2u] ^= 0x0000000100000000ULL;
        }

        for (size_t i = 0; i < words; i++) {
            const uint64_t expected = salt ^ ((uint64_t)i * 0x9e3779b97f4a7c15ULL);
            const uint64_t observed = buf[i];
            if (observed != expected) {
                ctx->errors++;
                ctx->expected_sample = (double)expected;
                ctx->observed_sample = (double)observed;
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "Memory/cache mismatch at word %zu (expected 0x%016llx, got 0x%016llx)",
                         i, (unsigned long long)expected, (unsigned long long)observed);
                ctx->ops_done = bytes;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        /* Write + read verification = 2x buffer bytes */
        bytes += (uint64_t)ctx->thread_buf_bytes * 2ULL;
    }

    ctx->ops_done = bytes;
    return CROSS_CPU_STATUS_OK;
}

static int32_t scalar_memory_pattern(cross_cpu_worker_ctx_t *ctx) {
    static const uint64_t base_patterns[] = {
        0x0000000000000000ULL,
        0xFFFFFFFFFFFFFFFFULL,
        0x5555555555555555ULL,
        0xAAAAAAAAAAAAAAAAULL,
        0x0F0F0F0F0F0F0F0FULL,
        0xF0F0F0F0F0F0F0F0ULL,
        0x00FF00FF00FF00FFULL,
        0xFF00FF00FF00FF00ULL
    };
    const size_t num_base = sizeof(base_patterns) / sizeof(base_patterns[0]);

    if (ctx->thread_buf == NULL || ctx->thread_buf_bytes < sizeof(uint64_t)) {
        snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                 "Working buffer not allocated for memory-pattern workload");
        return CROSS_CPU_STATUS_INIT_ERROR;
    }

    uint64_t *buf = (uint64_t *)ctx->thread_buf;
    const size_t words = ctx->thread_buf_bytes / sizeof(uint64_t);
    uint64_t bytes = 0;

    for (uint64_t pass = 0; pass < ctx->batch_iters; pass++) {
        const uint64_t sel = (ctx->seed + pass) % (num_base + 2u);
        uint64_t pat_even = 0;
        uint64_t pat_odd = 0;
        if (sel < num_base) {
            pat_even = base_patterns[sel];
            pat_odd = ~pat_even; /* Checkerboard adjacent words */
        } else if (sel == num_base) {
            pat_even = 1ULL << ((ctx->seed + pass) & 63u); /* Walking-1 */
            pat_odd = ~pat_even;
        } else {
            pat_even = ~(1ULL << ((ctx->seed + pass) & 63u)); /* Walking-0 */
            pat_odd = ~pat_even;
        }

        for (size_t i = 0; i < words; i++) {
            buf[i] = (i & 1u) ? pat_odd : pat_even;
        }

        cross_cpu_arch_flush_cache_range(buf, ctx->thread_buf_bytes);

        if (ctx->inject_fault && ctx->thread_idx == 0u && pass == 0ULL && words > 0u) {
            buf[words - 1u] ^= 0x8000000000000000ULL;
        }

        for (size_t i = 0; i < words; i++) {
            const uint64_t expected = (i & 1u) ? pat_odd : pat_even;
            const uint64_t observed = buf[i];
            if (observed != expected) {
                ctx->errors++;
                ctx->expected_sample = (double)expected;
                ctx->observed_sample = (double)observed;
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "memory-pattern bit error at word %zu (expected 0x%016llx, got 0x%016llx)",
                         i, (unsigned long long)expected, (unsigned long long)observed);
                ctx->ops_done = bytes;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        bytes += (uint64_t)ctx->thread_buf_bytes * 2ULL;
    }

    ctx->ops_done = bytes;
    return CROSS_CPU_STATUS_OK;
}

static int32_t scalar_memory_sweep(cross_cpu_worker_ctx_t *ctx) {
    if (ctx->thread_buf == NULL || ctx->thread_buf_bytes < sizeof(uint64_t)) {
        snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                 "Working buffer not allocated for memory-sweep workload");
        return CROSS_CPU_STATUS_INIT_ERROR;
    }

    uint64_t *buf = (uint64_t *)ctx->thread_buf;
    const size_t words = ctx->thread_buf_bytes / sizeof(uint64_t);
    uint64_t bytes = 0;

    for (uint64_t pass = 0; pass < ctx->batch_iters; pass++) {
        const uint64_t salt = ctx->seed ^ (pass * 0xbf58476d1ce4e5b9ULL);

        for (size_t i = 0; i < words; i++) {
            buf[i] = splitmix64(salt ^ ((uint64_t)i * 0x9e3779b97f4a7c15ULL));
        }

        cross_cpu_arch_flush_cache_range(buf, ctx->thread_buf_bytes);

        if (ctx->inject_fault && ctx->thread_idx == 0u && pass == 0ULL && words > 1u) {
            buf[1] ^= 0x1ULL;
        }

        for (size_t i = 0; i < words; i++) {
            const uint64_t expected = splitmix64(salt ^ ((uint64_t)i * 0x9e3779b97f4a7c15ULL));
            const uint64_t observed = buf[i];
            if (observed != expected) {
                ctx->errors++;
                ctx->expected_sample = (double)expected;
                ctx->observed_sample = (double)observed;
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "memory-sweep address-hash error at word %zu (expected 0x%016llx, got 0x%016llx)",
                         i, (unsigned long long)expected, (unsigned long long)observed);
                ctx->ops_done = bytes;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        bytes += (uint64_t)ctx->thread_buf_bytes * 2ULL;
    }

    ctx->ops_done = bytes;
    return CROSS_CPU_STATUS_OK;
}

static int32_t scalar_coherency_atomic(cross_cpu_worker_ctx_t *ctx) {
    if (ctx->shared_buf == NULL ||
        ctx->shared_buf_bytes < (size_t)CROSS_CPU_COHERENCY_SLOTS * CROSS_CPU_COHERENCY_SLOT_STRIDE) {
        snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                 "Shared coherency buffer not initialized");
        return CROSS_CPU_STATUS_INIT_ERROR;
    }

    uint64_t local_adds = 0;
    uint64_t total_atomic_ops = 0;

    /* Per-thread private atomic CAS loop verifies local atomic ordering in addition to shared contention */
    _Atomic uint64_t local_cas;
    atomic_init(&local_cas, 0ULL);

    const uint64_t iters = ctx->batch_iters;
    for (uint64_t i = 0; i < iters; i++) {
        uint32_t slot_idx = (uint32_t)((i + ctx->thread_idx) % CROSS_CPU_COHERENCY_SLOTS);
        _Atomic uint64_t *slot =
            (_Atomic uint64_t *)((uint8_t *)ctx->shared_buf +
                                 ((size_t)slot_idx * CROSS_CPU_COHERENCY_SLOT_STRIDE));

        /* Contended fetch_add across all threads on shared cache lines */
        if (ctx->inject_fault && ctx->thread_idx == 0u && i == 0ULL) {
            /* Deliberately skip 1 increment while counting it to trigger post-join coherency failure */
            local_adds += 1ULL;
        } else {
            (void)atomic_fetch_add_explicit(slot, 1ULL, memory_order_acq_rel);
            local_adds += 1ULL;
        }
        total_atomic_ops += 1ULL;

        /* CAS verification step */
        uint64_t expected = i;
        while (!atomic_compare_exchange_weak_explicit(
                   &local_cas, &expected, i + 1ULL,
                   memory_order_acq_rel, memory_order_relaxed)) {
            ctx->errors++;
            ctx->expected_sample = (double)i;
            ctx->observed_sample = (double)expected;
            snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                     "Local atomic CAS invariant broken (expected %llu, got %llu)",
                     (unsigned long long)i, (unsigned long long)expected);
            ctx->ops_done = total_atomic_ops;
            ctx->coherency_adds = local_adds;
            return CROSS_CPU_STATUS_COMPUTE_ERROR;
        }
        total_atomic_ops += 1ULL;
    }

    if (atomic_load_explicit(&local_cas, memory_order_acquire) != iters) {
        ctx->errors++;
        ctx->expected_sample = (double)iters;
        ctx->observed_sample = (double)atomic_load_explicit(&local_cas, memory_order_relaxed);
        snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                 "Local atomic counter mismatch at end of batch");
        ctx->ops_done = total_atomic_ops;
        ctx->coherency_adds = local_adds;
        return CROSS_CPU_STATUS_COMPUTE_ERROR;
    }

    ctx->ops_done = total_atomic_ops;
    ctx->coherency_adds = local_adds;
    return CROSS_CPU_STATUS_OK;
}

void cross_cpu_register_scalar_backend(cross_cpu_registry_t *reg) {
    cross_cpu_backend_t b;
    memset(&b, 0, sizeof(b));

    b.isa_id = CROSS_CPU_ISA_SCALAR;
    b.priority = 20u;
    b.available = 1u;
    b.compiled_in = 1u;
    snprintf(b.name, sizeof(b.name), "scalar");
    snprintf(b.detail, sizeof(b.detail), "Portable C11 scalar, memory, cache, and <stdatomic.h> kernels");

    b.kernels[CROSS_CPU_WL_COMPUTE_INT64]    = scalar_compute_int64;
    b.kernels[CROSS_CPU_WL_COMPUTE_INT32]    = scalar_compute_int32;
    b.kernels[CROSS_CPU_WL_COMPUTE_INT16]    = scalar_compute_int16;
    b.kernels[CROSS_CPU_WL_COMPUTE_INT8]     = scalar_compute_int8;
    b.kernels[CROSS_CPU_WL_COMPUTE_FP64]     = scalar_compute_fp64;
    b.kernels[CROSS_CPU_WL_COMPUTE_FP32]     = scalar_compute_fp32;
    b.kernels[CROSS_CPU_WL_CACHE_L1]         = scalar_cache_or_bandwidth;
    b.kernels[CROSS_CPU_WL_CACHE_L2]         = scalar_cache_or_bandwidth;
    b.kernels[CROSS_CPU_WL_CACHE_L3]         = scalar_cache_or_bandwidth;
    b.kernels[CROSS_CPU_WL_BANDWIDTH_RAM]    = scalar_cache_or_bandwidth;
    b.kernels[CROSS_CPU_WL_MEMORY_PATTERN]   = scalar_memory_pattern;
    b.kernels[CROSS_CPU_WL_MEMORY_SWEEP]     = scalar_memory_sweep;
    b.kernels[CROSS_CPU_WL_COHERENCY_ATOMIC] = scalar_coherency_atomic;

    /* Explicitly document zero-fake-emulation policy for hardware-only workloads */
    snprintf(b.unsupported_reasons[CROSS_CPU_WL_COMPUTE_VNNI], CROSS_CPU_MAX_REASON,
             "compute-vnni requires hardware dot-product instructions; scalar backend never emulates VNNI");
    snprintf(b.unsupported_reasons[CROSS_CPU_WL_COMPUTE_FP16], CROSS_CPU_MAX_REASON,
             "compute-fp16 requires native hardware FP16 instructions; scalar backend never upcasts FP16");
    snprintf(b.unsupported_reasons[CROSS_CPU_WL_COMPUTE_BF16], CROSS_CPU_MAX_REASON,
             "compute-bf16 requires native hardware BF16 instructions; scalar backend never upcasts BF16");
    snprintf(b.unsupported_reasons[CROSS_CPU_WL_COMPUTE_MATRIX_FP32], CROSS_CPU_MAX_REASON,
             "compute-matrix-fp32 requires hardware matrix tile engine (ARM SME); scalar emulation disabled");
    snprintf(b.unsupported_reasons[CROSS_CPU_WL_COMPUTE_MATRIX_BF16], CROSS_CPU_MAX_REASON,
             "compute-matrix-bf16 requires hardware matrix tile engine (Intel AMX/ARM SME); scalar emulation disabled");
    snprintf(b.unsupported_reasons[CROSS_CPU_WL_COMPUTE_MATRIX_INT8], CROSS_CPU_MAX_REASON,
             "compute-matrix-int8 requires hardware matrix tile engine (Intel AMX/ARM SME); scalar emulation disabled");
    snprintf(b.unsupported_reasons[CROSS_CPU_WL_CRYPTO_AES], CROSS_CPU_MAX_REASON,
             "crypto-aes requires hardware AES ISA instructions; software scalar AES is disabled");
    snprintf(b.unsupported_reasons[CROSS_CPU_WL_CRYPTO_SHA256], CROSS_CPU_MAX_REASON,
             "crypto-sha256 requires hardware SHA-256 ISA instructions; software scalar SHA-256 is disabled");

    (void)cross_cpu_registry_add(reg, &b);
}
