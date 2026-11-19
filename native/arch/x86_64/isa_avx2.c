// SPDX-License-Identifier: MIT
#include "cross_cpu_arch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if (defined(__AVX2__) && defined(__FMA__)) || defined(CROSS_CPU_HAVE_AVX2)
#include <immintrin.h>

static inline uint64_t splitmix64_avx2(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static void lcg_power_avx2(uint64_t a, uint64_t c, uint32_t steps,
                           uint64_t *out_mul, uint64_t *out_add) {
    uint64_t cur_m = a, cur_a = c, acc_m = 1ULL, acc_a = 0ULL;
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

static int32_t avx2_compute_int64(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint64_t inc_c = 0x9e3779b97f4a7c15ULL;
    const __m256i vinc = _mm256_set1_epi64x((long long)inc_c);

    uint64_t rm = 1ULL, ra = 0ULL;
    lcg_power_avx2(3ULL, inc_c, inner, &rm, &ra);

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        uint64_t init[8];
        for (int i = 0; i < 8; i++) {
            init[i] = splitmix64_avx2(ctx->seed + b * 8ULL + (uint64_t)i);
        }
        __m256i v0 = _mm256_loadu_si256((const __m256i *)&init[0]);
        __m256i v1 = _mm256_loadu_si256((const __m256i *)&init[4]);

        for (uint32_t i = 0; i < inner; i++) {
            __m256i s0 = _mm256_slli_epi64(v0, 1);
            __m256i s1 = _mm256_slli_epi64(v1, 1);
            v0 = _mm256_add_epi64(_mm256_add_epi64(v0, s0), vinc);
            v1 = _mm256_add_epi64(_mm256_add_epi64(v1, s1), vinc);
        }

        uint64_t out[8];
        _mm256_storeu_si256((__m256i *)&out[0], v0);
        _mm256_storeu_si256((__m256i *)&out[4], v1);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1ULL;
        }

        for (int i = 0; i < 8; i++) {
            uint64_t exp = init[i] * rm + ra;
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "avx2 int64 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 24ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t avx2_compute_int32(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint32_t mul_c = 1664525u;
    const uint32_t inc_c = 1013904223u;
    const __m256i vmul = _mm256_set1_epi32((int)mul_c);
    const __m256i vinc = _mm256_set1_epi32((int)inc_c);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_avx2(mul_c, inc_c, inner, &rm64, &ra64);
    const uint32_t rm = (uint32_t)rm64;
    const uint32_t ra = (uint32_t)ra64;

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        uint32_t init[16];
        for (int i = 0; i < 16; i++) {
            init[i] = (uint32_t)splitmix64_avx2(ctx->seed + b * 16ULL + (uint64_t)i);
        }
        __m256i v0 = _mm256_loadu_si256((const __m256i *)&init[0]);
        __m256i v1 = _mm256_loadu_si256((const __m256i *)&init[8]);

        for (uint32_t i = 0; i < inner; i++) {
            v0 = _mm256_add_epi32(_mm256_mullo_epi32(v0, vmul), vinc);
            v1 = _mm256_add_epi32(_mm256_mullo_epi32(v1, vmul), vinc);
        }

        uint32_t out[16];
        _mm256_storeu_si256((__m256i *)&out[0], v0);
        _mm256_storeu_si256((__m256i *)&out[8], v1);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1u;
        }

        for (int i = 0; i < 16; i++) {
            uint32_t exp = init[i] * rm + ra;
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "avx2 int32 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 32ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t avx2_compute_int16(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint16_t mul_c = 25173u;
    const uint16_t inc_c = 13849u;
    const __m256i vmul = _mm256_set1_epi16((short)mul_c);
    const __m256i vinc = _mm256_set1_epi16((short)inc_c);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_avx2(mul_c, inc_c, inner, &rm64, &ra64);
    const uint16_t rm = (uint16_t)rm64;
    const uint16_t ra = (uint16_t)ra64;

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        uint16_t init[32];
        for (int i = 0; i < 32; i++) {
            init[i] = (uint16_t)splitmix64_avx2(ctx->seed + b * 32ULL + (uint64_t)i);
        }
        __m256i v0 = _mm256_loadu_si256((const __m256i *)&init[0]);
        __m256i v1 = _mm256_loadu_si256((const __m256i *)&init[16]);

        for (uint32_t i = 0; i < inner; i++) {
            v0 = _mm256_add_epi16(_mm256_mullo_epi16(v0, vmul), vinc);
            v1 = _mm256_add_epi16(_mm256_mullo_epi16(v1, vmul), vinc);
        }

        uint16_t out[32];
        _mm256_storeu_si256((__m256i *)&out[0], v0);
        _mm256_storeu_si256((__m256i *)&out[16], v1);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1u;
        }

        for (int i = 0; i < 32; i++) {
            uint16_t exp = (uint16_t)(init[i] * rm + ra);
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "avx2 int16 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 64ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t avx2_compute_int8(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint8_t inc_c = 53u;
    const __m256i vinc = _mm256_set1_epi8((char)inc_c);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_avx2(3ULL, inc_c, inner, &rm64, &ra64);
    const uint8_t rm = (uint8_t)rm64;
    const uint8_t ra = (uint8_t)ra64;

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        uint8_t init[64];
        for (int i = 0; i < 64; i++) {
            init[i] = (uint8_t)splitmix64_avx2(ctx->seed + b * 64ULL + (uint64_t)i);
        }
        __m256i v0 = _mm256_loadu_si256((const __m256i *)&init[0]);
        __m256i v1 = _mm256_loadu_si256((const __m256i *)&init[32]);

        for (uint32_t i = 0; i < inner; i++) {
            __m256i d0 = _mm256_add_epi8(v0, v0);
            __m256i d1 = _mm256_add_epi8(v1, v1);
            v0 = _mm256_add_epi8(_mm256_add_epi8(v0, d0), vinc);
            v1 = _mm256_add_epi8(_mm256_add_epi8(v1, d1), vinc);
        }

        uint8_t out[64];
        _mm256_storeu_si256((__m256i *)&out[0], v0);
        _mm256_storeu_si256((__m256i *)&out[32], v1);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1u;
        }

        for (int i = 0; i < 64; i++) {
            uint8_t exp = (uint8_t)(init[i] * rm + ra);
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "avx2 int8 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 192ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

#if defined(__AVXVNNI__) || defined(CROSS_CPU_HAVE_X86_AVXVNNI)
__attribute__((target("avx2,avxvnni")))
static int32_t avx2_compute_vnni(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    uint8_t a_bytes[32];
    int8_t b_bytes[32];
    int32_t dot_ref[8];

    for (int lane = 0; lane < 8; lane++) {
        int32_t sum = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t av = (uint8_t)(1u + ((lane + j) % 7));
            int8_t bv = (int8_t)(((j & 1) ? -1 : 1) * (2 + (lane % 5)));
            a_bytes[lane * 4 + j] = av;
            b_bytes[lane * 4 + j] = bv;
            sum += (int32_t)av * (int32_t)bv;
        }
        dot_ref[lane] = sum;
    }

    const __m256i va = _mm256_loadu_si256((const __m256i *)a_bytes);
    const __m256i vb = _mm256_loadu_si256((const __m256i *)b_bytes);

    uint64_t ops = 0;
    for (uint64_t iter = 0; iter < ctx->batch_iters; iter++) {
        int32_t init0 = (int32_t)(ctx->seed + iter) & 0xFF;
        __m256i acc0 = _mm256_set1_epi32(init0);
        __m256i acc1 = _mm256_set1_epi32(init0 + 1);

        for (uint32_t i = 0; i < inner; i++) {
            acc0 = _mm256_dpbusd_avx_epi32(acc0, va, vb);
            acc1 = _mm256_dpbusd_avx_epi32(acc1, va, vb);
        }

        int32_t out0[8], out1[8];
        _mm256_storeu_si256((__m256i *)out0, acc0);
        _mm256_storeu_si256((__m256i *)out1, acc1);

        if (ctx->inject_fault && ctx->thread_idx == 0u && iter == 0ULL) {
            out0[0] += 1;
        }

        for (int lane = 0; lane < 8; lane++) {
            int32_t exp0 = init0 + (int32_t)inner * dot_ref[lane];
            int32_t exp1 = (init0 + 1) + (int32_t)inner * dot_ref[lane];
            if (out0[lane] != exp0 || out1[lane] != exp1) {
                ctx->errors++;
                ctx->expected_sample = (double)exp0;
                ctx->observed_sample = (double)out0[lane];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "avx2-vnni mismatch lane %d (expected %d, got %d)",
                         lane, exp0, out0[lane]);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        /* 2 accumulators * 8 lanes * 8 ops (4 mul + 4 add) * inner */
        ops += (uint64_t)inner * 128ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}
#endif

static int32_t avx2_compute_fp64(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 128u;
    const __m256d va = _mm256_set1_pd(0.75);
    const __m256d vb_p = _mm256_set_pd(0.50, 0.4375, 0.375, 0.3125);
    const __m256d vb_n = _mm256_set_pd(-0.50, -0.4375, -0.375, -0.3125);
    const double refs[4] = { 1.25, 1.50, 1.75, 2.00 };

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        double d = (double)((ctx->seed + b) & 0xFFu) * 0.00390625;
        __m256d vp = _mm256_set_pd(1.25 + d, 1.0 + d, 0.75 + d, 0.5 + d);
        __m256d vn = _mm256_set_pd(-(1.25 + d), -(1.0 + d), -(0.75 + d), -(0.5 + d));

        for (uint32_t i = 0; i < inner; i++) {
            vp = _mm256_fmadd_pd(va, vp, vb_p);
            vn = _mm256_fmadd_pd(va, vn, vb_n);
        }

        double op[4], on[4];
        _mm256_storeu_pd(op, vp);
        _mm256_storeu_pd(on, vn);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            op[0] += 1.0e-5;
        }

        for (int i = 0; i < 4; i++) {
            if (!isfinite(op[i]) || !isfinite(on[i]) ||
                fabs(op[i] + on[i]) > 1.0e-12 || fabs(op[i] - refs[i]) > 1.0e-12) {
                ctx->errors++;
                ctx->expected_sample = refs[i];
                ctx->observed_sample = op[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "avx2 fp64 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 16ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t avx2_compute_fp32(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 128u;
    const __m256 va = _mm256_set1_ps(0.75f);
    const float b_vals[8] = {
        0.25f, 0.3125f, 0.375f, 0.4375f, 0.50f, 0.5625f, 0.625f, 0.6875f
    };
    const float b_neg[8] = {
        -0.25f, -0.3125f, -0.375f, -0.4375f, -0.50f, -0.5625f, -0.625f, -0.6875f
    };
    const __m256 vb_p = _mm256_loadu_ps(b_vals);
    const __m256 vb_n = _mm256_loadu_ps(b_neg);

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        float d = (float)((ctx->seed + b) & 0xFFu) * 0.00390625f;
        __m256 vp = _mm256_set1_ps(0.75f + d);
        __m256 vn = _mm256_set1_ps(-(0.75f + d));

        for (uint32_t i = 0; i < inner; i++) {
            vp = _mm256_fmadd_ps(va, vp, vb_p);
            vn = _mm256_fmadd_ps(va, vn, vb_n);
        }

        float op[8], on[8];
        _mm256_storeu_ps(op, vp);
        _mm256_storeu_ps(on, vn);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            op[0] += 1.0e-3f;
        }

        for (int i = 0; i < 8; i++) {
            float ref = b_vals[i] * 4.0f;
            if (!isfinite(op[i]) || !isfinite(on[i]) ||
                fabsf(op[i] + on[i]) > 1.0e-5f || fabsf(op[i] - ref) > 1.0e-5f) {
                ctx->errors++;
                ctx->expected_sample = (double)ref;
                ctx->observed_sample = (double)op[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "avx2 fp32 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 32ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t avx2_cache_rw(cross_cpu_worker_ctx_t *ctx) {
    if (ctx->thread_buf == NULL || ctx->thread_buf_bytes < 32u) {
        return CROSS_CPU_STATUS_INIT_ERROR;
    }
    uint8_t *buf = (uint8_t *)ctx->thread_buf;
    const size_t vecs = ctx->thread_buf_bytes / 32u;
    uint64_t bytes = 0;

    for (uint64_t pass = 0; pass < ctx->batch_iters; pass++) {
        const uint64_t salt = splitmix64_avx2(ctx->seed + pass);
        const __m256i base = _mm256_set_epi64x((long long)(salt ^ 0x3333333333333333ULL),
                                               (long long)(salt ^ 0x5555555555555555ULL),
                                               (long long)(salt ^ 0xaaaaaaaaaaaaaaaaULL),
                                               (long long)salt);
        const __m256i step = _mm256_set1_epi64x(0x0102030405060708LL);

        __m256i cur = base;
        for (size_t i = 0; i < vecs; i++) {
            _mm256_store_si256((__m256i *)(buf + i * 32u), cur);
            cur = _mm256_add_epi64(cur, step);
        }

        if (ctx->inject_fault && ctx->thread_idx == 0u && pass == 0ULL) {
            buf[0] ^= 0x1u;
        }

        cur = base;
        for (size_t i = 0; i < vecs; i++) {
            __m256i got = _mm256_load_si256((const __m256i *)(buf + i * 32u));
            __m256i diff = _mm256_xor_si256(got, cur);
            if (!_mm256_testz_si256(diff, diff)) {
                ctx->errors++;
                ctx->expected_sample = (double)salt;
                ctx->observed_sample = 0.0;
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "avx2 cache verify mismatch at vector %zu", i);
                ctx->ops_done = bytes;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
            cur = _mm256_add_epi64(cur, step);
        }
        bytes += (uint64_t)(vecs * 32u) * 2ULL;
    }
    ctx->ops_done = bytes;
    return CROSS_CPU_STATUS_OK;
}

static int32_t avx2_bandwidth_ram(cross_cpu_worker_ctx_t *ctx) {
    if (ctx->thread_buf == NULL || ctx->thread_buf_bytes < 32u) {
        return CROSS_CPU_STATUS_INIT_ERROR;
    }
    uint8_t *buf = (uint8_t *)ctx->thread_buf;
    const size_t vecs = ctx->thread_buf_bytes / 32u;
    uint64_t bytes = 0;

    for (uint64_t pass = 0; pass < ctx->batch_iters; pass++) {
        const uint64_t salt = splitmix64_avx2(ctx->seed + pass);
        const __m256i pat = _mm256_set_epi64x((long long)(~salt), (long long)salt,
                                              (long long)(~salt), (long long)salt);

        for (size_t i = 0; i < vecs; i++) {
            _mm256_stream_si256((__m256i *)(buf + i * 32u), pat);
        }
        _mm_sfence();

        if (ctx->inject_fault && ctx->thread_idx == 0u && pass == 0ULL) {
            buf[0] ^= 0x80u;
        }

        for (size_t i = 0; i < vecs; i++) {
            __m256i got = _mm256_load_si256((const __m256i *)(buf + i * 32u));
            __m256i diff = _mm256_xor_si256(got, pat);
            if (!_mm256_testz_si256(diff, diff)) {
                ctx->errors++;
                ctx->expected_sample = (double)salt;
                ctx->observed_sample = 0.0;
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "avx2 non-temporal bandwidth-ram mismatch at vector %zu", i);
                ctx->ops_done = bytes;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        bytes += (uint64_t)(vecs * 32u) * 2ULL;
    }
    ctx->ops_done = bytes;
    return CROSS_CPU_STATUS_OK;
}
#endif

#if (defined(__AVX2__) && defined(__FMA__)) || defined(CROSS_CPU_HAVE_AVX2)
const uint32_t cross_cpu_x86_avx2_compiled = 1u;
#else
const uint32_t cross_cpu_x86_avx2_compiled = 0u;
#endif

#if ((defined(__AVX2__) && defined(__FMA__)) || defined(CROSS_CPU_HAVE_AVX2)) && (defined(__AVXVNNI__) || defined(CROSS_CPU_HAVE_X86_AVXVNNI))
const uint32_t cross_cpu_x86_avxvnni_compiled = 1u;
#else
const uint32_t cross_cpu_x86_avxvnni_compiled = 0u;
#endif

const cross_cpu_kernel_fn cross_cpu_x86_avx2_kernels[CROSS_CPU_WL_COUNT] = {
#if (defined(__AVX2__) && defined(__FMA__)) || defined(CROSS_CPU_HAVE_AVX2)
    [CROSS_CPU_WL_COMPUTE_INT64] = avx2_compute_int64,
    [CROSS_CPU_WL_COMPUTE_INT32] = avx2_compute_int32,
    [CROSS_CPU_WL_COMPUTE_INT16] = avx2_compute_int16,
    [CROSS_CPU_WL_COMPUTE_INT8]  = avx2_compute_int8,
    [CROSS_CPU_WL_COMPUTE_FP64]  = avx2_compute_fp64,
    [CROSS_CPU_WL_COMPUTE_FP32]  = avx2_compute_fp32,
    [CROSS_CPU_WL_CACHE_L1]      = avx2_cache_rw,
    [CROSS_CPU_WL_CACHE_L2]      = avx2_cache_rw,
    [CROSS_CPU_WL_CACHE_L3]      = avx2_cache_rw,
    [CROSS_CPU_WL_BANDWIDTH_RAM] = avx2_bandwidth_ram,
#  if defined(__AVXVNNI__) || defined(CROSS_CPU_HAVE_X86_AVXVNNI)
    [CROSS_CPU_WL_COMPUTE_VNNI]  = avx2_compute_vnni,
#  endif
#else
    NULL
#endif
};
