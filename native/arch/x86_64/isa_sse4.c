// SPDX-License-Identifier: MIT
#include "cross_cpu_arch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__SSE4_2__) || defined(CROSS_CPU_HAVE_SSE4)
#include <immintrin.h>

static inline uint64_t splitmix64_sse(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static void lcg_power_sse(uint64_t a, uint64_t c, uint32_t steps,
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

static int32_t sse4_compute_int64(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint64_t inc_c = 0x9e3779b97f4a7c15ULL;
    const __m128i vinc = _mm_set1_epi64x((long long)inc_c);
    uint64_t ops = 0;

    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        uint64_t init[4];
        for (int i = 0; i < 4; i++) {
            init[i] = splitmix64_sse(ctx->seed + b * 4ULL + (uint64_t)i);
        }
        __m128i v0 = _mm_loadu_si128((const __m128i *)&init[0]);
        __m128i v1 = _mm_loadu_si128((const __m128i *)&init[2]);

        for (uint32_t i = 0; i < inner; i++) {
            /* x = x + (x << 1) + inc = 3*x + inc (exact 64-bit modular recurrence) */
            __m128i s0 = _mm_slli_epi64(v0, 1);
            __m128i s1 = _mm_slli_epi64(v1, 1);
            v0 = _mm_add_epi64(_mm_add_epi64(v0, s0), vinc);
            v1 = _mm_add_epi64(_mm_add_epi64(v1, s1), vinc);
        }

        uint64_t out[4];
        _mm_storeu_si128((__m128i *)&out[0], v0);
        _mm_storeu_si128((__m128i *)&out[2], v1);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1ULL;
        }

        uint64_t rm = 1ULL, ra = 0ULL;
        lcg_power_sse(3ULL, inc_c, inner, &rm, &ra);
        for (int i = 0; i < 4; i++) {
            uint64_t exp = init[i] * rm + ra;
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sse4 int64 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        /* 4 lanes * 3 ops (shift + 2 adds) * inner */
        ops += (uint64_t)inner * 12ULL;
    }

    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sse4_compute_int32(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint32_t mul_c = 1664525u;
    const uint32_t inc_c = 1013904223u;
    const __m128i vmul = _mm_set1_epi32((int)mul_c);
    const __m128i vinc = _mm_set1_epi32((int)inc_c);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_sse(mul_c, inc_c, inner, &rm64, &ra64);
    const uint32_t rm = (uint32_t)rm64;
    const uint32_t ra = (uint32_t)ra64;

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        uint32_t init[8];
        for (int i = 0; i < 8; i++) {
            init[i] = (uint32_t)splitmix64_sse(ctx->seed + b * 8ULL + (uint64_t)i);
        }
        __m128i v0 = _mm_loadu_si128((const __m128i *)&init[0]);
        __m128i v1 = _mm_loadu_si128((const __m128i *)&init[4]);

        for (uint32_t i = 0; i < inner; i++) {
            v0 = _mm_add_epi32(_mm_mullo_epi32(v0, vmul), vinc);
            v1 = _mm_add_epi32(_mm_mullo_epi32(v1, vmul), vinc);
        }

        uint32_t out[8];
        _mm_storeu_si128((__m128i *)&out[0], v0);
        _mm_storeu_si128((__m128i *)&out[4], v1);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1u;
        }

        for (int i = 0; i < 8; i++) {
            uint32_t exp = init[i] * rm + ra;
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sse4 int32 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 16ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sse4_compute_int16(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint16_t mul_c = 25173u;
    const uint16_t inc_c = 13849u;
    const __m128i vmul = _mm_set1_epi16((short)mul_c);
    const __m128i vinc = _mm_set1_epi16((short)inc_c);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_sse(mul_c, inc_c, inner, &rm64, &ra64);
    const uint16_t rm = (uint16_t)rm64;
    const uint16_t ra = (uint16_t)ra64;

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        uint16_t init[16];
        for (int i = 0; i < 16; i++) {
            init[i] = (uint16_t)splitmix64_sse(ctx->seed + b * 16ULL + (uint64_t)i);
        }
        __m128i v0 = _mm_loadu_si128((const __m128i *)&init[0]);
        __m128i v1 = _mm_loadu_si128((const __m128i *)&init[8]);

        for (uint32_t i = 0; i < inner; i++) {
            v0 = _mm_add_epi16(_mm_mullo_epi16(v0, vmul), vinc);
            v1 = _mm_add_epi16(_mm_mullo_epi16(v1, vmul), vinc);
        }

        uint16_t out[16];
        _mm_storeu_si128((__m128i *)&out[0], v0);
        _mm_storeu_si128((__m128i *)&out[8], v1);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1u;
        }

        for (int i = 0; i < 16; i++) {
            uint16_t exp = (uint16_t)(init[i] * rm + ra);
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sse4 int16 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 32ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sse4_compute_int8(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint8_t inc_c = 53u;
    const __m128i vinc = _mm_set1_epi8((char)inc_c);
    uint64_t ops = 0;

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_sse(3ULL, inc_c, inner, &rm64, &ra64);
    const uint8_t rm = (uint8_t)rm64;
    const uint8_t ra = (uint8_t)ra64;

    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        uint8_t init[32];
        for (int i = 0; i < 32; i++) {
            init[i] = (uint8_t)splitmix64_sse(ctx->seed + b * 32ULL + (uint64_t)i);
        }
        __m128i v0 = _mm_loadu_si128((const __m128i *)&init[0]);
        __m128i v1 = _mm_loadu_si128((const __m128i *)&init[16]);

        for (uint32_t i = 0; i < inner; i++) {
            /* x = 3*x + inc using byte adds */
            __m128i d0 = _mm_add_epi8(v0, v0);
            __m128i d1 = _mm_add_epi8(v1, v1);
            v0 = _mm_add_epi8(_mm_add_epi8(v0, d0), vinc);
            v1 = _mm_add_epi8(_mm_add_epi8(v1, d1), vinc);
        }

        uint8_t out[32];
        _mm_storeu_si128((__m128i *)&out[0], v0);
        _mm_storeu_si128((__m128i *)&out[16], v1);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1u;
        }

        for (int i = 0; i < 32; i++) {
            uint8_t exp = (uint8_t)(init[i] * rm + ra);
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sse4 int8 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 96ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sse4_compute_fp64(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 128u;
    const __m128d va = _mm_set1_pd(0.75);
    const __m128d vb_p = _mm_set_pd(0.4375, 0.3125);
    const __m128d vb_n = _mm_set_pd(-0.4375, -0.3125);
    const double refs[2] = { 0.3125 / 0.25, 0.4375 / 0.25 };

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        double d = (double)((ctx->seed + b) & 0xFFu) * 0.00390625;
        __m128d vp = _mm_set_pd(1.0 + d, 0.5 + d);
        __m128d vn = _mm_set_pd(-(1.0 + d), -(0.5 + d));

        for (uint32_t i = 0; i < inner; i++) {
            vp = _mm_add_pd(_mm_mul_pd(va, vp), vb_p);
            vn = _mm_add_pd(_mm_mul_pd(va, vn), vb_n);
        }

        double op[2], on[2];
        _mm_storeu_pd(op, vp);
        _mm_storeu_pd(on, vn);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            op[0] += 1.0e-5;
        }

        for (int i = 0; i < 2; i++) {
            if (!isfinite(op[i]) || !isfinite(on[i]) ||
                fabs(op[i] + on[i]) > 1.0e-12 || fabs(op[i] - refs[i]) > 1.0e-12) {
                ctx->errors++;
                ctx->expected_sample = refs[i];
                ctx->observed_sample = op[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sse4 fp64 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 8ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sse4_compute_fp32(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 128u;
    const __m128 va = _mm_set1_ps(0.75f);
    const __m128 vb_p = _mm_set_ps(0.50f, 0.4375f, 0.375f, 0.3125f);
    const __m128 vb_n = _mm_set_ps(-0.50f, -0.4375f, -0.375f, -0.3125f);
    const float refs[4] = { 1.25f, 1.50f, 1.75f, 2.00f };

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        float d = (float)((ctx->seed + b) & 0xFFu) * 0.00390625f;
        __m128 vp = _mm_set_ps(1.25f + d, 1.0f + d, 0.75f + d, 0.5f + d);
        __m128 vn = _mm_set_ps(-(1.25f + d), -(1.0f + d), -(0.75f + d), -(0.5f + d));

        for (uint32_t i = 0; i < inner; i++) {
            vp = _mm_add_ps(_mm_mul_ps(va, vp), vb_p);
            vn = _mm_add_ps(_mm_mul_ps(va, vn), vb_n);
        }

        float op[4], on[4];
        _mm_storeu_ps(op, vp);
        _mm_storeu_ps(on, vn);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            op[0] += 1.0e-3f;
        }

        for (int i = 0; i < 4; i++) {
            if (!isfinite(op[i]) || !isfinite(on[i]) ||
                fabsf(op[i] + on[i]) > 1.0e-5f || fabsf(op[i] - refs[i]) > 1.0e-5f) {
                ctx->errors++;
                ctx->expected_sample = (double)refs[i];
                ctx->observed_sample = (double)op[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sse4 fp32 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 16ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sse4_cache_rw(cross_cpu_worker_ctx_t *ctx) {
    if (ctx->thread_buf == NULL || ctx->thread_buf_bytes < 16u) {
        return CROSS_CPU_STATUS_INIT_ERROR;
    }
    uint8_t *buf = (uint8_t *)ctx->thread_buf;
    const size_t vecs = ctx->thread_buf_bytes / 16u;
    uint64_t bytes = 0;

    for (uint64_t pass = 0; pass < ctx->batch_iters; pass++) {
        const uint64_t salt = splitmix64_sse(ctx->seed + pass);
        const __m128i base = _mm_set_epi64x((long long)(salt ^ 0x55aa55aa55aa55aaULL), (long long)salt);
        const __m128i step = _mm_set_epi64x(0x0102030405060708LL, 0x1112131415161718LL);

        __m128i cur = base;
        for (size_t i = 0; i < vecs; i++) {
            _mm_store_si128((__m128i *)(buf + i * 16u), cur);
            cur = _mm_add_epi64(cur, step);
        }

        if (ctx->inject_fault && ctx->thread_idx == 0u && pass == 0ULL) {
            buf[0] ^= 0x1u;
        }

        cur = base;
        for (size_t i = 0; i < vecs; i++) {
            __m128i got = _mm_load_si128((const __m128i *)(buf + i * 16u));
            __m128i diff = _mm_xor_si128(got, cur);
            if (!_mm_testz_si128(diff, diff)) {
                ctx->errors++;
                ctx->expected_sample = (double)salt;
                ctx->observed_sample = 0.0;
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sse4 cache verify mismatch at vector %zu", i);
                ctx->ops_done = bytes;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
            cur = _mm_add_epi64(cur, step);
        }
        bytes += (uint64_t)(vecs * 16u) * 2ULL;
    }
    ctx->ops_done = bytes;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sse4_bandwidth_ram(cross_cpu_worker_ctx_t *ctx) {
    if (ctx->thread_buf == NULL || ctx->thread_buf_bytes < 16u) {
        return CROSS_CPU_STATUS_INIT_ERROR;
    }
    uint8_t *buf = (uint8_t *)ctx->thread_buf;
    const size_t vecs = ctx->thread_buf_bytes / 16u;
    uint64_t bytes = 0;

    for (uint64_t pass = 0; pass < ctx->batch_iters; pass++) {
        const uint64_t salt = splitmix64_sse(ctx->seed + pass);
        const __m128i pat = _mm_set_epi64x((long long)(~salt), (long long)salt);

        for (size_t i = 0; i < vecs; i++) {
            _mm_stream_si128((__m128i *)(buf + i * 16u), pat);
        }
        _mm_sfence();

        if (ctx->inject_fault && ctx->thread_idx == 0u && pass == 0ULL) {
            buf[0] ^= 0x80u;
        }

        for (size_t i = 0; i < vecs; i++) {
            __m128i got = _mm_load_si128((const __m128i *)(buf + i * 16u));
            __m128i diff = _mm_xor_si128(got, pat);
            if (!_mm_testz_si128(diff, diff)) {
                ctx->errors++;
                ctx->expected_sample = (double)salt;
                ctx->observed_sample = 0.0;
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sse4 bandwidth-ram mismatch at vector %zu", i);
                ctx->ops_done = bytes;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        bytes += (uint64_t)(vecs * 16u) * 2ULL;
    }
    ctx->ops_done = bytes;
    return CROSS_CPU_STATUS_OK;
}

#if defined(__AES__) || defined(CROSS_CPU_HAVE_X86_AES)
__attribute__((target("aes")))
static int32_t sse4_crypto_aes(cross_cpu_worker_ctx_t *ctx) {
    /* NIST FIPS 197 Appendix C.1 AES-128 round keys */
    static const uint8_t rk_c1[11][16] = {
        { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f },
        { 0xd6, 0xaa, 0x74, 0xfd, 0xd2, 0xaf, 0x72, 0xfa, 0xda, 0xa6, 0x78, 0xf1, 0xd6, 0xab, 0x76, 0xfe },
        { 0xb6, 0x92, 0xcf, 0x0b, 0x64, 0x3d, 0xbd, 0xf1, 0xbe, 0x9b, 0xc5, 0x00, 0x68, 0x30, 0xb3, 0xfe },
        { 0xb6, 0xff, 0x74, 0x4e, 0xd2, 0xc2, 0xc9, 0xbf, 0x6c, 0x59, 0x0c, 0xbf, 0x04, 0x69, 0xbf, 0x41 },
        { 0x47, 0xf7, 0xf7, 0xbc, 0x95, 0x35, 0x3e, 0x03, 0xf9, 0x6c, 0x32, 0xbc, 0xfd, 0x05, 0x8d, 0xfd },
        { 0x3c, 0xaa, 0xa3, 0xe8, 0xa9, 0x9f, 0x9d, 0xeb, 0x50, 0xf3, 0xaf, 0x57, 0xad, 0xf6, 0x22, 0xaa },
        { 0x5e, 0x39, 0x0f, 0x7d, 0xf7, 0xa6, 0x92, 0x96, 0xa7, 0x55, 0x3d, 0xc1, 0x0a, 0xa3, 0x1f, 0x6b },
        { 0x14, 0xf9, 0x70, 0x1a, 0xe3, 0x5f, 0xe2, 0x8c, 0x44, 0x0a, 0xdf, 0x4d, 0x4e, 0xa9, 0xc0, 0x26 },
        { 0x47, 0x43, 0x87, 0x35, 0xa4, 0x1c, 0x65, 0xb9, 0xe0, 0x16, 0xba, 0xf4, 0xae, 0xbf, 0x7a, 0xd2 },
        { 0x54, 0x99, 0x32, 0xd1, 0xf0, 0x85, 0x57, 0x68, 0x10, 0x93, 0xed, 0x9c, 0xbe, 0x2c, 0x97, 0x4e },
        { 0x13, 0x11, 0x1d, 0x7f, 0xe3, 0x94, 0x4a, 0x17, 0xf3, 0x07, 0xa7, 0x8b, 0x4d, 0x2b, 0x30, 0xc5 }
    };
    static const uint8_t pt_c1[16] = {
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff
    };
    static const uint8_t ct_c1[16] = {
        0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a
    };

    __m128i rk[11];
    for (int r = 0; r < 11; r++) {
        rk[r] = _mm_loadu_si128((const __m128i *)rk_c1[r]);
    }
    const __m128i pt = _mm_loadu_si128((const __m128i *)pt_c1);
    const __m128i expected = _mm_loadu_si128((const __m128i *)ct_c1);

    uint64_t bytes = 0;
    __m128i carry0 = _mm_setzero_si128();
    __m128i carry1 = _mm_setzero_si128();
    __m128i carry2 = _mm_setzero_si128();
    __m128i carry3 = _mm_setzero_si128();
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        for (uint32_t k = 0; k < 64u; k++) {
            __m128i s0 = _mm_xor_si128(_mm_xor_si128(pt, carry0), rk[0]);
            __m128i s1 = _mm_xor_si128(_mm_xor_si128(pt, carry1), rk[0]);
            __m128i s2 = _mm_xor_si128(_mm_xor_si128(pt, carry2), rk[0]);
            __m128i s3 = _mm_xor_si128(_mm_xor_si128(pt, carry3), rk[0]);
            for (int r = 1; r < 10; r++) {
                s0 = _mm_aesenc_si128(s0, rk[r]);
                s1 = _mm_aesenc_si128(s1, rk[r]);
                s2 = _mm_aesenc_si128(s2, rk[r]);
                s3 = _mm_aesenc_si128(s3, rk[r]);
            }
            s0 = _mm_aesenclast_si128(s0, rk[10]);
            s1 = _mm_aesenclast_si128(s1, rk[10]);
            s2 = _mm_aesenclast_si128(s2, rk[10]);
            s3 = _mm_aesenclast_si128(s3, rk[10]);

            if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL && k == 0u) {
                s0 = _mm_xor_si128(s0, _mm_set1_epi8(1));
            }

            carry0 = _mm_xor_si128(s0, expected);
            carry1 = _mm_xor_si128(s1, expected);
            carry2 = _mm_xor_si128(s2, expected);
            carry3 = _mm_xor_si128(s3, expected);
            __asm__ volatile("" : "+x"(carry0), "+x"(carry1), "+x"(carry2), "+x"(carry3));

            __m128i err = _mm_or_si128(_mm_or_si128(carry0, carry1),
                                       _mm_or_si128(carry2, carry3));
            if (!_mm_testz_si128(err, err)) {
                ctx->errors++;
                ctx->expected_sample = 1.0;
                ctx->observed_sample = 0.0;
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "x86 AES-NI NIST FIPS 197 KAT ciphertext mismatch");
                ctx->ops_done = bytes;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
            bytes += 64ULL;
        }
    }
    ctx->ops_done = bytes;
    return CROSS_CPU_STATUS_OK;
}
#endif

#if defined(__SHA__) || defined(CROSS_CPU_HAVE_X86_SHA)
__attribute__((target("sha,ssse3,sse4.1")))
static int32_t sse4_crypto_sha256(cross_cpu_worker_ctx_t *ctx) {
    static const uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };
    /* Padded SHA-256 block for "abc" (16 big-endian 32-bit words) */
    static const uint32_t msg_abc[16] = {
        0x61626380u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0x00000018u
    };
    static const uint32_t exp_digest[8] = {
        0xba7816bfu, 0x8f01cfeau, 0x414140deu, 0x5dae2223u,
        0xb00361a3u, 0x96177a9cu, 0xb410ff61u, 0xf20015adu
    };

    const __m128i abef_init = _mm_set_epi32((int)0x6a09e667u, (int)0xbb67ae85u,
                                            (int)0x510e527fu, (int)0x9b05688cu);
    const __m128i cdgh_init = _mm_set_epi32((int)0x3c6ef372u, (int)0xa54ff53au,
                                            (int)0x1f83d9abu, (int)0x5be0cd19u);
    const __m128i exp_abef = _mm_set_epi32((int)exp_digest[0], (int)exp_digest[1],
                                           (int)exp_digest[4], (int)exp_digest[5]);
    const __m128i exp_cdgh = _mm_set_epi32((int)exp_digest[2], (int)exp_digest[3],
                                           (int)exp_digest[6], (int)exp_digest[7]);

    uint64_t bytes = 0;
    __m128i carry = _mm_setzero_si128();
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        for (uint32_t k = 0; k < 32u; k++) {
            __m128i abef = _mm_xor_si128(abef_init, carry);
            __m128i cdgh = cdgh_init;
            __m128i m0 = _mm_loadu_si128((const __m128i *)&msg_abc[0]);
            __m128i m1 = _mm_loadu_si128((const __m128i *)&msg_abc[4]);
            __m128i m2 = _mm_loadu_si128((const __m128i *)&msg_abc[8]);
            __m128i m3 = _mm_loadu_si128((const __m128i *)&msg_abc[12]);

            for (int r = 0; r < 64; r += 4) {
                __m128i kv = _mm_loadu_si128((const __m128i *)&K[r]);
                __m128i msg = _mm_add_epi32(m0, kv);
                cdgh = _mm_sha256rnds2_epu32(cdgh, abef, msg);
                msg = _mm_shuffle_epi32(msg, 0x0E);
                abef = _mm_sha256rnds2_epu32(abef, cdgh, msg);

                if (r < 48) {
                    __m128i next = _mm_sha256msg1_epu32(m0, m1);
                    __m128i tmp = _mm_alignr_epi8(m3, m2, 4);
                    next = _mm_add_epi32(next, tmp);
                    next = _mm_sha256msg2_epu32(next, m3);
                    m0 = m1;
                    m1 = m2;
                    m2 = m3;
                    m3 = next;
                } else {
                    m0 = m1;
                    m1 = m2;
                    m2 = m3;
                }
            }

            abef = _mm_add_epi32(abef, abef_init);
            cdgh = _mm_add_epi32(cdgh, cdgh_init);

            if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL && k == 0u) {
                abef = _mm_xor_si128(abef, _mm_set1_epi32(1));
            }

            carry = _mm_or_si128(_mm_xor_si128(abef, exp_abef),
                                 _mm_xor_si128(cdgh, exp_cdgh));
            __asm__ volatile("" : "+x"(carry));
            if (!_mm_testz_si128(carry, carry)) {
                ctx->errors++;
                ctx->expected_sample = (double)exp_digest[0];
                ctx->observed_sample = 0.0;
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "x86 SHA-NI NIST FIPS 180-4 KAT digest mismatch");
                ctx->ops_done = bytes;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
            bytes += 64ULL;
        }
    }
    ctx->ops_done = bytes;
    return CROSS_CPU_STATUS_OK;
}
#endif

#endif /* __SSE4_2__ || CROSS_CPU_HAVE_SSE4 */

#if defined(__SSE4_2__) || defined(CROSS_CPU_HAVE_SSE4)
const uint32_t cross_cpu_x86_sse4_compiled = 1u;
#else
const uint32_t cross_cpu_x86_sse4_compiled = 0u;
#endif

#if (defined(__SSE4_2__) || defined(CROSS_CPU_HAVE_SSE4)) && (defined(__AES__) || defined(CROSS_CPU_HAVE_X86_AES))
const uint32_t cross_cpu_x86_aes_compiled = 1u;
#else
const uint32_t cross_cpu_x86_aes_compiled = 0u;
#endif

#if (defined(__SSE4_2__) || defined(CROSS_CPU_HAVE_SSE4)) && (defined(__SHA__) || defined(CROSS_CPU_HAVE_X86_SHA))
const uint32_t cross_cpu_x86_sha_compiled = 1u;
#else
const uint32_t cross_cpu_x86_sha_compiled = 0u;
#endif

const cross_cpu_kernel_fn cross_cpu_x86_sse4_kernels[CROSS_CPU_WL_COUNT] = {
#if defined(__SSE4_2__) || defined(CROSS_CPU_HAVE_SSE4)
    [CROSS_CPU_WL_COMPUTE_INT64] = sse4_compute_int64,
    [CROSS_CPU_WL_COMPUTE_INT32] = sse4_compute_int32,
    [CROSS_CPU_WL_COMPUTE_INT16] = sse4_compute_int16,
    [CROSS_CPU_WL_COMPUTE_INT8]  = sse4_compute_int8,
    [CROSS_CPU_WL_COMPUTE_FP64]  = sse4_compute_fp64,
    [CROSS_CPU_WL_COMPUTE_FP32]  = sse4_compute_fp32,
    [CROSS_CPU_WL_CACHE_L1]      = sse4_cache_rw,
    [CROSS_CPU_WL_CACHE_L2]      = sse4_cache_rw,
    [CROSS_CPU_WL_CACHE_L3]      = sse4_cache_rw,
    [CROSS_CPU_WL_BANDWIDTH_RAM] = sse4_bandwidth_ram,
#  if defined(__AES__) || defined(CROSS_CPU_HAVE_X86_AES)
    [CROSS_CPU_WL_CRYPTO_AES]    = sse4_crypto_aes,
#  endif
#  if defined(__SHA__) || defined(CROSS_CPU_HAVE_X86_SHA)
    [CROSS_CPU_WL_CRYPTO_SHA256] = sse4_crypto_sha256,
#  endif
#else
    NULL
#endif
};
