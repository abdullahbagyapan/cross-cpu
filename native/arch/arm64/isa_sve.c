// SPDX-License-Identifier: MIT
#include "cross_cpu_arch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__ARM_FEATURE_SVE) || defined(CROSS_CPU_HAVE_SVE)
#include <arm_sve.h>

#define SVE_MAX_BYTES 256u

static inline uint64_t splitmix64_sve(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static void lcg_power_sve(uint64_t a, uint64_t c, uint32_t steps,
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

static int32_t sve_compute_int64(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint64_t mul_c = 6364136223846793005ULL;
    const uint64_t inc_c = 1442695040888963407ULL;
    const uint64_t lanes = svcntd();
    const svbool_t pg = svptrue_b64();
    const svuint64_t vmul = svdup_u64(mul_c);
    const svuint64_t vinc = svdup_u64(inc_c);

    uint64_t rm = 1ULL, ra = 0ULL;
    lcg_power_sve(mul_c, inc_c, inner, &rm, &ra);

    uint64_t init[SVE_MAX_BYTES / 8u];
    uint64_t out[SVE_MAX_BYTES / 8u];
    uint64_t ops = 0;

    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        for (uint64_t i = 0; i < lanes; i++) {
            init[i] = splitmix64_sve(ctx->seed + b * lanes + i);
        }
        svuint64_t v = svld1_u64(pg, init);
        for (uint32_t i = 0; i < inner; i++) {
            v = svmla_u64_x(pg, vinc, v, vmul);
        }
        svst1_u64(pg, out, v);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1ULL;
        }

        for (uint64_t i = 0; i < lanes; i++) {
            uint64_t exp = init[i] * rm + ra;
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sve int64 mismatch lane %llu", (unsigned long long)i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * lanes * 2ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sve_compute_int32(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint32_t mul_c = 1664525u;
    const uint32_t inc_c = 1013904223u;
    const uint64_t lanes = svcntw();
    const svbool_t pg = svptrue_b32();
    const svuint32_t vmul = svdup_u32(mul_c);
    const svuint32_t vinc = svdup_u32(inc_c);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_sve(mul_c, inc_c, inner, &rm64, &ra64);
    const uint32_t rm = (uint32_t)rm64;
    const uint32_t ra = (uint32_t)ra64;

    uint32_t init[SVE_MAX_BYTES / 4u];
    uint32_t out[SVE_MAX_BYTES / 4u];
    uint64_t ops = 0;

    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        for (uint64_t i = 0; i < lanes; i++) {
            init[i] = (uint32_t)splitmix64_sve(ctx->seed + b * lanes + i);
        }
        svuint32_t v = svld1_u32(pg, init);
        for (uint32_t i = 0; i < inner; i++) {
            v = svmla_u32_x(pg, vinc, v, vmul);
        }
        svst1_u32(pg, out, v);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1u;
        }

        for (uint64_t i = 0; i < lanes; i++) {
            uint32_t exp = init[i] * rm + ra;
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sve int32 mismatch lane %llu", (unsigned long long)i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * lanes * 2ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sve_compute_int16(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint16_t mul_c = 25173u;
    const uint16_t inc_c = 13849u;
    const uint64_t lanes = svcnth();
    const svbool_t pg = svptrue_b16();
    const svuint16_t vmul = svdup_u16(mul_c);
    const svuint16_t vinc = svdup_u16(inc_c);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_sve(mul_c, inc_c, inner, &rm64, &ra64);
    const uint16_t rm = (uint16_t)rm64;
    const uint16_t ra = (uint16_t)ra64;

    uint16_t init[SVE_MAX_BYTES / 2u];
    uint16_t out[SVE_MAX_BYTES / 2u];
    uint64_t ops = 0;

    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        for (uint64_t i = 0; i < lanes; i++) {
            init[i] = (uint16_t)splitmix64_sve(ctx->seed + b * lanes + i);
        }
        svuint16_t v = svld1_u16(pg, init);
        for (uint32_t i = 0; i < inner; i++) {
            v = svmla_u16_x(pg, vinc, v, vmul);
        }
        svst1_u16(pg, out, v);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1u;
        }

        for (uint64_t i = 0; i < lanes; i++) {
            uint16_t exp = (uint16_t)(init[i] * rm + ra);
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sve int16 mismatch lane %llu", (unsigned long long)i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * lanes * 2ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sve_compute_int8(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint8_t mul_c = 141u;
    const uint8_t inc_c = 53u;
    const uint64_t lanes = svcntb();
    const svbool_t pg = svptrue_b8();
    const svuint8_t vmul = svdup_u8(mul_c);
    const svuint8_t vinc = svdup_u8(inc_c);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_sve(mul_c, inc_c, inner, &rm64, &ra64);
    const uint8_t rm = (uint8_t)rm64;
    const uint8_t ra = (uint8_t)ra64;

    uint8_t init[SVE_MAX_BYTES];
    uint8_t out[SVE_MAX_BYTES];
    uint64_t ops = 0;

    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        for (uint64_t i = 0; i < lanes; i++) {
            init[i] = (uint8_t)splitmix64_sve(ctx->seed + b * lanes + i);
        }
        svuint8_t v = svld1_u8(pg, init);
        for (uint32_t i = 0; i < inner; i++) {
            v = svmla_u8_x(pg, vinc, v, vmul);
        }
        svst1_u8(pg, out, v);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1u;
        }

        for (uint64_t i = 0; i < lanes; i++) {
            uint8_t exp = (uint8_t)(init[i] * rm + ra);
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sve int8 mismatch lane %llu", (unsigned long long)i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * lanes * 2ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sve_compute_vnni(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint64_t lanes = svcntw();
    const svbool_t pg32 = svptrue_b32();
    const svbool_t pg8  = svptrue_b8();

    int8_t a_bytes[SVE_MAX_BYTES];
    int8_t b_bytes[SVE_MAX_BYTES];
    int32_t dot_ref[SVE_MAX_BYTES / 4u];

    for (uint64_t lane = 0; lane < lanes; lane++) {
        int32_t s = 0;
        for (int j = 0; j < 4; j++) {
            int8_t av = (int8_t)(1 + ((lane + (uint64_t)j) % 7u));
            int8_t bv = (int8_t)(((j & 1) ? -1 : 1) * (2 + (int)(lane % 5u)));
            a_bytes[lane * 4u + (uint64_t)j] = av;
            b_bytes[lane * 4u + (uint64_t)j] = bv;
            s += (int32_t)av * (int32_t)bv;
        }
        dot_ref[lane] = s;
    }

    const svint8_t va = svld1_s8(pg8, a_bytes);
    const svint8_t vb = svld1_s8(pg8, b_bytes);
    int32_t out[SVE_MAX_BYTES / 4u];
    uint64_t ops = 0;

    for (uint64_t iter = 0; iter < ctx->batch_iters; iter++) {
        int32_t init0 = (int32_t)((ctx->seed + iter) & 0xFFu);
        svint32_t acc = svdup_s32(init0);
        for (uint32_t i = 0; i < inner; i++) {
            acc = svdot_s32(acc, va, vb);
        }
        svst1_s32(pg32, out, acc);

        if (ctx->inject_fault && ctx->thread_idx == 0u && iter == 0ULL) {
            out[0] += 1;
        }

        for (uint64_t lane = 0; lane < lanes; lane++) {
            int32_t exp = init0 + (int32_t)inner * dot_ref[lane];
            if (out[lane] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[lane];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sve dotprod mismatch lane %llu", (unsigned long long)lane);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * lanes * 8ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sve_compute_fp64(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 128u;
    const uint64_t lanes = svcntd();
    const svbool_t pg = svptrue_b64();
    const svfloat64_t va = svdup_f64(0.75);

    double b_pos[SVE_MAX_BYTES / 8u], b_neg[SVE_MAX_BYTES / 8u];
    for (uint64_t i = 0; i < lanes; i++) {
        b_pos[i] = 0.25 + (double)(i & 7u) * 0.0625;
        b_neg[i] = -b_pos[i];
    }
    const svfloat64_t vbp = svld1_f64(pg, b_pos);
    const svfloat64_t vbn = svld1_f64(pg, b_neg);

    double op[SVE_MAX_BYTES / 8u], on[SVE_MAX_BYTES / 8u];
    uint64_t ops = 0;

    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        double d = (double)((ctx->seed + b) & 0xFFu) * 0.00390625;
        svfloat64_t vp = svdup_f64(0.75 + d);
        svfloat64_t vn = svdup_f64(-(0.75 + d));
        for (uint32_t i = 0; i < inner; i++) {
            vp = svmla_f64_x(pg, vbp, va, vp);
            vn = svmla_f64_x(pg, vbn, va, vn);
        }
        svst1_f64(pg, op, vp);
        svst1_f64(pg, on, vn);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            op[0] += 1.0e-5;
        }

        for (uint64_t i = 0; i < lanes; i++) {
            double ref = b_pos[i] * 4.0;
            if (!isfinite(op[i]) || !isfinite(on[i]) ||
                fabs(op[i] + on[i]) > 1.0e-12 || fabs(op[i] - ref) > 1.0e-12) {
                ctx->errors++;
                ctx->expected_sample = ref;
                ctx->observed_sample = op[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sve fp64 mismatch lane %llu", (unsigned long long)i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * lanes * 4ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sve_compute_fp32(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 128u;
    const uint64_t lanes = svcntw();
    const svbool_t pg = svptrue_b32();
    const svfloat32_t va = svdup_f32(0.75f);

    float b_pos[SVE_MAX_BYTES / 4u], b_neg[SVE_MAX_BYTES / 4u];
    for (uint64_t i = 0; i < lanes; i++) {
        b_pos[i] = 0.25f + (float)(i & 7u) * 0.0625f;
        b_neg[i] = -b_pos[i];
    }
    const svfloat32_t vbp = svld1_f32(pg, b_pos);
    const svfloat32_t vbn = svld1_f32(pg, b_neg);

    float op[SVE_MAX_BYTES / 4u], on[SVE_MAX_BYTES / 4u];
    uint64_t ops = 0;

    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        float d = (float)((ctx->seed + b) & 0xFFu) * 0.00390625f;
        svfloat32_t vp = svdup_f32(0.75f + d);
        svfloat32_t vn = svdup_f32(-(0.75f + d));
        for (uint32_t i = 0; i < inner; i++) {
            vp = svmla_f32_x(pg, vbp, va, vp);
            vn = svmla_f32_x(pg, vbn, va, vn);
        }
        svst1_f32(pg, op, vp);
        svst1_f32(pg, on, vn);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            op[0] += 1.0e-3f;
        }

        for (uint64_t i = 0; i < lanes; i++) {
            float ref = b_pos[i] * 4.0f;
            if (!isfinite(op[i]) || !isfinite(on[i]) ||
                fabsf(op[i] + on[i]) > 1.0e-5f || fabsf(op[i] - ref) > 1.0e-5f) {
                ctx->errors++;
                ctx->expected_sample = (double)ref;
                ctx->observed_sample = (double)op[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sve fp32 mismatch lane %llu", (unsigned long long)i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * lanes * 4ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sve_compute_fp16(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 64u;
    const uint64_t lanes = svcnth();
    const svbool_t pg = svptrue_b16();
    const svfloat16_t va = svdup_f16((float16_t)0.75f);

    float16_t b_pos[SVE_MAX_BYTES / 2u], b_neg[SVE_MAX_BYTES / 2u];
    for (uint64_t i = 0; i < lanes; i++) {
        float v = 0.25f + (float)(i & 7u) * 0.0625f;
        b_pos[i] = (float16_t)v;
        b_neg[i] = (float16_t)(-v);
    }
    const svfloat16_t vbp = svld1_f16(pg, b_pos);
    const svfloat16_t vbn = svld1_f16(pg, b_neg);

    float16_t op[SVE_MAX_BYTES / 2u], on[SVE_MAX_BYTES / 2u];
    uint64_t ops = 0;

    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        svfloat16_t vp = svdup_f16((float16_t)1.0f);
        svfloat16_t vn = svdup_f16((float16_t)-1.0f);
        for (uint32_t i = 0; i < inner; i++) {
            vp = svmla_f16_x(pg, vbp, va, vp);
            vn = svmla_f16_x(pg, vbn, va, vn);
        }
        svst1_f16(pg, op, vp);
        svst1_f16(pg, on, vn);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            op[0] = (float16_t)((float)op[0] + 0.25f);
        }

        for (uint64_t i = 0; i < lanes; i++) {
            float p = (float)op[i], n = (float)on[i];
            float ref = (float)b_pos[i] * 4.0f;
            if (!isfinite(p) || !isfinite(n) ||
                fabsf(p + n) > 1.0e-2f || fabsf(p - ref) > 2.0e-2f) {
                ctx->errors++;
                ctx->expected_sample = (double)ref;
                ctx->observed_sample = (double)p;
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sve fp16 mismatch lane %llu", (unsigned long long)i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * lanes * 4ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sve_cache_rw(cross_cpu_worker_ctx_t *ctx) {
    if (ctx->thread_buf == NULL || ctx->thread_buf_bytes < 64u) {
        return CROSS_CPU_STATUS_INIT_ERROR;
    }
    uint64_t *buf = (uint64_t *)ctx->thread_buf;
    const size_t words = ctx->thread_buf_bytes / sizeof(uint64_t);
    const uint64_t step = svcntd();
    uint64_t bytes = 0;

    for (uint64_t pass = 0; pass < ctx->batch_iters; pass++) {
        const uint64_t salt = splitmix64_sve(ctx->seed + pass);
        const svuint64_t base = svdup_u64(salt);
        const svuint64_t inc = svdup_u64(0x0102030405060708ULL);

        svuint64_t cur = base;
        for (size_t i = 0; i < words; i += (size_t)step) {
            svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)words);
            svst1_u64(pg, buf + i, cur);
            cur = svadd_u64_x(pg, cur, inc);
        }

        if (ctx->inject_fault && ctx->thread_idx == 0u && pass == 0ULL) {
            buf[0] ^= 1ULL;
        }

        cur = base;
        for (size_t i = 0; i < words; i += (size_t)step) {
            svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)words);
            svuint64_t got = svld1_u64(pg, buf + i);
            svbool_t neq = svcmpne_u64(pg, got, cur);
            if (svptest_any(pg, neq)) {
                ctx->errors++;
                ctx->expected_sample = (double)salt;
                ctx->observed_sample = (double)buf[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sve cache mismatch near word %zu", i);
                ctx->ops_done = bytes;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
            cur = svadd_u64_x(pg, cur, inc);
        }
        bytes += (uint64_t)ctx->thread_buf_bytes * 2ULL;
    }
    ctx->ops_done = bytes;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sve_bandwidth_ram(cross_cpu_worker_ctx_t *ctx) {
    if (ctx->thread_buf == NULL || ctx->thread_buf_bytes < 64u) {
        return CROSS_CPU_STATUS_INIT_ERROR;
    }
    uint64_t *buf = (uint64_t *)ctx->thread_buf;
    const size_t words = ctx->thread_buf_bytes / sizeof(uint64_t);
    const uint64_t step = svcntd();
    uint64_t bytes = 0;

    for (uint64_t pass = 0; pass < ctx->batch_iters; pass++) {
        const uint64_t salt = splitmix64_sve(ctx->seed + pass);
        const svuint64_t pat = svdup_u64(salt);

        for (size_t i = 0; i < words; i += (size_t)step) {
            svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)words);
            svstnt1_u64(pg, buf + i, pat);
        }
        __asm__ volatile("dmb ish" ::: "memory");

        if (ctx->inject_fault && ctx->thread_idx == 0u && pass == 0ULL) {
            buf[0] ^= 0x80ULL;
        }

        for (size_t i = 0; i < words; i += (size_t)step) {
            svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)words);
            svuint64_t got = svld1_u64(pg, buf + i);
            svbool_t neq = svcmpne_u64(pg, got, pat);
            if (svptest_any(pg, neq)) {
                ctx->errors++;
                ctx->expected_sample = (double)salt;
                ctx->observed_sample = (double)buf[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "sve non-temporal bandwidth-ram mismatch near word %zu", i);
                ctx->ops_done = bytes;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        bytes += (uint64_t)ctx->thread_buf_bytes * 2ULL;
    }
    ctx->ops_done = bytes;
    return CROSS_CPU_STATUS_OK;
}
#endif

#if defined(__ARM_FEATURE_SVE) || defined(CROSS_CPU_HAVE_SVE)
const uint32_t cross_cpu_arm64_sve_compiled = 1u;
#else
const uint32_t cross_cpu_arm64_sve_compiled = 0u;
#endif

const cross_cpu_kernel_fn cross_cpu_arm64_sve_kernels[CROSS_CPU_WL_COUNT] = {
#if defined(__ARM_FEATURE_SVE) || defined(CROSS_CPU_HAVE_SVE)
    [CROSS_CPU_WL_COMPUTE_INT64] = sve_compute_int64,
    [CROSS_CPU_WL_COMPUTE_INT32] = sve_compute_int32,
    [CROSS_CPU_WL_COMPUTE_INT16] = sve_compute_int16,
    [CROSS_CPU_WL_COMPUTE_INT8]  = sve_compute_int8,
    [CROSS_CPU_WL_COMPUTE_VNNI]  = sve_compute_vnni,
    [CROSS_CPU_WL_COMPUTE_FP64]  = sve_compute_fp64,
    [CROSS_CPU_WL_COMPUTE_FP32]  = sve_compute_fp32,
    [CROSS_CPU_WL_COMPUTE_FP16]  = sve_compute_fp16,
    [CROSS_CPU_WL_CACHE_L1]      = sve_cache_rw,
    [CROSS_CPU_WL_CACHE_L2]      = sve_cache_rw,
    [CROSS_CPU_WL_CACHE_L3]      = sve_cache_rw,
    [CROSS_CPU_WL_BANDWIDTH_RAM] = sve_bandwidth_ram,
#else
    NULL
#endif
};
