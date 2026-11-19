// SPDX-License-Identifier: MIT
#include "cross_cpu_arch.h"

#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static inline uint64_t splitmix64_neon(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static void lcg_power_neon(uint64_t a, uint64_t c, uint32_t steps,
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

static int32_t neon_compute_int64(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint64_t inc_c = 0x9e3779b97f4a7c15ULL;
    const uint64x2_t vinc = vdupq_n_u64(inc_c);

    uint64_t rm = 1ULL, ra = 0ULL;
    lcg_power_neon(3ULL, inc_c, inner, &rm, &ra);

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        uint64_t init[4];
        for (int i = 0; i < 4; i++) {
            init[i] = splitmix64_neon(ctx->seed + b * 4ULL + (uint64_t)i);
        }
        uint64x2_t v0 = vld1q_u64(&init[0]);
        uint64x2_t v1 = vld1q_u64(&init[2]);

        for (uint32_t i = 0; i < inner; i++) {
            v0 = vaddq_u64(vaddq_u64(v0, vshlq_n_u64(v0, 1)), vinc);
            v1 = vaddq_u64(vaddq_u64(v1, vshlq_n_u64(v1, 1)), vinc);
        }

        uint64_t out[4];
        vst1q_u64(&out[0], v0);
        vst1q_u64(&out[2], v1);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1ULL;
        }

        for (int i = 0; i < 4; i++) {
            uint64_t exp = init[i] * rm + ra;
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "neon int64 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 12ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t neon_compute_int32(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint32_t mul_c = 1664525u;
    const uint32_t inc_c = 1013904223u;
    const uint32x4_t vmul = vdupq_n_u32(mul_c);
    const uint32x4_t vinc = vdupq_n_u32(inc_c);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_neon(mul_c, inc_c, inner, &rm64, &ra64);
    const uint32_t rm = (uint32_t)rm64;
    const uint32_t ra = (uint32_t)ra64;

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        uint32_t init[8];
        for (int i = 0; i < 8; i++) {
            init[i] = (uint32_t)splitmix64_neon(ctx->seed + b * 8ULL + (uint64_t)i);
        }
        uint32x4_t v0 = vld1q_u32(&init[0]);
        uint32x4_t v1 = vld1q_u32(&init[4]);

        for (uint32_t i = 0; i < inner; i++) {
            v0 = vmlaq_u32(vinc, v0, vmul);
            v1 = vmlaq_u32(vinc, v1, vmul);
        }

        uint32_t out[8];
        vst1q_u32(&out[0], v0);
        vst1q_u32(&out[4], v1);

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
                         "neon int32 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 16ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t neon_compute_int16(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint16_t mul_c = 25173u;
    const uint16_t inc_c = 13849u;
    const uint16x8_t vmul = vdupq_n_u16(mul_c);
    const uint16x8_t vinc = vdupq_n_u16(inc_c);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_neon(mul_c, inc_c, inner, &rm64, &ra64);
    const uint16_t rm = (uint16_t)rm64;
    const uint16_t ra = (uint16_t)ra64;

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        uint16_t init[16];
        for (int i = 0; i < 16; i++) {
            init[i] = (uint16_t)splitmix64_neon(ctx->seed + b * 16ULL + (uint64_t)i);
        }
        uint16x8_t v0 = vld1q_u16(&init[0]);
        uint16x8_t v1 = vld1q_u16(&init[8]);

        for (uint32_t i = 0; i < inner; i++) {
            v0 = vmlaq_u16(vinc, v0, vmul);
            v1 = vmlaq_u16(vinc, v1, vmul);
        }

        uint16_t out[16];
        vst1q_u16(&out[0], v0);
        vst1q_u16(&out[8], v1);

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
                         "neon int16 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 32ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t neon_compute_int8(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint8_t mul_c = 141u;
    const uint8_t inc_c = 53u;
    const uint8x16_t vmul = vdupq_n_u8(mul_c);
    const uint8x16_t vinc = vdupq_n_u8(inc_c);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_neon(mul_c, inc_c, inner, &rm64, &ra64);
    const uint8_t rm = (uint8_t)rm64;
    const uint8_t ra = (uint8_t)ra64;

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        uint8_t init[32];
        for (int i = 0; i < 32; i++) {
            init[i] = (uint8_t)splitmix64_neon(ctx->seed + b * 32ULL + (uint64_t)i);
        }
        uint8x16_t v0 = vld1q_u8(&init[0]);
        uint8x16_t v1 = vld1q_u8(&init[16]);

        for (uint32_t i = 0; i < inner; i++) {
            v0 = vmlaq_u8(vinc, v0, vmul);
            v1 = vmlaq_u8(vinc, v1, vmul);
        }

        uint8_t out[32];
        vst1q_u8(&out[0], v0);
        vst1q_u8(&out[16], v1);

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
                         "neon int8 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 64ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t neon_compute_fp64(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 128u;
    const float64x2_t va = vdupq_n_f64(0.75);
    const double b_vals[2] = { 0.3125, 0.4375 };
    const double b_neg[2]  = { -0.3125, -0.4375 };
    const float64x2_t vbp = vld1q_f64(b_vals);
    const float64x2_t vbn = vld1q_f64(b_neg);

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        double d = (double)((ctx->seed + b) & 0xFFu) * 0.00390625;
        double p_init[2] = { 0.5 + d, 1.0 + d };
        double n_init[2] = { -(0.5 + d), -(1.0 + d) };
        float64x2_t vp0 = vld1q_f64(p_init);
        float64x2_t vn0 = vld1q_f64(n_init);
        float64x2_t vp1 = vld1q_f64(p_init);
        float64x2_t vn1 = vld1q_f64(n_init);

        for (uint32_t i = 0; i < inner; i++) {
            vp0 = vfmaq_f64(vbp, va, vp0);
            vn0 = vfmaq_f64(vbn, va, vn0);
            vp1 = vfmaq_f64(vbp, va, vp1);
            vn1 = vfmaq_f64(vbn, va, vn1);
        }

        double op0[2], on0[2], op1[2], on1[2];
        vst1q_f64(op0, vp0);
        vst1q_f64(on0, vn0);
        vst1q_f64(op1, vp1);
        vst1q_f64(on1, vn1);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            op0[0] += 1.0e-5;
        }

        for (int i = 0; i < 2; i++) {
            double ref = b_vals[i] * 4.0;
            if (!isfinite(op0[i]) || !isfinite(on0[i]) ||
                !isfinite(op1[i]) || !isfinite(on1[i]) ||
                fabs(op0[i] + on0[i]) > 1.0e-12 || fabs(op0[i] - ref) > 1.0e-12 ||
                fabs(op1[i] + on1[i]) > 1.0e-12 || fabs(op1[i] - ref) > 1.0e-12) {
                ctx->errors++;
                ctx->expected_sample = ref;
                ctx->observed_sample = op0[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "neon fp64 mismatch lane %d (expected %.12g, got %.12g)",
                         i, ref, op0[i]);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        /* 4 chains * 2 lanes * 2 FLOPs * inner */
        ops += (uint64_t)inner * 16ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t neon_compute_fp32(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 128u;
    const float32x4_t va = vdupq_n_f32(0.75f);
    const float b_vals[4] = { 0.3125f, 0.375f, 0.4375f, 0.50f };
    const float b_neg[4]  = { -0.3125f, -0.375f, -0.4375f, -0.50f };
    const float32x4_t vbp = vld1q_f32(b_vals);
    const float32x4_t vbn = vld1q_f32(b_neg);

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        float d = (float)((ctx->seed + b) & 0xFFu) * 0.00390625f;
        float32x4_t vp0 = vdupq_n_f32(0.75f + d);
        float32x4_t vn0 = vdupq_n_f32(-(0.75f + d));
        float32x4_t vp1 = vdupq_n_f32(1.25f + d);
        float32x4_t vn1 = vdupq_n_f32(-(1.25f + d));

        for (uint32_t i = 0; i < inner; i++) {
            vp0 = vfmaq_f32(vbp, va, vp0);
            vn0 = vfmaq_f32(vbn, va, vn0);
            vp1 = vfmaq_f32(vbp, va, vp1);
            vn1 = vfmaq_f32(vbn, va, vn1);
        }

        float op0[4], on0[4], op1[4], on1[4];
        vst1q_f32(op0, vp0);
        vst1q_f32(on0, vn0);
        vst1q_f32(op1, vp1);
        vst1q_f32(on1, vn1);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            op0[0] += 1.0e-3f;
        }

        for (int i = 0; i < 4; i++) {
            float ref = b_vals[i] * 4.0f;
            if (!isfinite(op0[i]) || !isfinite(on0[i]) ||
                !isfinite(op1[i]) || !isfinite(on1[i]) ||
                fabsf(op0[i] + on0[i]) > 1.0e-5f || fabsf(op0[i] - ref) > 1.0e-5f ||
                fabsf(op1[i] + on1[i]) > 1.0e-5f || fabsf(op1[i] - ref) > 1.0e-5f) {
                ctx->errors++;
                ctx->expected_sample = (double)ref;
                ctx->observed_sample = (double)op0[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "neon fp32 mismatch lane %d (expected %.8g, got %.8g)",
                         i, (double)ref, (double)op0[i]);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        /* 4 chains * 4 lanes * 2 FLOPs * inner */
        ops += (uint64_t)inner * 32ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(CROSS_CPU_HAVE_ARM_FP16)
__attribute__((target("fp16")))
static int32_t neon_compute_fp16(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 64u;
    const float16x8_t va = vdupq_n_f16((float16_t)0.75f);
    float16_t b_pos[8], b_neg[8];
    for (int i = 0; i < 8; i++) {
        float v = 0.25f + (float)i * 0.0625f;
        b_pos[i] = (float16_t)v;
        b_neg[i] = (float16_t)(-v);
    }
    const float16x8_t vbp = vld1q_f16(b_pos);
    const float16x8_t vbn = vld1q_f16(b_neg);

    uint64_t ops = 0;
    for (uint64_t iter = 0; iter < ctx->batch_iters; iter++) {
        float16x8_t vp0 = vdupq_n_f16((float16_t)1.0f);
        float16x8_t vn0 = vdupq_n_f16((float16_t)-1.0f);
        float16x8_t vp1 = vdupq_n_f16((float16_t)1.5f);
        float16x8_t vn1 = vdupq_n_f16((float16_t)-1.5f);

        for (uint32_t i = 0; i < inner; i++) {
            vp0 = vfmaq_f16(vbp, va, vp0);
            vn0 = vfmaq_f16(vbn, va, vn0);
            vp1 = vfmaq_f16(vbp, va, vp1);
            vn1 = vfmaq_f16(vbn, va, vn1);
        }

        float16_t op0[8], on0[8], op1[8], on1[8];
        vst1q_f16(op0, vp0);
        vst1q_f16(on0, vn0);
        vst1q_f16(op1, vp1);
        vst1q_f16(on1, vn1);

        if (ctx->inject_fault && ctx->thread_idx == 0u && iter == 0ULL) {
            op0[0] = (float16_t)((float)op0[0] + 0.25f);
        }

        for (int i = 0; i < 8; i++) {
            float p0 = (float)op0[i], n0 = (float)on0[i];
            float p1 = (float)op1[i], n1 = (float)on1[i];
            float ref = (float)b_pos[i] * 4.0f;
            if (!isfinite(p0) || !isfinite(n0) || !isfinite(p1) || !isfinite(n1) ||
                fabsf(p0 + n0) > 1.0e-2f || fabsf(p0 - ref) > 2.0e-2f ||
                fabsf(p1 + n1) > 1.0e-2f || fabsf(p1 - ref) > 2.0e-2f) {
                ctx->errors++;
                ctx->expected_sample = (double)ref;
                ctx->observed_sample = (double)p0;
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "neon fp16 mismatch lane %d (expected %.4f, got %.4f)",
                         i, (double)ref, (double)p0);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        /* 4 chains * 8 lanes * 2 FLOPs * inner */
        ops += (uint64_t)inner * 64ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(CROSS_CPU_HAVE_ARM_BF16)
__attribute__((target("bf16")))
static int32_t neon_compute_bf16(cross_cpu_worker_ctx_t *ctx) {
    /* Exact BF16 bit patterns: 1.0, 1.5, 2.0, 0.5 */
    static const uint16_t bf_lut[4] = { 0x3f80u, 0x3fc0u, 0x4000u, 0x3f00u };
    static const float    f_lut[4]  = { 1.0f,    1.5f,    2.0f,    0.5f    };
    const uint32_t inner = 256u;

    uint16_t a_raw[8], b_raw[8];
    float dot_ref[4];
    for (int lane = 0; lane < 4; lane++) {
        int i0 = lane & 3;
        int i1 = (lane + 1) & 3;
        a_raw[lane * 2 + 0] = bf_lut[i0];
        a_raw[lane * 2 + 1] = bf_lut[i1];
        b_raw[lane * 2 + 0] = bf_lut[i1];
        b_raw[lane * 2 + 1] = bf_lut[i0];
        dot_ref[lane] = f_lut[i0] * f_lut[i1] + f_lut[i1] * f_lut[i0];
    }

    bfloat16x8_t va, vb;
    memcpy(&va, a_raw, sizeof(va));
    memcpy(&vb, b_raw, sizeof(vb));

    uint64_t ops = 0;
    for (uint64_t iter = 0; iter < ctx->batch_iters; iter++) {
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(1.0f);

        for (uint32_t i = 0; i < inner; i++) {
            acc0 = vbfdotq_f32(acc0, va, vb);
            acc1 = vbfdotq_f32(acc1, va, vb);
        }

        float out0[4], out1[4];
        vst1q_f32(out0, acc0);
        vst1q_f32(out1, acc1);

        if (ctx->inject_fault && ctx->thread_idx == 0u && iter == 0ULL) {
            out0[0] += 1.0f;
        }

        for (int lane = 0; lane < 4; lane++) {
            float exp0 = (float)inner * dot_ref[lane];
            float exp1 = 1.0f + exp0;
            if (!isfinite(out0[lane]) || !isfinite(out1[lane]) ||
                fabsf(out0[lane] - exp0) > 1.0e-3f || fabsf(out1[lane] - exp1) > 1.0e-3f) {
                ctx->errors++;
                ctx->expected_sample = (double)exp0;
                ctx->observed_sample = (double)out0[lane];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "neon bf16 mismatch lane %d (expected %.4f, got %.4f)",
                         lane, (double)exp0, (double)out0[lane]);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        /* 2 accs * 4 lanes * 4 FLOPs (2 mul + 2 add) * inner */
        ops += (uint64_t)inner * 32ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}
#endif

#if defined(__ARM_FEATURE_DOTPROD) || defined(CROSS_CPU_HAVE_ARM_DOTPROD)
__attribute__((target("dotprod")))
static int32_t neon_compute_vnni_dotprod(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    int8_t a_bytes[16], b_bytes[16];
    int32_t dot_ref[4];

    for (int lane = 0; lane < 4; lane++) {
        int32_t sum = 0;
        for (int j = 0; j < 4; j++) {
            int8_t av = (int8_t)(1 + ((lane + j) % 7));
            int8_t bv = (int8_t)(((j & 1) ? -1 : 1) * (2 + (lane % 5)));
            a_bytes[lane * 4 + j] = av;
            b_bytes[lane * 4 + j] = bv;
            sum += (int32_t)av * (int32_t)bv;
        }
        dot_ref[lane] = sum;
    }

    const int8x16_t va = vld1q_s8(a_bytes);
    const int8x16_t vb = vld1q_s8(b_bytes);

    uint64_t ops = 0;
    for (uint64_t iter = 0; iter < ctx->batch_iters; iter++) {
        int32_t init0 = (int32_t)((ctx->seed + iter) & 0xFFu);
        int32x4_t acc0 = vdupq_n_s32(init0);
        int32x4_t acc1 = vdupq_n_s32(init0 + 1);

        for (uint32_t i = 0; i < inner; i++) {
            acc0 = vdotq_s32(acc0, va, vb);
            acc1 = vdotq_s32(acc1, va, vb);
        }

        int32_t out0[4], out1[4];
        vst1q_s32(out0, acc0);
        vst1q_s32(out1, acc1);

        if (ctx->inject_fault && ctx->thread_idx == 0u && iter == 0ULL) {
            out0[0] += 1;
        }

        for (int lane = 0; lane < 4; lane++) {
            int32_t exp0 = init0 + (int32_t)inner * dot_ref[lane];
            int32_t exp1 = (init0 + 1) + (int32_t)inner * dot_ref[lane];
            if (out0[lane] != exp0 || out1[lane] != exp1) {
                ctx->errors++;
                ctx->expected_sample = (double)exp0;
                ctx->observed_sample = (double)out0[lane];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "neon dotprod mismatch lane %d (expected %d, got %d)",
                         lane, exp0, out0[lane]);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        /* 2 accs * 4 lanes * 8 ops * inner */
        ops += (uint64_t)inner * 64ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8) || defined(CROSS_CPU_HAVE_ARM_I8MM)
__attribute__((target("i8mm")))
static int32_t neon_compute_vnni_i8mm(cross_cpu_worker_ctx_t *ctx) {
    /* SMMLA computes C[2x2] += A[2x8] * B^T[8x2] (32 muls + 32 adds = 64 ops/inst) */
    const uint32_t inner = 256u;
    static const int8_t a_mat[16] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        2, 3, 4, 5, 6, 7, 8, 9
    };
    static const int8_t b_mat[16] = {
        1, -1, 2, -2, 3, -3, 4, -4,
        2,  1, 3,  1, 4,  1, 5,  1
    };

    int32_t step_ref[4];
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 2; c++) {
            int32_t s = 0;
            for (int k = 0; k < 8; k++) {
                s += (int32_t)a_mat[r * 8 + k] * (int32_t)b_mat[c * 8 + k];
            }
            step_ref[r * 2 + c] = s;
        }
    }

    const int8x16_t va = vld1q_s8(a_mat);
    const int8x16_t vb = vld1q_s8(b_mat);

    uint64_t ops = 0;
    for (uint64_t iter = 0; iter < ctx->batch_iters; iter++) {
        int32_t init0 = (int32_t)((ctx->seed + iter) & 0xFFu);
        int32x4_t acc0 = vdupq_n_s32(init0);
        int32x4_t acc1 = vdupq_n_s32(init0 + 1);

        for (uint32_t i = 0; i < inner; i++) {
            acc0 = vmmlaq_s32(acc0, va, vb);
            acc1 = vmmlaq_s32(acc1, va, vb);
        }

        int32_t out0[4], out1[4];
        vst1q_s32(out0, acc0);
        vst1q_s32(out1, acc1);

        if (ctx->inject_fault && ctx->thread_idx == 0u && iter == 0ULL) {
            out0[0] += 1;
        }

        for (int lane = 0; lane < 4; lane++) {
            int32_t exp0 = init0 + (int32_t)inner * step_ref[lane];
            int32_t exp1 = (init0 + 1) + (int32_t)inner * step_ref[lane];
            if (out0[lane] != exp0 || out1[lane] != exp1) {
                ctx->errors++;
                ctx->expected_sample = (double)exp0;
                ctx->observed_sample = (double)out0[lane];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "neon i8mm mismatch lane %d (expected %d, got %d)",
                         lane, exp0, out0[lane]);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        /* 2 accs * 64 ops * inner */
        ops += (uint64_t)inner * 128ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}
#endif

#if defined(__ARM_FEATURE_AES) || defined(CROSS_CPU_HAVE_ARM_AES)
__attribute__((target("aes")))
static int32_t neon_crypto_aes(cross_cpu_worker_ctx_t *ctx) {
    /* NIST FIPS 197 Appendix C.1 AES-128 KAT */
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

    uint8x16_t rk[11];
    for (int r = 0; r < 11; r++) {
        rk[r] = vld1q_u8(rk_c1[r]);
    }
    const uint8x16_t pt = vld1q_u8(pt_c1);
    const uint8x16_t expected = vld1q_u8(ct_c1);

    uint64_t bytes = 0;
    uint8x16_t carry0 = vdupq_n_u8(0u);
    uint8x16_t carry1 = vdupq_n_u8(0u);
    uint8x16_t carry2 = vdupq_n_u8(0u);
    uint8x16_t carry3 = vdupq_n_u8(0u);
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        for (uint32_t k = 0; k < 64u; k++) {
            uint8x16_t s0 = veorq_u8(pt, carry0);
            uint8x16_t s1 = veorq_u8(pt, carry1);
            uint8x16_t s2 = veorq_u8(pt, carry2);
            uint8x16_t s3 = veorq_u8(pt, carry3);
            for (int r = 0; r < 9; r++) {
                s0 = vaesmcq_u8(vaeseq_u8(s0, rk[r]));
                s1 = vaesmcq_u8(vaeseq_u8(s1, rk[r]));
                s2 = vaesmcq_u8(vaeseq_u8(s2, rk[r]));
                s3 = vaesmcq_u8(vaeseq_u8(s3, rk[r]));
            }
            s0 = veorq_u8(vaeseq_u8(s0, rk[9]), rk[10]);
            s1 = veorq_u8(vaeseq_u8(s1, rk[9]), rk[10]);
            s2 = veorq_u8(vaeseq_u8(s2, rk[9]), rk[10]);
            s3 = veorq_u8(vaeseq_u8(s3, rk[9]), rk[10]);

            if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL && k == 0u) {
                s0 = veorq_u8(s0, vdupq_n_u8(1u));
            }

            carry0 = veorq_u8(s0, expected);
            carry1 = veorq_u8(s1, expected);
            carry2 = veorq_u8(s2, expected);
            carry3 = veorq_u8(s3, expected);
            __asm__ volatile("" : "+w"(carry0), "+w"(carry1), "+w"(carry2), "+w"(carry3));

            uint8x16_t err = vorrq_u8(vorrq_u8(carry0, carry1), vorrq_u8(carry2, carry3));
            if (vmaxvq_u8(err) != 0u) {
                ctx->errors++;
                ctx->expected_sample = 1.0;
                ctx->observed_sample = 0.0;
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "ARM64 FEAT_AES NIST FIPS 197 KAT ciphertext mismatch");
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

#if defined(__ARM_FEATURE_SHA2) || defined(CROSS_CPU_HAVE_ARM_SHA2)
__attribute__((target("sha2")))
static int32_t neon_crypto_sha256(cross_cpu_worker_ctx_t *ctx) {
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
    static const uint32_t init_h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    static const uint32_t msg_abc[16] = {
        0x61626380u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0x00000018u
    };
    static const uint32_t exp_digest[8] = {
        0xba7816bfu, 0x8f01cfeau, 0x414140deu, 0x5dae2223u,
        0xb00361a3u, 0x96177a9cu, 0xb410ff61u, 0xf20015adu
    };

    const uint32x4_t h0_init = vld1q_u32(&init_h[0]);
    const uint32x4_t h1_init = vld1q_u32(&init_h[4]);
    const uint32x4_t exp0 = vld1q_u32(&exp_digest[0]);
    const uint32x4_t exp1 = vld1q_u32(&exp_digest[4]);

    uint64_t bytes = 0;
    uint32x4_t carry = vdupq_n_u32(0u);
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        for (uint32_t k = 0; k < 32u; k++) {
            uint32x4_t s0 = veorq_u32(h0_init, carry);
            uint32x4_t s1 = h1_init;
            uint32x4_t w0 = vld1q_u32(&msg_abc[0]);
            uint32x4_t w1 = vld1q_u32(&msg_abc[4]);
            uint32x4_t w2 = vld1q_u32(&msg_abc[8]);
            uint32x4_t w3 = vld1q_u32(&msg_abc[12]);

            for (int r = 0; r < 64; r += 4) {
                uint32x4_t wk = vaddq_u32(w0, vld1q_u32(&K[r]));
                uint32x4_t s0_prev = s0;
                s0 = vsha256hq_u32(s0, s1, wk);
                s1 = vsha256h2q_u32(s1, s0_prev, wk);

                if (r < 48) {
                    uint32x4_t next = vsha256su0q_u32(w0, w1);
                    next = vsha256su1q_u32(next, w2, w3);
                    w0 = w1;
                    w1 = w2;
                    w2 = w3;
                    w3 = next;
                } else {
                    w0 = w1;
                    w1 = w2;
                    w2 = w3;
                }
            }

            s0 = vaddq_u32(s0, h0_init);
            s1 = vaddq_u32(s1, h1_init);

            if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL && k == 0u) {
                s0 = veorq_u32(s0, vdupq_n_u32(1u));
            }

            carry = vorrq_u32(veorq_u32(s0, exp0), veorq_u32(s1, exp1));
            __asm__ volatile("" : "+w"(carry));
            if (vmaxvq_u32(carry) != 0u) {
                ctx->errors++;
                ctx->expected_sample = (double)exp_digest[0];
                ctx->observed_sample = (double)vgetq_lane_u32(s0, 0);
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "ARM64 FEAT_SHA256 NIST FIPS 180-4 KAT digest mismatch");
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

static int32_t neon_cache_rw(cross_cpu_worker_ctx_t *ctx) {
    if (ctx->thread_buf == NULL || ctx->thread_buf_bytes < 16u) {
        return CROSS_CPU_STATUS_INIT_ERROR;
    }
    uint64_t *buf = (uint64_t *)ctx->thread_buf;
    const size_t vecs = ctx->thread_buf_bytes / 16u;
    uint64_t bytes = 0;

    for (uint64_t pass = 0; pass < ctx->batch_iters; pass++) {
        const uint64_t salt = splitmix64_neon(ctx->seed + pass);
        uint64_t init_pair[2] = { salt, salt ^ 0x55aa55aa55aa55aaULL };
        const uint64x2_t base = vld1q_u64(init_pair);
        const uint64x2_t step = vdupq_n_u64(0x0102030405060708ULL);

        uint64x2_t cur = base;
        for (size_t i = 0; i < vecs; i++) {
            vst1q_u64(buf + i * 2u, cur);
            cur = vaddq_u64(cur, step);
        }

        if (ctx->inject_fault && ctx->thread_idx == 0u && pass == 0ULL) {
            buf[0] ^= 1ULL;
        }

        cur = base;
        for (size_t i = 0; i < vecs; i++) {
            uint64x2_t got = vld1q_u64(buf + i * 2u);
            uint32x4_t diff = vreinterpretq_u32_u64(veorq_u64(got, cur));
            if (vmaxvq_u32(diff) != 0u) {
                ctx->errors++;
                ctx->expected_sample = (double)vgetq_lane_u64(cur, 0);
                ctx->observed_sample = (double)vgetq_lane_u64(got, 0);
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "neon cache verify mismatch at vector %zu", i);
                ctx->ops_done = bytes;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
            cur = vaddq_u64(cur, step);
        }
        bytes += (uint64_t)(vecs * 16u) * 2ULL;
    }
    ctx->ops_done = bytes;
    return CROSS_CPU_STATUS_OK;
}

static int32_t neon_bandwidth_ram(cross_cpu_worker_ctx_t *ctx) {
    if (ctx->thread_buf == NULL || ctx->thread_buf_bytes < 64u) {
        return CROSS_CPU_STATUS_INIT_ERROR;
    }
    uint8_t *buf = (uint8_t *)ctx->thread_buf;
    const size_t chunks = ctx->thread_buf_bytes / 64u;
    uint64_t bytes = 0;

    for (uint64_t pass = 0; pass < ctx->batch_iters; pass++) {
        const uint64_t p0 = splitmix64_neon(ctx->seed + pass);
        const uint64_t p1 = ~p0;

        /* Non-temporal pair stores (stnp) bypass cache allocation */
        for (size_t i = 0; i < chunks; i++) {
            uint8_t *dst = buf + i * 64u;
            __asm__ volatile(
                "stnp %0, %1, [%2, #0]\n\t"
                "stnp %0, %1, [%2, #16]\n\t"
                "stnp %0, %1, [%2, #32]\n\t"
                "stnp %0, %1, [%2, #48]\n\t"
                :
                : "r"(p0), "r"(p1), "r"(dst)
                : "memory");
        }
        __asm__ volatile("dmb ish" ::: "memory");

        if (ctx->inject_fault && ctx->thread_idx == 0u && pass == 0ULL) {
            buf[0] ^= 0x80u;
        }

        uint64_t exp_pair[2] = { p0, p1 };
        const uint64x2_t vexp = vld1q_u64(exp_pair);
        const uint64_t *rbuf = (const uint64_t *)buf;
        const size_t vecs = chunks * 4u;
        for (size_t i = 0; i < vecs; i++) {
            uint64x2_t got = vld1q_u64(rbuf + i * 2u);
            uint32x4_t diff = vreinterpretq_u32_u64(veorq_u64(got, vexp));
            if (vmaxvq_u32(diff) != 0u) {
                ctx->errors++;
                ctx->expected_sample = (double)p0;
                ctx->observed_sample = (double)vgetq_lane_u64(got, 0);
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "neon stnp bandwidth-ram mismatch at vector %zu", i);
                ctx->ops_done = bytes;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        bytes += (uint64_t)(chunks * 64u) * 2ULL;
    }
    ctx->ops_done = bytes;
    return CROSS_CPU_STATUS_OK;
}

const uint32_t cross_cpu_arm64_neon_compiled = 1u;

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(CROSS_CPU_HAVE_ARM_FP16)
const uint32_t cross_cpu_arm64_fp16_compiled = 1u;
#else
const uint32_t cross_cpu_arm64_fp16_compiled = 0u;
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(CROSS_CPU_HAVE_ARM_BF16)
const uint32_t cross_cpu_arm64_bf16_compiled = 1u;
#else
const uint32_t cross_cpu_arm64_bf16_compiled = 0u;
#endif

#if defined(__ARM_FEATURE_DOTPROD) || defined(CROSS_CPU_HAVE_ARM_DOTPROD)
const uint32_t cross_cpu_arm64_dotprod_compiled = 1u;
const cross_cpu_kernel_fn cross_cpu_arm64_vnni_dotprod_kernel = neon_compute_vnni_dotprod;
#else
const uint32_t cross_cpu_arm64_dotprod_compiled = 0u;
const cross_cpu_kernel_fn cross_cpu_arm64_vnni_dotprod_kernel = NULL;
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8) || defined(CROSS_CPU_HAVE_ARM_I8MM)
const uint32_t cross_cpu_arm64_i8mm_compiled = 1u;
const cross_cpu_kernel_fn cross_cpu_arm64_vnni_i8mm_kernel = neon_compute_vnni_i8mm;
#else
const uint32_t cross_cpu_arm64_i8mm_compiled = 0u;
const cross_cpu_kernel_fn cross_cpu_arm64_vnni_i8mm_kernel = NULL;
#endif

#if defined(__ARM_FEATURE_AES) || defined(CROSS_CPU_HAVE_ARM_AES)
const uint32_t cross_cpu_arm64_aes_compiled = 1u;
#else
const uint32_t cross_cpu_arm64_aes_compiled = 0u;
#endif

#if defined(__ARM_FEATURE_SHA2) || defined(CROSS_CPU_HAVE_ARM_SHA2)
const uint32_t cross_cpu_arm64_sha2_compiled = 1u;
#else
const uint32_t cross_cpu_arm64_sha2_compiled = 0u;
#endif

const cross_cpu_kernel_fn cross_cpu_arm64_neon_kernels[CROSS_CPU_WL_COUNT] = {
    [CROSS_CPU_WL_COMPUTE_INT64] = neon_compute_int64,
    [CROSS_CPU_WL_COMPUTE_INT32] = neon_compute_int32,
    [CROSS_CPU_WL_COMPUTE_INT16] = neon_compute_int16,
    [CROSS_CPU_WL_COMPUTE_INT8]  = neon_compute_int8,
    [CROSS_CPU_WL_COMPUTE_FP64]  = neon_compute_fp64,
    [CROSS_CPU_WL_COMPUTE_FP32]  = neon_compute_fp32,
    [CROSS_CPU_WL_CACHE_L1]      = neon_cache_rw,
    [CROSS_CPU_WL_CACHE_L2]      = neon_cache_rw,
    [CROSS_CPU_WL_CACHE_L3]      = neon_cache_rw,
    [CROSS_CPU_WL_BANDWIDTH_RAM] = neon_bandwidth_ram,
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(CROSS_CPU_HAVE_ARM_FP16)
    [CROSS_CPU_WL_COMPUTE_FP16]  = neon_compute_fp16,
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(CROSS_CPU_HAVE_ARM_BF16)
    [CROSS_CPU_WL_COMPUTE_BF16]  = neon_compute_bf16,
#endif
#if defined(__ARM_FEATURE_AES) || defined(CROSS_CPU_HAVE_ARM_AES)
    [CROSS_CPU_WL_CRYPTO_AES]    = neon_crypto_aes,
#endif
#if defined(__ARM_FEATURE_SHA2) || defined(CROSS_CPU_HAVE_ARM_SHA2)
    [CROSS_CPU_WL_CRYPTO_SHA256] = neon_crypto_sha256,
#endif
};
