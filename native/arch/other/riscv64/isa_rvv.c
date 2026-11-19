// SPDX-License-Identifier: MIT
#include "cross_cpu_arch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__riscv_vector) || defined(CROSS_CPU_HAVE_RVV)
#include <riscv_vector.h>

#define RVV_MAX_ELEMS 256u

static inline uint64_t splitmix64_rvv(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static void lcg_power_rvv(uint64_t a, uint64_t c, uint32_t steps,
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

static int32_t rvv_compute_int64(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint64_t mul_c = 6364136223846793005ULL;
    const uint64_t inc_c = 1442695040888963407ULL;
    const size_t vl = __riscv_vsetvl_e64m1(RVV_MAX_ELEMS);

    uint64_t rm = 1ULL, ra = 0ULL;
    lcg_power_rvv(mul_c, inc_c, inner, &rm, &ra);

    uint64_t init[RVV_MAX_ELEMS], out[RVV_MAX_ELEMS];
    uint64_t ops = 0;

    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        for (size_t i = 0; i < vl; i++) {
            init[i] = splitmix64_rvv(ctx->seed + b * (uint64_t)vl + (uint64_t)i);
        }
        vuint64m1_t v = __riscv_vle64_v_u64m1(init, vl);
        vuint64m1_t vinc = __riscv_vmv_v_x_u64m1(inc_c, vl);
        for (uint32_t i = 0; i < inner; i++) {
            v = __riscv_vmacc_vx_u64m1(vinc, mul_c, v, vl);
        }
        __riscv_vse64_v_u64m1(out, v, vl);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1ULL;
        }

        for (size_t i = 0; i < vl; i++) {
            uint64_t exp = init[i] * rm + ra;
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "rvv int64 mismatch lane %zu", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * (uint64_t)vl * 2ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t rvv_compute_int32(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint32_t mul_c = 1664525u;
    const uint32_t inc_c = 1013904223u;
    const size_t vl = __riscv_vsetvl_e32m1(RVV_MAX_ELEMS);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_rvv(mul_c, inc_c, inner, &rm64, &ra64);
    const uint32_t rm = (uint32_t)rm64;
    const uint32_t ra = (uint32_t)ra64;

    uint32_t init[RVV_MAX_ELEMS], out[RVV_MAX_ELEMS];
    uint64_t ops = 0;

    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        for (size_t i = 0; i < vl; i++) {
            init[i] = (uint32_t)splitmix64_rvv(ctx->seed + b * (uint64_t)vl + (uint64_t)i);
        }
        vuint32m1_t v = __riscv_vle32_v_u32m1(init, vl);
        vuint32m1_t vinc = __riscv_vmv_v_x_u32m1(inc_c, vl);
        for (uint32_t i = 0; i < inner; i++) {
            v = __riscv_vmacc_vx_u32m1(vinc, mul_c, v, vl);
        }
        __riscv_vse32_v_u32m1(out, v, vl);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1u;
        }

        for (size_t i = 0; i < vl; i++) {
            uint32_t exp = init[i] * rm + ra;
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "rvv int32 mismatch lane %zu", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * (uint64_t)vl * 2ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t rvv_compute_int16(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint16_t mul_c = 25173u;
    const uint16_t inc_c = 13849u;
    const size_t vl = __riscv_vsetvl_e16m1(RVV_MAX_ELEMS);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_rvv(mul_c, inc_c, inner, &rm64, &ra64);
    const uint16_t rm = (uint16_t)rm64;
    const uint16_t ra = (uint16_t)ra64;

    uint16_t init[RVV_MAX_ELEMS], out[RVV_MAX_ELEMS];
    uint64_t ops = 0;

    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        for (size_t i = 0; i < vl; i++) {
            init[i] = (uint16_t)splitmix64_rvv(ctx->seed + b * (uint64_t)vl + (uint64_t)i);
        }
        vuint16m1_t v = __riscv_vle16_v_u16m1(init, vl);
        vuint16m1_t vinc = __riscv_vmv_v_x_u16m1(inc_c, vl);
        for (uint32_t i = 0; i < inner; i++) {
            v = __riscv_vmacc_vx_u16m1(vinc, mul_c, v, vl);
        }
        __riscv_vse16_v_u16m1(out, v, vl);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1u;
        }

        for (size_t i = 0; i < vl; i++) {
            uint16_t exp = (uint16_t)(init[i] * rm + ra);
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "rvv int16 mismatch lane %zu", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * (uint64_t)vl * 2ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t rvv_compute_int8(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const uint8_t mul_c = 141u;
    const uint8_t inc_c = 53u;
    const size_t vl = __riscv_vsetvl_e8m1(RVV_MAX_ELEMS);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_rvv(mul_c, inc_c, inner, &rm64, &ra64);
    const uint8_t rm = (uint8_t)rm64;
    const uint8_t ra = (uint8_t)ra64;

    uint8_t init[RVV_MAX_ELEMS], out[RVV_MAX_ELEMS];
    uint64_t ops = 0;

    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        for (size_t i = 0; i < vl; i++) {
            init[i] = (uint8_t)splitmix64_rvv(ctx->seed + b * (uint64_t)vl + (uint64_t)i);
        }
        vuint8m1_t v = __riscv_vle8_v_u8m1(init, vl);
        vuint8m1_t vinc = __riscv_vmv_v_x_u8m1(inc_c, vl);
        for (uint32_t i = 0; i < inner; i++) {
            v = __riscv_vmacc_vx_u8m1(vinc, mul_c, v, vl);
        }
        __riscv_vse8_v_u8m1(out, v, vl);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1u;
        }

        for (size_t i = 0; i < vl; i++) {
            uint8_t exp = (uint8_t)(init[i] * rm + ra);
            if (out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "rvv int8 mismatch lane %zu", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * (uint64_t)vl * 2ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t rvv_compute_fp64(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 128u;
    const size_t vl = __riscv_vsetvl_e64m1(RVV_MAX_ELEMS);
    double b_pos[RVV_MAX_ELEMS], b_neg[RVV_MAX_ELEMS];
    for (size_t i = 0; i < vl; i++) {
        b_pos[i] = 0.25 + (double)(i & 7u) * 0.0625;
        b_neg[i] = -b_pos[i];
    }
    const vfloat64m1_t vbp = __riscv_vle64_v_f64m1(b_pos, vl);
    const vfloat64m1_t vbn = __riscv_vle64_v_f64m1(b_neg, vl);

    double op[RVV_MAX_ELEMS], on[RVV_MAX_ELEMS];
    uint64_t ops = 0;

    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        double d = (double)((ctx->seed + b) & 0xFFu) * 0.00390625;
        vfloat64m1_t vp = __riscv_vfmv_v_f_f64m1(0.75 + d, vl);
        vfloat64m1_t vn = __riscv_vfmv_v_f_f64m1(-(0.75 + d), vl);
        for (uint32_t i = 0; i < inner; i++) {
            vp = __riscv_vfmacc_vf_f64m1(vbp, 0.75, vp, vl);
            vn = __riscv_vfmacc_vf_f64m1(vbn, 0.75, vn, vl);
        }
        __riscv_vse64_v_f64m1(op, vp, vl);
        __riscv_vse64_v_f64m1(on, vn, vl);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            op[0] += 1.0e-5;
        }

        for (size_t i = 0; i < vl; i++) {
            double ref = b_pos[i] * 4.0;
            if (!isfinite(op[i]) || !isfinite(on[i]) ||
                fabs(op[i] + on[i]) > 1.0e-12 || fabs(op[i] - ref) > 1.0e-12) {
                ctx->errors++;
                ctx->expected_sample = ref;
                ctx->observed_sample = op[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "rvv fp64 mismatch lane %zu", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * (uint64_t)vl * 4ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t rvv_compute_fp32(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 128u;
    const size_t vl = __riscv_vsetvl_e32m1(RVV_MAX_ELEMS);
    float b_pos[RVV_MAX_ELEMS], b_neg[RVV_MAX_ELEMS];
    for (size_t i = 0; i < vl; i++) {
        b_pos[i] = 0.25f + (float)(i & 7u) * 0.0625f;
        b_neg[i] = -b_pos[i];
    }
    const vfloat32m1_t vbp = __riscv_vle32_v_f32m1(b_pos, vl);
    const vfloat32m1_t vbn = __riscv_vle32_v_f32m1(b_neg, vl);

    float op[RVV_MAX_ELEMS], on[RVV_MAX_ELEMS];
    uint64_t ops = 0;

    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        float d = (float)((ctx->seed + b) & 0xFFu) * 0.00390625f;
        vfloat32m1_t vp = __riscv_vfmv_v_f_f32m1(0.75f + d, vl);
        vfloat32m1_t vn = __riscv_vfmv_v_f_f32m1(-(0.75f + d), vl);
        for (uint32_t i = 0; i < inner; i++) {
            vp = __riscv_vfmacc_vf_f32m1(vbp, 0.75f, vp, vl);
            vn = __riscv_vfmacc_vf_f32m1(vbn, 0.75f, vn, vl);
        }
        __riscv_vse32_v_f32m1(op, vp, vl);
        __riscv_vse32_v_f32m1(on, vn, vl);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            op[0] += 1.0e-3f;
        }

        for (size_t i = 0; i < vl; i++) {
            float ref = b_pos[i] * 4.0f;
            if (!isfinite(op[i]) || !isfinite(on[i]) ||
                fabsf(op[i] + on[i]) > 1.0e-5f || fabsf(op[i] - ref) > 1.0e-5f) {
                ctx->errors++;
                ctx->expected_sample = (double)ref;
                ctx->observed_sample = (double)op[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "rvv fp32 mismatch lane %zu", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * (uint64_t)vl * 4ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t rvv_memory_rw(cross_cpu_worker_ctx_t *ctx) {
    if (ctx->thread_buf == NULL || ctx->thread_buf_bytes < 64u) {
        return CROSS_CPU_STATUS_INIT_ERROR;
    }
    uint64_t *buf = (uint64_t *)ctx->thread_buf;
    const size_t words = ctx->thread_buf_bytes / sizeof(uint64_t);
    uint64_t bytes = 0;

    for (uint64_t pass = 0; pass < ctx->batch_iters; pass++) {
        const uint64_t salt = splitmix64_rvv(ctx->seed + pass);
        for (size_t i = 0; i < words; ) {
            size_t vl = __riscv_vsetvl_e64m1(words - i);
            vuint64m1_t pat = __riscv_vmv_v_x_u64m1(salt ^ (uint64_t)i, vl);
            __riscv_vse64_v_u64m1(buf + i, pat, vl);
            i += vl;
        }

        if (ctx->inject_fault && ctx->thread_idx == 0u && pass == 0ULL) {
            buf[0] ^= 1ULL;
        }

        for (size_t i = 0; i < words; ) {
            size_t vl = __riscv_vsetvl_e64m1(words - i);
            vuint64m1_t exp = __riscv_vmv_v_x_u64m1(salt ^ (uint64_t)i, vl);
            vuint64m1_t got = __riscv_vle64_v_u64m1(buf + i, vl);
            vbool64_t neq = __riscv_vmsne_vv_u64m1_b64(got, exp, vl);
            if (__riscv_vcpop_m_b64(neq, vl) != 0u) {
                ctx->errors++;
                ctx->expected_sample = (double)(salt ^ (uint64_t)i);
                ctx->observed_sample = (double)buf[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "rvv memory/cache mismatch near word %zu", i);
                ctx->ops_done = bytes;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
            i += vl;
        }
        bytes += (uint64_t)ctx->thread_buf_bytes * 2ULL;
    }
    ctx->ops_done = bytes;
    return CROSS_CPU_STATUS_OK;
}
#endif

#if defined(__riscv_vector) || defined(CROSS_CPU_HAVE_RVV)
const uint32_t cross_cpu_riscv_rvv_compiled = 1u;
#else
const uint32_t cross_cpu_riscv_rvv_compiled = 0u;
#endif

const cross_cpu_kernel_fn cross_cpu_riscv_rvv_kernels[CROSS_CPU_WL_COUNT] = {
#if defined(__riscv_vector) || defined(CROSS_CPU_HAVE_RVV)
    [CROSS_CPU_WL_COMPUTE_INT64] = rvv_compute_int64,
    [CROSS_CPU_WL_COMPUTE_INT32] = rvv_compute_int32,
    [CROSS_CPU_WL_COMPUTE_INT16] = rvv_compute_int16,
    [CROSS_CPU_WL_COMPUTE_INT8]  = rvv_compute_int8,
    [CROSS_CPU_WL_COMPUTE_FP64]  = rvv_compute_fp64,
    [CROSS_CPU_WL_COMPUTE_FP32]  = rvv_compute_fp32,
    [CROSS_CPU_WL_CACHE_L1]      = rvv_memory_rw,
    [CROSS_CPU_WL_CACHE_L2]      = rvv_memory_rw,
    [CROSS_CPU_WL_CACHE_L3]      = rvv_memory_rw,
    [CROSS_CPU_WL_BANDWIDTH_RAM] = rvv_memory_rw,
#else
    NULL
#endif
};
