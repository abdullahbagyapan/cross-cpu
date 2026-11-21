// SPDX-License-Identifier: MIT
#include "cross_cpu_arch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__VEC__) || defined(CROSS_CPU_HAVE_VX)
#include <vecintrin.h>

static inline uint64_t splitmix64_vx(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static void lcg_power_vx(uint64_t a, uint64_t c, uint32_t steps,
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

static int32_t vx_compute_int64(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const unsigned long long inc_c = 0x9e3779b97f4a7c15ULL;
    const __vector unsigned long long vinc = vec_splats(inc_c);

    uint64_t rm = 1ULL, ra = 0ULL;
    lcg_power_vx(3ULL, (uint64_t)inc_c, inner, &rm, &ra);

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        unsigned long long init[4];
        for (int i = 0; i < 4; i++) {
            init[i] = (unsigned long long)splitmix64_vx(ctx->seed + b * 4ULL + (uint64_t)i);
        }
        __vector unsigned long long v0 = vec_xl(0, &init[0]);
        __vector unsigned long long v1 = vec_xl(0, &init[2]);

        for (uint32_t i = 0; i < inner; i++) {
            __vector unsigned long long d0 = v0 + v0;
            __vector unsigned long long d1 = v1 + v1;
            v0 = (v0 + d0) + vinc;
            v1 = (v1 + d1) + vinc;
        }

        unsigned long long out[4];
        vec_xst(v0, 0, &out[0]);
        vec_xst(v1, 0, &out[2]);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1ULL;
        }

        for (int i = 0; i < 4; i++) {
            uint64_t exp = (uint64_t)init[i] * rm + ra;
            if ((uint64_t)out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "s390x vx int64 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 12ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t vx_compute_int32(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const unsigned int mul_c = 1664525u;
    const unsigned int inc_c = 1013904223u;
    const __vector unsigned int vmul = vec_splats(mul_c);
    const __vector unsigned int vinc = vec_splats(inc_c);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_vx(mul_c, inc_c, inner, &rm64, &ra64);
    const uint32_t rm = (uint32_t)rm64;
    const uint32_t ra = (uint32_t)ra64;

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        unsigned int init[8];
        for (int i = 0; i < 8; i++) {
            init[i] = (unsigned int)splitmix64_vx(ctx->seed + b * 8ULL + (uint64_t)i);
        }
        __vector unsigned int v0 = vec_xl(0, &init[0]);
        __vector unsigned int v1 = vec_xl(0, &init[4]);

        for (uint32_t i = 0; i < inner; i++) {
            v0 = vec_mladd(v0, vmul, vinc);
            v1 = vec_mladd(v1, vmul, vinc);
        }

        unsigned int out[8];
        vec_xst(v0, 0, &out[0]);
        vec_xst(v1, 0, &out[4]);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1u;
        }

        for (int i = 0; i < 8; i++) {
            uint32_t exp = (uint32_t)init[i] * rm + ra;
            if ((uint32_t)out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "s390x vx int32 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 16ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t vx_compute_int16(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const unsigned short mul_c = 25173u;
    const unsigned short inc_c = 13849u;
    const __vector unsigned short vmul = vec_splats(mul_c);
    const __vector unsigned short vinc = vec_splats(inc_c);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_vx(mul_c, inc_c, inner, &rm64, &ra64);
    const uint16_t rm = (uint16_t)rm64;
    const uint16_t ra = (uint16_t)ra64;

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        unsigned short init[16];
        for (int i = 0; i < 16; i++) {
            init[i] = (unsigned short)splitmix64_vx(ctx->seed + b * 16ULL + (uint64_t)i);
        }
        __vector unsigned short v0 = vec_xl(0, &init[0]);
        __vector unsigned short v1 = vec_xl(0, &init[8]);

        for (uint32_t i = 0; i < inner; i++) {
            v0 = vec_mladd(v0, vmul, vinc);
            v1 = vec_mladd(v1, vmul, vinc);
        }

        unsigned short out[16];
        vec_xst(v0, 0, &out[0]);
        vec_xst(v1, 0, &out[8]);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1u;
        }

        for (int i = 0; i < 16; i++) {
            uint16_t exp = (uint16_t)((uint16_t)init[i] * rm + ra);
            if ((uint16_t)out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "s390x vx int16 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 32ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t vx_compute_int8(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 256u;
    const unsigned char mul_c = 141u;
    const unsigned char inc_c = 53u;
    const __vector unsigned char vmul = vec_splats(mul_c);
    const __vector unsigned char vinc = vec_splats(inc_c);

    uint64_t rm64 = 1ULL, ra64 = 0ULL;
    lcg_power_vx(mul_c, inc_c, inner, &rm64, &ra64);
    const uint8_t rm = (uint8_t)rm64;
    const uint8_t ra = (uint8_t)ra64;

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        unsigned char init[32];
        for (int i = 0; i < 32; i++) {
            init[i] = (unsigned char)splitmix64_vx(ctx->seed + b * 32ULL + (uint64_t)i);
        }
        __vector unsigned char v0 = vec_xl(0, &init[0]);
        __vector unsigned char v1 = vec_xl(0, &init[16]);

        for (uint32_t i = 0; i < inner; i++) {
            v0 = vec_mladd(v0, vmul, vinc);
            v1 = vec_mladd(v1, vmul, vinc);
        }

        unsigned char out[32];
        vec_xst(v0, 0, &out[0]);
        vec_xst(v1, 0, &out[16]);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            out[0] ^= 1u;
        }

        for (int i = 0; i < 32; i++) {
            uint8_t exp = (uint8_t)((uint8_t)init[i] * rm + ra);
            if ((uint8_t)out[i] != exp) {
                ctx->errors++;
                ctx->expected_sample = (double)exp;
                ctx->observed_sample = (double)out[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "s390x vx int8 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 64ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t vx_compute_fp64(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 128u;
    const __vector double va = vec_splats(0.75);
    const double b_vals[2] = { 0.3125, 0.4375 };
    const double b_neg[2]  = { -0.3125, -0.4375 };
    const __vector double vbp = vec_xl(0, b_vals);
    const __vector double vbn = vec_xl(0, b_neg);

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        double d = (double)((ctx->seed + b) & 0xFFu) * 0.00390625;
        __vector double vp = vec_splats(0.75 + d);
        __vector double vn = vec_splats(-(0.75 + d));
        for (uint32_t i = 0; i < inner; i++) {
            vp = vec_madd(va, vp, vbp);
            vn = vec_madd(va, vn, vbn);
        }
        double op[2], on[2];
        vec_xst(vp, 0, op);
        vec_xst(vn, 0, on);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            op[0] += 1.0e-5;
        }

        for (int i = 0; i < 2; i++) {
            double ref = b_vals[i] * 4.0;
            if (!isfinite(op[i]) || !isfinite(on[i]) ||
                fabs(op[i] + on[i]) > 1.0e-12 || fabs(op[i] - ref) > 1.0e-12) {
                ctx->errors++;
                ctx->expected_sample = ref;
                ctx->observed_sample = op[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "s390x vx fp64 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 8ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}

#if (defined(__ARCH__) && (__ARCH__ >= 12)) || defined(CROSS_CPU_HAVE_VXE)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("arch=z14")))
#endif
static int32_t vx_compute_fp32(cross_cpu_worker_ctx_t *ctx) {
    const uint32_t inner = 128u;
    const __vector float va = vec_splats(0.75f);
    const float b_vals[4] = { 0.3125f, 0.375f, 0.4375f, 0.50f };
    const float b_neg[4]  = { -0.3125f, -0.375f, -0.4375f, -0.50f };
    const __vector float vbp = vec_xl(0, b_vals);
    const __vector float vbn = vec_xl(0, b_neg);

    uint64_t ops = 0;
    for (uint64_t b = 0; b < ctx->batch_iters; b++) {
        float d = (float)((ctx->seed + b) & 0xFFu) * 0.00390625f;
        __vector float vp = vec_splats(0.75f + d);
        __vector float vn = vec_splats(-(0.75f + d));
        for (uint32_t i = 0; i < inner; i++) {
            vp = vec_madd(va, vp, vbp);
            vn = vec_madd(va, vn, vbn);
        }
        float op[4], on[4];
        vec_xst(vp, 0, op);
        vec_xst(vn, 0, on);

        if (ctx->inject_fault && ctx->thread_idx == 0u && b == 0ULL) {
            op[0] += 1.0e-3f;
        }

        for (int i = 0; i < 4; i++) {
            float ref = b_vals[i] * 4.0f;
            if (!isfinite(op[i]) || !isfinite(on[i]) ||
                fabsf(op[i] + on[i]) > 1.0e-5f || fabsf(op[i] - ref) > 1.0e-5f) {
                ctx->errors++;
                ctx->expected_sample = (double)ref;
                ctx->observed_sample = (double)op[i];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "s390x vxe fp32 mismatch lane %d", i);
                ctx->ops_done = ops;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
        }
        ops += (uint64_t)inner * 16ULL;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}
#endif

static int32_t vx_memory_rw(cross_cpu_worker_ctx_t *ctx) {
    if (ctx->thread_buf == NULL || ctx->thread_buf_bytes < 16u) {
        return CROSS_CPU_STATUS_INIT_ERROR;
    }
    unsigned long long *buf = (unsigned long long *)ctx->thread_buf;
    const size_t vecs = ctx->thread_buf_bytes / 16u;
    uint64_t bytes = 0;

    for (uint64_t pass = 0; pass < ctx->batch_iters; pass++) {
        const unsigned long long salt = (unsigned long long)splitmix64_vx(ctx->seed + pass);
        const __vector unsigned long long base = vec_splats(salt);
        const __vector unsigned long long step = vec_splats(0x0102030405060708ULL);

        __vector unsigned long long cur = base;
        for (size_t i = 0; i < vecs; i++) {
            vec_xst(cur, 0, buf + i * 2u);
            cur = cur + step;
        }

        if (ctx->workload_id == CROSS_CPU_WL_BANDWIDTH_RAM) {
            cross_cpu_arch_flush_cache_range(buf, ctx->thread_buf_bytes);
        }

        if (ctx->inject_fault && ctx->thread_idx == 0u && pass == 0ULL) {
            buf[0] ^= 1ULL;
        }

        cur = base;
        for (size_t i = 0; i < vecs; i++) {
            __vector unsigned long long got = vec_xl(0, buf + i * 2u);
            if (vec_any_ne(got, cur)) {
                ctx->errors++;
                ctx->expected_sample = (double)salt;
                ctx->observed_sample = (double)buf[i * 2u];
                snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                         "s390x vx memory/cache mismatch at vector %zu", i);
                ctx->ops_done = bytes;
                return CROSS_CPU_STATUS_COMPUTE_ERROR;
            }
            cur = cur + step;
        }
        bytes += (uint64_t)(vecs * 16u) * 2ULL;
    }
    ctx->ops_done = bytes;
    return CROSS_CPU_STATUS_OK;
}
#endif

#if defined(__VEC__) || defined(CROSS_CPU_HAVE_VX)
const uint32_t cross_cpu_s390x_vx_compiled = 1u;
#else
const uint32_t cross_cpu_s390x_vx_compiled = 0u;
#endif

#if (defined(__VEC__) || defined(CROSS_CPU_HAVE_VX)) && ((defined(__ARCH__) && (__ARCH__ >= 12)) || defined(CROSS_CPU_HAVE_VXE))
const uint32_t cross_cpu_s390x_vxe_compiled = 1u;
#else
const uint32_t cross_cpu_s390x_vxe_compiled = 0u;
#endif

const cross_cpu_kernel_fn cross_cpu_s390x_vx_kernels[CROSS_CPU_WL_COUNT] = {
#if defined(__VEC__) || defined(CROSS_CPU_HAVE_VX)
    [CROSS_CPU_WL_COMPUTE_INT64] = vx_compute_int64,
    [CROSS_CPU_WL_COMPUTE_INT32] = vx_compute_int32,
    [CROSS_CPU_WL_COMPUTE_INT16] = vx_compute_int16,
    [CROSS_CPU_WL_COMPUTE_INT8]  = vx_compute_int8,
    [CROSS_CPU_WL_COMPUTE_FP64]  = vx_compute_fp64,
#  if (defined(__ARCH__) && (__ARCH__ >= 12)) || defined(CROSS_CPU_HAVE_VXE)
    [CROSS_CPU_WL_COMPUTE_FP32]  = vx_compute_fp32,
#  endif
    [CROSS_CPU_WL_CACHE_L1]      = vx_memory_rw,
    [CROSS_CPU_WL_CACHE_L2]      = vx_memory_rw,
    [CROSS_CPU_WL_CACHE_L3]      = vx_memory_rw,
    [CROSS_CPU_WL_BANDWIDTH_RAM] = vx_memory_rw,
#else
    NULL
#endif
};
