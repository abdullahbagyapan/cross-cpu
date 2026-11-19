// SPDX-License-Identifier: MIT
#include "cross_cpu_arch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__ARM_FEATURE_SME) || defined(CROSS_CPU_HAVE_SME)
#include <arm_sme.h>

#define SME_MAX_SVLW 64u /* Up to 2048-bit streaming vector length */

__arm_new("za")
static uint64_t sme_exec_fp32_tile(const float *a_vec,
                                   const float *b_vec,
                                   float *c_mat,
                                   uint32_t repeats,
                                   uint32_t *out_svlw) __arm_streaming {
    const uint64_t svlw = svcntw();
    const svbool_t pg = svptrue_b32();
    const svfloat32_t va = svld1_f32(pg, a_vec);
    const svfloat32_t vb = svld1_f32(pg, b_vec);

    svzero_za();
    for (uint32_t r = 0; r < repeats; r++) {
        svmopa_za32_f32_m(0, pg, pg, va, vb);
    }

    for (uint32_t row = 0; row < (uint32_t)svlw; row++) {
        svst1_hor_za32(0, row, pg, c_mat + (size_t)row * (size_t)svlw);
    }
    *out_svlw = (uint32_t)svlw;
    /* 2 * M(svlw) * N(svlw) * K(1) * repeats */
    return 2ULL * svlw * svlw * (uint64_t)repeats;
}

__arm_new("za")
static uint64_t sme_exec_bf16_tile(const uint16_t *a_vec,
                                   const uint16_t *b_vec,
                                   float *c_mat,
                                   uint32_t repeats,
                                   uint32_t *out_svlw) __arm_streaming {
    const uint64_t svlw = svcntw();
    const svbool_t pg16 = svptrue_b16();
    const svbool_t pg32 = svptrue_b32();
    const svbfloat16_t va = svreinterpret_bf16_u16(svld1_u16(pg16, a_vec));
    const svbfloat16_t vb = svreinterpret_bf16_u16(svld1_u16(pg16, b_vec));

    svzero_za();
    for (uint32_t r = 0; r < repeats; r++) {
        svmopa_za32_bf16_m(0, pg16, pg16, va, vb);
    }

    for (uint32_t row = 0; row < (uint32_t)svlw; row++) {
        svst1_hor_za32(0, row, pg32, c_mat + (size_t)row * (size_t)svlw);
    }
    *out_svlw = (uint32_t)svlw;
    /* 2 * M(svlw) * N(svlw) * K(2) * repeats */
    return 4ULL * svlw * svlw * (uint64_t)repeats;
}

__arm_new("za")
static uint64_t sme_exec_int8_tile(const int8_t *a_vec,
                                   const int8_t *b_vec,
                                   int32_t *c_mat,
                                   uint32_t repeats,
                                   uint32_t *out_svlw) __arm_streaming {
    const uint64_t svlw = svcntw();
    const svbool_t pg8  = svptrue_b8();
    const svbool_t pg32 = svptrue_b32();
    const svint8_t va = svld1_s8(pg8, a_vec);
    const svint8_t vb = svld1_s8(pg8, b_vec);

    svzero_za();
    for (uint32_t r = 0; r < repeats; r++) {
        svmopa_za32_s8_m(0, pg8, pg8, va, vb);
    }

    for (uint32_t row = 0; row < (uint32_t)svlw; row++) {
        svst1_hor_za32(0, row, pg32, c_mat + (size_t)row * (size_t)svlw);
    }
    *out_svlw = (uint32_t)svlw;
    /* 2 * M(svlw) * N(svlw) * K(4) * repeats */
    return 8ULL * svlw * svlw * (uint64_t)repeats;
}

static int32_t sme_compute_matrix_fp32(cross_cpu_worker_ctx_t *ctx) {
    float a_vec[SME_MAX_SVLW];
    float b_vec[SME_MAX_SVLW];
    float c_mat[SME_MAX_SVLW * SME_MAX_SVLW];
    for (uint32_t i = 0; i < SME_MAX_SVLW; i++) {
        a_vec[i] = (float)((i & 3u) + 1u);
        b_vec[i] = (float)(((i + 1u) & 3u) + 1u);
    }

    const uint32_t repeats = 32u;
    uint64_t flops = 0;

    for (uint64_t iter = 0; iter < ctx->batch_iters; iter++) {
        uint32_t svlw = 0;
        uint64_t step_flops = sme_exec_fp32_tile(a_vec, b_vec, c_mat, repeats, &svlw);
        if (svlw == 0u || svlw > SME_MAX_SVLW) {
            snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                     "Unexpected SME SVLw=%u", svlw);
            return CROSS_CPU_STATUS_INIT_ERROR;
        }

        if (ctx->inject_fault && ctx->thread_idx == 0u && iter == 0ULL) {
            c_mat[0] += 1.0f;
        }

        for (uint32_t r = 0; r < svlw; r++) {
            for (uint32_t c = 0; c < svlw; c++) {
                float expected = (float)repeats * a_vec[r] * b_vec[c];
                float got = c_mat[(size_t)r * svlw + c];
                if (!isfinite(got) || fabsf(got - expected) > 1.0e-3f) {
                    ctx->errors++;
                    ctx->expected_sample = (double)expected;
                    ctx->observed_sample = (double)got;
                    snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                             "ARM SME FP32 tile mismatch at (%u,%u): expected %.2f, got %.2f",
                             r, c, (double)expected, (double)got);
                    ctx->ops_done = flops;
                    return CROSS_CPU_STATUS_COMPUTE_ERROR;
                }
            }
        }
        flops += step_flops;
    }
    ctx->ops_done = flops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sme_compute_matrix_bf16(cross_cpu_worker_ctx_t *ctx) {
    static const uint16_t bf_lut[4] = { 0x3f80u, 0x4000u, 0x4040u, 0x4080u };
    static const float    f_lut[4]  = { 1.0f,    2.0f,    3.0f,    4.0f    };

    uint16_t a_vec[SME_MAX_SVLW * 2u];
    uint16_t b_vec[SME_MAX_SVLW * 2u];
    float    c_mat[SME_MAX_SVLW * SME_MAX_SVLW];
    for (uint32_t i = 0; i < SME_MAX_SVLW; i++) {
        a_vec[i * 2u + 0u] = bf_lut[i & 3u];
        a_vec[i * 2u + 1u] = bf_lut[i & 3u];
        b_vec[i * 2u + 0u] = bf_lut[(i + 1u) & 3u];
        b_vec[i * 2u + 1u] = bf_lut[(i + 1u) & 3u];
    }

    const uint32_t repeats = 32u;
    uint64_t flops = 0;

    for (uint64_t iter = 0; iter < ctx->batch_iters; iter++) {
        uint32_t svlw = 0;
        uint64_t step_flops = sme_exec_bf16_tile(a_vec, b_vec, c_mat, repeats, &svlw);
        if (svlw == 0u || svlw > SME_MAX_SVLW) {
            return CROSS_CPU_STATUS_INIT_ERROR;
        }

        if (ctx->inject_fault && ctx->thread_idx == 0u && iter == 0ULL) {
            c_mat[0] += 1.0f;
        }

        for (uint32_t r = 0; r < svlw; r++) {
            for (uint32_t c = 0; c < svlw; c++) {
                float expected = (float)(repeats * 2u) * f_lut[r & 3u] * f_lut[(c + 1u) & 3u];
                float got = c_mat[(size_t)r * svlw + c];
                if (!isfinite(got) || fabsf(got - expected) > 1.0e-3f) {
                    ctx->errors++;
                    ctx->expected_sample = (double)expected;
                    ctx->observed_sample = (double)got;
                    snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                             "ARM SME BF16 tile mismatch at (%u,%u)", r, c);
                    ctx->ops_done = flops;
                    return CROSS_CPU_STATUS_COMPUTE_ERROR;
                }
            }
        }
        flops += step_flops;
    }
    ctx->ops_done = flops;
    return CROSS_CPU_STATUS_OK;
}

static int32_t sme_compute_matrix_int8(cross_cpu_worker_ctx_t *ctx) {
    int8_t  a_vec[SME_MAX_SVLW * 4u];
    int8_t  b_vec[SME_MAX_SVLW * 4u];
    int32_t c_mat[SME_MAX_SVLW * SME_MAX_SVLW];
    for (uint32_t i = 0; i < SME_MAX_SVLW; i++) {
        int8_t av = (int8_t)((i & 3u) + 1u);
        int8_t bv = (int8_t)(((i + 1u) & 3u) + 1u);
        for (uint32_t j = 0; j < 4u; j++) {
            a_vec[i * 4u + j] = av;
            b_vec[i * 4u + j] = bv;
        }
    }

    const uint32_t repeats = 32u;
    uint64_t ops = 0;

    for (uint64_t iter = 0; iter < ctx->batch_iters; iter++) {
        uint32_t svlw = 0;
        uint64_t step_ops = sme_exec_int8_tile(a_vec, b_vec, c_mat, repeats, &svlw);
        if (svlw == 0u || svlw > SME_MAX_SVLW) {
            return CROSS_CPU_STATUS_INIT_ERROR;
        }

        if (ctx->inject_fault && ctx->thread_idx == 0u && iter == 0ULL) {
            c_mat[0] += 1;
        }

        for (uint32_t r = 0; r < svlw; r++) {
            for (uint32_t c = 0; c < svlw; c++) {
                int32_t expected = (int32_t)(repeats * 4u) *
                                   (int32_t)((r & 3u) + 1u) *
                                   (int32_t)(((c + 1u) & 3u) + 1u);
                int32_t got = c_mat[(size_t)r * svlw + c];
                if (got != expected) {
                    ctx->errors++;
                    ctx->expected_sample = (double)expected;
                    ctx->observed_sample = (double)got;
                    snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                             "ARM SME INT8 tile mismatch at (%u,%u)", r, c);
                    ctx->ops_done = ops;
                    return CROSS_CPU_STATUS_COMPUTE_ERROR;
                }
            }
        }
        ops += step_ops;
    }
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}
#endif

#if defined(__ARM_FEATURE_SME) || defined(CROSS_CPU_HAVE_SME)
const uint32_t cross_cpu_arm64_sme_compiled = 1u;
#else
const uint32_t cross_cpu_arm64_sme_compiled = 0u;
#endif

const cross_cpu_kernel_fn cross_cpu_arm64_sme_kernels[CROSS_CPU_WL_COUNT] = {
#if defined(__ARM_FEATURE_SME) || defined(CROSS_CPU_HAVE_SME)
    [CROSS_CPU_WL_COMPUTE_MATRIX_FP32] = sme_compute_matrix_fp32,
    [CROSS_CPU_WL_COMPUTE_MATRIX_BF16] = sme_compute_matrix_bf16,
    [CROSS_CPU_WL_COMPUTE_MATRIX_INT8] = sme_compute_matrix_int8,
#else
    NULL
#endif
};
