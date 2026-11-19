// SPDX-License-Identifier: MIT
#include "cross_cpu_arch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if (defined(__AMX_TILE__) && (defined(__AMX_BF16__) || defined(__AMX_INT8__))) || defined(CROSS_CPU_HAVE_AMX)
#include <immintrin.h>

typedef struct amx_tile_config {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved0[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
} amx_tile_config_t;

static void init_amx_cfg_16x64(amx_tile_config_t *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->palette_id = 1u;
    /* Tile 0: C (16x16 fp32/int32 = 16 rows x 64 bytes) */
    cfg->rows[0] = 16u;
    cfg->colsb[0] = 64u;
    /* Tile 1: A (16 rows x 64 bytes) */
    cfg->rows[1] = 16u;
    cfg->colsb[1] = 64u;
    /* Tile 2: B (16 rows x 64 bytes in VNNI layout) */
    cfg->rows[2] = 16u;
    cfg->colsb[2] = 64u;
}

#if defined(__AMX_BF16__) || defined(CROSS_CPU_HAVE_AMX_BF16)
static int32_t amx_compute_matrix_bf16(cross_cpu_worker_ctx_t *ctx) {
    /* Exact BF16 representations: 1.0, 2.0, 3.0, 4.0 */
    static const uint16_t bf_lut[4] = { 0x3f80u, 0x4000u, 0x4040u, 0x4080u };
    static const float    f_lut[4]  = { 1.0f,    2.0f,    3.0f,    4.0f    };

    uint16_t tile_a[16 * 32];
    uint16_t tile_b[16 * 32];
    float    tile_c[16 * 16];

    for (int r = 0; r < 16; r++) {
        uint16_t av = bf_lut[r & 3];
        for (int k = 0; k < 32; k++) {
            tile_a[r * 32 + k] = av;
        }
    }
    /* Tile B is 16 K-pairs x 16 N-columns x 2 BF16 elements */
    for (int kp = 0; kp < 16; kp++) {
        for (int c = 0; c < 16; c++) {
            uint16_t bv = bf_lut[c & 3];
            tile_b[kp * 32 + c * 2 + 0] = bv;
            tile_b[kp * 32 + c * 2 + 1] = bv;
        }
    }

    amx_tile_config_t cfg;
    init_amx_cfg_16x64(&cfg);
    _tile_loadconfig(&cfg);

    const uint32_t repeats = 16u;
    uint64_t flops = 0;

    for (uint64_t iter = 0; iter < ctx->batch_iters; iter++) {
        _tile_zero(0);
        _tile_loadd(1, tile_a, 64);
        _tile_loadd(2, tile_b, 64);

        for (uint32_t rep = 0; rep < repeats; rep++) {
            _tile_dpbf16ps(0, 1, 2);
        }
        _tile_stored(0, tile_c, 64);

        if (ctx->inject_fault && ctx->thread_idx == 0u && iter == 0ULL) {
            tile_c[0] += 1.0f;
        }

        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                float expected = (float)(repeats * 32u) * f_lut[r & 3] * f_lut[c & 3];
                float got = tile_c[r * 16 + c];
                if (!isfinite(got) || fabsf(got - expected) > 1.0e-3f) {
                    _tile_release();
                    ctx->errors++;
                    ctx->expected_sample = (double)expected;
                    ctx->observed_sample = (double)got;
                    snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                             "AMX-BF16 tile mismatch at (%d,%d): expected %.2f, got %.2f",
                             r, c, (double)expected, (double)got);
                    ctx->ops_done = flops;
                    return CROSS_CPU_STATUS_COMPUTE_ERROR;
                }
            }
        }
        /* 2 * M(16) * N(16) * K(32) * repeats = 16384 * repeats FLOPs */
        flops += (uint64_t)repeats * 16384ULL;
    }

    _tile_release();
    ctx->ops_done = flops;
    return CROSS_CPU_STATUS_OK;
}
#endif

#if defined(__AMX_INT8__) || defined(CROSS_CPU_HAVE_AMX_INT8)
static int32_t amx_compute_matrix_int8(cross_cpu_worker_ctx_t *ctx) {
    int8_t  tile_a[16 * 64];
    int8_t  tile_b[16 * 64];
    int32_t tile_c[16 * 16];

    for (int r = 0; r < 16; r++) {
        int8_t av = (int8_t)((r & 3) + 1);
        for (int k = 0; k < 64; k++) {
            tile_a[r * 64 + k] = av;
        }
    }
    /* Tile B: 16 K-quads x 16 N-cols x 4 int8 elements */
    for (int kq = 0; kq < 16; kq++) {
        for (int c = 0; c < 16; c++) {
            int8_t bv = (int8_t)((c & 3) + 1);
            for (int j = 0; j < 4; j++) {
                tile_b[kq * 64 + c * 4 + j] = bv;
            }
        }
    }

    amx_tile_config_t cfg;
    init_amx_cfg_16x64(&cfg);
    _tile_loadconfig(&cfg);

    const uint32_t repeats = 16u;
    uint64_t ops = 0;

    for (uint64_t iter = 0; iter < ctx->batch_iters; iter++) {
        _tile_zero(0);
        _tile_loadd(1, tile_a, 64);
        _tile_loadd(2, tile_b, 64);

        for (uint32_t rep = 0; rep < repeats; rep++) {
            _tile_dpbssd(0, 1, 2);
        }
        _tile_stored(0, tile_c, 64);

        if (ctx->inject_fault && ctx->thread_idx == 0u && iter == 0ULL) {
            tile_c[0] += 1;
        }

        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                int32_t expected = (int32_t)(repeats * 64u) * ((r & 3) + 1) * ((c & 3) + 1);
                int32_t got = tile_c[r * 16 + c];
                if (got != expected) {
                    _tile_release();
                    ctx->errors++;
                    ctx->expected_sample = (double)expected;
                    ctx->observed_sample = (double)got;
                    snprintf(ctx->error_detail, sizeof(ctx->error_detail),
                             "AMX-INT8 tile mismatch at (%d,%d): expected %d, got %d",
                             r, c, expected, got);
                    ctx->ops_done = ops;
                    return CROSS_CPU_STATUS_COMPUTE_ERROR;
                }
            }
        }
        /* 2 * M(16) * N(16) * K(64) * repeats = 32768 * repeats ops */
        ops += (uint64_t)repeats * 32768ULL;
    }

    _tile_release();
    ctx->ops_done = ops;
    return CROSS_CPU_STATUS_OK;
}
#endif

#endif /* AMX */

#if (defined(__AMX_TILE__) && (defined(__AMX_BF16__) || defined(__AMX_INT8__))) || defined(CROSS_CPU_HAVE_AMX)
const uint32_t cross_cpu_x86_amx_compiled = 1u;
#else
const uint32_t cross_cpu_x86_amx_compiled = 0u;
#endif

#if (defined(__AMX_TILE__) && defined(__AMX_BF16__)) || (defined(CROSS_CPU_HAVE_AMX) && defined(CROSS_CPU_HAVE_AMX_BF16))
const uint32_t cross_cpu_x86_amx_bf16_compiled = 1u;
#else
const uint32_t cross_cpu_x86_amx_bf16_compiled = 0u;
#endif

#if (defined(__AMX_TILE__) && defined(__AMX_INT8__)) || (defined(CROSS_CPU_HAVE_AMX) && defined(CROSS_CPU_HAVE_AMX_INT8))
const uint32_t cross_cpu_x86_amx_int8_compiled = 1u;
#else
const uint32_t cross_cpu_x86_amx_int8_compiled = 0u;
#endif

const cross_cpu_kernel_fn cross_cpu_x86_amx_kernels[CROSS_CPU_WL_COUNT] = {
#if (defined(__AMX_TILE__) && defined(__AMX_BF16__)) || (defined(CROSS_CPU_HAVE_AMX) && defined(CROSS_CPU_HAVE_AMX_BF16))
    [CROSS_CPU_WL_COMPUTE_MATRIX_BF16] = amx_compute_matrix_bf16,
#endif
#if (defined(__AMX_TILE__) && defined(__AMX_INT8__)) || (defined(CROSS_CPU_HAVE_AMX) && defined(CROSS_CPU_HAVE_AMX_INT8))
    [CROSS_CPU_WL_COMPUTE_MATRIX_INT8] = amx_compute_matrix_int8,
#endif
    [CROSS_CPU_WL_COMPUTE_MATRIX_FP32] = NULL
};
