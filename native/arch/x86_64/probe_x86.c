// SPDX-License-Identifier: MIT
#if defined(__linux__)
#  ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#  endif
#endif

#include "cross_cpu_arch.h"

#include <cpuid.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__linux__)
#  include <sys/syscall.h>
#  include <unistd.h>
#  ifndef CROSS_CPU_ARCH_REQ_XCOMP_PERM
#    define CROSS_CPU_ARCH_REQ_XCOMP_PERM 0x1023
#  endif
#  ifndef CROSS_CPU_XFEATURE_XTILEDATA
#    define CROSS_CPU_XFEATURE_XTILEDATA 18
#  endif
#endif

/* Read-only kernel tables defined in isolated per-ISA translation units.
 * No executable code inside any ISA TU is ever invoked unless runtime probing
 * confirms silicon + OS support (preventing compiler auto-vectorized SIGILL). */
extern const uint32_t cross_cpu_x86_sse4_compiled;
extern const uint32_t cross_cpu_x86_aes_compiled;
extern const uint32_t cross_cpu_x86_sha_compiled;
extern const cross_cpu_kernel_fn cross_cpu_x86_sse4_kernels[CROSS_CPU_WL_COUNT];

extern const uint32_t cross_cpu_x86_avx2_compiled;
extern const uint32_t cross_cpu_x86_avxvnni_compiled;
extern const cross_cpu_kernel_fn cross_cpu_x86_avx2_kernels[CROSS_CPU_WL_COUNT];

extern const uint32_t cross_cpu_x86_avx512_compiled;
extern const uint32_t cross_cpu_x86_avx512_vnni_compiled;
extern const uint32_t cross_cpu_x86_avx512_bf16_compiled;
extern const uint32_t cross_cpu_x86_avx512_fp16_compiled;
extern const cross_cpu_kernel_fn cross_cpu_x86_avx512_kernels[CROSS_CPU_WL_COUNT];

extern const uint32_t cross_cpu_x86_amx_compiled;
extern const uint32_t cross_cpu_x86_amx_bf16_compiled;
extern const uint32_t cross_cpu_x86_amx_int8_compiled;
extern const cross_cpu_kernel_fn cross_cpu_x86_amx_kernels[CROSS_CPU_WL_COUNT];

typedef struct x86_probe_state {
    int probed;
    int has_clflush;
    int has_clflushopt;
    size_t clflush_line;

    int has_sse42;
    int has_aes;
    int has_sha;

    int has_avx2_fma;
    int has_avx_vnni;

    int has_avx512_base;
    int has_avx512_vnni;
    int has_avx512_bf16;
    int has_avx512_fp16;

    int has_amx_tile;
    int has_amx_bf16;
    int has_amx_int8;

    uint64_t xcr0;
    uint64_t cpuid1_ecx;
    uint64_t cpuid7_ebx;
    uint64_t cpuid7_ecx_edx;
} x86_probe_state_t;

static x86_probe_state_t g_x86;

static uint64_t read_xcr0(void) {
    uint32_t eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0u));
    return ((uint64_t)edx << 32) | (uint64_t)eax;
}

static void run_x86_probe(void) {
    if (g_x86.probed) {
        return;
    }
    memset(&g_x86, 0, sizeof(g_x86));
    g_x86.clflush_line = 64u;

    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    unsigned int max_leaf = __get_cpuid_max(0u, NULL);

    if (max_leaf >= 1u) {
        __cpuid_count(1u, 0u, eax, ebx, ecx, edx);
        g_x86.cpuid1_ecx = ecx;

        g_x86.has_sse42 = (ecx & (1u << 20)) != 0u;
        g_x86.has_aes = (ecx & (1u << 25)) != 0u;
        g_x86.has_clflush = (edx & (1u << 19)) != 0u;

        unsigned int line_words = (ebx >> 8) & 0xFFu;
        if (line_words > 0u) {
            g_x86.clflush_line = (size_t)line_words * 8u;
        }

        int has_osxsave = (ecx & (1u << 27)) != 0u;
        int has_avx = (ecx & (1u << 28)) != 0u;
        int has_fma = (ecx & (1u << 12)) != 0u;

        if (has_osxsave) {
            g_x86.xcr0 = read_xcr0();
        }

        int os_avx = has_osxsave && ((g_x86.xcr0 & 0x6ULL) == 0x6ULL);
        int os_avx512 = has_osxsave && ((g_x86.xcr0 & 0xE6ULL) == 0xE6ULL);
        int os_amx = has_osxsave && ((g_x86.xcr0 & 0x60000ULL) == 0x60000ULL);

        if (max_leaf >= 7u) {
            unsigned int eax7 = 0, ebx7 = 0, ecx7 = 0, edx7 = 0;
            __cpuid_count(7u, 0u, eax7, ebx7, ecx7, edx7);
            g_x86.cpuid7_ebx = ebx7;
            g_x86.cpuid7_ecx_edx = ((uint64_t)edx7 << 32) | (uint64_t)ecx7;

            g_x86.has_clflushopt = (ebx7 & (1u << 23)) != 0u;
            g_x86.has_sha = (ebx7 & (1u << 29)) != 0u;

            int hw_avx2 = (ebx7 & (1u << 5)) != 0u;
            if (os_avx && has_avx && has_fma && hw_avx2) {
                g_x86.has_avx2_fma = 1;
            }

            int hw_avx512f = (ebx7 & (1u << 16)) != 0u;
            int hw_avx512bw = (ebx7 & (1u << 30)) != 0u;
            int hw_avx512vl = (ebx7 & (1u << 31)) != 0u;
            if (os_avx512 && hw_avx512f && hw_avx512bw && hw_avx512vl) {
                g_x86.has_avx512_base = 1;
                g_x86.has_avx512_vnni = (ecx7 & (1u << 11)) != 0u;
                g_x86.has_avx512_fp16 = (edx7 & (1u << 23)) != 0u;
            }

            int hw_amx_tile = (edx7 & (1u << 24)) != 0u;
            int hw_amx_bf16 = (edx7 & (1u << 22)) != 0u;
            int hw_amx_int8 = (edx7 & (1u << 25)) != 0u;

            if (eax7 >= 1u) {
                unsigned int eax71 = 0, ebx71 = 0, ecx71 = 0, edx71 = 0;
                __cpuid_count(7u, 1u, eax71, ebx71, ecx71, edx71);
                if (g_x86.has_avx2_fma && (eax71 & (1u << 4)) != 0u) {
                    g_x86.has_avx_vnni = 1;
                }
                if (g_x86.has_avx512_base && (eax71 & (1u << 5)) != 0u) {
                    g_x86.has_avx512_bf16 = 1;
                }
            }

            if (os_amx && hw_amx_tile && (hw_amx_bf16 || hw_amx_int8)) {
#if defined(__linux__) && defined(SYS_arch_prctl)
                if (syscall(SYS_arch_prctl,
                            (unsigned long)CROSS_CPU_ARCH_REQ_XCOMP_PERM,
                            (unsigned long)CROSS_CPU_XFEATURE_XTILEDATA) == 0) {
                    g_x86.has_amx_tile = 1;
                    g_x86.has_amx_bf16 = hw_amx_bf16;
                    g_x86.has_amx_int8 = hw_amx_int8;
                }
#else
                (void)hw_amx_bf16;
                (void)hw_amx_int8;
#endif
            }
        }
    }

    g_x86.probed = 1;
}

void cross_cpu_arch_probe(cross_cpu_caps_t *caps) {
    run_x86_probe();
    if (caps == NULL) {
        return;
    }

    caps->arch_id = CROSS_CPU_ARCH_X86_64;
    snprintf(caps->arch_name, sizeof(caps->arch_name), "x86_64");

    caps->raw_probe[0] = g_x86.xcr0;
    caps->raw_probe[1] = g_x86.cpuid1_ecx;
    caps->raw_probe[2] = g_x86.cpuid7_ebx;
    caps->raw_probe[3] = g_x86.cpuid7_ecx_edx;

    uint64_t ff = 0;
    if (g_x86.has_sse42)       ff |= (1ULL << 0);
    if (g_x86.has_aes)         ff |= (1ULL << 1);
    if (g_x86.has_sha)         ff |= (1ULL << 2);
    if (g_x86.has_avx2_fma)    ff |= (1ULL << 3);
    if (g_x86.has_avx_vnni)    ff |= (1ULL << 4);
    if (g_x86.has_avx512_base) ff |= (1ULL << 5);
    if (g_x86.has_avx512_vnni) ff |= (1ULL << 6);
    if (g_x86.has_avx512_bf16) ff |= (1ULL << 7);
    if (g_x86.has_avx512_fp16) ff |= (1ULL << 8);
    if (g_x86.has_amx_tile)    ff |= (1ULL << 9);
    if (g_x86.has_amx_bf16)    ff |= (1ULL << 10);
    if (g_x86.has_amx_int8)    ff |= (1ULL << 11);
    if (g_x86.has_clflushopt)  ff |= (1ULL << 12);
    caps->feature_flags = ff;

    snprintf(caps->probe_summary, sizeof(caps->probe_summary),
             "CPUID/XGETBV xcr0=0x%llx sse4.2=%d aes=%d sha=%d avx2=%d vnni=%d "
             "avx512=%d(vnni=%d,bf16=%d,fp16=%d) amx=%d(bf16=%d,int8=%d) flush=%s",
             (unsigned long long)g_x86.xcr0,
             g_x86.has_sse42, g_x86.has_aes, g_x86.has_sha,
             g_x86.has_avx2_fma, g_x86.has_avx_vnni,
             g_x86.has_avx512_base, g_x86.has_avx512_vnni,
             g_x86.has_avx512_bf16, g_x86.has_avx512_fp16,
             g_x86.has_amx_tile, g_x86.has_amx_bf16, g_x86.has_amx_int8,
             g_x86.has_clflushopt ? "clflushopt" : (g_x86.has_clflush ? "clflush" : "generic"));
}

void cross_cpu_arch_register_backends(cross_cpu_registry_t *reg) {
    run_x86_probe();

    /* SSE4.2 + AES-NI + SHA-NI */
    {
        cross_cpu_backend_t b;
        memset(&b, 0, sizeof(b));
        b.isa_id = CROSS_CPU_ISA_SSE4;
        b.priority = 60u;
        b.compiled_in = cross_cpu_x86_sse4_compiled;
        b.available = (cross_cpu_x86_sse4_compiled && g_x86.has_sse42) ? 1u : 0u;
        snprintf(b.name, sizeof(b.name), "sse4");
        snprintf(b.detail, sizeof(b.detail),
                 "x86_64 SSE4.2 (aes=%d sha=%d)", g_x86.has_aes, g_x86.has_sha);
        if (b.available) {
            memcpy(b.kernels, cross_cpu_x86_sse4_kernels, sizeof(b.kernels));
            if (!(cross_cpu_x86_aes_compiled && g_x86.has_aes)) {
                b.kernels[CROSS_CPU_WL_CRYPTO_AES] = NULL;
                snprintf(b.unsupported_reasons[CROSS_CPU_WL_CRYPTO_AES], CROSS_CPU_MAX_REASON,
                         "CPU or build lacks AES-NI hardware crypto support");
            }
            if (!(cross_cpu_x86_sha_compiled && g_x86.has_sha)) {
                b.kernels[CROSS_CPU_WL_CRYPTO_SHA256] = NULL;
                snprintf(b.unsupported_reasons[CROSS_CPU_WL_CRYPTO_SHA256], CROSS_CPU_MAX_REASON,
                         "CPU or build lacks SHA-NI hardware crypto support");
            }
        }
        (void)cross_cpu_registry_add(reg, &b);
    }

    /* AVX2 + FMA + AVX-VNNI */
    {
        cross_cpu_backend_t b;
        memset(&b, 0, sizeof(b));
        b.isa_id = CROSS_CPU_ISA_AVX2;
        b.priority = 80u;
        b.compiled_in = cross_cpu_x86_avx2_compiled;
        b.available = (cross_cpu_x86_avx2_compiled && g_x86.has_avx2_fma) ? 1u : 0u;
        snprintf(b.name, sizeof(b.name), "avx2");
        snprintf(b.detail, sizeof(b.detail),
                 "x86_64 AVX2 + FMA (avx-vnni=%d)", g_x86.has_avx_vnni);
        if (b.available) {
            memcpy(b.kernels, cross_cpu_x86_avx2_kernels, sizeof(b.kernels));
            if (!(cross_cpu_x86_avxvnni_compiled && g_x86.has_avx_vnni)) {
                b.kernels[CROSS_CPU_WL_COMPUTE_VNNI] = NULL;
                snprintf(b.unsupported_reasons[CROSS_CPU_WL_COMPUTE_VNNI], CROSS_CPU_MAX_REASON,
                         "CPU or build lacks AVX-VNNI support");
            }
        }
        (void)cross_cpu_registry_add(reg, &b);
    }

    /* AVX-512F/BW/VL + VNNI/BF16/FP16 */
    {
        cross_cpu_backend_t b;
        memset(&b, 0, sizeof(b));
        b.isa_id = CROSS_CPU_ISA_AVX512;
        b.priority = 100u;
        b.compiled_in = cross_cpu_x86_avx512_compiled;
        b.available = (cross_cpu_x86_avx512_compiled && g_x86.has_avx512_base) ? 1u : 0u;
        snprintf(b.name, sizeof(b.name), "avx512");
        snprintf(b.detail, sizeof(b.detail),
                 "x86_64 AVX-512F/BW/VL (vnni=%d bf16=%d fp16=%d)",
                 g_x86.has_avx512_vnni, g_x86.has_avx512_bf16, g_x86.has_avx512_fp16);
        if (b.available) {
            memcpy(b.kernels, cross_cpu_x86_avx512_kernels, sizeof(b.kernels));
            if (!(cross_cpu_x86_avx512_vnni_compiled && g_x86.has_avx512_vnni)) {
                b.kernels[CROSS_CPU_WL_COMPUTE_VNNI] = NULL;
            }
            if (!(cross_cpu_x86_avx512_bf16_compiled && g_x86.has_avx512_bf16)) {
                b.kernels[CROSS_CPU_WL_COMPUTE_BF16] = NULL;
            }
            if (!(cross_cpu_x86_avx512_fp16_compiled && g_x86.has_avx512_fp16)) {
                b.kernels[CROSS_CPU_WL_COMPUTE_FP16] = NULL;
            }
        }
        (void)cross_cpu_registry_add(reg, &b);
    }

    /* Intel AMX-TILE/BF16/INT8 */
    {
        cross_cpu_backend_t b;
        memset(&b, 0, sizeof(b));
        b.isa_id = CROSS_CPU_ISA_AMX;
        b.priority = 120u;
        b.compiled_in = cross_cpu_x86_amx_compiled;
        b.available = (cross_cpu_x86_amx_compiled && g_x86.has_amx_tile &&
                       (g_x86.has_amx_bf16 || g_x86.has_amx_int8)) ? 1u : 0u;
        snprintf(b.name, sizeof(b.name), "amx");
        snprintf(b.detail, sizeof(b.detail),
                 "x86_64 Intel AMX (tile=%d bf16=%d int8=%d)",
                 g_x86.has_amx_tile, g_x86.has_amx_bf16, g_x86.has_amx_int8);
        snprintf(b.unsupported_reasons[CROSS_CPU_WL_COMPUTE_MATRIX_FP32], CROSS_CPU_MAX_REASON,
                 "Intel AMX supports BF16 and INT8 tiles; hardware FP32 matrix tile engine is unavailable");
        if (b.available) {
            memcpy(b.kernels, cross_cpu_x86_amx_kernels, sizeof(b.kernels));
            if (!(cross_cpu_x86_amx_bf16_compiled && g_x86.has_amx_bf16)) {
                b.kernels[CROSS_CPU_WL_COMPUTE_MATRIX_BF16] = NULL;
            }
            if (!(cross_cpu_x86_amx_int8_compiled && g_x86.has_amx_int8)) {
                b.kernels[CROSS_CPU_WL_COMPUTE_MATRIX_INT8] = NULL;
            }
        }
        (void)cross_cpu_registry_add(reg, &b);
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("clflushopt")))
static void flush_with_clflushopt(const uint8_t *p, size_t len, size_t step) {
    _mm_mfence();
    for (size_t off = 0; off < len; off += step) {
        _mm_clflushopt((void *)(p + off));
    }
    _mm_mfence();
}
#endif

void cross_cpu_arch_flush_cache_range(const void *ptr, size_t len) {
    run_x86_probe();
    if (ptr == NULL || len == 0u) {
        return;
    }
    const uint8_t *p = (const uint8_t *)ptr;
    const size_t step = (g_x86.clflush_line >= 32u) ? g_x86.clflush_line : 64u;

#if defined(__GNUC__) || defined(__clang__)
    if (g_x86.has_clflushopt) {
        flush_with_clflushopt(p, len, step);
        return;
    }
#endif
    if (g_x86.has_clflush) {
        _mm_mfence();
        for (size_t off = 0; off < len; off += step) {
            _mm_clflush(p + off);
        }
        _mm_mfence();
        return;
    }

    cross_cpu_cache_flush_generic(ptr, len);
}
