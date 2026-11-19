// SPDX-License-Identifier: MIT
#include "cross_cpu_arch.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__linux__)
#  include <sys/auxv.h>
#  include <unistd.h>
#  if defined(__has_include)
#    if __has_include(<asm/hwcap.h>)
#      include <asm/hwcap.h>
#    endif
#  endif
#  ifndef AT_HWCAP
#    define AT_HWCAP 16
#  endif
#  ifndef AT_HWCAP2
#    define AT_HWCAP2 26
#  endif
#  ifndef HWCAP_ASIMD
#    define HWCAP_ASIMD (1UL << 1)
#  endif
#  ifndef HWCAP_AES
#    define HWCAP_AES (1UL << 3)
#  endif
#  ifndef HWCAP_SHA2
#    define HWCAP_SHA2 (1UL << 6)
#  endif
#  ifndef HWCAP_FPHP
#    define HWCAP_FPHP (1UL << 9)
#  endif
#  ifndef HWCAP_ASIMDHP
#    define HWCAP_ASIMDHP (1UL << 10)
#  endif
#  ifndef HWCAP_ASIMDDP
#    define HWCAP_ASIMDDP (1UL << 20)
#  endif
#  ifndef HWCAP_SVE
#    define HWCAP_SVE (1UL << 22)
#  endif
#  ifndef HWCAP2_SVE2
#    define HWCAP2_SVE2 (1UL << 1)
#  endif
#  ifndef HWCAP2_SVEBF16
#    define HWCAP2_SVEBF16 (1UL << 12)
#  endif
#  ifndef HWCAP2_I8MM
#    define HWCAP2_I8MM (1UL << 13)
#  endif
#  ifndef HWCAP2_BF16
#    define HWCAP2_BF16 (1UL << 14)
#  endif
#  ifndef HWCAP2_SME
#    define HWCAP2_SME (1UL << 23)
#  endif
#elif defined(__APPLE__)
#  include <sys/sysctl.h>
#endif

/* Read-only kernel tables defined in isolated per-ISA translation units.
 * No code inside isa_sve.c or isa_sme.c is ever invoked unless runtime probing
 * confirms silicon + OS support (preventing compiler auto-vectorized SIGILL). */
extern const uint32_t cross_cpu_arm64_neon_compiled;
extern const uint32_t cross_cpu_arm64_fp16_compiled;
extern const uint32_t cross_cpu_arm64_bf16_compiled;
extern const uint32_t cross_cpu_arm64_dotprod_compiled;
extern const uint32_t cross_cpu_arm64_i8mm_compiled;
extern const uint32_t cross_cpu_arm64_aes_compiled;
extern const uint32_t cross_cpu_arm64_sha2_compiled;
extern const cross_cpu_kernel_fn cross_cpu_arm64_vnni_dotprod_kernel;
extern const cross_cpu_kernel_fn cross_cpu_arm64_vnni_i8mm_kernel;
extern const cross_cpu_kernel_fn cross_cpu_arm64_neon_kernels[CROSS_CPU_WL_COUNT];

extern const uint32_t cross_cpu_arm64_sve_compiled;
extern const cross_cpu_kernel_fn cross_cpu_arm64_sve_kernels[CROSS_CPU_WL_COUNT];

extern const uint32_t cross_cpu_arm64_sme_compiled;
extern const cross_cpu_kernel_fn cross_cpu_arm64_sme_kernels[CROSS_CPU_WL_COUNT];

typedef struct arm64_probe_state {
    int probed;
    size_t dcache_line;

    int has_neon;
    int has_fp16;
    int has_bf16;
    int has_dotprod;
    int has_i8mm;
    int has_aes;
    int has_sha2;

    int has_sve;
    int has_sve2;
    int has_sve_bf16;

    int has_sme;

    uint64_t hwcap;
    uint64_t hwcap2;
} arm64_probe_state_t;

static arm64_probe_state_t g_arm64;

#if defined(__APPLE__)
static int sysctl_bool(const char *name) {
    int val = 0;
    size_t sz = sizeof(val);
    if (sysctlbyname(name, &val, &sz, NULL, 0) == 0 && val != 0) {
        return 1;
    }
    return 0;
}
#endif

static void run_arm64_probe(void) {
    if (g_arm64.probed) {
        return;
    }
    memset(&g_arm64, 0, sizeof(g_arm64));
    g_arm64.dcache_line = 64u;

#if defined(__linux__)
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    g_arm64.hwcap = (uint64_t)hwcap;
    g_arm64.hwcap2 = (uint64_t)hwcap2;

    g_arm64.has_neon    = (hwcap & HWCAP_ASIMD) != 0UL;
    g_arm64.has_aes     = (hwcap & HWCAP_AES) != 0UL;
    g_arm64.has_sha2    = (hwcap & HWCAP_SHA2) != 0UL;
    g_arm64.has_fp16    = ((hwcap & HWCAP_FPHP) != 0UL) && ((hwcap & HWCAP_ASIMDHP) != 0UL);
    g_arm64.has_dotprod = (hwcap & HWCAP_ASIMDDP) != 0UL;
    g_arm64.has_sve     = (hwcap & HWCAP_SVE) != 0UL;

    g_arm64.has_sve2     = (hwcap2 & HWCAP2_SVE2) != 0UL;
    g_arm64.has_sve_bf16 = (hwcap2 & HWCAP2_SVEBF16) != 0UL;
    g_arm64.has_i8mm     = (hwcap2 & HWCAP2_I8MM) != 0UL;
    g_arm64.has_bf16     = (hwcap2 & HWCAP2_BF16) != 0UL;
    g_arm64.has_sme      = (hwcap2 & HWCAP2_SME) != 0UL;

#  ifdef _SC_LEVEL1_DCACHE_LINESIZE
    long cls = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (cls >= 32L && cls <= 512L) {
        g_arm64.dcache_line = (size_t)cls;
    }
#  endif
#elif defined(__APPLE__)
    g_arm64.has_neon    = sysctl_bool("hw.optional.arm.AdvSIMD") || sysctl_bool("hw.optional.neon");
    g_arm64.has_fp16    = sysctl_bool("hw.optional.arm.FEAT_FP16") || sysctl_bool("hw.optional.neon_fp16");
    g_arm64.has_bf16    = sysctl_bool("hw.optional.arm.FEAT_BF16");
    g_arm64.has_dotprod = sysctl_bool("hw.optional.arm.FEAT_DotProd");
    g_arm64.has_i8mm    = sysctl_bool("hw.optional.arm.FEAT_I8MM");
    g_arm64.has_aes     = sysctl_bool("hw.optional.arm.FEAT_AES");
    g_arm64.has_sha2    = sysctl_bool("hw.optional.arm.FEAT_SHA256");
    g_arm64.has_sve     = 0; /* SVE is not exposed on macOS */
    g_arm64.has_sve2    = 0;
    g_arm64.has_sve_bf16 = 0;
    g_arm64.has_sme     = sysctl_bool("hw.optional.arm.FEAT_SME");

    uint64_t cls = 0;
    size_t sz = sizeof(cls);
    if (sysctlbyname("hw.cachelinesize", &cls, &sz, NULL, 0) == 0 && cls >= 32ULL && cls <= 512ULL) {
        g_arm64.dcache_line = (size_t)cls;
    }

    uint64_t mask = 0;
    if (g_arm64.has_neon)    mask |= (1ULL << 0);
    if (g_arm64.has_fp16)    mask |= (1ULL << 1);
    if (g_arm64.has_bf16)    mask |= (1ULL << 2);
    if (g_arm64.has_dotprod) mask |= (1ULL << 3);
    if (g_arm64.has_i8mm)    mask |= (1ULL << 4);
    if (g_arm64.has_aes)     mask |= (1ULL << 5);
    if (g_arm64.has_sha2)    mask |= (1ULL << 6);
    if (g_arm64.has_sme)     mask |= (1ULL << 7);
    g_arm64.hwcap = mask;
    g_arm64.hwcap2 = 0;
#else
    g_arm64.has_neon = 1;
#endif

    g_arm64.probed = 1;
}

void cross_cpu_arch_probe(cross_cpu_caps_t *caps) {
    run_arm64_probe();
    if (caps == NULL) {
        return;
    }

    caps->arch_id = CROSS_CPU_ARCH_ARM64;
    snprintf(caps->arch_name, sizeof(caps->arch_name), "arm64");

    caps->raw_probe[0] = g_arm64.hwcap;
    caps->raw_probe[1] = g_arm64.hwcap2;
    caps->raw_probe[2] = (uint64_t)g_arm64.dcache_line;
    caps->raw_probe[3] = 0ULL;

    uint64_t ff = 0;
    if (g_arm64.has_neon)     ff |= (1ULL << 0);
    if (g_arm64.has_fp16)     ff |= (1ULL << 1);
    if (g_arm64.has_bf16)     ff |= (1ULL << 2);
    if (g_arm64.has_dotprod)  ff |= (1ULL << 3);
    if (g_arm64.has_i8mm)     ff |= (1ULL << 4);
    if (g_arm64.has_aes)      ff |= (1ULL << 5);
    if (g_arm64.has_sha2)     ff |= (1ULL << 6);
    if (g_arm64.has_sve)      ff |= (1ULL << 7);
    if (g_arm64.has_sve2)     ff |= (1ULL << 8);
    if (g_arm64.has_sve_bf16) ff |= (1ULL << 9);
    if (g_arm64.has_sme)      ff |= (1ULL << 10);
    caps->feature_flags = ff;

    snprintf(caps->probe_summary, sizeof(caps->probe_summary),
             "%s hwcap=0x%llx hwcap2=0x%llx neon=%d fp16=%d bf16=%d dotprod=%d "
             "i8mm=%d aes=%d sha2=%d sve=%d sve2=%d sme=%d dline=%zuB",
#if defined(__APPLE__)
             "sysctlbyname",
#else
             "getauxval",
#endif
             (unsigned long long)g_arm64.hwcap,
             (unsigned long long)g_arm64.hwcap2,
             g_arm64.has_neon, g_arm64.has_fp16, g_arm64.has_bf16,
             g_arm64.has_dotprod, g_arm64.has_i8mm,
             g_arm64.has_aes, g_arm64.has_sha2,
             g_arm64.has_sve, g_arm64.has_sve2, g_arm64.has_sme,
             g_arm64.dcache_line);
}

void cross_cpu_arch_register_backends(cross_cpu_registry_t *reg) {
    run_arm64_probe();

    /* NEON / ASIMD + optional ARMv8.2-v8.6 sub-features */
    {
        cross_cpu_backend_t b;
        memset(&b, 0, sizeof(b));
        b.isa_id = CROSS_CPU_ISA_NEON;
        b.priority = 80u;
        b.compiled_in = cross_cpu_arm64_neon_compiled;
        b.available = (cross_cpu_arm64_neon_compiled && g_arm64.has_neon) ? 1u : 0u;
        snprintf(b.name, sizeof(b.name), "neon");
        snprintf(b.detail, sizeof(b.detail),
                 "ARM64 NEON/ASIMD (fp16=%d bf16=%d dotprod=%d i8mm=%d aes=%d sha2=%d)",
                 g_arm64.has_fp16, g_arm64.has_bf16, g_arm64.has_dotprod,
                 g_arm64.has_i8mm, g_arm64.has_aes, g_arm64.has_sha2);
        if (b.available) {
            memcpy(b.kernels, cross_cpu_arm64_neon_kernels, sizeof(b.kernels));

            if (!(cross_cpu_arm64_fp16_compiled && g_arm64.has_fp16)) {
                b.kernels[CROSS_CPU_WL_COMPUTE_FP16] = NULL;
                snprintf(b.unsupported_reasons[CROSS_CPU_WL_COMPUTE_FP16], CROSS_CPU_MAX_REASON,
                         "CPU or build lacks ARMv8.2 FP16 hardware arithmetic");
            }
            if (!(cross_cpu_arm64_bf16_compiled && g_arm64.has_bf16)) {
                b.kernels[CROSS_CPU_WL_COMPUTE_BF16] = NULL;
                snprintf(b.unsupported_reasons[CROSS_CPU_WL_COMPUTE_BF16], CROSS_CPU_MAX_REASON,
                         "CPU or build lacks ARMv8.6 BF16 hardware arithmetic");
            }
            if (cross_cpu_arm64_i8mm_compiled && g_arm64.has_i8mm &&
                cross_cpu_arm64_vnni_i8mm_kernel != NULL) {
                b.kernels[CROSS_CPU_WL_COMPUTE_VNNI] = cross_cpu_arm64_vnni_i8mm_kernel;
            } else if (cross_cpu_arm64_dotprod_compiled && g_arm64.has_dotprod &&
                       cross_cpu_arm64_vnni_dotprod_kernel != NULL) {
                b.kernels[CROSS_CPU_WL_COMPUTE_VNNI] = cross_cpu_arm64_vnni_dotprod_kernel;
            } else {
                b.kernels[CROSS_CPU_WL_COMPUTE_VNNI] = NULL;
                snprintf(b.unsupported_reasons[CROSS_CPU_WL_COMPUTE_VNNI], CROSS_CPU_MAX_REASON,
                         "CPU or build lacks ARM DotProd/I8MM integer dot-product instructions");
            }
            if (!(cross_cpu_arm64_aes_compiled && g_arm64.has_aes)) {
                b.kernels[CROSS_CPU_WL_CRYPTO_AES] = NULL;
                snprintf(b.unsupported_reasons[CROSS_CPU_WL_CRYPTO_AES], CROSS_CPU_MAX_REASON,
                         "CPU or build lacks ARM FEAT_AES hardware crypto instructions");
            }
            if (!(cross_cpu_arm64_sha2_compiled && g_arm64.has_sha2)) {
                b.kernels[CROSS_CPU_WL_CRYPTO_SHA256] = NULL;
                snprintf(b.unsupported_reasons[CROSS_CPU_WL_CRYPTO_SHA256], CROSS_CPU_MAX_REASON,
                         "CPU or build lacks ARM FEAT_SHA256 hardware crypto instructions");
            }
        }
        (void)cross_cpu_registry_add(reg, &b);
    }

    /* ARM64 SVE / SVE2 (Linux only; not exposed on macOS) */
    {
        cross_cpu_backend_t b;
        memset(&b, 0, sizeof(b));
        b.isa_id = CROSS_CPU_ISA_SVE;
        b.priority = 100u;
        b.compiled_in = cross_cpu_arm64_sve_compiled;
        b.available = (cross_cpu_arm64_sve_compiled && g_arm64.has_sve) ? 1u : 0u;
        snprintf(b.name, sizeof(b.name), "sve");
        snprintf(b.detail, sizeof(b.detail),
                 "ARM64 SVE (sve=%d sve2=%d sve-bf16=%d)",
                 g_arm64.has_sve, g_arm64.has_sve2, g_arm64.has_sve_bf16);
        if (b.available) {
            memcpy(b.kernels, cross_cpu_arm64_sve_kernels, sizeof(b.kernels));
        }
        (void)cross_cpu_registry_add(reg, &b);
    }

    /* ARM64 SME (Linux / Apple M4+) */
    {
        cross_cpu_backend_t b;
        memset(&b, 0, sizeof(b));
        b.isa_id = CROSS_CPU_ISA_SME;
        b.priority = 120u;
        b.compiled_in = cross_cpu_arm64_sme_compiled;
        b.available = (cross_cpu_arm64_sme_compiled && g_arm64.has_sme) ? 1u : 0u;
        snprintf(b.name, sizeof(b.name), "sme");
        snprintf(b.detail, sizeof(b.detail),
                 "ARM64 SME matrix tile engine (sme=%d)", g_arm64.has_sme);
        if (b.available) {
            memcpy(b.kernels, cross_cpu_arm64_sme_kernels, sizeof(b.kernels));
        }
        (void)cross_cpu_registry_add(reg, &b);
    }
}

void cross_cpu_arch_flush_cache_range(const void *ptr, size_t len) {
    run_arm64_probe();
    if (ptr == NULL || len == 0u) {
        return;
    }
    const uint8_t *p = (const uint8_t *)ptr;
    const size_t step = (g_arm64.dcache_line >= 32u) ? g_arm64.dcache_line : 64u;

    __asm__ volatile("dsb ish" ::: "memory");
    for (size_t off = 0; off < len; off += step) {
        __asm__ volatile("dc civac, %0" :: "r"(p + off) : "memory");
    }
    __asm__ volatile("dsb ish" ::: "memory");
}
