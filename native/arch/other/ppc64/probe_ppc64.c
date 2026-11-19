// SPDX-License-Identifier: MIT
#include "cross_cpu_arch.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__linux__)
#  include <sys/auxv.h>
#  ifndef AT_HWCAP
#    define AT_HWCAP 16
#  endif
#  ifndef AT_HWCAP2
#    define AT_HWCAP2 26
#  endif
#  ifndef PPC_FEATURE_HAS_ALTIVEC
#    define PPC_FEATURE_HAS_ALTIVEC 0x10000000UL
#  endif
#  ifndef PPC_FEATURE_HAS_VSX
#    define PPC_FEATURE_HAS_VSX     0x00000080UL
#  endif
#  ifndef PPC_FEATURE2_ARCH_3_00
#    define PPC_FEATURE2_ARCH_3_00  0x00800000UL
#  endif
#  ifndef PPC_FEATURE2_ARCH_3_1
#    define PPC_FEATURE2_ARCH_3_1   0x00040000UL
#  endif
#  ifndef PPC_FEATURE2_MMA
#    define PPC_FEATURE2_MMA        0x00020000UL
#  endif
#endif

extern const uint32_t cross_cpu_ppc64_vsx_compiled;
extern const cross_cpu_kernel_fn cross_cpu_ppc64_vsx_kernels[CROSS_CPU_WL_COUNT];

typedef struct ppc64_probe_state {
    int probed;
    int has_altivec;
    int has_vsx;
    int has_power9;
    int has_power10;
    int has_mma;
    uint64_t hwcap;
    uint64_t hwcap2;
} ppc64_probe_state_t;

static ppc64_probe_state_t g_ppc;

static void run_ppc64_probe(void) {
    if (g_ppc.probed) {
        return;
    }
    memset(&g_ppc, 0, sizeof(g_ppc));

#if defined(__linux__)
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    g_ppc.hwcap = (uint64_t)hwcap;
    g_ppc.hwcap2 = (uint64_t)hwcap2;
    g_ppc.has_altivec = (hwcap & PPC_FEATURE_HAS_ALTIVEC) != 0UL;
    g_ppc.has_vsx     = (hwcap & PPC_FEATURE_HAS_VSX) != 0UL;
    g_ppc.has_power9  = (hwcap2 & PPC_FEATURE2_ARCH_3_00) != 0UL;
    g_ppc.has_power10 = (hwcap2 & PPC_FEATURE2_ARCH_3_1) != 0UL;
    g_ppc.has_mma     = (hwcap2 & PPC_FEATURE2_MMA) != 0UL;
#endif

    g_ppc.probed = 1;
}

void cross_cpu_arch_probe(cross_cpu_caps_t *caps) {
    run_ppc64_probe();
    if (caps == NULL) {
        return;
    }

    caps->arch_id = CROSS_CPU_ARCH_PPC64;
    if (caps->endianness == 1u) {
        snprintf(caps->arch_name, sizeof(caps->arch_name), "ppc64");
    } else {
        snprintf(caps->arch_name, sizeof(caps->arch_name), "ppc64le");
    }

    caps->raw_probe[0] = g_ppc.hwcap;
    caps->raw_probe[1] = g_ppc.hwcap2;
    caps->raw_probe[2] = 0ULL;
    caps->raw_probe[3] = 0ULL;

    uint64_t ff = 0;
    if (g_ppc.has_altivec) ff |= (1ULL << 0);
    if (g_ppc.has_vsx)     ff |= (1ULL << 1);
    if (g_ppc.has_power9)  ff |= (1ULL << 2);
    if (g_ppc.has_power10) ff |= (1ULL << 3);
    if (g_ppc.has_mma)     ff |= (1ULL << 4);
    caps->feature_flags = ff;

    snprintf(caps->probe_summary, sizeof(caps->probe_summary),
             "getauxval hwcap=0x%llx hwcap2=0x%llx altivec=%d vsx=%d p9=%d p10=%d mma=%d",
             (unsigned long long)g_ppc.hwcap,
             (unsigned long long)g_ppc.hwcap2,
             g_ppc.has_altivec, g_ppc.has_vsx,
             g_ppc.has_power9, g_ppc.has_power10, g_ppc.has_mma);
}

void cross_cpu_arch_register_backends(cross_cpu_registry_t *reg) {
    run_ppc64_probe();

    cross_cpu_backend_t b;
    memset(&b, 0, sizeof(b));
    b.isa_id = CROSS_CPU_ISA_VSX;
    b.priority = 80u;
    b.compiled_in = cross_cpu_ppc64_vsx_compiled;
    b.available = (cross_cpu_ppc64_vsx_compiled && g_ppc.has_vsx) ? 1u : 0u;
    snprintf(b.name, sizeof(b.name), "vsx");
    snprintf(b.detail, sizeof(b.detail),
             "PowerPC AltiVec/VSX (vsx=%d mma=%d compiled=%u)",
             g_ppc.has_vsx, g_ppc.has_mma, cross_cpu_ppc64_vsx_compiled);

    if (b.available) {
        memcpy(b.kernels, cross_cpu_ppc64_vsx_kernels, sizeof(b.kernels));
    }

    (void)cross_cpu_registry_add(reg, &b);
}

void cross_cpu_arch_flush_cache_range(const void *ptr, size_t len) {
    cross_cpu_cache_flush_generic(ptr, len);
}
