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
#  ifndef HWCAP_S390_VX
#    define HWCAP_S390_VX   2048UL /* (1UL << 11) */
#  endif
#  ifndef HWCAP_S390_VXE
#    define HWCAP_S390_VXE  8192UL /* (1UL << 13) */
#  endif
#  ifndef HWCAP_S390_VXE2
#    define HWCAP_S390_VXE2 32768UL /* (1UL << 15) */
#  endif
#endif

extern const uint32_t cross_cpu_s390x_vx_compiled;
extern const uint32_t cross_cpu_s390x_vxe_compiled;
extern const cross_cpu_kernel_fn cross_cpu_s390x_vx_kernels[CROSS_CPU_WL_COUNT];

typedef struct s390x_probe_state {
    int probed;
    int has_vx;
    int has_vxe;
    int has_vxe2;
    uint64_t hwcap;
} s390x_probe_state_t;

static s390x_probe_state_t g_s390;

static void run_s390x_probe(void) {
    if (g_s390.probed) {
        return;
    }
    memset(&g_s390, 0, sizeof(g_s390));

#if defined(__linux__)
    unsigned long hwcap = getauxval(AT_HWCAP);
    g_s390.hwcap = (uint64_t)hwcap;
    g_s390.has_vx   = (hwcap & HWCAP_S390_VX) != 0UL;
    g_s390.has_vxe  = (hwcap & HWCAP_S390_VXE) != 0UL;
    g_s390.has_vxe2 = (hwcap & HWCAP_S390_VXE2) != 0UL;
#endif

    g_s390.probed = 1;
}

void cross_cpu_arch_probe(cross_cpu_caps_t *caps) {
    run_s390x_probe();
    if (caps == NULL) {
        return;
    }

    caps->arch_id = CROSS_CPU_ARCH_S390X;
    snprintf(caps->arch_name, sizeof(caps->arch_name), "s390x");
    caps->raw_probe[0] = g_s390.hwcap;
    caps->raw_probe[1] = 0ULL;
    caps->raw_probe[2] = 0ULL;
    caps->raw_probe[3] = 0ULL;

    uint64_t ff = 0;
    if (g_s390.has_vx)   ff |= (1ULL << 0);
    if (g_s390.has_vxe)  ff |= (1ULL << 1);
    if (g_s390.has_vxe2) ff |= (1ULL << 2);
    caps->feature_flags = ff;

    snprintf(caps->probe_summary, sizeof(caps->probe_summary),
             "getauxval hwcap=0x%llx vx=%d vxe=%d vxe2=%d (big-endian)",
             (unsigned long long)g_s390.hwcap,
             g_s390.has_vx, g_s390.has_vxe, g_s390.has_vxe2);
}

void cross_cpu_arch_register_backends(cross_cpu_registry_t *reg) {
    run_s390x_probe();

    cross_cpu_backend_t b;
    memset(&b, 0, sizeof(b));
    b.isa_id = CROSS_CPU_ISA_VX;
    b.priority = 80u;
    b.compiled_in = cross_cpu_s390x_vx_compiled;
    b.available = (cross_cpu_s390x_vx_compiled && g_s390.has_vx) ? 1u : 0u;
    snprintf(b.name, sizeof(b.name), "vx");
    snprintf(b.detail, sizeof(b.detail),
             "IBM z/Vector (vx=%d vxe=%d vxe2=%d compiled=%u)",
             g_s390.has_vx, g_s390.has_vxe, g_s390.has_vxe2, cross_cpu_s390x_vx_compiled);

    if (b.available) {
        memcpy(b.kernels, cross_cpu_s390x_vx_kernels, sizeof(b.kernels));
        if (!(cross_cpu_s390x_vxe_compiled && g_s390.has_vxe)) {
            b.kernels[CROSS_CPU_WL_COMPUTE_FP32] = NULL;
            snprintf(b.unsupported_reasons[CROSS_CPU_WL_COMPUTE_FP32], CROSS_CPU_MAX_REASON,
                     "IBM z/Vector FP32 vectors require VXE (z14+); CPU reports VX only");
        }
    }

    (void)cross_cpu_registry_add(reg, &b);
}

void cross_cpu_arch_flush_cache_range(const void *ptr, size_t len) {
    cross_cpu_cache_flush_generic(ptr, len);
}
