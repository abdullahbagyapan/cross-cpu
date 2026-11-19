// SPDX-License-Identifier: MIT
#if defined(__linux__)
#  ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#  endif
#endif

#include "cross_cpu_arch.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__linux__)
#  include <sys/auxv.h>
#  include <sys/syscall.h>
#  include <unistd.h>
#  ifndef AT_HWCAP
#    define AT_HWCAP 16
#  endif
#  ifndef CROSS_CPU_RV_HWCAP_V
#    define CROSS_CPU_RV_HWCAP_V (1UL << ('V' - 'A'))
#  endif
#  ifndef SYS_riscv_hwprobe
#    define SYS_riscv_hwprobe 258
#  endif
#  define CROSS_CPU_RV_HWPROBE_KEY_IMA_EXT_0 4LL
#  define CROSS_CPU_RV_HWPROBE_IMA_V         (1ULL << 2)
#  define CROSS_CPU_RV_HWPROBE_EXT_ZVKNED    (1ULL << 21)
#  define CROSS_CPU_RV_HWPROBE_EXT_ZVKNHB    (1ULL << 23)

struct cross_cpu_rv_hwprobe {
    int64_t  key;
    uint64_t value;
};
#endif

extern const uint32_t cross_cpu_riscv_rvv_compiled;
extern const cross_cpu_kernel_fn cross_cpu_riscv_rvv_kernels[CROSS_CPU_WL_COUNT];

typedef struct riscv64_probe_state {
    int probed;
    int has_rvv;
    int has_zvk;
    uint64_t hwcap;
    uint64_t hwprobe_ext0;
} riscv64_probe_state_t;

static riscv64_probe_state_t g_rv;

static void run_riscv64_probe(void) {
    if (g_rv.probed) {
        return;
    }
    memset(&g_rv, 0, sizeof(g_rv));

#if defined(__linux__)
    unsigned long hwcap = getauxval(AT_HWCAP);
    g_rv.hwcap = (uint64_t)hwcap;
    if ((hwcap & CROSS_CPU_RV_HWCAP_V) != 0UL) {
        g_rv.has_rvv = 1;
    }

    struct cross_cpu_rv_hwprobe pair;
    pair.key = CROSS_CPU_RV_HWPROBE_KEY_IMA_EXT_0;
    pair.value = 0ULL;
    if (syscall(SYS_riscv_hwprobe, &pair, (size_t)1u, (size_t)0u, NULL, 0u) == 0 &&
        pair.key == CROSS_CPU_RV_HWPROBE_KEY_IMA_EXT_0) {
        g_rv.hwprobe_ext0 = pair.value;
        if ((pair.value & CROSS_CPU_RV_HWPROBE_IMA_V) != 0ULL) {
            g_rv.has_rvv = 1;
        }
        if ((pair.value & (CROSS_CPU_RV_HWPROBE_EXT_ZVKNED | CROSS_CPU_RV_HWPROBE_EXT_ZVKNHB)) != 0ULL) {
            g_rv.has_zvk = 1;
        }
    }
#endif

    g_rv.probed = 1;
}

void cross_cpu_arch_probe(cross_cpu_caps_t *caps) {
    run_riscv64_probe();
    if (caps == NULL) {
        return;
    }

    caps->arch_id = CROSS_CPU_ARCH_RISCV64;
    snprintf(caps->arch_name, sizeof(caps->arch_name), "riscv64");
    caps->raw_probe[0] = g_rv.hwcap;
    caps->raw_probe[1] = g_rv.hwprobe_ext0;
    caps->raw_probe[2] = 0ULL;
    caps->raw_probe[3] = 0ULL;

    uint64_t ff = 0;
    if (g_rv.has_rvv) ff |= (1ULL << 0);
    if (g_rv.has_zvk) ff |= (1ULL << 1);
    caps->feature_flags = ff;

    snprintf(caps->probe_summary, sizeof(caps->probe_summary),
             "getauxval/riscv_hwprobe hwcap=0x%llx ext0=0x%llx rvv=%d zvk=%d",
             (unsigned long long)g_rv.hwcap,
             (unsigned long long)g_rv.hwprobe_ext0,
             g_rv.has_rvv, g_rv.has_zvk);
}

void cross_cpu_arch_register_backends(cross_cpu_registry_t *reg) {
    run_riscv64_probe();

    cross_cpu_backend_t b;
    memset(&b, 0, sizeof(b));
    b.isa_id = CROSS_CPU_ISA_RVV;
    b.priority = 80u;
    b.compiled_in = cross_cpu_riscv_rvv_compiled;
    b.available = (cross_cpu_riscv_rvv_compiled && g_rv.has_rvv) ? 1u : 0u;
    snprintf(b.name, sizeof(b.name), "rvv");
    snprintf(b.detail, sizeof(b.detail),
             "RISC-V Vector 1.0 (rvv=%d zvk=%d compiled=%u)",
             g_rv.has_rvv, g_rv.has_zvk, cross_cpu_riscv_rvv_compiled);

    if (b.available) {
        memcpy(b.kernels, cross_cpu_riscv_rvv_kernels, sizeof(b.kernels));
    }

    (void)cross_cpu_registry_add(reg, &b);
}

void cross_cpu_arch_flush_cache_range(const void *ptr, size_t len) {
    cross_cpu_cache_flush_generic(ptr, len);
}
