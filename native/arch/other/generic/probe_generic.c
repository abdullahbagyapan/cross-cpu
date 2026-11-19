// SPDX-License-Identifier: MIT
#include "cross_cpu_arch.h"

#include <stdio.h>
#include <string.h>

void cross_cpu_arch_probe(cross_cpu_caps_t *caps) {
    if (caps == NULL) {
        return;
    }

    caps->arch_id = CROSS_CPU_ARCH_GENERIC;
    snprintf(caps->arch_name, sizeof(caps->arch_name), "generic");
    caps->raw_probe[0] = 0ULL;
    caps->raw_probe[1] = 0ULL;
    caps->raw_probe[2] = 0ULL;
    caps->raw_probe[3] = 0ULL;
    caps->feature_flags = 0ULL;

    snprintf(caps->probe_summary, sizeof(caps->probe_summary),
             "Portable POSIX generic backend (scalar C11 only, endian=%s)",
             caps->endianness ? "big" : "little");
}

void cross_cpu_arch_register_backends(cross_cpu_registry_t *reg) {
    /* Scalar backend is already registered by cross_cpu_register_scalar_backend() */
    (void)reg;
}

void cross_cpu_arch_flush_cache_range(const void *ptr, size_t len) {
    cross_cpu_cache_flush_generic(ptr, len);
}
