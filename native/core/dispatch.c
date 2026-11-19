// SPDX-License-Identifier: MIT
#include "cross_cpu_arch.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;
static cross_cpu_caps_t g_caps;
static cross_cpu_registry_t g_registry;
static int g_initialized = 0;

static uint32_t detect_endianness(void) {
    const uint32_t v = 0x01020304u;
    const uint8_t *b = (const uint8_t *)&v;
    return (b[0] == 0x01u) ? 1u : 0u;
}

static const char *workload_name_by_id(uint32_t workload_id) {
    static const char *const names[CROSS_CPU_WL_COUNT] = {
        "compute-int64",
        "compute-int32",
        "compute-int16",
        "compute-int8",
        "compute-vnni",
        "compute-fp64",
        "compute-fp32",
        "compute-fp16",
        "compute-bf16",
        "compute-matrix-fp32",
        "compute-matrix-bf16",
        "compute-matrix-int8",
        "crypto-aes",
        "crypto-sha256",
        "cache-l1",
        "cache-l2",
        "cache-l3",
        "bandwidth-ram",
        "memory-pattern",
        "memory-sweep",
        "coherency-atomic"
    };
    if (workload_id < CROSS_CPU_WL_COUNT) {
        return names[workload_id];
    }
    return "unknown";
}

static const char *isa_name_by_id(uint32_t isa_id) {
    switch (isa_id) {
        case CROSS_CPU_ISA_AUTO:   return "auto";
        case CROSS_CPU_ISA_SCALAR: return "scalar";
        case CROSS_CPU_ISA_SSE4:   return "sse4";
        case CROSS_CPU_ISA_AVX2:   return "avx2";
        case CROSS_CPU_ISA_AVX512: return "avx512";
        case CROSS_CPU_ISA_AMX:    return "amx";
        case CROSS_CPU_ISA_NEON:   return "neon";
        case CROSS_CPU_ISA_SVE:    return "sve";
        case CROSS_CPU_ISA_SME:    return "sme";
        case CROSS_CPU_ISA_RVV:    return "rvv";
        case CROSS_CPU_ISA_VSX:    return "vsx";
        case CROSS_CPU_ISA_VX:     return "vx";
        default:                   return "unknown";
    }
}

static const char *default_unsupported_reason(uint32_t workload_id) {
    switch (workload_id) {
        case CROSS_CPU_WL_COMPUTE_VNNI:
            return "Requires native hardware integer dot-product ISA (x86 AVX-VNNI/AVX512-VNNI or ARM DotProd/I8MM); scalar emulation is disabled";
        case CROSS_CPU_WL_COMPUTE_FP16:
            return "Requires native hardware FP16 arithmetic ISA (ARM FP16/SVE or x86 AVX-512FP16); scalar upcasting is disabled";
        case CROSS_CPU_WL_COMPUTE_BF16:
            return "Requires native hardware BF16 arithmetic ISA (ARM BF16 or x86 AVX-512BF16); scalar upcasting is disabled";
        case CROSS_CPU_WL_COMPUTE_MATRIX_FP32:
            return "Requires native hardware FP32 matrix tile engine (ARM SME); scalar emulation is disabled";
        case CROSS_CPU_WL_COMPUTE_MATRIX_BF16:
            return "Requires native hardware BF16 matrix tile engine (Intel AMX-BF16 or ARM SME); scalar emulation is disabled";
        case CROSS_CPU_WL_COMPUTE_MATRIX_INT8:
            return "Requires native hardware INT8 matrix tile engine (Intel AMX-INT8 or ARM SME); scalar emulation is disabled";
        case CROSS_CPU_WL_CRYPTO_AES:
            return "Requires hardware AES ISA instructions (x86 AES-NI or ARM FEAT_AES); software AES fallback is disabled";
        case CROSS_CPU_WL_CRYPTO_SHA256:
            return "Requires hardware SHA-256 ISA instructions (x86 SHA-NI or ARM FEAT_SHA256); software SHA-256 fallback is disabled";
        default:
            return "No supported native ISA backend is available for this workload on the current host";
    }
}

int32_t cross_cpu_registry_add(cross_cpu_registry_t *reg,
                               const cross_cpu_backend_t *backend) {
    if (reg == NULL || backend == NULL) {
        return -1;
    }
    /* Update existing entry with same isa_id if already registered */
    for (uint32_t i = 0; i < reg->count; i++) {
        if (reg->backends[i].isa_id == backend->isa_id) {
            reg->backends[i] = *backend;
            return 0;
        }
    }
    if (reg->count >= CROSS_CPU_MAX_ISAS) {
        return -1;
    }
    reg->backends[reg->count] = *backend;
    reg->count++;
    return 0;
}

static void sort_registry_by_priority(cross_cpu_registry_t *reg) {
    for (uint32_t i = 1; i < reg->count; i++) {
        cross_cpu_backend_t key = reg->backends[i];
        uint32_t j = i;
        while (j > 0 && reg->backends[j - 1].priority < key.priority) {
            reg->backends[j] = reg->backends[j - 1];
            j--;
        }
        reg->backends[j] = key;
    }
}

static uint32_t compute_workload_mask(const cross_cpu_backend_t *backend) {
    uint32_t mask = 0;
    if (!backend->available) {
        return 0;
    }
    for (uint32_t w = 0; w < CROSS_CPU_WL_COUNT && w < 32u; w++) {
        if (backend->kernels[w] != NULL) {
            mask |= (1u << w);
        }
    }
    return mask;
}

static void do_init_once(void) {
    memset(&g_caps, 0, sizeof(g_caps));
    memset(&g_registry, 0, sizeof(g_registry));

    g_caps.abi_version = CROSS_CPU_ABI_VERSION;
    g_caps.endianness = detect_endianness();

    /* Always register portable C11 scalar backend first */
    cross_cpu_register_scalar_backend(&g_registry);

    /* Probe host architecture and register architecture-specific ISA backends */
    cross_cpu_arch_probe(&g_caps);
    cross_cpu_arch_register_backends(&g_registry);

    sort_registry_by_priority(&g_registry);

    g_caps.isa_count = g_registry.count;
    for (uint32_t i = 0; i < g_registry.count && i < CROSS_CPU_MAX_ISAS; i++) {
        const cross_cpu_backend_t *b = &g_registry.backends[i];
        cross_cpu_isa_info_t *info = &g_caps.isas[i];
        info->isa_id = b->isa_id;
        info->priority = b->priority;
        info->available = b->available;
        info->compiled_in = b->compiled_in;
        info->workload_mask = compute_workload_mask(b);
        info->reserved = 0;
        snprintf(info->name, sizeof(info->name), "%s", b->name);
        snprintf(info->detail, sizeof(info->detail), "%s", b->detail);
    }

    g_initialized = 1;
}

uint32_t cross_cpu_abi_version(void) {
    return CROSS_CPU_ABI_VERSION;
}

int32_t cross_cpu_init(void) {
    pthread_once(&g_init_once, do_init_once);
    return g_initialized ? CROSS_CPU_STATUS_OK : CROSS_CPU_STATUS_INIT_ERROR;
}

int32_t cross_cpu_get_caps(cross_cpu_caps_t *out_caps) {
    if (out_caps == NULL) {
        return CROSS_CPU_STATUS_INIT_ERROR;
    }
    (void)cross_cpu_init();
    *out_caps = g_caps;
    return CROSS_CPU_STATUS_OK;
}

static const cross_cpu_backend_t *find_backend_by_id(uint32_t isa_id) {
    for (uint32_t i = 0; i < g_registry.count; i++) {
        if (g_registry.backends[i].isa_id == isa_id) {
            return &g_registry.backends[i];
        }
    }
    return NULL;
}

static int32_t resolve_backend(uint32_t workload_id,
                               uint32_t requested_isa,
                               const cross_cpu_backend_t **out_backend,
                               char *out_reason,
                               size_t reason_len) {
    if (out_reason != NULL && reason_len > 0) {
        out_reason[0] = '\0';
    }
    if (out_backend != NULL) {
        *out_backend = NULL;
    }
    if (workload_id >= CROSS_CPU_WL_COUNT) {
        if (out_reason != NULL && reason_len > 0) {
            snprintf(out_reason, reason_len, "Unknown workload ID: %u", workload_id);
        }
        return CROSS_CPU_STATUS_INIT_ERROR;
    }

    if (requested_isa == CROSS_CPU_ISA_AUTO) {
        /* Registry is sorted by descending priority */
        for (uint32_t i = 0; i < g_registry.count; i++) {
            const cross_cpu_backend_t *b = &g_registry.backends[i];
            if (b->available && b->kernels[workload_id] != NULL) {
                if (out_backend != NULL) {
                    *out_backend = b;
                }
                return CROSS_CPU_STATUS_OK;
            }
        }
        /* Provide best diagnostic reason why no available ISA supports this workload */
        for (uint32_t i = 0; i < g_registry.count; i++) {
            const cross_cpu_backend_t *b = &g_registry.backends[i];
            if (b->unsupported_reasons[workload_id][0] != '\0') {
                if (out_reason != NULL && reason_len > 0) {
                    snprintf(out_reason, reason_len, "%s", b->unsupported_reasons[workload_id]);
                }
                return CROSS_CPU_STATUS_UNSUPPORTED;
            }
        }
        if (out_reason != NULL && reason_len > 0) {
            snprintf(out_reason, reason_len, "%s", default_unsupported_reason(workload_id));
        }
        return CROSS_CPU_STATUS_UNSUPPORTED;
    }

    const cross_cpu_backend_t *forced = find_backend_by_id(requested_isa);
    if (forced == NULL) {
        if (out_reason != NULL && reason_len > 0) {
            snprintf(out_reason, reason_len,
                     "Requested ISA '%s' is not available on this host architecture (%s)",
                     isa_name_by_id(requested_isa), g_caps.arch_name);
        }
        return CROSS_CPU_STATUS_INIT_ERROR;
    }
    if (!forced->available) {
        if (out_reason != NULL && reason_len > 0) {
            if (forced->detail[0] != '\0') {
                snprintf(out_reason, reason_len,
                         "Requested ISA '%s' is unavailable on this host: %s",
                         forced->name, forced->detail);
            } else {
                snprintf(out_reason, reason_len,
                         "Requested ISA '%s' is not supported by this CPU or OS",
                         forced->name);
            }
        }
        return CROSS_CPU_STATUS_INIT_ERROR;
    }
    if (forced->kernels[workload_id] == NULL) {
        if (out_reason != NULL && reason_len > 0) {
            if (forced->unsupported_reasons[workload_id][0] != '\0') {
                snprintf(out_reason, reason_len, "%s", forced->unsupported_reasons[workload_id]);
            } else {
                snprintf(out_reason, reason_len,
                         "Workload '%s' is not natively supported by ISA '%s' (no silent fallback)",
                         workload_name_by_id(workload_id), forced->name);
            }
        }
        return CROSS_CPU_STATUS_UNSUPPORTED;
    }

    if (out_backend != NULL) {
        *out_backend = forced;
    }
    return CROSS_CPU_STATUS_OK;
}

int32_t cross_cpu_resolve_isa(uint32_t workload_id,
                              uint32_t requested_isa,
                              uint32_t *out_isa_id,
                              char *out_reason,
                              size_t reason_len) {
    (void)cross_cpu_init();
    const cross_cpu_backend_t *backend = NULL;
    int32_t rc = resolve_backend(workload_id, requested_isa, &backend, out_reason, reason_len);
    if (out_isa_id != NULL) {
        *out_isa_id = (backend != NULL) ? backend->isa_id : CROSS_CPU_ISA_AUTO;
    }
    return rc;
}

int32_t cross_cpu_run_workload(const cross_cpu_run_config_t *cfg,
                               cross_cpu_run_result_t *out_result) {
    if (cfg == NULL || out_result == NULL) {
        return CROSS_CPU_STATUS_INIT_ERROR;
    }

    memset(out_result, 0, sizeof(*out_result));
    out_result->abi_version = CROSS_CPU_ABI_VERSION;
    out_result->failed_thread = UINT32_MAX;
    out_result->failed_core = UINT32_MAX;

    if (cfg->abi_version != CROSS_CPU_ABI_VERSION) {
        out_result->status = CROSS_CPU_STATUS_INIT_ERROR;
        snprintf(out_result->reason, sizeof(out_result->reason),
                 "ABI version mismatch: caller=%u library=%u",
                 cfg->abi_version, CROSS_CPU_ABI_VERSION);
        return CROSS_CPU_STATUS_INIT_ERROR;
    }

    (void)cross_cpu_init();

    const cross_cpu_backend_t *backend = NULL;
    int32_t rc = resolve_backend(cfg->workload_id,
                                 cfg->requested_isa,
                                 &backend,
                                 out_result->reason,
                                 sizeof(out_result->reason));
    if (rc != CROSS_CPU_STATUS_OK || backend == NULL) {
        out_result->status = rc;
        out_result->selected_isa = CROSS_CPU_ISA_AUTO;
        snprintf(out_result->isa_name, sizeof(out_result->isa_name), "none");
        return rc;
    }

    return cross_cpu_pool_execute(cfg, backend, backend->kernels[cfg->workload_id], out_result);
}
