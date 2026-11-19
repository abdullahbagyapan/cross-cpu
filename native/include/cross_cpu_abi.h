// SPDX-License-Identifier: MIT
#ifndef CROSS_CPU_ABI_H
#define CROSS_CPU_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CROSS_CPU_ABI_VERSION 1u
#define CROSS_CPU_MAX_ISAS    16u
#define CROSS_CPU_MAX_NAME    32u
#define CROSS_CPU_MAX_REASON  256u
#define CROSS_CPU_MAX_CORES   1024u

/* Architecture family identifiers */
typedef enum cross_cpu_arch_id {
    CROSS_CPU_ARCH_GENERIC = 0,
    CROSS_CPU_ARCH_X86_64  = 1,
    CROSS_CPU_ARCH_ARM64   = 2,
    CROSS_CPU_ARCH_RISCV64 = 3,
    CROSS_CPU_ARCH_PPC64   = 4,
    CROSS_CPU_ARCH_S390X   = 5
} cross_cpu_arch_id_t;

/* ISA backend identifiers */
typedef enum cross_cpu_isa_id {
    CROSS_CPU_ISA_AUTO   = 0,
    CROSS_CPU_ISA_SCALAR = 1,
    CROSS_CPU_ISA_SSE4   = 2,
    CROSS_CPU_ISA_AVX2   = 3,
    CROSS_CPU_ISA_AVX512 = 4,
    CROSS_CPU_ISA_AMX    = 5,
    CROSS_CPU_ISA_NEON   = 6,
    CROSS_CPU_ISA_SVE    = 7,
    CROSS_CPU_ISA_SME    = 8,
    CROSS_CPU_ISA_RVV    = 9,
    CROSS_CPU_ISA_VSX    = 10,
    CROSS_CPU_ISA_VX     = 11
} cross_cpu_isa_id_t;

/* Canonical workload identifiers */
typedef enum cross_cpu_workload_id {
    CROSS_CPU_WL_COMPUTE_INT64       = 0,
    CROSS_CPU_WL_COMPUTE_INT32       = 1,
    CROSS_CPU_WL_COMPUTE_INT16       = 2,
    CROSS_CPU_WL_COMPUTE_INT8        = 3,
    CROSS_CPU_WL_COMPUTE_VNNI        = 4,
    CROSS_CPU_WL_COMPUTE_FP64        = 5,
    CROSS_CPU_WL_COMPUTE_FP32        = 6,
    CROSS_CPU_WL_COMPUTE_FP16        = 7,
    CROSS_CPU_WL_COMPUTE_BF16        = 8,
    CROSS_CPU_WL_COMPUTE_MATRIX_FP32 = 9,
    CROSS_CPU_WL_COMPUTE_MATRIX_BF16 = 10,
    CROSS_CPU_WL_COMPUTE_MATRIX_INT8 = 11,
    CROSS_CPU_WL_CRYPTO_AES          = 12,
    CROSS_CPU_WL_CRYPTO_SHA256       = 13,
    CROSS_CPU_WL_CACHE_L1            = 14,
    CROSS_CPU_WL_CACHE_L2            = 15,
    CROSS_CPU_WL_CACHE_L3            = 16,
    CROSS_CPU_WL_BANDWIDTH_RAM       = 17,
    CROSS_CPU_WL_MEMORY_PATTERN      = 18,
    CROSS_CPU_WL_MEMORY_SWEEP        = 19,
    CROSS_CPU_WL_COHERENCY_ATOMIC    = 20,
    CROSS_CPU_WL_COUNT               = 21
} cross_cpu_workload_id_t;

/* Execution status codes returned by native functions */
typedef enum cross_cpu_status {
    CROSS_CPU_STATUS_OK            = 0,
    CROSS_CPU_STATUS_UNSUPPORTED   = 1,
    CROSS_CPU_STATUS_INIT_ERROR    = 2,
    CROSS_CPU_STATUS_COMPUTE_ERROR = 3,
    CROSS_CPU_STATUS_STOPPED       = 4
} cross_cpu_status_t;

/* macOS QoS scheduling hints (ignored on Linux where affinity pinning is used) */
typedef enum cross_cpu_qos_hint {
    CROSS_CPU_QOS_DEFAULT     = 0,
    CROSS_CPU_QOS_INTERACTIVE = 1,
    CROSS_CPU_QOS_BACKGROUND  = 2
} cross_cpu_qos_hint_t;

/* Per-ISA capability descriptor (fixed alignment, 312 bytes) */
typedef struct cross_cpu_isa_info {
    uint32_t isa_id;
    uint32_t priority;
    uint32_t available;
    uint32_t compiled_in;
    uint32_t workload_mask;
    uint32_t reserved;
    char     name[CROSS_CPU_MAX_NAME];
    char     detail[CROSS_CPU_MAX_REASON];
} cross_cpu_isa_info_t;

/* Aggregated host capability structure */
typedef struct cross_cpu_caps {
    uint32_t abi_version;
    uint32_t arch_id;
    uint32_t endianness; /* 0 = little-endian, 1 = big-endian */
    uint32_t isa_count;
    uint64_t raw_probe[4];
    uint64_t feature_flags;
    char     arch_name[CROSS_CPU_MAX_NAME];
    char     probe_summary[CROSS_CPU_MAX_REASON];
    cross_cpu_isa_info_t isas[CROSS_CPU_MAX_ISAS];
} cross_cpu_caps_t;

/* Configuration for a workload execution */
typedef struct cross_cpu_run_config {
    uint32_t abi_version;
    uint32_t workload_id;
    uint32_t requested_isa;
    uint32_t num_threads;
    uint64_t duration_ns;
    uint64_t batch_ns;
    uint64_t buffer_bytes;
    uint64_t seed;
    uint32_t qos_class;
    uint32_t inject_fault;
    int32_t  core_ids[CROSS_CPU_MAX_CORES];
} cross_cpu_run_config_t;

/* Result returned by cross_cpu_run_workload() */
typedef struct cross_cpu_run_result {
    uint32_t abi_version;
    int32_t  status;
    uint32_t selected_isa;
    uint32_t threads_executed;
    uint64_t total_ops;
    uint64_t elapsed_ns;
    uint64_t error_count;
    uint64_t iterations;
    double   expected_sample;
    double   observed_sample;
    uint32_t failed_thread;
    uint32_t failed_core;
    char     isa_name[CROSS_CPU_MAX_NAME];
    char     reason[CROSS_CPU_MAX_REASON];
} cross_cpu_run_result_t;

/* Public exported ABI symbols */
uint32_t cross_cpu_abi_version(void);
int32_t  cross_cpu_init(void);
int32_t  cross_cpu_get_caps(cross_cpu_caps_t *out_caps);
int32_t  cross_cpu_resolve_isa(uint32_t workload_id,
                               uint32_t requested_isa,
                               uint32_t *out_isa_id,
                               char *out_reason,
                               size_t reason_len);
int32_t  cross_cpu_run_workload(const cross_cpu_run_config_t *cfg,
                                cross_cpu_run_result_t *out_result);
void     cross_cpu_request_stop(void);
void     cross_cpu_reset_stop(void);
uint64_t cross_cpu_get_progress_ops(void);

#ifdef __cplusplus
}
#endif

#endif /* CROSS_CPU_ABI_H */
