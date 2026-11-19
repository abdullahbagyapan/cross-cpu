// SPDX-License-Identifier: MIT
#ifndef CROSS_CPU_ARCH_H
#define CROSS_CPU_ARCH_H

#include "cross_cpu_abi.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CROSS_CPU_COHERENCY_SLOTS        8u
#define CROSS_CPU_COHERENCY_SLOT_STRIDE  128u

/* Per-worker execution context passed into each kernel batch */
typedef struct cross_cpu_worker_ctx {
    uint32_t thread_idx;
    uint32_t num_threads;
    int32_t  core_id;
    uint32_t workload_id;
    uint32_t isa_id;
    uint32_t inject_fault;
    uint64_t seed;
    uint64_t buffer_bytes;
    void    *thread_buf;
    size_t   thread_buf_bytes;
    void    *shared_buf;
    size_t   shared_buf_bytes;
    uint64_t batch_iters;
    uint64_t ops_done;
    uint64_t coherency_adds;
    uint64_t errors;
    double   expected_sample;
    double   observed_sample;
    char     error_detail[CROSS_CPU_MAX_REASON];
} cross_cpu_worker_ctx_t;

/* Kernel batch function signature */
typedef int32_t (*cross_cpu_kernel_fn)(cross_cpu_worker_ctx_t *ctx);

/* Registered ISA backend descriptor */
typedef struct cross_cpu_backend {
    uint32_t isa_id;
    uint32_t priority;
    uint32_t available;
    uint32_t compiled_in;
    char     name[CROSS_CPU_MAX_NAME];
    char     detail[CROSS_CPU_MAX_REASON];
    cross_cpu_kernel_fn kernels[CROSS_CPU_WL_COUNT];
    char     unsupported_reasons[CROSS_CPU_WL_COUNT][CROSS_CPU_MAX_REASON];
} cross_cpu_backend_t;

/* Registry populated during cross_cpu_init() */
typedef struct cross_cpu_registry {
    uint32_t count;
    cross_cpu_backend_t backends[CROSS_CPU_MAX_ISAS];
} cross_cpu_registry_t;

/* Architecture hook interface implemented by native/arch/<target>/ */
void cross_cpu_arch_probe(cross_cpu_caps_t *caps);
void cross_cpu_arch_register_backends(cross_cpu_registry_t *reg);
void cross_cpu_arch_flush_cache_range(const void *ptr, size_t len);

/* Common baseline helpers implemented in native/arch/common/ */
void cross_cpu_register_scalar_backend(cross_cpu_registry_t *reg);
void cross_cpu_cache_flush_generic(const void *ptr, size_t len);

/* Core registry & worker-pool internal helpers */
int32_t cross_cpu_registry_add(cross_cpu_registry_t *reg,
                               const cross_cpu_backend_t *backend);
int32_t cross_cpu_pool_execute(const cross_cpu_run_config_t *cfg,
                               const cross_cpu_backend_t *backend,
                               cross_cpu_kernel_fn kernel,
                               cross_cpu_run_result_t *out_result);

#ifdef __cplusplus
}
#endif

#endif /* CROSS_CPU_ARCH_H */
