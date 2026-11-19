// SPDX-License-Identifier: MIT
#if defined(__linux__)
#  ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#  endif
#elif defined(__APPLE__)
#  ifndef _DARWIN_C_SOURCE
#    define _DARWIN_C_SOURCE
#  endif
#else
#  ifndef _POSIX_C_SOURCE
#    define _POSIX_C_SOURCE 200809L
#  endif
#endif

#include "cross_cpu_arch.h"

#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__linux__)
#  include <sched.h>
#elif defined(__APPLE__)
#  include <pthread/qos.h>
#endif

static _Atomic int g_stop_flag = 0;
static _Atomic uint64_t g_progress_ops = 0;

void cross_cpu_request_stop(void) {
    atomic_store_explicit(&g_stop_flag, 1, memory_order_release);
}

void cross_cpu_reset_stop(void) {
    atomic_store_explicit(&g_stop_flag, 0, memory_order_release);
    atomic_store_explicit(&g_progress_ops, 0, memory_order_relaxed);
}

uint64_t cross_cpu_get_progress_ops(void) {
    return atomic_load_explicit(&g_progress_ops, memory_order_relaxed);
}

static uint64_t monotonic_now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
}

/* Portable POSIX barrier (macOS does not provide pthread_barrier_t) */
typedef struct cross_cpu_barrier {
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
    uint32_t        count;
    uint32_t        waiting;
    uint32_t        generation;
} cross_cpu_barrier_t;

static int barrier_init(cross_cpu_barrier_t *b, uint32_t count) {
    if (count == 0) {
        return -1;
    }
    if (pthread_mutex_init(&b->mutex, NULL) != 0) {
        return -1;
    }
    if (pthread_cond_init(&b->cond, NULL) != 0) {
        pthread_mutex_destroy(&b->mutex);
        return -1;
    }
    b->count = count;
    b->waiting = 0;
    b->generation = 0;
    return 0;
}

static void barrier_destroy(cross_cpu_barrier_t *b) {
    pthread_cond_destroy(&b->cond);
    pthread_mutex_destroy(&b->mutex);
}

static void barrier_wait(cross_cpu_barrier_t *b) {
    pthread_mutex_lock(&b->mutex);
    uint32_t gen = b->generation;
    b->waiting++;
    if (b->waiting == b->count) {
        b->generation++;
        b->waiting = 0;
        pthread_cond_broadcast(&b->cond);
    } else {
        while (gen == b->generation) {
            pthread_cond_wait(&b->cond, &b->mutex);
        }
    }
    pthread_mutex_unlock(&b->mutex);
}

static int workload_needs_thread_buffer(uint32_t workload_id) {
    switch (workload_id) {
        case CROSS_CPU_WL_CACHE_L1:
        case CROSS_CPU_WL_CACHE_L2:
        case CROSS_CPU_WL_CACHE_L3:
        case CROSS_CPU_WL_BANDWIDTH_RAM:
        case CROSS_CPU_WL_MEMORY_PATTERN:
        case CROSS_CPU_WL_MEMORY_SWEEP:
            return 1;
        default:
            return 0;
    }
}

static uint64_t default_initial_iters(uint32_t workload_id) {
    switch (workload_id) {
        case CROSS_CPU_WL_CACHE_L1:
            return 64ULL;
        case CROSS_CPU_WL_CACHE_L2:
            return 8ULL;
        case CROSS_CPU_WL_CACHE_L3:
        case CROSS_CPU_WL_BANDWIDTH_RAM:
        case CROSS_CPU_WL_MEMORY_PATTERN:
        case CROSS_CPU_WL_MEMORY_SWEEP:
            return 1ULL;
        case CROSS_CPU_WL_COHERENCY_ATOMIC:
            return 4096ULL;
        case CROSS_CPU_WL_CRYPTO_AES:
        case CROSS_CPU_WL_CRYPTO_SHA256:
            return 256ULL;
        case CROSS_CPU_WL_COMPUTE_MATRIX_FP32:
        case CROSS_CPU_WL_COMPUTE_MATRIX_BF16:
        case CROSS_CPU_WL_COMPUTE_MATRIX_INT8:
            return 128ULL;
        default:
            return 256ULL;
    }
}

static uint64_t max_iters_for_workload(uint32_t workload_id) {
    switch (workload_id) {
        case CROSS_CPU_WL_CACHE_L2:
        case CROSS_CPU_WL_CACHE_L3:
        case CROSS_CPU_WL_BANDWIDTH_RAM:
        case CROSS_CPU_WL_MEMORY_PATTERN:
        case CROSS_CPU_WL_MEMORY_SWEEP:
            return 64ULL;
        case CROSS_CPU_WL_CACHE_L1:
            return 4096ULL;
        default:
            return 1000000ULL;
    }
}

typedef struct worker_thread_state {
    const cross_cpu_run_config_t *cfg;
    cross_cpu_kernel_fn           kernel;
    cross_cpu_barrier_t          *start_barrier;
    _Atomic int                  *shared_init_error;
    _Atomic uint64_t             *shared_start_ns;
    void                         *shared_buf;
    size_t                        shared_buf_bytes;
    uint32_t                      selected_isa;
    uint32_t                      thread_idx;
    int32_t                       core_id;

    /* Per-thread outputs */
    int32_t  status;
    uint64_t total_ops;
    uint64_t total_coherency_adds;
    uint64_t total_errors;
    uint64_t iterations;
    double   expected_sample;
    double   observed_sample;
    char     error_detail[CROSS_CPU_MAX_REASON];
} worker_thread_state_t;

static void apply_thread_placement(int32_t core_id, uint32_t qos_class) {
#if defined(__linux__)
    (void)qos_class;
    if (core_id >= 0 && core_id < CPU_SETSIZE) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        (void)sched_setaffinity(0, sizeof(cpuset), &cpuset);
    }
#elif defined(__APPLE__)
    (void)core_id;
    qos_class_t qos = QOS_CLASS_USER_INITIATED;
    if (qos_class == CROSS_CPU_QOS_INTERACTIVE) {
        qos = QOS_CLASS_USER_INTERACTIVE;
    } else if (qos_class == CROSS_CPU_QOS_BACKGROUND) {
        qos = QOS_CLASS_BACKGROUND;
    }
    (void)pthread_set_qos_class_self_np(qos, 0);
#else
    (void)core_id;
    (void)qos_class;
#endif
}

static size_t align_up_128(size_t n) {
    size_t rem = n % 128u;
    if (rem == 0u) {
        return n;
    }
    return n + (128u - rem);
}

static void *worker_thread_main(void *arg) {
    worker_thread_state_t *st = (worker_thread_state_t *)arg;
    const cross_cpu_run_config_t *cfg = st->cfg;

    /* Pin / set QoS BEFORE allocating memory for Linux NUMA first-touch locality */
    apply_thread_placement(st->core_id, cfg->qos_class);

    void *thread_buf = NULL;
    size_t thread_buf_bytes = 0;

    if (workload_needs_thread_buffer(cfg->workload_id)) {
        size_t req = (size_t)cfg->buffer_bytes;
        if (req < 4096u) {
            req = 4096u;
        }
        req = align_up_128(req);
        if (posix_memalign(&thread_buf, 128u, req) != 0 || thread_buf == NULL) {
            st->status = CROSS_CPU_STATUS_INIT_ERROR;
            snprintf(st->error_detail, sizeof(st->error_detail),
                     "Failed to allocate %zu bytes aligned working buffer on worker %u",
                     req, st->thread_idx);
            atomic_store_explicit(st->shared_init_error, 1, memory_order_release);
        } else {
            thread_buf_bytes = req;
            /* First-touch pages on the pinned core's NUMA node */
            memset(thread_buf, 0, thread_buf_bytes);
        }
    }

    barrier_wait(st->start_barrier);

    if (atomic_load_explicit(st->shared_init_error, memory_order_acquire) != 0) {
        free(thread_buf);
        return NULL;
    }

    if (st->thread_idx == 0) {
        atomic_store_explicit(st->shared_start_ns, monotonic_now_ns(), memory_order_release);
    }
    barrier_wait(st->start_barrier);

    const uint64_t start_ns = atomic_load_explicit(st->shared_start_ns, memory_order_acquire);
    const uint64_t duration_ns = cfg->duration_ns;
    const uint64_t deadline_ns = (duration_ns > 0ULL) ? (start_ns + duration_ns) : start_ns;

    uint64_t target_slice_ns = (cfg->batch_ns > 0ULL) ? cfg->batch_ns : 100000000ULL;
    if (duration_ns > 0ULL) {
        uint64_t quarter = duration_ns / 4ULL;
        if (quarter < 5000000ULL) {
            quarter = 5000000ULL;
        }
        if (target_slice_ns > quarter) {
            target_slice_ns = quarter;
        }
    }

    uint64_t batch_iters = default_initial_iters(cfg->workload_id);
    const uint64_t max_iters = max_iters_for_workload(cfg->workload_id);

    cross_cpu_worker_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.thread_idx = st->thread_idx;
    ctx.num_threads = cfg->num_threads;
    ctx.core_id = st->core_id;
    ctx.workload_id = cfg->workload_id;
    ctx.isa_id = st->selected_isa;
    ctx.inject_fault = cfg->inject_fault;
    ctx.seed = cfg->seed ^ ((uint64_t)(st->thread_idx + 1u) * 0x9e3779b97f4a7c15ULL);
    ctx.buffer_bytes = (uint64_t)thread_buf_bytes;
    ctx.thread_buf = thread_buf;
    ctx.thread_buf_bytes = thread_buf_bytes;
    ctx.shared_buf = st->shared_buf;
    ctx.shared_buf_bytes = st->shared_buf_bytes;

    st->status = CROSS_CPU_STATUS_OK;

    uint64_t batch_num = 0;
    for (;;) {
        if (batch_num > 0 && atomic_load_explicit(&g_stop_flag, memory_order_acquire) != 0) {
            break;
        }

        ctx.batch_iters = batch_iters;
        ctx.ops_done = 0;
        ctx.coherency_adds = 0;
        ctx.errors = 0;
        ctx.expected_sample = 0.0;
        ctx.observed_sample = 0.0;
        ctx.error_detail[0] = '\0';
        ctx.seed += 0x9e3779b97f4a7c15ULL;

        const uint64_t t0 = monotonic_now_ns();
        const int32_t rc = st->kernel(&ctx);
        const uint64_t t1 = monotonic_now_ns();
        const uint64_t dt_ns = (t1 > t0) ? (t1 - t0) : 1ULL;

        st->total_ops += ctx.ops_done;
        st->total_coherency_adds += ctx.coherency_adds;
        st->total_errors += ctx.errors;
        st->iterations++;
        batch_num++;

        if (ctx.ops_done > 0ULL) {
            atomic_fetch_add_explicit(&g_progress_ops, ctx.ops_done, memory_order_relaxed);
        }

        if (rc == CROSS_CPU_STATUS_COMPUTE_ERROR || ctx.errors > 0ULL) {
            st->status = CROSS_CPU_STATUS_COMPUTE_ERROR;
            if (st->total_errors == 0ULL) {
                st->total_errors = 1ULL;
            }
            st->expected_sample = ctx.expected_sample;
            st->observed_sample = ctx.observed_sample;
            if (ctx.error_detail[0] != '\0') {
                snprintf(st->error_detail, sizeof(st->error_detail), "%s", ctx.error_detail);
            } else {
                snprintf(st->error_detail, sizeof(st->error_detail),
                         "Verification mismatch on worker %u", st->thread_idx);
            }
            break;
        }

        if (rc == CROSS_CPU_STATUS_UNSUPPORTED || rc == CROSS_CPU_STATUS_INIT_ERROR) {
            st->status = rc;
            if (ctx.error_detail[0] != '\0') {
                snprintf(st->error_detail, sizeof(st->error_detail), "%s", ctx.error_detail);
            }
            break;
        }

        if (duration_ns == 0ULL) {
            break;
        }
        if (atomic_load_explicit(&g_stop_flag, memory_order_acquire) != 0) {
            break;
        }
        if (t1 >= deadline_ns) {
            break;
        }

        /* Dynamically tune batch_iters toward target_slice_ns without overshooting remaining_ns */
        const uint64_t remaining_ns = deadline_ns - t1;
        const uint64_t desired_ns = (remaining_ns < target_slice_ns) ? remaining_ns : target_slice_ns;
        double ratio = (double)desired_ns / (double)dt_ns;
        if (ratio > 4.0) {
            ratio = 4.0;
        } else if (ratio < 0.25) {
            ratio = 0.25;
        }
        uint64_t next_iters = (uint64_t)((double)batch_iters * ratio);
        if (next_iters < 1ULL) {
            next_iters = 1ULL;
        } else if (next_iters > max_iters) {
            next_iters = max_iters;
        }
        batch_iters = next_iters;
    }

    free(thread_buf);
    return NULL;
}

int32_t cross_cpu_pool_execute(const cross_cpu_run_config_t *cfg,
                               const cross_cpu_backend_t *backend,
                               cross_cpu_kernel_fn kernel,
                               cross_cpu_run_result_t *out_result) {
    if (cfg == NULL || backend == NULL || kernel == NULL || out_result == NULL) {
        return CROSS_CPU_STATUS_INIT_ERROR;
    }

    uint32_t num_threads = cfg->num_threads;
    if (num_threads == 0u || num_threads > CROSS_CPU_MAX_CORES) {
        out_result->status = CROSS_CPU_STATUS_INIT_ERROR;
        snprintf(out_result->reason, sizeof(out_result->reason),
                 "Invalid thread count: %u (must be 1..%u)",
                 num_threads, CROSS_CPU_MAX_CORES);
        return CROSS_CPU_STATUS_INIT_ERROR;
    }

    void *shared_buf = NULL;
    size_t shared_buf_bytes = 0u;
    if (cfg->workload_id == CROSS_CPU_WL_COHERENCY_ATOMIC) {
        shared_buf_bytes = (size_t)CROSS_CPU_COHERENCY_SLOTS * CROSS_CPU_COHERENCY_SLOT_STRIDE;
        if (posix_memalign(&shared_buf, 128u, shared_buf_bytes) != 0 || shared_buf == NULL) {
            out_result->status = CROSS_CPU_STATUS_INIT_ERROR;
            snprintf(out_result->reason, sizeof(out_result->reason),
                     "Failed to allocate shared coherency buffer");
            return CROSS_CPU_STATUS_INIT_ERROR;
        }
        memset(shared_buf, 0, shared_buf_bytes);
        for (uint32_t s = 0; s < CROSS_CPU_COHERENCY_SLOTS; s++) {
            _Atomic uint64_t *slot =
                (_Atomic uint64_t *)((uint8_t *)shared_buf + ((size_t)s * CROSS_CPU_COHERENCY_SLOT_STRIDE));
            atomic_init(slot, 0ULL);
        }
    }

    pthread_t *threads = (pthread_t *)calloc(num_threads, sizeof(pthread_t));
    worker_thread_state_t *states =
        (worker_thread_state_t *)calloc(num_threads, sizeof(worker_thread_state_t));
    if (threads == NULL || states == NULL) {
        free(threads);
        free(states);
        free(shared_buf);
        out_result->status = CROSS_CPU_STATUS_INIT_ERROR;
        snprintf(out_result->reason, sizeof(out_result->reason),
                 "Failed to allocate worker pool metadata");
        return CROSS_CPU_STATUS_INIT_ERROR;
    }

    cross_cpu_barrier_t start_barrier;
    if (barrier_init(&start_barrier, num_threads) != 0) {
        free(threads);
        free(states);
        free(shared_buf);
        out_result->status = CROSS_CPU_STATUS_INIT_ERROR;
        snprintf(out_result->reason, sizeof(out_result->reason),
                 "Failed to initialize worker synchronization barrier");
        return CROSS_CPU_STATUS_INIT_ERROR;
    }

    _Atomic int shared_init_error = 0;
    _Atomic uint64_t shared_start_ns = 0ULL;
    atomic_store_explicit(&g_progress_ops, 0ULL, memory_order_relaxed);

    uint32_t spawned = 0;
    int spawn_failed = 0;

    for (uint32_t i = 0; i < num_threads; i++) {
        states[i].cfg = cfg;
        states[i].kernel = kernel;
        states[i].start_barrier = &start_barrier;
        states[i].shared_init_error = &shared_init_error;
        states[i].shared_start_ns = &shared_start_ns;
        states[i].shared_buf = shared_buf;
        states[i].shared_buf_bytes = shared_buf_bytes;
        states[i].selected_isa = backend->isa_id;
        states[i].thread_idx = i;
        states[i].core_id = cfg->core_ids[i];
        states[i].status = CROSS_CPU_STATUS_OK;

        if (pthread_create(&threads[i], NULL, worker_thread_main, &states[i]) != 0) {
            spawn_failed = 1;
            atomic_store_explicit(&shared_init_error, 1, memory_order_release);
            /* Unblock already-spawned threads waiting on the barrier */
            pthread_mutex_lock(&start_barrier.mutex);
            start_barrier.count = spawned;
            if (start_barrier.waiting >= start_barrier.count) {
                start_barrier.generation++;
                start_barrier.waiting = 0;
                pthread_cond_broadcast(&start_barrier.cond);
            }
            pthread_mutex_unlock(&start_barrier.mutex);
            break;
        }
        spawned++;
    }

    for (uint32_t i = 0; i < spawned; i++) {
        (void)pthread_join(threads[i], NULL);
    }

    const uint64_t end_ns = monotonic_now_ns();
    const uint64_t start_ns = atomic_load_explicit(&shared_start_ns, memory_order_acquire);
    const uint64_t elapsed_ns = (start_ns > 0ULL && end_ns > start_ns) ? (end_ns - start_ns) : 1ULL;

    barrier_destroy(&start_barrier);

    uint64_t total_ops = 0;
    uint64_t total_coherency_adds = 0;
    uint64_t total_errors = 0;
    uint64_t total_iters = 0;
    int32_t final_status = CROSS_CPU_STATUS_OK;
    uint32_t failed_thread = UINT32_MAX;
    uint32_t failed_core = UINT32_MAX;
    double expected_sample = 0.0;
    double observed_sample = 0.0;
    char reason[CROSS_CPU_MAX_REASON];
    reason[0] = '\0';

    for (uint32_t i = 0; i < spawned; i++) {
        total_ops += states[i].total_ops;
        total_coherency_adds += states[i].total_coherency_adds;
        total_errors += states[i].total_errors;
        total_iters += states[i].iterations;

        if (states[i].status == CROSS_CPU_STATUS_COMPUTE_ERROR &&
            final_status != CROSS_CPU_STATUS_COMPUTE_ERROR) {
            final_status = CROSS_CPU_STATUS_COMPUTE_ERROR;
            failed_thread = i;
            failed_core = (states[i].core_id >= 0) ? (uint32_t)states[i].core_id : UINT32_MAX;
            expected_sample = states[i].expected_sample;
            observed_sample = states[i].observed_sample;
            snprintf(reason, sizeof(reason), "%s", states[i].error_detail);
        } else if (states[i].status != CROSS_CPU_STATUS_OK &&
                   final_status == CROSS_CPU_STATUS_OK) {
            final_status = states[i].status;
            failed_thread = i;
            failed_core = (states[i].core_id >= 0) ? (uint32_t)states[i].core_id : UINT32_MAX;
            snprintf(reason, sizeof(reason), "%s", states[i].error_detail);
        }
    }

    if (spawn_failed) {
        final_status = CROSS_CPU_STATUS_INIT_ERROR;
        snprintf(reason, sizeof(reason),
                 "Failed to spawn worker thread %u of %u", spawned, num_threads);
    }

    /* Post-join global coherency verification for coherency-atomic */
    if (cfg->workload_id == CROSS_CPU_WL_COHERENCY_ATOMIC &&
        shared_buf != NULL &&
        final_status == CROSS_CPU_STATUS_OK) {
        uint64_t observed_total = 0;
        for (uint32_t s = 0; s < CROSS_CPU_COHERENCY_SLOTS; s++) {
            _Atomic uint64_t *slot =
                (_Atomic uint64_t *)((uint8_t *)shared_buf + ((size_t)s * CROSS_CPU_COHERENCY_SLOT_STRIDE));
            observed_total += atomic_load_explicit(slot, memory_order_seq_cst);
        }
        if (observed_total != total_coherency_adds) {
            final_status = CROSS_CPU_STATUS_COMPUTE_ERROR;
            total_errors++;
            failed_thread = 0;
            failed_core = (cfg->core_ids[0] >= 0) ? (uint32_t)cfg->core_ids[0] : UINT32_MAX;
            expected_sample = (double)total_coherency_adds;
            observed_sample = (double)observed_total;
            snprintf(reason, sizeof(reason),
                     "Atomic coherency mismatch: expected %llu total increments, observed %llu",
                     (unsigned long long)total_coherency_adds,
                     (unsigned long long)observed_total);
        }
    }

    if (final_status == CROSS_CPU_STATUS_OK &&
        atomic_load_explicit(&g_stop_flag, memory_order_acquire) != 0) {
        final_status = CROSS_CPU_STATUS_STOPPED;
        snprintf(reason, sizeof(reason), "Stopped by signal before full duration elapsed");
    }

    out_result->abi_version = CROSS_CPU_ABI_VERSION;
    out_result->status = final_status;
    out_result->selected_isa = backend->isa_id;
    out_result->threads_executed = spawned;
    out_result->total_ops = total_ops;
    out_result->elapsed_ns = elapsed_ns;
    out_result->error_count = total_errors;
    out_result->iterations = total_iters;
    out_result->expected_sample = expected_sample;
    out_result->observed_sample = observed_sample;
    out_result->failed_thread = failed_thread;
    out_result->failed_core = failed_core;
    snprintf(out_result->isa_name, sizeof(out_result->isa_name), "%s", backend->name);
    snprintf(out_result->reason, sizeof(out_result->reason), "%s", reason);

    free(threads);
    free(states);
    free(shared_buf);

    return final_status;
}
