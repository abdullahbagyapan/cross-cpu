# System Architecture & C11 ABI

`CrossCPU` is divided into two strictly decoupled layers:

1. **Python 3.10+ Standard Library Orchestrator (`src/`):**
   - CLI parsing (`src/cli.py`), `<80`-column ASCII banner (`src/banner.py`), exit-code aggregation (`src/exit_codes.py`), Linux `sysfs` & macOS `sysctl` topology discovery (`src/topology.py`), dynamic cache/memory working-set sizing (`src/workloads.py`), warmup/measurement runner (`src/runner.py`), and 1 Hz filtered telemetry (`src/telemetry/`).
   - Zero external runtime dependencies (`ctypes`, `argparse`, `json`, `threading`, `signal`, `difflib`, `pathlib`, `dataclasses`, `struct`, `time`, `math`, `os`, `sys`, `abc`, `re`, `tarfile`, `unittest`).
   - Uses **relative imports only** inside `src/` so the source tree works both in-place (`./cross-cpu`) and installed (`site-packages/cross_cpu/`).

2. **Modular Multi-Architecture C11 + POSIX `pthreads` Shared Library (`native/`):**
   - Compiled to `_native.so` (Linux) or `_native.dylib` (macOS) with `-std=c11 -O2 -fPIC -lpthread -lm` and zero external C dependencies (no `libnuma`, `hwloc`, `OpenSSL`, or `OpenMP`).

## Strict Architecture Isolation

`native/core/` contains **zero** architecture-specific `#ifdef` directives (`__x86_64__`, `__aarch64__`, `__riscv`, `__powerpc__`, `__s390x__`, etc.):

```text
native/
  include/
    cross_cpu_abi.h              # Public C11 ctypes ABI (fixed-width stdint.h structs)
    cross_cpu_arch.h             # Internal arch hook contract
  core/
    worker_pool.c                # POSIX pthreads pool, affinity/QoS, barrier, atomic stop flag
    dispatch.c                   # Arch-agnostic priority resolution & zero-fake-emulation dispatch
  arch/
    common/
      isa_scalar.c               # Portable C11 scalar, cache, RAM, pattern, sweep, <stdatomic.h>
      cache_flush_generic.c      # Contention-free oversized buffer sweep cache-eviction fallback
    x86_64/                      # CPUID/XGETBV probe + SSE4.2, AVX2, AVX-512, Intel AMX
    arm64/                       # getauxval/sysctlbyname probe + NEON, SVE, SME
    other/
      riscv64/                   # getauxval/riscv_hwprobe + RVV 1.0
      ppc64/                     # AT_HWCAP/AT_HWCAP2 + PowerPC VSX
      s390x/                     # AT_HWCAP + IBM z/Vector (VX) — big-endian safe
      generic/                   # Fallback for any other POSIX CPU (exposes scalar cleanly)
```

Every target architecture implements the exact same three-function C hook contract declared in `native/include/cross_cpu_arch.h`:

```c
void cross_cpu_arch_probe(cross_cpu_caps_t *caps);
void cross_cpu_arch_register_backends(cross_cpu_registry_t *reg);
void cross_cpu_arch_flush_cache_range(const void *ptr, size_t len);
```

## Isolated Per-ISA Translation Units & `SIGILL` Prevention

Compilers given advanced SIMD/matrix flags (such as `-mavx512f` or `-march=armv9-a+sme`) can emit vector instructions inside ordinary C functions (for example, auto-vectorizing struct initialization). To guarantee **zero `SIGILL` crashes** on older CPUs:

1. `native/core/*`, `native/arch/common/*`, and `native/arch/<target>/probe_*.c` are compiled strictly with **baseline compiler flags** (never `-march=native`).
2. Per-ISA kernel files (`isa_sse4.c`, `isa_avx2.c`, `isa_avx512.c`, `isa_amx.c`, `isa_neon.c`, `isa_sve.c`, `isa_sme.c`, `isa_rvv.c`, `isa_vsx.c`, `isa_vx.c`) are compiled in isolated translation units with probed per-file ISA flags, and optional sub-extensions use `__attribute__((target("...")))`.
3. Each per-ISA translation unit exports only a **read-only `const` function pointer table** (`cross_cpu_*_kernels[CROSS_CPU_WL_COUNT]`). No executable instruction inside an ISA translation unit is ever called unless `probe_*.c` has confirmed both silicon and OS save-state support at runtime.

## Worker Pool, First-Touch NUMA Locality & Signal Responsiveness

- **Thread Placement & Memory Locality:** Each worker thread in `native/core/worker_pool.c` applies Linux `sched_setaffinity()` (or macOS `pthread_set_qos_class_self_np()`) *before* allocating and zero-touching its 128-byte-aligned per-thread buffer (`posix_memalign`), ensuring Linux first-touch NUMA page locality.
- **Portable Barrier Synchronization:** Because macOS does not provide `pthread_barrier_t`, `worker_pool.c` implements a portable `pthread_mutex_t` + `pthread_cond_t` barrier so all threads begin timed measurement simultaneously.
- **Dynamic Watchdog Slices & `SIGINT` Bridge:** Workloads execute in dynamically calibrated ~100 ms slices (clamped below `--duration` for sub-second `ms` runs). Python's `SIGINT`/`SIGTERM` handler invokes `cross_cpu_request_stop()`, setting a C11 `_Atomic int` flag checked between every batch.
