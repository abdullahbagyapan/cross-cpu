# Changelog

All notable changes to `CrossCPU` (`cross-cpu`) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-19

### Added
- **Architecture-Agnostic C11 Core (`native/core/`):**
  - Flat fixed-width public ABI (`native/include/cross_cpu_abi.h`) and internal hook contract (`native/include/cross_cpu_arch.h`).
  - POSIX `pthreads` worker pool (`worker_pool.c`) with Linux `sched_setaffinity()` + first-touch NUMA allocation, macOS `pthread_set_qos_class_self_np()` QoS hints, portable barrier synchronization, dynamic batch calibration, and C11 atomic stop flag.
  - Priority-sorted ISA dispatcher (`dispatch.c`) enforcing strict zero-fake-emulation rules.
- **Portable Baseline (`native/arch/common/`):**
  - C11 `scalar` integer, floating-point contractive FMA, cache read/write, memory bandwidth, adversarial `memory-pattern`, SplitMix64 `memory-sweep`, and multi-core `<stdatomic.h>` `coherency-atomic` kernels.
  - Contention-free oversized buffer sweep cache-eviction fallback (`cache_flush_generic.c`).
- **x86_64 Backends (`native/arch/x86_64/`):**
  - `CPUID` + `XGETBV` (`XCR0`) + Linux `arch_prctl(ARCH_REQ_XCOMP_PERM)` runtime probing and `clflushopt`/`clflush` cache line flushing (`probe_x86.c`).
  - Isolated per-ISA translation units for `sse4` (SSE4.2 + AES-NI + SHA-NI), `avx2` (AVX2 + FMA + AVX-VNNI + `_mm256_stream_si256`), `avx512` (AVX-512F/BW/VL/VNNI/BF16/FP16), and `amx` (Intel AMX-TILE/BF16/INT8).
- **ARM64 Backends (`native/arch/arm64/`):**
  - Linux `getauxval(AT_HWCAP/AT_HWCAP2)` and macOS `sysctlbyname` probing with `dc civac` user-space cache flushing (`probe_arm64.c`).
  - Isolated per-ISA translation units for `neon` (ASIMD + FP16 + BF16 + DotProd + I8MM + FEAT_AES + FEAT_SHA256 + `stnp`), `sve` (Linux SVE/SVE2), and `sme` (ARMv9 SME matrix engine).
- **Other Architectures (`native/arch/other/`):**
  - `riscv64` (`getauxval`/`riscv_hwprobe` + RVV 1.0 `rvv`), `ppc64`/`ppc64le` (`AT_HWCAP`/`AT_HWCAP2` + AltiVec/VSX `vsx`), `s390x` (big-endian safe `AT_HWCAP` + IBM z/Vector `vx`), and `generic` POSIX fallback.
- **Python 3.10+ Stdlib CLI (`src/`):**
  - `cross-cpu list`, `cross-cpu info`, and `cross-cpu run` subcommands with `ms`/`s`/`m`/`h` duration parser, `difflib` typo suggestions, `<80`-column ASCII banner with `NO_COLOR`/`TERM=dumb`/TTY/JSON guards, single-document `--json` output, and severity-ordered exit codes (`0..4`).
  - Linux `sysfs` and macOS `sysctl` topology discovery (`src/topology.py`).
- **Packaging, Completions & Documentation:**
  - Static `bash`, `zsh`, and `fish` completions (`completions/`), offline `Makefile` with FHS `install`/`uninstall`, PEP 621 `pyproject.toml`, per-arch C11 `setup.py` (including macOS `universal2` slice compilation), `docs/` suite, and `cross-cpu(1)` man page.
