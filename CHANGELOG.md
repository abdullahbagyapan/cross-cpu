# Changelog

All notable changes to `CrossCPU` (`cross-cpu`) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3a1] - 2026-11-24

### Added
- **PyPI Trusted Publishing (`.github/workflows/release.yml`):** Added automated `publish-pypi` job using GitHub OIDC trusted publishing (`pypa/gh-action-pypi-publish@release/v1`) with strict `twine check` validation for source distributions.
- **Package Metadata (`pyproject.toml`):** Added explicit `readme = "README.md"` and project URLs (`Homepage`, `Repository`, `Issues`, and `"Hardware Matrix"`).

## [0.1.3] - 2026-11-24

### Added
- **Emulation and Multi-Architecture Testing Documentation (`docs/emulation.md`):** Added a dedicated reference guide covering the Docker-only test harness (`scripts/emulate.sh`), Colima and multi-architecture Docker setup on macOS hosts, QEMU per-ISA gating, adversarial sandboxing, and CI matrix replication.

### Fixed
- **Hermetic CI Emulation Runner (`.github/workflows/emulation.yml`):** Unified all emulation matrix jobs to execute within hermetic Debian containers via `scripts/emulate.sh`, eliminating host environment pollution and SIGILL traps on legacy QEMU CPU models (such as `qemu64`).
- **Scalar Fallback Emulation Matrix:** Added `arch-generic` (pure scalar fallback backend) to CI emulation jobs.
- **x86_64 Cache Flush Compiler Warning:** Explicitly cast pointer in `_mm_clflushopt` to void pointer (`native/arch/x86_64/probe_x86.c`) to eliminate discarded-qualifiers warnings under GCC/Clang.

## [0.1.2] - 2026-11-23

### Added
- **Multi-Architecture Emulation CI Workflow (`.github/workflows/emulation.yml`):** Added GitHub Actions emulation matrix covering x86_64/ARM64 QEMU per-ISA gating, foreign architectures (`linux/arm64`, `linux/riscv64`, `linux/ppc64le`, big-endian `linux/s390x`), multi-distro libc/compiler validation (Alpine musl, Debian Trixie hardened, Fedora, Arch Linux with GCC + Clang + offline wheels), and adversarial locked-down containers (offline, read-only, non-root UID 65534, non-contiguous cpuset).
- **Docker-Only Local Emulation Test Harness (`scripts/emulate.sh`):** Added zero-host-assumption, testing-only Docker script targeting all 6 `native/arch/*` backends (`amd64`, `arm64`, `riscv64`, `ppc64le`, big-endian `s390x`, and `generic`), QEMU per-ISA gating, distros, and adversarial containers with Rosetta-safe `binfmt_misc` probing and structured per-step logs + Markdown summaries.

### Fixed
- **CI Emulation Diagnostics & Compatibility:** Fixed CLI module entry point (`python3 -m src`), separated verbose `-v` probe traces from strict `--json` checks (with automatic `-v` diagnostic retries on silent `--json` failures), added uncollapsed `::error::` annotations + failure details in `$GITHUB_STEP_SUMMARY`, corrected `native.arch_name` JSON assertions, added `redhat-rpm-config` on Fedora and `--disable-sandbox` on Arch Linux, and moved adversarial wheel pre-builds into an ABI-matched `python:3.12-slim` container.

## [0.1.1] - 2026-11-22

### Fixed
- **Native Worker Pool Responsiveness:** Immediately check C11 atomic stop flag in inner iteration batches to eliminate exit lag during early cancellation and timeouts.
- **DRAM Cache-Flush Verification:** Explicitly evict cache lines in scalar `bandwidth-ram` and architecture-specific SIMD memory loops (`RVV`, `VSX`, `VX`) to guarantee main memory bus saturation rather than L1/L2 hits.
- **s390x z14 Vector Instruction Isolation:** Guard vector float and double precision built-ins with `#pragma GCC target("vx,vxe")` and `__attribute__((target("vxe")))` on s390x to prevent illegal instruction traps on z13 hardware.
- **ARM64 SVE/SME Vector Length Safety:** Enforce positive vector length bounds guards (`svcntb() > 0`, `svcntd() > 0`) preventing division-by-zero or infinite loops on unconfigured SVE/SME hardware.
- **Dynamic DRAM Buffer Sizing:** Dynamically scale DRAM memory buffers strictly above LLC capacity per thread while enforcing low-RAM OOM caps to prevent thrashing.
- **Generic Topology Fallback:** Provide portable CPU topology discovery on non-Linux, non-macOS POSIX environments.
- **CI & Packaging:** Added GitHub issue templates, set explicit workflow permissions for security scanners, and cleaned up legacy test assertions.

### Added
- **Release Audit Regression Tests:** Comprehensive test suite for C11 ABI struct offsets, early stop cancellation responsiveness, full `--test all` workload invocation, fault injection on crypto and sweep kernels, and dynamic LLC buffer sizing.

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
