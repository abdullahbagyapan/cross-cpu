# CrossCPU Documentation Suite

Welcome to the `CrossCPU` (`cross-cpu`) documentation. `CrossCPU` is a modular, zero-dependency, multi-architecture CPU stress-testing and Silent Data Corruption (SDC) validation suite for Linux and macOS.

## Table of Contents

1. [**CLI Reference & Exit Codes (`cli.md`)**](https://github.com/abdullahbagyapan/cross-cpu/blob/main/docs/cli.md)
   - Command synopsis (`list`, `info`, `run`), flags, `ms`/`s`/`m`/`h` timing syntax, severity-ordered exit codes (`0..4`), stream hygiene (`stdout` vs `stderr`), and `JSON` output schema.
2. [**System Architecture & C11 ABI (`architecture.md`)**](https://github.com/abdullahbagyapan/cross-cpu/blob/main/docs/architecture.md)
   - Strict separation between Python 3.10+ stdlib orchestration (`src/`), architecture-agnostic C11 core (`native/core/`), and isolated per-architecture native backends (`native/arch/`).
3. [**ISA Backends (`isa_backends.md`)**](https://github.com/abdullahbagyapan/cross-cpu/blob/main/docs/isa_backends.md)
   - Priority matrices, silicon + OS save-state detection gates, per-file compiler flag probing, and `SIGILL` prevention across:
     - **x86_64** (`amx`, `avx512`, `avx2`, `sse4`, `scalar`)
     - **ARM64 / `aarch64`** (`sme`, `sve`, `neon`, `scalar`)
     - **Other Architectures** (`riscv64` `rvv`, `ppc64le`/`ppc64` `vsx`, `s390x` `vx`, and `generic` POSIX fallback)
4. [**Workloads, Fixed Units & SDC Verification (`workloads.md`)**](https://github.com/abdullahbagyapan/cross-cpu/blob/main/docs/workloads.md)
   - Mathematical specifications, fixed SI units (`GOP/s`, `GFLOP/s`, `GB/s`), topology-driven buffer sizing, and deterministic verification invariants for all 21 workloads.
5. [**Packaging, Distro Builds & FHS Installation (`packaging.md`)**](https://github.com/abdullahbagyapan/cross-cpu/blob/main/docs/packaging.md)
   - `pyproject.toml` `src/` -> `cross_cpu` mapping, offline wheel and `sdist` builds, multi-arch `Makefile` FHS installation, and shell completions.
6. [**Testing & Multi-Architecture Emulation (`emulation.md`)**](https://github.com/abdullahbagyapan/cross-cpu/blob/main/docs/emulation.md)
   - Offline unit & SDC fault-injection test suite (`tests/`), Docker-only multi-architecture emulation harness (`scripts/emulate.sh`) covering all 6 `native/arch/*` backends, QEMU per-ISA gating, multi-distro GCC/Clang builds, Colima setup, and GitHub Actions emulation CI.
7. [**Contributing Guide (`contributing.md`)**](https://github.com/abdullahbagyapan/cross-cpu/blob/main/docs/contributing.md)
   - Coding standards, SPDX headers, adding new ISA backends or workloads, fault-injection testing, and Conventional Commits workflow.
8. [**Unix Manual Page (`cross-cpu.1`)**](https://github.com/abdullahbagyapan/cross-cpu/blob/main/docs/cross-cpu.1)
   - Standard `troff`/`man(1)` manual page installed to `$(PREFIX)/share/man/man1/cross-cpu.1`.
