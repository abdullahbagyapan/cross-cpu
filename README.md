# CrossCPU (`cross-cpu`)

`CrossCPU` is a production-grade CPU stress-testing and Silent Data Corruption (SDC) validation suite targeting **Linux** and **macOS** across **x86_64**, **ARM64 (`aarch64`)**, and **other POSIX architectures (`riscv64`, `ppc64le`/`ppc64`, `s390x`, and generic CPUs)**.

It combines a **zero-dependency Python 3.10+ standard-library CLI orchestrator** with a **strictly architecture-isolated C11 + POSIX `pthreads` native shared library** loaded via `ctypes`.

## Highlights

- **Zero External Runtime Dependencies:** Python 3.10+ stdlib only (`ctypes`, `argparse`, `json`, `threading`, `signal`, `difflib`, etc.) + standard C11 (`-std=c11`) and POSIX `pthreads`.
- **Strict Architecture Isolation (`native/arch/`):** `native/core/` is 100% architecture-agnostic (`worker_pool.c`, `dispatch.c`), while architecture-specific probing, cache-flush primitives, and SIMD/matrix/crypto kernels live in isolated translation units under `native/arch/{common,x86_64,arm64,other}`.
- **No `SIGILL` & Zero Fake Emulation:** Per-ISA translation units are compiled with probed per-file compiler flags and exposed through read-only kernel dispatch tables. Hardware-only workloads (`compute-vnni`, `compute-fp16`, `compute-bf16`, `compute-matrix-*`, `crypto-*`) never silently fall back to scalar C loops.
- **21 Deterministic SDC-Verified Workloads:** Integer (`GOP/s`), Floating-Point & ML (`GFLOP/s`), Matrix Engines (`GFLOP/s` / `GOP/s`), Hardware Crypto KATs (`GB/s`), Cache Tiers (`GB/s`), Main Memory (`GB/s`), and Multi-Core Atomic Coherency (`GOP/s`).
- **Filtered 1 Hz Telemetry & Severity-Ordered Exit Codes (`0..4`):** Sanitized Linux (`sysfs`/`procfs`/`hwmon`/`RAPL`/`EDAC`) and unprivileged macOS Mach telemetry without `N/A` noise or `null` JSON keys.

## Quick Start

```bash
# Build native C11 shared library offline and run unit tests
make build
make test

# Inspect host CPU topology, cache hierarchy, NUMA nodes, and ISA backends
./cross-cpu info

# List all 21 canonical workloads and host ISA readiness
./cross-cpu list

# Run targeted stress and SDC validation workloads
./cross-cpu run --test compute-fp32,memory-pattern --duration 500ms
./cross-cpu run --cores p-cores --isa auto --warmup 5s --duration 60s
./cross-cpu run --test all --duration 10s --json
```

## Documentation Suite

- [Documentation Index](docs/README.md)
- [CLI Reference & Exit Codes](docs/cli.md)
- [System Architecture & C11 ABI](docs/architecture.md)
- [ISA Backends (`x86_64`, `ARM64`, and `Other Architectures`)](docs/isa_backends.md)
- [Workloads, Fixed Units & SDC Verification](docs/workloads.md)
- [1 Hz Filtered Telemetry & EDAC Monitoring](docs/telemetry.md)
- [Packaging, Distro Builds & FHS Installation](docs/packaging.md)
- [Contributing Guide](docs/contributing.md)
- [Unix `man(1)` Page (`cross-cpu.1`)](docs/cross-cpu.1)

## License

MIT License. See [LICENSE](LICENSE).
