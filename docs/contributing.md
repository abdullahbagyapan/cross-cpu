# Contributing to CrossCPU

Thank you for contributing to `CrossCPU` (`cross-cpu`). Please follow these architectural and testing guidelines to keep the suite portable, deterministic, and dependency-free.

## Core Invariants

1. **Zero External Runtime Dependencies:**
   - Python code under `src/` and `tests/` must use only the Python 3.10+ standard library.
   - Native code under `native/` must use only standard C11 (`-std=c11`), POSIX `pthreads`, and OS/ISA headers.
2. **Strict Architecture Isolation:**
   - Never add architecture-specific `#ifdef` macros (`__x86_64__`, `__aarch64__`, `__riscv`, `__powerpc__`, `__s390x__`) to `native/core/`.
   - Place architecture-specific probing and kernels in their respective `native/arch/<target>/` directory behind the `native/include/cross_cpu_arch.h` contract.
3. **No `SIGILL` & Read-Only Kernel Dispatch Tables:**
   - Never use `-march=native`.
   - Per-ISA translation units (`isa_*.c`) must export only read-only `const` kernel function pointer tables and compiled-in flags so no instruction inside an ISA translation unit executes before runtime probing verifies both silicon and OS register save-state support.
4. **Zero Fake Emulation:**
   - Never fall back to scalar C loops or FP32 upcasting when an ISA or hardware datatype (`FP16`, `BF16`, `VNNI`, `AMX`, `SME`, `AES`, `SHA-256`) is unavailable. Return `CROSS_CPU_STATUS_UNSUPPORTED` with an explicit diagnostic reason.
5. **SPDX Headers & Relative Imports:**
   - Every source file must carry `# SPDX-License-Identifier: MIT` or `// SPDX-License-Identifier: MIT` (preserving `#!` shebangs and `#compdef cross-cpu` on line 1).
   - Python files inside `src/` must use relative imports only.

## Verification Checklist

Before submitting changes, run the full offline verification suite:

```bash
# 1. Build native C11 shared library and run unit tests
make clean && make build
python3 -m compileall -q src tests
python3 -m unittest discover -s tests -v

# 2. Verify shell completions and FHS staging
bash -n completions/cross-cpu.bash
make install DESTDIR=/tmp/cross-cpu-stage PREFIX=/usr

# 3. Verify offline wheel and sdist packaging
python3 setup.py sdist --dist-dir /tmp/cross-cpu-dist

# 4. (Optional, Docker-only) Multi-arch & per-ISA emulation verification
./scripts/emulate.sh --suite arch
./scripts/emulate.sh --suite isa
```

See [Testing & Multi-Architecture Emulation](https://github.com/abdullahbagyapan/cross-cpu/blob/main/docs/emulation.md) for the full Docker emulation matrix, Colima setup, and GitHub Actions CI details.

## Conventional Commits

Format commit messages following [Conventional Commits](https://www.conventionalcommits.org/):

- `feat(native): ...`
- `feat(arch-x86): ...`
- `feat(arch-arm64): ...`
- `feat(arch-other): ...`
- `feat(cli): ...`
- `fix(runner): ...`
- `test(cross-cpu): ...`
- `docs(cross-cpu): ...`
