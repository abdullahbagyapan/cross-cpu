# Packaging, Distro Builds & FHS Installation

`CrossCPU` is structured for clean offline builds across Linux distributions (`.deb`, `.rpm`, Alpine `APKBUILD`, Arch `PKGBUILD`), macOS, and standard Python `pip` / wheel workflows.

## Flat `src/` to `cross_cpu` Package Mapping

On disk, Python modules reside directly under `src/` and use **relative imports only**:

```toml
[tool.setuptools]
packages = ["cross_cpu", "cross_cpu.telemetry"]
package-dir = {"cross_cpu" = "src"}
```

When installed into a virtual environment or system `site-packages/`, files are placed under `site-packages/cross_cpu/` (alongside the compiled `_native*.so` / `_native*.dylib`) and **never** create a top-level `site-packages/src/` collision.

## Per-Architecture C11 Compilation (`setup.py` & `Makefile`)

Both `setup.py` (`CrossCPUBuildExt`) and `Makefile` detect the target architecture (`x86_64`, `arm64`/`aarch64`, `riscv64`, `ppc64le`/`ppc64`, `s390x`, or `generic` fallback) and compile:

- `native/core/worker_pool.c` and `native/core/dispatch.c` with baseline C11 flags only.
- `native/arch/common/isa_scalar.c` and `native/arch/common/cache_flush_generic.c` with baseline C11 flags only.
- The matching `native/arch/<target>/*` directory, probing compiler support for each per-ISA flag (`-msse4.2`, `-mavx2 -mfma`, `-mavx512f -mavx512bw -mavx512vl`, `-mamx-tile -mamx-bf16 -mamx-int8`, `-march=armv8.2-a+sve`, `-march=armv9-a+sme`, `-march=rv64gcv`, `-mvsx`, `-mzvector`) and applying it strictly to the isolated `.c` translation unit.
- On macOS with `universal2` Python builds, `setup.py` compiles the `arm64` slice from `native/arch/arm64/` and the `x86_64` slice from `native/arch/x86_64/` separately and combines them via `lipo -create`.

## Offline Wheel & Source Distribution (`sdist`)

```bash
# Offline wheel install in an isolated virtual environment
python3 -m venv /tmp/cross-cpu-venv
/tmp/cross-cpu-venv/bin/pip install setuptools wheel
/tmp/cross-cpu-venv/bin/pip install --no-build-isolation --no-deps .

# Build complete source distribution (includes all native/arch/* directories)
python3 setup.py sdist
```

`MANIFEST.in` includes **all** `native/arch/*` directories (`common`, `x86_64`, `arm64`, `other/riscv64`, `other/ppc64`, `other/s390x`, `other/generic`), `src/`, `completions/`, `docs/`, `tests/`, `Makefile`, `LICENSE`, and `CHANGELOG.md` while excluding host build artifacts (`*.so`, `*.dylib`, `*.o`, `__pycache__`).

## Makefile FHS Installation (`PREFIX` & `DESTDIR`)

For OS package maintainers building without `pip`:

```bash
make build
make test
make install DESTDIR=/tmp/cross-cpu-stage PREFIX=/usr
```

Installed layout under `$(DESTDIR)$(PREFIX)`:

- `$(PREFIX)/bin/cross-cpu` — Executable CLI entry script
- `$(PREFIX)/lib/cross-cpu/cross_cpu/` — Python package + compiled `_native.so`/`.dylib`
- `$(PREFIX)/share/man/man1/cross-cpu.1` — Unix `man(1)` manual page
- `$(PREFIX)/share/bash-completion/completions/cross-cpu` — Bash completion
- `$(PREFIX)/share/zsh/site-functions/_cross-cpu` — Zsh completion
- `$(PREFIX)/share/fish/vendor_completions.d/cross-cpu.fish` — Fish completion
