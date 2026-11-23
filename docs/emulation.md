# Testing & Multi-Architecture Emulation

`CrossCPU` (`cross-cpu`) separates **zero-dependency offline unit tests** from **Docker-based multi-architecture emulation testing**.

> [!NOTE]
> Docker and QEMU emulation (`scripts/emulate.sh` and `.github/workflows/emulation.yml`) exist **strictly for testing and validation**. They are **not** runtime or build dependencies of `cross-cpu` and are excluded from packaged wheels and `sdist` archives.

---

## 1. Offline Unit & Fault-Injection Test Suite (`tests/`)

The offline `unittest` suite requires only Python 3.10+ stdlib and a C11 compiler (`make build`), with zero network access or external packages:

```bash
make clean && make build
python3 -m compileall -q src tests
python3 -m unittest discover -s tests -v
```

### Test Modules (39 Tests)

| Module | Coverage |
|---|---|
| `tests/test_cli.py` | CLI flags, `ms`/`s`/`m`/`h` duration parser, `--warmup`, `difflib` typo hints, `<80`-column banner + `NO_COLOR`/`TERM=dumb` guards, strict single-document `--json` output with empty `stderr` |
| `tests/test_exit_codes.py` | Severity-ordered `ExitCode` values (`0..4`) and max-severity aggregation |
| `tests/test_isa_dispatch.py` | Zero arch `#ifdef`s in `native/core/`, `native/arch/` file layout, C11 ABI struct sizes & field offsets, `-march=native` / global `-march=z14` ban, priority ordering, unavailable forced ISA (`INIT_ERROR`), zero fake emulation under `scalar` |
| `tests/test_topology.py` | Linux `sysfs` multi-socket/NUMA/cache/hybrid (`cpu_core`/`cpu_atom` and ARM `cpu_capacity`) discovery, macOS `sysctl` P-core/E-core performance levels, generic POSIX fallback, and CPU list parsing |
| `tests/test_workloads.py` | All 21 canonical workloads, fixed SI `10^9` scaling, dynamic LLC/DRAM buffer sizing with low-RAM OOM caps, native execution across supported workloads, deterministic SDC fault injection (`ExitCode.COMPUTE_ERROR = 3`), `--test all` skipping, and C11 atomic stop-flag cancellation |
| `tests/test_packaging.py` | SPDX license headers across all source/build/workflow files, relative-only stdlib imports in `src/`, Bash/Zsh/Fish completion parity + `bash -n`, `pyproject.toml` zero-dependency mapping, and offline `sdist` completeness across all `native/arch/*` directories |

---

## 2. Docker-Only Multi-Architecture Emulation Harness (`scripts/emulate.sh`)

To validate every architecture backend, endianness, QEMU CPU tier, Linux distribution/libc/compiler, and restricted container environment without host assumptions, run:

```bash
./scripts/emulate.sh --list
```

### Design Principles

- **Docker is the sole host dependency:** Pure Bash 3.2+ host script with zero reliance on host Python, host C compilers, host `make`, host `qemu-user`, Rosetta 2, or host OS/arch detection.
- **Read-only repository mount:** The working tree is mounted strictly read-only (`/repo:ro`) and copied to `/tmp/work` inside each container, leaving the host repo untouched.
- **Rosetta-safe `binfmt_misc` management:** Probes `tonistiigi/binfmt` first and registers only missing architectures (`amd64`, `arm64`, `riscv64`, `ppc64le`, `s390x`) without overwriting pre-existing handlers such as Apple VZ Rosetta.
- **Clean interrupt handling:** Labels every spawned container (`cross-cpu-emulate=<run_id>`) and wheel volume (`<run_id>-wheels`), stopping and removing them automatically on `SIGINT`/`SIGTERM`.

### Suites & Targets

| Suite (`--suite`) | Targets (`--target`) | What Is Verified |
|---|---|---|
| `arch` | `arch-amd64`, `arch-arm64`, `arch-riscv64`, `arch-ppc64le`, `arch-s390x`, `arch-generic` | All 6 `native/arch/*` backends via explicit Docker `--platform` (`linux/amd64`, `linux/arm64`, `linux/riscv64`, `linux/ppc64le`, big-endian `linux/s390x`, plus `ARCH=generic` scalar-only build). Runs `compileall`, `make build`, all 39 unit tests, `info -v`, strict `info --json` (asserting `native.arch_name`, `native.endianness`, empty `stderr`), `run --json`, `run --test all --json`, and `generic` zero-emulation gates (`crypto-aes` → exit `2`). |
| `isa` | `isa-x86_64`, `isa-arm64` | Hermetic `debian:trixie-slim` containers executing `qemu-x86_64` and `qemu-aarch64` across progressive CPU tiers:<br>• **x86_64:** `qemu64` (`scalar`), `Nehalem` (`sse4`), `Haswell` (`avx2`), `max` (`avx2`/`avx512` + `--test all`), plus 6 negative exit-`2` gates (`sse4`, `avx2`, `avx512`, `amx`, `crypto-aes`).<br>• **ARM64:** `cortex-a53` (`neon`), `max,sve=off,sme=off` (`neon`), `max,sve=on,sme=off` (`sve` when compiled), plus 5 negative exit-`2` gates (`compute-fp16`, `sve`, `sme`). |
| `distros` | `distro-alpine`, `distro-debian`, `distro-fedora`, `distro-arch` | Compiles and runs the full 39-test suite with both **GCC** and **Clang** across musl (`alpine:latest`), hardened glibc (`debian:trixie-slim` with `-D_FORTIFY_SOURCE=3 -fstack-protector-strong -Werror=format-security`), Fedora (`fedora:latest` with `redhat-rpm-config`), and Arch Linux (`archlinux:latest` with `pacman --disable-sandbox`), followed by an offline wheel build (`python3 -m build --wheel --no-isolation`) verifying `cross_cpu/_native*.so` inside the wheel. |
| `adversarial` | `adversarial` | Pre-builds an ABI-matched wheel inside `python:3.12-slim` into a temporary Docker volume, then runs a locked-down `python:3.12-slim` container (`--network=none`, `--read-only`, `--tmpfs /tmp:rw,exec,mode=1777,size=256m`, `--user 65534:65534`, `--cpuset-cpus=...` sized from Docker's `NCPU`), validating offline `venv` install, `--cores <cpuset>`, `--cores all`, and empty `stderr` in `--json` mode. |
| `all` | All 13 targets above | Complete multi-architecture, per-ISA, multi-distro, and adversarial validation matrix. |

### Common Commands & Options

```bash
# Inspect all suites and targets
./scripts/emulate.sh --list

# Preview selected targets without running containers
./scripts/emulate.sh --dry-run --suite all

# Run all 6 native/arch/* backends (x86_64, arm64, riscv64, ppc64le, s390x, generic)
./scripts/emulate.sh --suite arch

# Run x86_64 and ARM64 QEMU per-ISA SIGILL & zero-emulation gates
./scripts/emulate.sh --suite isa

# Run specific targets with custom duration and log output directory
./scripts/emulate.sh --target arch-s390x,arch-riscv64,arch-generic --duration 100ms --log-dir ~/crosscpu-logs

# Override platform for distro and adversarial suites (default: linux/amd64)
./scripts/emulate.sh --suite distros,adversarial --distro-platform linux/arm64

# Skip automatic tonistiigi/binfmt registration (e.g. rootless Docker or pre-configured CI runner)
./scripts/emulate.sh --suite arch --skip-binfmt

# Stop on first failing target and stream full container output
./scripts/emulate.sh --suite all --fail-fast -v
```

### Logs, Diagnostics & Reports

By default, each run writes structured logs under `~/.cache/cross-cpu-emulation/<YYYYMMDD-HHMMSS>/`:

- `<target>/<step>.cmd` — exact command executed
- `<target>/<step>.rc` and `<step>.expected_rc` — actual and expected exit codes
- `<target>/<step>.stdout.log` and `<step>.stderr.log` — full captured streams
- `summary.md` — consolidated Markdown report with per-target timings and `## Failure Details` (`tail -n 200` of `stdout` and `stderr`) for every failed step

On any failure:
- Crash signals are decoded explicitly (e.g. `132 (SIGILL)`, `139 (SIGSEGV)`, `137 (SIGKILL/OOM)`, `124 (TIMEOUT)`).
- Because `cross-cpu ... --json` suppresses `stderr` by design, any unexpected `--json` failure with empty `stdout` and `stderr` automatically triggers a diagnostic retry (`<step>_diag_verbose`) without `--json` and with `-v` to capture the full error message and ISA probe trace.

---

## 3. Setting Up Docker for Multi-Arch Emulation

### Linux (Docker Engine)

Ensure the Docker daemon is running and accessible to your user (`docker info`). Running `./scripts/emulate.sh` automatically registers missing QEMU `binfmt_misc` handlers via `tonistiigi/binfmt`.

### macOS Apple Silicon (Colima)

Colima with Apple's Virtualization framework (`vz`) and Rosetta provides fast native `linux/arm64` execution, near-native `linux/amd64` translation via Rosetta, and full QEMU `binfmt_misc` emulation for `riscv64`, `ppc64le`, and `s390x`:

```bash
colima start \
  --cpu 4 \
  --memory 6 \
  --disk 40 \
  --vm-type vz \
  --vz-rosetta \
  --mount-type virtiofs \
  --runtime docker \
  --activate
```

> [!IMPORTANT]
> When `--vz-rosetta` is enabled in Colima, do **not** run `tonistiigi/binfmt --install all` manually—overwriting the `amd64` `binfmt_misc` entry conflicts with Apple's Rosetta handler. `scripts/emulate.sh` detects Rosetta automatically and registers only missing architectures (`riscv64,ppc64le,s390x`). If registering manually, run:
>
> ```bash
> docker run --privileged --rm tonistiigi/binfmt --install riscv64,ppc64le,s390x
> ```

When finished testing, stop the VM to reclaim memory:

```bash
colima stop
```

---

## 4. GitHub Actions Emulation Workflow (`.github/workflows/emulation.yml`)

GitHub Actions runs the exact same [`scripts/emulate.sh`](../scripts/emulate.sh) targets (single source of truth) on:
- Manual dispatch (`workflow_dispatch`)
- Weekly schedule (`Sunday 04:00 UTC`)
- Release tags (`v*.*.*`)

| CI Job | Runner | Targets Executed |
|---|---|---|
| `x86-isa-sigill-matrix` | `ubuntu-latest` | `isa-x86_64` inside hermetic `debian:trixie-slim` |
| `arm64-isa-sigill-matrix` | `ubuntu-24.04-arm` | `isa-arm64` inside hermetic `debian:trixie-slim` |
| `foreign-arch-matrix` | `ubuntu-latest` | `arch-arm64`, `arch-riscv64`, `arch-ppc64le`, `arch-s390x`, `arch-generic` |
| `distro-libc-compiler-matrix` | `ubuntu-latest` | `distro-alpine`, `distro-debian`, `distro-fedora`, `distro-arch` |
| `adversarial-container-constraints` | `ubuntu-latest` | `adversarial` |

Each job appends `summary.md` directly to the GitHub Actions Job Summary (`$GITHUB_STEP_SUMMARY`), emits `::error::` annotations on target failures, and uploads the entire log directory as an artifact (`retention-days: 14`).

---

## 5. Troubleshooting

- **Docker daemon unreachable (`exit 2`):** Start Docker Engine, Docker Desktop, or Colima (`colima start`) and re-run `docker info`.
- **Rootless Docker / unprivileged environment:** Pass `--skip-binfmt` if `--privileged` containers are disallowed and register `qemu-user-static` on the host instead.
- **Custom `--log-dir` on macOS VMs:** Choose a directory inside `$HOME` (such as the default `~/.cache/cross-cpu-emulation/...`) because Colima's VirtioFS mounts `$HOME` by default rather than `/tmp` or `/var/folders`.
- **Slow foreign-arch runs (`s390x`, `riscv64`, `ppc64le`):** Emulated `apt-get` + GCC + 39 unit tests typically take 2–4 minutes per foreign architecture; pass `--duration 80ms` for faster local smoke iterations.
