# CLI Reference & Exit Codes

## Synopsis

```text
cross-cpu list [--format text|json] [--json] [-v|--verbose]
cross-cpu info [--format text|json] [--json] [-v|--verbose]
cross-cpu run [--cores all|physical|p-cores|e-cores|LIST]
              [--numa all|N] [--isa auto|ISA_NAME]
              [--test TEST|all] [--duration TIME] [--warmup [TIME]]
              [--format text|json] [--json] [-v|--verbose] [--no-banner]
cross-cpu --help
cross-cpu --version
```

## Subcommands

### `cross-cpu list`

Displays all 21 canonical workloads, their category, fixed measurement units (`GOP/s`, `GFLOP/s`, `GB/s`), resolved `--isa auto` backend on the current host, readiness status (`ready` vs `unsupported`), and description.

### `cross-cpu info`

Discovers and prints host CPU topology (sockets, physical/logical cores, NUMA nodes, hybrid P-core/E-core clusters, L1i/L1d/L2/L3 cache hierarchy, system memory, thread placement mechanism) alongside native architecture probe results (`CPUID`/`XGETBV`, `getauxval`, or `sysctlbyname`) and registered ISA backends in descending priority order.

### `cross-cpu run`

Executes selected stress and Silent Data Corruption (SDC) verification workloads across the chosen CPU cores.

| Flag | Default | Description |
| --- | --- | --- |
| `--cores SPEC` | `all` | `all`, `physical` (one thread per physical core), `p-cores`, `e-cores`, or Linux-style CPU list (`0-3,6,8-9`). |
| `--numa NODE` | `all` | `all` or integer NUMA node ID (`0`, `1`, ...). |
| `--isa ISA` | `auto` | `auto` (highest-priority available ISA per workload) or explicit backend (`amx`, `avx512`, `avx2`, `sse4`, `sme`, `sve`, `neon`, `rvv`, `vsx`, `vx`, `scalar`). |
| `--test TEST` | `all` | Workload name(s) (comma-separated or repeated `--test`) or `all`. |
| `--duration TIME` | `10s` | Timed execution duration per workload (`ms`, `s`, `m`, `h`; must be `> 0`). |
| `--warmup [TIME]` | `0s` | Optional warmup phase before timing. Omitted = `0s`; bare `--warmup` = `5s`; explicit `--warmup 0ms` / `--warmup 2s`. |
| `--format text\|json` | `text` | Output format on `stdout`. |
| `--json` | `false` | Shorthand for `--format json` (conflicts with `--format text`). |
| `-v`, `--verbose` | `false` | Emit architecture probe bits, core affinity, buffer sizing, and ISA dispatch diagnostics to `stderr`. |
| `--no-banner` | `false` | Suppress the `<80`-column interactive ASCII banner in `text` mode. |

## Duration Syntax & Sub-Second Progress

- Supported suffixes: `ms` (milliseconds), `s` (seconds), `m` (minutes), `h` (hours).
- The parser matches `ms` before single-character suffixes (`500ms`, `1.5s`, `5m`, `1h`).
- Sub-second interactive progress is formatted in milliseconds (e.g. `[01/02 compute-fp32 | 250ms/500ms]`, never `0s/0s`).

## Severity-Ordered Exit Codes (`max()` Aggregation)

When multiple workloads execute in a single invocation, `cross-cpu` aggregates all workload outcomes and hardware fault checks using `max(codes)`:

| Code | Enum Name | Condition |
| --- | --- | --- |
| `0` | `SUCCESS` | All executed workloads passed verification with zero hardware faults. |
| `1` | `USAGE_ERROR` | Invalid CLI syntax, unknown flag/subcommand/workload/ISA/preset, `--duration <= 0`, or conflicting `--format text --json`. |
| `2` | `INIT_ERROR` | Native shared library load error, requested `--isa`/`--cores`/`--numa` unavailable on host, buffer allocation failure, or an explicitly requested `--test <name>` is unsupported. |
| `3` | `COMPUTE_ERROR` | Numerical or bitwise SDC mismatch, unexpected `NaN`/`Inf`, NIST crypto KAT failure, DRAM bit error, or multi-core atomic coherency mismatch. |
| `4` | `HARDWARE_ERROR` | Linux EDAC uncorrectable ECC memory error (`ue_count` delta `> 0`) or fatal hardware fault during stress execution. |

## Stream Hygiene & Typo Ergonomics

- **`stdout`:** Strictly reserved for primary command output (`text` tables or valid `json`), `--help`, and `--version`.
- **`stderr`:** Strictly reserved for usage/init/compute/hardware error diagnostics, unsupported workload reasons, and `-v`/`--verbose` output.
- **`difflib` Hints:** Mistyped subcommands, `--test` names, `--isa` backends, and `--cores` presets suggest the closest valid match.
