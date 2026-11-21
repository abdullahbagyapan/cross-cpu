# SPDX-License-Identifier: MIT
"""Command-line interface for CrossCPU (zero external runtime dependencies)."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import sys
from typing import Any, Sequence, TextIO

from . import __version__
from .banner import maybe_print_banner, supports_color
from .exit_codes import ExitCode
from .models import CPUTopology, NativeCapabilities, WorkloadResult, WorkloadSpec
from .native_bridge import ISA_NAME_TO_ID, NativeBridgeError, NativeLibrary
from .runner import RunReport, WorkloadRunner, format_duration_progress
from .topology import (
    TopologyUnavailableError,
    TopologyUsageError,
    discover_topology,
    select_cores,
)
from .workloads import ALL_WORKLOAD_NAMES, WORKLOAD_BY_NAME, WORKLOAD_SPECS

SUBCOMMANDS = ("list", "info", "run")
ALL_ISA_NAMES = tuple(ISA_NAME_TO_ID.keys())

_DURATION_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(ms|s|m|h)\s*$", re.IGNORECASE)
_DURATION_MULTIPLIERS = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}


class CLIUsageError(ValueError):
    """Raised on CLI argument errors; always maps to ExitCode.USAGE_ERROR (1)."""


def parse_duration(raw: str, *, allow_zero: bool = False, flag_name: str = "--duration") -> float:
    """
    Parse duration strings with ``ms``, ``s``, ``m``, or ``h`` suffix into seconds.

    Matches ``ms`` before single-character suffixes. Rejects ``<= 0`` unless
    ``allow_zero=True`` (for ``--warmup``).
    """
    m = _DURATION_RE.match(raw)
    if not m:
        raise CLIUsageError(
            f"Invalid {flag_name} value {raw!r}: expected <number><ms|s|m|h> "
            "(examples: 500ms, 1.5s, 5m, 1h)"
        )
    magnitude = float(m.group(1))
    unit = m.group(2).lower()
    if not math.isfinite(magnitude):
        raise CLIUsageError(f"Invalid {flag_name} value {raw!r}: must be finite")
    seconds = magnitude * _DURATION_MULTIPLIERS[unit]
    if allow_zero:
        if seconds < 0.0:
            raise CLIUsageError(f"Invalid {flag_name} value {raw!r}: must be >= 0")
    else:
        if seconds <= 0.0:
            raise CLIUsageError(f"Invalid {flag_name} value {raw!r}: must be > 0")
    return seconds


def _resolve_output_format(fmt: str | None, json_flag: bool) -> str:
    if fmt == "text" and json_flag:
        raise CLIUsageError("Conflicting output flags: cannot combine '--format text' with '--json'")
    if json_flag or fmt == "json":
        return "json"
    return "text"


def _has_json_flag(raw_argv: Sequence[str]) -> bool:
    """Return True if ``--json`` or ``--format json`` appears in ``raw_argv``."""
    if "--json" in raw_argv:
        return True
    for idx, tok in enumerate(raw_argv):
        if tok == "--format" and idx + 1 < len(raw_argv) and raw_argv[idx + 1] == "json":
            return True
        if tok == "--format=json":
            return True
    return False


def _parse_test_selection(raw_tests: Sequence[str]) -> tuple[list[WorkloadSpec], bool]:
    """Parse ``--test`` flag values into ``(specs, explicit_tests)``."""
    tokens: list[str] = []
    for item in raw_tests:
        for piece in item.split(","):
            cleaned = piece.strip().lower()
            if cleaned:
                tokens.append(cleaned)

    if not tokens:
        return list(WORKLOAD_SPECS), False

    if "all" in tokens:
        if len(tokens) > 1:
            raise CLIUsageError("Cannot combine '--test all' with specific workload names")
        return list(WORKLOAD_SPECS), False

    selected: list[WorkloadSpec] = []
    seen: set[str] = set()
    valid_choices = ("all",) + ALL_WORKLOAD_NAMES
    for name in tokens:
        spec = WORKLOAD_BY_NAME.get(name)
        if spec is None:
            matches = difflib.get_close_matches(name, valid_choices, n=1, cutoff=0.5)
            hint = f" Did you mean '{matches[0]}'?" if matches else ""
            raise CLIUsageError(
                f"Unknown workload {name!r} (run 'cross-cpu list' to see all {len(WORKLOAD_SPECS)} workloads).{hint}"
            )
        if name not in seen:
            seen.add(name)
            selected.append(spec)
    return selected, True


def _validate_isa_name(raw_isa: str) -> str:
    name = raw_isa.strip().lower()
    if name not in ISA_NAME_TO_ID:
        matches = difflib.get_close_matches(name, ALL_ISA_NAMES, n=1, cutoff=0.5)
        hint = f" Did you mean '{matches[0]}'?" if matches else ""
        valid = ", ".join(ALL_ISA_NAMES)
        raise CLIUsageError(f"Unknown --isa value {raw_isa!r} (valid: {valid}).{hint}")
    return name


class _CrossCPUArgumentParser(argparse.ArgumentParser):
    """ArgumentParser subclass raising CLIUsageError (exit code 1) instead of exit(2)."""

    def error(self, message: str) -> None:  # type: ignore[override]
        hint = ""
        m = re.search(r"invalid choice:\s*'([^']+)'", message)
        if m:
            bad = m.group(1)
            matches = difflib.get_close_matches(bad, SUBCOMMANDS, n=1, cutoff=0.45)
            if matches:
                hint = f" Did you mean '{matches[0]}'?"
        raise CLIUsageError(f"{message}{hint}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the CrossCPU argument parser with concrete examples."""
    root_examples = (
        "Examples:\n"
        "  cross-cpu list\n"
        "  cross-cpu info --json\n"
        "  cross-cpu run --test compute-fp32,memory-pattern --duration 500ms\n"
        "  cross-cpu run --cores p-cores --isa auto --warmup 5s --duration 60s\n"
        "  cross-cpu run --test all --duration 10s --format json\n"
    )
    parser = _CrossCPUArgumentParser(
        prog="cross-cpu",
        description="CrossCPU — Cross-Platform CPU Stress & Silent Data Corruption Validation Suite.",
        epilog=root_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"cross-cpu {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # list
    p_list = subparsers.add_parser(
        "list",
        help="List all canonical workloads, fixed units, and host ISA readiness",
        epilog=(
            "Examples:\n"
            "  cross-cpu list\n"
            "  cross-cpu list --json\n"
            "  cross-cpu list -v\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_output_flags(p_list)

    # info
    p_info = subparsers.add_parser(
        "info",
        help="Inspect CPU topology, cache hierarchy, NUMA nodes, and detected ISA backends",
        epilog=(
            "Examples:\n"
            "  cross-cpu info\n"
            "  cross-cpu info --json\n"
            "  cross-cpu info -v\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_output_flags(p_info)

    # run
    p_run = subparsers.add_parser(
        "run",
        help="Execute CPU stress and Silent Data Corruption validation workloads",
        epilog=(
            "Examples:\n"
            "  cross-cpu run --test compute-fp32 --duration 500ms\n"
            "  cross-cpu run --test compute-fp32,memory-pattern --duration 250ms --json\n"
            "  cross-cpu run --cores physical --numa 0 --warmup --duration 30s\n"
            "  cross-cpu run --cores p-cores --isa auto --test all --duration 1m\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_run.add_argument(
        "--cores",
        default="all",
        metavar="SPEC",
        help="Core selection: all, physical, p-cores, e-cores, or CPU list (e.g. 0-3,6) [default: all]",
    )
    p_run.add_argument(
        "--numa",
        default="all",
        metavar="NODE",
        help="NUMA node filter: all or integer node ID [default: all]",
    )
    p_run.add_argument(
        "--isa",
        default="auto",
        metavar="ISA",
        help="ISA backend: auto or explicit name (amx, avx512, avx2, sse4, sme, sve, neon, rvv, vsx, vx, scalar)",
    )
    p_run.add_argument(
        "--test",
        action="append",
        default=[],
        metavar="TEST",
        help="Workload name(s) (comma-separated or repeated) or 'all' [default: all]",
    )
    p_run.add_argument(
        "--duration",
        default="10s",
        metavar="TIME",
        help="Timed measurement duration per workload (e.g. 250ms, 1.5s, 5m, 1h) [default: 10s]",
    )
    p_run.add_argument(
        "--warmup",
        nargs="?",
        const="5s",
        default=None,
        metavar="TIME",
        help="Optional warmup duration before timing (bare --warmup defaults to 5s; omitted is 0s)",
    )
    p_run.add_argument(
        "--no-banner",
        action="store_true",
        default=False,
        help="Suppress the interactive ASCII banner",
    )
    _add_common_output_flags(p_run)

    return parser


def _add_common_output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default=None,
        help="Output format (text or json) [default: text]",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Shorthand for --format json (conflicts with --format text)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Emit detailed architecture probe, affinity, and ISA dispatch diagnostics to stderr",
    )


def _format_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "none"
    if num_bytes >= 1024 * 1024 * 1024 and num_bytes % (1024 * 1024 * 1024) == 0:
        return f"{num_bytes // (1024 * 1024 * 1024)} GiB"
    if num_bytes >= 1024 * 1024 and num_bytes % (1024 * 1024) == 0:
        return f"{num_bytes // (1024 * 1024)} MiB"
    if num_bytes >= 1024 and num_bytes % 1024 == 0:
        return f"{num_bytes // 1024} KiB"
    return f"{num_bytes} B"


def _emit_verbose_probe(
    caps: NativeCapabilities,
    topology: CPUTopology,
    stderr: TextIO,
) -> None:
    stderr.write(
        f"[verbose] arch={caps.arch_name} endian={caps.endianness} "
        f"abi={caps.abi_version} os={topology.os_name}\n"
    )
    stderr.write(f"[verbose] probe: {caps.probe_summary}\n")
    raw_hex = ", ".join(hex(x) for x in caps.raw_probe)
    stderr.write(f"[verbose] raw_probe=[{raw_hex}] feature_flags={hex(caps.feature_flags)}\n")
    for b in caps.backends:
        stderr.write(
            f"[verbose] backend={b.name} priority={b.priority} "
            f"available={b.available} compiled={b.compiled_in} "
            f"mask={hex(b.workload_mask)} ({b.detail})\n"
        )
    stderr.flush()


def _cmd_list(
    native: NativeLibrary,
    caps: NativeCapabilities,
    topology: CPUTopology,
    *,
    output_format: str,
    verbose: bool,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    if verbose and output_format != "json":
        _emit_verbose_probe(caps, topology, stderr)

    rows: list[dict[str, Any]] = []
    for spec in WORKLOAD_SPECS:
        status, isa_name, reason = native.resolve_isa(spec.workload_id, "auto")
        # Check cache topology availability for cache-l3
        from .workloads import compute_workload_buffer_bytes

        _, buf_reason = compute_workload_buffer_bytes(spec, topology, topology.logical_cores)
        if buf_reason is not None:
            supported = False
            isa_name = "none"
            reason = buf_reason
        else:
            supported = status == 0

        rows.append(
            {
                "name": spec.name,
                "category": spec.category,
                "rate_unit": spec.rate_unit,
                "total_unit": spec.total_unit,
                "hardware_only": spec.hardware_only,
                "supported": supported,
                "auto_isa": isa_name if supported else "none",
                "reason": "" if supported else reason,
                "description": spec.description,
            }
        )

    if output_format == "json":
        payload = {
            "command": "list",
            "version": __version__,
            "arch": caps.arch_name,
            "workloads": rows,
        }
        stdout.write(json.dumps(payload, indent=2) + "\n")
        return int(ExitCode.SUCCESS)

    stdout.write(
        f"{'WORKLOAD':<22} {'CATEGORY':<15} {'UNITS':<14} {'AUTO ISA':<10} {'STATUS':<12} DESCRIPTION\n"
    )
    stdout.write("-" * 105 + "\n")
    for r in rows:
        units = f"{r['rate_unit']} / {r['total_unit']}"
        st = "ready" if r["supported"] else "unsupported"
        stdout.write(
            f"{r['name']:<22} {r['category']:<15} {units:<14} "
            f"{r['auto_isa']:<10} {st:<12} {r['description']}\n"
        )
    return int(ExitCode.SUCCESS)


def _cmd_info(
    caps: NativeCapabilities,
    topology: CPUTopology,
    *,
    output_format: str,
    verbose: bool,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    if verbose and output_format != "json":
        _emit_verbose_probe(caps, topology, stderr)

    if output_format == "json":
        payload = {
            "command": "info",
            "version": __version__,
            "topology": topology.to_dict(),
            "native": caps.to_dict(),
        }
        stdout.write(json.dumps(payload, indent=2) + "\n")
        return int(ExitCode.SUCCESS)

    stdout.write(f"CrossCPU v{__version__} Host Information\n")
    stdout.write("=" * 60 + "\n")
    stdout.write(f"OS / Machine     : {topology.os_name} ({topology.machine})\n")
    stdout.write(f"CPU Model        : {topology.model_name}\n")
    stdout.write(f"Arch / Endian    : {caps.arch_name} ({caps.endianness}-endian)\n")
    stdout.write(
        f"Sockets / Cores  : {topology.sockets} socket(s), "
        f"{topology.physical_cores} physical, {topology.logical_cores} logical\n"
    )
    stdout.write(f"NUMA Nodes       : {', '.join(str(n) for n in topology.numa_nodes)}\n")
    if topology.is_hybrid:
        stdout.write(
            f"Hybrid Clusters  : {len(topology.p_core_ids)} P-core(s), "
            f"{len(topology.e_core_ids)} E-core(s)\n"
        )
    else:
        stdout.write("Hybrid Clusters  : homogeneous\n")
    if topology.memory_bytes > 0:
        stdout.write(f"System Memory    : {_format_bytes(topology.memory_bytes)}\n")
    caches = topology.caches
    stdout.write(
        f"Cache Hierarchy  : L1i={_format_bytes(caches.l1i_bytes)}, "
        f"L1d={_format_bytes(caches.l1d_bytes)}, "
        f"L2={_format_bytes(caches.l2_bytes)}, "
        f"L3={_format_bytes(caches.l3_bytes)} (line={caches.line_size_bytes}B)\n"
    )
    stdout.write(f"Thread Placement : {topology.pinning_note}\n")
    stdout.write(f"ISA Probe        : {caps.probe_summary}\n")
    stdout.write("\nRegistered ISA Backends (priority order):\n")
    for b in caps.backends:
        state = "available" if b.available else ("compiled-only" if b.compiled_in else "not-compiled")
        stdout.write(
            f"  - {b.name:<8} priority={b.priority:<3} status={state:<13} {b.detail}\n"
        )
    return int(ExitCode.SUCCESS)


def _status_badge(status: str, use_color: bool) -> str:
    label = status.upper()
    if not use_color:
        return label
    colors = {
        "passed": "\033[32;1m",
        "failed": "\033[31;1m",
        "unsupported": "\033[33m",
        "init_error": "\033[31m",
        "stopped": "\033[36m",
    }
    prefix = colors.get(status, "")
    return f"{prefix}{label}\033[0m" if prefix else label


def _cmd_run(
    args: argparse.Namespace,
    native: NativeLibrary,
    caps: NativeCapabilities,
    topology: CPUTopology,
    *,
    output_format: str,
    verbose: bool,
    stdout: TextIO,
    stderr: TextIO,
    inject_fault_workloads: set[str] | None = None,
) -> int:
    json_mode = output_format == "json"
    duration_s = parse_duration(args.duration, allow_zero=False, flag_name="--duration")
    warmup_s = (
        parse_duration(args.warmup, allow_zero=True, flag_name="--warmup")
        if args.warmup is not None
        else 0.0
    )
    requested_isa = _validate_isa_name(args.isa)
    workloads, explicit_tests = _parse_test_selection(args.test)

    try:
        core_ids, qos_hint, core_notes = select_cores(topology, args.cores, args.numa)
    except TopologyUsageError as exc:
        raise CLIUsageError(str(exc)) from exc
    except TopologyUnavailableError as exc:
        if not json_mode:
            stderr.write(f"Error: {exc}\n")
        return int(ExitCode.INIT_ERROR)

    if verbose and not json_mode:
        _emit_verbose_probe(caps, topology, stderr)
        stderr.write(
            f"[verbose] cores={core_ids} qos_hint={qos_hint} "
            f"requested_isa={requested_isa} duration={duration_s}s warmup={warmup_s}s\n"
        )
        for note in core_notes:
            stderr.write(f"[verbose] placement: {note}\n")
        stderr.flush()

    use_color = supports_color(stdout)
    is_tty = False
    try:
        is_tty = bool(stdout.isatty())
    except Exception:
        is_tty = False

    if not json_mode:
        maybe_print_banner(stdout, no_banner=args.no_banner, json_mode=False)
        stdout.write(
            f"Host: {topology.model_name} ({caps.arch_name}, {topology.os_name}) | "
            f"Threads: {len(core_ids)} | ISA: {requested_isa} | "
            f"Duration: {args.duration}"
            + (f" | Warmup: {args.warmup}" if warmup_s > 0.0 else "")
            + "\n\n"
        )
        stdout.write(
            f"{'WORKLOAD':<22} {'ISA':<8} {'STATUS':<12} {'RATE':>16} {'TOTAL':>16}  DETAIL\n"
        )
        stdout.write("-" * 92 + "\n")
        stdout.flush()

    progress_active = [False]

    def _clear_progress() -> None:
        if progress_active[0] and is_tty and not json_mode:
            stdout.write("\r\033[K")
            stdout.flush()
            progress_active[0] = False

    def _on_progress(
        idx: int,
        total: int,
        label: str,
        elapsed_s: float,
        target_s: float,
    ) -> None:
        if json_mode or not is_tty:
            return
        prog = format_duration_progress(elapsed_s, target_s)
        line = f"[{idx:02d}/{total:02d} {label} | {prog}]"
        stdout.write(f"\r\033[K{line}")
        stdout.flush()
        progress_active[0] = True

    def _on_result(_idx: int, _total: int, res: WorkloadResult) -> None:
        if json_mode:
            return
        if res.status in ("unsupported", "init_error", "failed") and res.reason:
            stderr.write(f"[{res.status}] {res.name}: {res.reason}\n")
            stderr.flush()

        _clear_progress()
        badge = _status_badge(res.status, use_color)
        if res.status in ("passed", "failed", "stopped"):
            rate_str = f"{res.rate:.2f} {res.rate_unit}"
            total_str = f"{res.total_work:.3f} {res.total_unit}"
        else:
            rate_str = "-"
            total_str = "-"
        detail = res.reason if res.reason else f"{res.threads}T / {res.iterations} batches"
        # Pad raw status width cleanly when ANSI escapes are active
        pad_extra = len(badge) - len(res.status)
        stdout.write(
            f"{res.name:<22} {res.isa:<8} {badge:<{12 + pad_extra}} "
            f"{rate_str:>16} {total_str:>16}  {detail}\n"
        )
        stdout.flush()

    def _on_diag(msg: str) -> None:
        if verbose and not json_mode:
            stderr.write(f"[verbose] {msg}\n")
            stderr.flush()

    runner = WorkloadRunner(
        native=native,
        topology=topology,
        caps=caps,
    )
    report: RunReport = runner.run(
        workloads,
        requested_isa=requested_isa,
        core_ids=core_ids,
        qos_class=qos_hint,
        duration_s=duration_s,
        warmup_s=warmup_s,
        explicit_tests=explicit_tests,
        inject_fault_workloads=inject_fault_workloads,
        on_progress=_on_progress,
        on_result=_on_result,
        on_diag=_on_diag,
    )

    _clear_progress()

    if report.hardware_fault_reason and not json_mode:
        stderr.write(f"[hardware_error] {report.hardware_fault_reason}\n")
        stderr.flush()

    counts = {
        "passed": sum(1 for r in report.results if r.status == "passed"),
        "failed": sum(1 for r in report.results if r.status == "failed"),
        "unsupported": sum(1 for r in report.results if r.status == "unsupported"),
        "init_error": sum(1 for r in report.results if r.status == "init_error"),
        "stopped": sum(1 for r in report.results if r.status == "stopped"),
    }

    if json_mode:
        payload: dict[str, Any] = {
            "command": "run",
            "version": __version__,
            "host": {
                "os": topology.os_name,
                "machine": topology.machine,
                "model_name": topology.model_name,
                "arch": caps.arch_name,
                "endianness": caps.endianness,
            },
            "config": {
                "cores": args.cores,
                "selected_core_ids": core_ids,
                "numa": args.numa,
                "requested_isa": requested_isa,
                "duration_s": duration_s,
                "warmup_s": warmup_s,
            },
            "results": [r.to_dict() for r in report.results],
            "summary": {
                **counts,
                "interrupted": report.interrupted,
                "exit_code": int(report.exit_code),
                "exit_name": report.exit_code.name,
            },
        }
        if report.hardware_fault_reason:
            payload["summary"]["hardware_fault_reason"] = report.hardware_fault_reason
        stdout.write(json.dumps(payload, indent=2) + "\n")
        return int(report.exit_code)

    stdout.write("-" * 92 + "\n")
    summary_parts = [
        f"{counts['passed']} passed",
        f"{counts['failed']} failed",
        f"{counts['unsupported']} unsupported",
    ]
    if counts["init_error"]:
        summary_parts.append(f"{counts['init_error']} init_error")
    if counts["stopped"]:
        summary_parts.append(f"{counts['stopped']} stopped")
    stdout.write(
        f"Summary: {', '.join(summary_parts)} | "
        f"Exit: {report.exit_code.name} ({int(report.exit_code)})\n"
    )
    stdout.flush()
    return int(report.exit_code)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    native_library: NativeLibrary | None = None,
    topology_override: CPUTopology | None = None,
    inject_fault_workloads: set[str] | None = None,
) -> int:
    """Main CLI entry point returning a severity-ordered ExitCode integer (0..4)."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    if not raw_argv:
        parser.print_help(file=out)
        return int(ExitCode.USAGE_ERROR)

    # Intercept --help / --version cleanly while letting custom stdout/stderr work in tests
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = out
        sys.stderr = err
        try:
            args = parser.parse_args(raw_argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0
            return int(ExitCode.SUCCESS) if code == 0 else int(ExitCode.USAGE_ERROR)
        except CLIUsageError as exc:
            if not _has_json_flag(raw_argv):
                err.write(f"Usage error: {exc}\n")
            return int(ExitCode.USAGE_ERROR)

        if not args.command:
            parser.print_help(file=out)
            return int(ExitCode.USAGE_ERROR)

        try:
            output_format = _resolve_output_format(args.format, args.json)
        except CLIUsageError as exc:
            err.write(f"Usage error: {exc}\n")
            return int(ExitCode.USAGE_ERROR)

        json_mode = output_format == "json"

        # Validate run syntax before loading native library so bad CLI args always exit 1
        if args.command == "run":
            try:
                parse_duration(args.duration, allow_zero=False, flag_name="--duration")
                if args.warmup is not None:
                    parse_duration(args.warmup, allow_zero=True, flag_name="--warmup")
                _validate_isa_name(args.isa)
                _parse_test_selection(args.test)
            except CLIUsageError as exc:
                if not json_mode:
                    err.write(f"Usage error: {exc}\n")
                return int(ExitCode.USAGE_ERROR)

        try:
            native = native_library if native_library is not None else NativeLibrary()
            caps = native.get_caps()
        except NativeBridgeError as exc:
            if not json_mode:
                err.write(f"Initialization error: {exc}\n")
            return int(ExitCode.INIT_ERROR)

        topology = topology_override if topology_override is not None else discover_topology()

        try:
            if args.command == "list":
                return _cmd_list(
                    native,
                    caps,
                    topology,
                    output_format=output_format,
                    verbose=args.verbose,
                    stdout=out,
                    stderr=err,
                )
            if args.command == "info":
                return _cmd_info(
                    caps,
                    topology,
                    output_format=output_format,
                    verbose=args.verbose,
                    stdout=out,
                    stderr=err,
                )
            if args.command == "run":
                return _cmd_run(
                    args,
                    native,
                    caps,
                    topology,
                    output_format=output_format,
                    verbose=args.verbose,
                    stdout=out,
                    stderr=err,
                    inject_fault_workloads=inject_fault_workloads,
                )
        except CLIUsageError as exc:
            if not json_mode:
                err.write(f"Usage error: {exc}\n")
            return int(ExitCode.USAGE_ERROR)

        if not json_mode:
            err.write(f"Usage error: unknown command {args.command!r}\n")
        return int(ExitCode.USAGE_ERROR)
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
