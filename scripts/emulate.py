#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Local multi-environment emulation runner for CrossCPU.

Mirrors `.github/workflows/emulation.yml` on a developer's host machine (macOS or Linux),
plus local host native and macOS Apple Silicon Rosetta 2 (`x86_64`) emulation without
requiring Docker.

Zero external dependencies (Python 3.9+ standard library only).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

EXTRA_BIN_DIRS = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/opt/local/bin",
)

SIGNAL_NAMES = {
    132: "SIGILL",
    134: "SIGABRT",
    136: "SIGFPE",
    137: "SIGKILL",
    139: "SIGSEGV",
    143: "SIGTERM",
}

SUITE_TARGETS: dict[str, tuple[str, ...]] = {
    "host": ("native", "rosetta"),
    "isa": ("x86-isa", "arm64-isa"),
    "foreign-arch": ("arch-arm64", "arch-riscv64", "arch-ppc64le", "arch-s390x"),
    "distros": ("distro-alpine", "distro-debian", "distro-fedora", "distro-arch"),
    "adversarial": ("adversarial",),
}

ALL_TARGETS: tuple[str, ...] = (
    "native",
    "rosetta",
    "x86-isa",
    "arm64-isa",
    "arch-arm64",
    "arch-riscv64",
    "arch-ppc64le",
    "arch-s390x",
    "distro-alpine",
    "distro-debian",
    "distro-fedora",
    "distro-arch",
    "adversarial",
)

TARGET_DESCRIPTIONS: dict[str, str] = {
    "native": "Host native C11 build, 39 unit tests, verbose probe, JSON + CLI smoke",
    "rosetta": "macOS Apple Silicon Rosetta 2 (x86_64) build, 39 unit tests, SSE4/AES + negative ISA gates",
    "x86-isa": "x86_64 QEMU user-mode ISA tiers (qemu64/Nehalem/Haswell/max) + negative exit-2 gates",
    "arm64-isa": "ARM64 QEMU user-mode ISA tiers (cortex-a53/neon_only/sve_on) + negative exit-2 gates",
    "arch-arm64": "Container linux/arm64 (debian:trixie-slim) build, unit tests, info & run checks",
    "arch-riscv64": "Container linux/riscv64 (debian:trixie-slim) build, unit tests, info & run checks",
    "arch-ppc64le": "Container linux/ppc64le (debian:trixie-slim) build, unit tests, info & run checks",
    "arch-s390x": "Container linux/s390x (debian:trixie-slim) big-endian build, unit tests, info & run checks",
    "distro-alpine": "Alpine Linux (musl) GCC + Clang builds, unit tests, offline wheel verification",
    "distro-debian": "Debian Trixie hardened CFLAGS GCC + Clang builds, unit tests, offline wheel",
    "distro-fedora": "Fedora latest (redhat-rpm-config) GCC + Clang builds, unit tests, offline wheel",
    "distro-arch": "Arch Linux rolling (pacman --disable-sandbox) GCC + Clang builds, tests, wheel",
    "adversarial": "Offline, read-only, UID 65534, non-contiguous cpuset python:3.12-slim container",
}


def decode_rc(rc: int) -> str:
    if rc in SIGNAL_NAMES:
        return f"{rc} ({SIGNAL_NAMES[rc]})"
    if 128 < rc < 192:
        return f"{rc} (SIG{rc - 128})"
    return str(rc)


def build_augmented_path() -> str:
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    for extra in EXTRA_BIN_DIRS:
        if extra not in parts and Path(extra).is_dir():
            parts.append(extra)
    return os.pathsep.join(parts)


def which_tool(name: str, env_path: str) -> str | None:
    return shutil.which(name, path=env_path)


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _color(text: str, code: str, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


@dataclass
class HostCapabilities:
    os_name: str
    machine: str
    env_path: str
    make_bin: str | None
    cc_bin: str | None
    host_python: str
    host_python_version: str
    rosetta_available: bool
    rosetta_python: str | None
    qemu_x86_64: str | None
    qemu_aarch64: str | None
    container_runtime: str | None  # "docker" or "podman" binary path
    container_runtime_name: str | None  # "docker" or "podman"
    container_daemon_ready: bool
    container_daemon_error: str
    container_ncpu: int
    container_arch: str
    colima_bin: str | None
    colima_running: bool


@dataclass
class CommandRecord:
    name: str
    cmd: str
    rc: int
    expected_rc: int
    stdout_path: Path
    stderr_path: Path
    elapsed_s: float

    @property
    def passed(self) -> bool:
        return self.rc == self.expected_rc


@dataclass
class TargetResult:
    name: str
    status: str  # "passed", "failed", "skipped"
    elapsed_s: float = 0.0
    detail: str = ""
    records: list[CommandRecord] = field(default_factory=list)


def _probe_python_version(py_bin: str, prefix: Sequence[str] = ()) -> tuple[int, int, int] | None:
    try:
        proc = subprocess.run(
            [*prefix, py_bin, "-c", "import sys; print(*sys.version_info[:3])"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0:
            return None
        nums = proc.stdout.strip().split()
        if len(nums) != 3:
            return None
        return (int(nums[0]), int(nums[1]), int(nums[2]))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _select_host_python(env_path: str) -> tuple[str, str]:
    candidates: list[str] = []
    for candidate in (
        shutil.which("python3", path=env_path),
        "/usr/local/bin/python3",
        "/opt/homebrew/bin/python3",
        sys.executable,
        "/usr/bin/python3",
    ):
        if candidate and Path(candidate).is_file() and candidate not in candidates:
            candidates.append(candidate)

    best_bin = sys.executable
    best_ver = (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    for cand in candidates:
        ver = _probe_python_version(cand)
        if ver is not None and ver > best_ver:
            best_bin = cand
            best_ver = ver
    return best_bin, ".".join(str(x) for x in best_ver)


def _select_rosetta_python(env_path: str) -> tuple[bool, str | None]:
    if platform.system() != "Darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        return False, None
    arch_bin = which_tool("arch", env_path)
    if not arch_bin:
        return False, None
    try:
        probe = subprocess.run(
            [arch_bin, "-x86_64", "/usr/bin/uname", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if probe.returncode != 0 or probe.stdout.strip() != "x86_64":
            return False, None
    except (OSError, subprocess.SubprocessError):
        return False, None

    candidates: list[str] = []
    for cand in (
        "/usr/local/bin/python3",
        shutil.which("python3", path=env_path),
        sys.executable,
        "/usr/bin/python3",
    ):
        if cand and Path(cand).is_file() and cand not in candidates:
            candidates.append(cand)

    best_py: str | None = None
    best_ver: tuple[int, int, int] = (0, 0, 0)
    for cand in candidates:
        ver = _probe_python_version(cand, prefix=(arch_bin, "-x86_64"))
        if ver is not None and ver > best_ver:
            best_py = cand
            best_ver = ver
    return True, best_py


def _probe_container_runtime(
    requested: str,
    env_path: str,
) -> tuple[str | None, str | None, bool, str, int, str]:
    env = dict(os.environ, PATH=env_path)
    order = ("docker", "podman") if requested == "auto" else (requested,)
    chosen_bin: str | None = None
    chosen_name: str | None = None
    last_err = "no container runtime (docker/podman) found on PATH"

    for name in order:
        bin_path = which_tool(name, env_path)
        if not bin_path:
            continue
        chosen_bin = bin_path
        chosen_name = name
        try:
            proc = subprocess.run(
                [bin_path, "info", "--format", "{{.NCPU}}|{{.Architecture}}"],
                capture_output=True,
                text=True,
                env=env,
                timeout=15,
                check=False,
            )
            if proc.returncode == 0:
                parts = proc.stdout.strip().split("|")
                ncpu = int(parts[0]) if parts and parts[0].isdigit() else 2
                arch = parts[1].strip() if len(parts) > 1 and parts[1].strip() else platform.machine()
                return bin_path, name, True, "", ncpu, arch
            err_msg = (proc.stderr or proc.stdout).strip().splitlines()
            last_err = err_msg[-1] if err_msg else f"{name} info exited {proc.returncode}"
        except (OSError, subprocess.SubprocessError) as exc:
            last_err = str(exc)

    return chosen_bin, chosen_name, False, last_err, 0, ""


def detect_host_capabilities(runtime_choice: str = "auto") -> HostCapabilities:
    env_path = build_augmented_path()
    os_name = platform.system()
    machine = platform.machine().lower()
    host_py, host_py_ver = _select_host_python(env_path)
    rosetta_ok, rosetta_py = _select_rosetta_python(env_path)
    rt_bin, rt_name, rt_ready, rt_err, rt_ncpu, rt_arch = _probe_container_runtime(
        runtime_choice, env_path
    )

    colima_bin = which_tool("colima", env_path)
    colima_running = False
    if colima_bin:
        try:
            cp = subprocess.run(
                [colima_bin, "status"],
                capture_output=True,
                text=True,
                env=dict(os.environ, PATH=env_path),
                timeout=10,
                check=False,
            )
            colima_running = cp.returncode == 0
        except (OSError, subprocess.SubprocessError):
            colima_running = False

    return HostCapabilities(
        os_name=os_name,
        machine=machine,
        env_path=env_path,
        make_bin=which_tool("make", env_path),
        cc_bin=which_tool("cc", env_path) or which_tool("clang", env_path) or which_tool("gcc", env_path),
        host_python=host_py,
        host_python_version=host_py_ver,
        rosetta_available=rosetta_ok,
        rosetta_python=rosetta_py,
        qemu_x86_64=which_tool("qemu-x86_64", env_path),
        qemu_aarch64=which_tool("qemu-aarch64", env_path),
        container_runtime=rt_bin,
        container_runtime_name=rt_name,
        container_daemon_ready=rt_ready,
        container_daemon_error=rt_err,
        container_ncpu=rt_ncpu,
        container_arch=rt_arch,
        colima_bin=colima_bin,
        colima_running=colima_running,
    )


def target_readiness(target: str, caps: HostCapabilities) -> tuple[bool, str]:
    if target == "native":
        if not caps.make_bin or not caps.cc_bin:
            return False, "missing host C compiler or make"
        return True, f"host {caps.os_name}/{caps.machine} ({caps.host_python} {caps.host_python_version})"

    if target == "rosetta":
        if caps.os_name != "Darwin" or caps.machine not in ("arm64", "aarch64"):
            return False, "requires Apple Silicon macOS host"
        if not caps.rosetta_available:
            return False, "Rosetta 2 not installed (run: softwareupdate --install-rosetta)"
        if not caps.rosetta_python:
            return False, "no x86_64/universal2 python3 found for Rosetta 2"
        if not caps.make_bin or not caps.cc_bin:
            return False, "missing Xcode Command Line Tools (cc/make)"
        return True, f"Rosetta 2 x86_64 ({caps.rosetta_python})"

    if target == "x86-isa":
        if caps.os_name == "Linux" and caps.machine in ("x86_64", "amd64") and caps.qemu_x86_64:
            return True, f"native Linux {caps.qemu_x86_64}"
        if caps.container_daemon_ready:
            return True, f"via {caps.container_runtime_name} (linux/amd64 + qemu-x86_64)"
        return False, _container_hint(caps)

    if target == "arm64-isa":
        if caps.os_name == "Linux" and caps.machine in ("arm64", "aarch64") and caps.qemu_aarch64:
            return True, f"native Linux {caps.qemu_aarch64}"
        if caps.container_daemon_ready:
            return True, f"via {caps.container_runtime_name} (linux/arm64 + qemu-aarch64)"
        return False, _container_hint(caps)

    if target in (
        "arch-arm64",
        "arch-riscv64",
        "arch-ppc64le",
        "arch-s390x",
        "distro-alpine",
        "distro-debian",
        "distro-fedora",
        "distro-arch",
        "adversarial",
    ):
        if caps.container_daemon_ready:
            return True, f"via {caps.container_runtime_name} ({caps.container_arch}, {caps.container_ncpu} vCPU)"
        return False, _container_hint(caps)

    return False, f"unknown target {target!r}"


def _container_hint(caps: HostCapabilities) -> str:
    if caps.colima_bin and not caps.colima_running:
        if caps.os_name == "Darwin" and caps.machine in ("arm64", "aarch64"):
            return "colima stopped — start with: colima start --cpu 4 --memory 4 --vm-type vz --vz-rosetta"
        return "colima stopped — start with: colima start --cpu 4 --memory 4"
    if caps.container_runtime_name:
        return f"{caps.container_runtime_name} daemon unavailable ({caps.container_daemon_error})"
    return "requires Docker or Podman (not found on PATH)"


def copy_repo_clean(dst: Path) -> None:
    ignore = shutil.ignore_patterns(
        ".git",
        "build",
        "dist",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.o",
        "*.so",
        "*.dylib",
        "*.egg-info",
        ".DS_Store",
        ".venv",
        "venv",
    )
    shutil.copytree(REPO_ROOT, dst, ignore=ignore, symlinks=True)


class TargetLogger:
    """Per-target command runner with structured logs, signal decoding, and JSON fallback."""

    def __init__(
        self,
        target: str,
        log_dir: Path,
        *,
        env_path: str,
        verbose: bool = False,
        use_color: bool = False,
    ) -> None:
        self.target = target
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.env_path = env_path
        self.verbose = verbose
        self.use_color = use_color
        self.records: list[CommandRecord] = []

    def run_expect(
        self,
        expected_rc: int,
        name: str,
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        env_extra: Mapping[str, str] | None = None,
        stdin_text: str | None = None,
        timeout_s: float = 1800.0,
    ) -> CommandRecord:
        out_path = self.log_dir / f"{name}.stdout.log"
        err_path = self.log_dir / f"{name}.stderr.log"
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        (self.log_dir / f"{name}.cmd").write_text(cmd_str + "\n", encoding="utf-8")
        (self.log_dir / f"{name}.expected_rc").write_text(f"{expected_rc}\n", encoding="utf-8")

        run_env = dict(os.environ)
        run_env["PATH"] = self.env_path
        if env_extra:
            run_env.update(env_extra)

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                list(cmd),
                cwd=str(cwd) if cwd else None,
                env=run_env,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            rc = proc.returncode
            stdout_txt = proc.stdout or ""
            stderr_txt = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            rc = 124
            stdout_txt = (exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
            stderr_txt = (
                (exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
                + f"\n[TIMEOUT] Command exceeded {timeout_s:.0f}s\n"
            )
        elapsed = time.monotonic() - t0

        out_path.write_text(stdout_txt, encoding="utf-8")
        err_path.write_text(stderr_txt, encoding="utf-8")
        (self.log_dir / f"{name}.rc").write_text(f"{rc}\n", encoding="utf-8")

        rec = CommandRecord(
            name=name,
            cmd=cmd_str,
            rc=rc,
            expected_rc=expected_rc,
            stdout_path=out_path,
            stderr_path=err_path,
            elapsed_s=elapsed,
        )
        self.records.append(rec)

        if rec.passed:
            badge = _color("PASS", "32;1", self.use_color)
            print(f"  [{badge}] {self.target}/{name} (rc={rc}, {elapsed:.2f}s)")
            if self.verbose:
                if stdout_txt.strip():
                    print(f"    --- stdout ({name}) ---\n{stdout_txt.rstrip()}")
                if stderr_txt.strip():
                    print(f"    --- stderr ({name}) ---\n{stderr_txt.rstrip()}")
        else:
            badge = _color("FAIL", "31;1", self.use_color)
            print(f"  [{badge}] {self.target}/{name} (rc={decode_rc(rc)}, expected={expected_rc}, {elapsed:.2f}s)")
            print(f"    Command : {cmd_str}")
            print(f"    --- stdout ({out_path}) ---")
            print(_tail_text(stdout_txt, 80) or "(empty)")
            print(f"    --- stderr ({err_path}) ---")
            print(_tail_text(stderr_txt, 80) or "(empty)")
        sys.stdout.flush()
        return rec

    def run(
        self,
        name: str,
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        env_extra: Mapping[str, str] | None = None,
        stdin_text: str | None = None,
        timeout_s: float = 1800.0,
    ) -> CommandRecord:
        return self.run_expect(
            0,
            name,
            cmd,
            cwd=cwd,
            env_extra=env_extra,
            stdin_text=stdin_text,
            timeout_s=timeout_s,
        )

    def run_cli_json_expect(
        self,
        expected_rc: int,
        name: str,
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        env_extra: Mapping[str, str] | None = None,
    ) -> CommandRecord:
        rec = self.run_expect(expected_rc, name, cmd, cwd=cwd, env_extra=env_extra)
        if not rec.passed:
            out_empty = rec.stdout_path.stat().st_size == 0
            err_empty = rec.stderr_path.stat().st_size == 0
            if out_empty and err_empty:
                diag_cmd = [tok for tok in cmd if tok != "--json"] + ["-v"]
                print(f"    >>> Silent --json failure detected; running verbose diagnostic retry for {name}:")
                self.run_expect(0, f"{name}_diag_verbose", diag_cmd, cwd=cwd, env_extra=env_extra)
        return rec

    def record_assertion(self, name: str, fn: Callable[[], str]) -> CommandRecord:
        out_path = self.log_dir / f"{name}.stdout.log"
        err_path = self.log_dir / f"{name}.stderr.log"
        (self.log_dir / f"{name}.cmd").write_text(f"<python assertion: {name}>\n", encoding="utf-8")
        (self.log_dir / f"{name}.expected_rc").write_text("0\n", encoding="utf-8")
        t0 = time.monotonic()
        try:
            msg = fn()
            rc = 0
            out_path.write_text(msg + "\n", encoding="utf-8")
            err_path.write_text("", encoding="utf-8")
        except Exception as exc:
            rc = 1
            out_path.write_text("", encoding="utf-8")
            err_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            msg = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - t0
        (self.log_dir / f"{name}.rc").write_text(f"{rc}\n", encoding="utf-8")
        rec = CommandRecord(
            name=name,
            cmd=f"<assertion: {name}>",
            rc=rc,
            expected_rc=0,
            stdout_path=out_path,
            stderr_path=err_path,
            elapsed_s=elapsed,
        )
        self.records.append(rec)
        if rec.passed:
            badge = _color("PASS", "32;1", self.use_color)
            print(f"  [{badge}] {self.target}/{name}: {msg}")
        else:
            badge = _color("FAIL", "31;1", self.use_color)
            print(f"  [{badge}] {self.target}/{name}: {msg}")
        sys.stdout.flush()
        return rec

    def harvest_container_logs(self) -> None:
        """Import step records written inside a container mounted to self.log_dir."""
        known = {r.name for r in self.records}
        for rc_file in sorted(self.log_dir.glob("*.rc")):
            name = rc_file.stem
            if name in known:
                continue
            try:
                rc = int(rc_file.read_text(encoding="utf-8").strip())
            except ValueError:
                rc = 1
            exp_file = self.log_dir / f"{name}.expected_rc"
            try:
                exp = int(exp_file.read_text(encoding="utf-8").strip()) if exp_file.is_file() else 0
            except ValueError:
                exp = 0
            cmd_file = self.log_dir / f"{name}.cmd"
            cmd_txt = cmd_file.read_text(encoding="utf-8").strip() if cmd_file.is_file() else "(container step)"
            self.records.append(
                CommandRecord(
                    name=name,
                    cmd=cmd_txt,
                    rc=rc,
                    expected_rc=exp,
                    stdout_path=self.log_dir / f"{name}.stdout.log",
                    stderr_path=self.log_dir / f"{name}.stderr.log",
                    elapsed_s=0.0,
                )
            )


def _tail_text(text: str, max_lines: int = 80) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return f"... ({len(lines) - max_lines} earlier lines omitted) ...\n" + "\n".join(lines[-max_lines:])


# Shared container helper prelude (bash) matching .github/workflows/emulation.yml
_BASH_LOG_HELPERS = r"""
decode_rc() {
  local code="$1"
  case "${code}" in
    132) echo "132 (SIGILL)" ;;
    134) echo "134 (SIGABRT)" ;;
    136) echo "136 (SIGFPE)" ;;
    137) echo "137 (SIGKILL)" ;;
    139) echo "139 (SIGSEGV)" ;;
    143) echo "143 (SIGTERM)" ;;
    *)
      if (( code > 128 && code < 192 )); then
        echo "${code} (SIG$((code - 128)))"
      else
        echo "${code}"
      fi
      ;;
  esac
}

run_logged_expect() {
  local expected="$1"; shift
  local name="$1"; shift
  local out="/logs/${name}.stdout.log"
  local err="/logs/${name}.stderr.log"
  local rc=0
  printf '%s\n' "$*" > "/logs/${name}.cmd"
  printf '%s\n' "${expected}" > "/logs/${name}.expected_rc"
  "$@" >"${out}" 2>"${err}" || rc=$?
  printf '%s\n' "${rc}" > "/logs/${name}.rc"

  if [[ "${rc}" -eq "${expected}" ]]; then
    printf '[PASS] %s (rc=%s)\n' "${name}" "${rc}"
    return 0
  fi

  local rc_desc
  rc_desc="$(decode_rc "${rc}")"
  printf '%s\n' "================================================================"
  printf '[FAIL] %s\n' "${name}"
  printf 'Command : %s\n' "$*"
  printf 'Exit    : %s (expected %s)\n' "${rc_desc}" "${expected}"
  printf '%s\n' "--- stdout (${name}.stdout.log) ---"
  if [[ -s "${out}" ]]; then cat "${out}"; else printf '%s\n' "(empty)"; fi
  printf '%s\n' "--- stderr (${name}.stderr.log) ---"
  if [[ -s "${err}" ]]; then cat "${err}"; else printf '%s\n' "(empty)"; fi
  printf '%s\n' "================================================================"
  return 1
}

run_logged() {
  local name="$1"; shift
  run_logged_expect 0 "${name}" "$@"
}

run_cli_json_expect() {
  local expected="$1"; shift
  local name="$1"; shift
  if run_logged_expect "${expected}" "${name}" "$@"; then
    return 0
  fi
  if [[ ! -s "/logs/${name}.stdout.log" && ! -s "/logs/${name}.stderr.log" ]]; then
    local diag_cmd=()
    local arg
    for arg in "$@"; do
      if [[ "${arg}" != "--json" ]]; then
        diag_cmd+=("${arg}")
      fi
    done
    diag_cmd+=("-v")
    printf '%s\n' ">>> Diagnostic retry without --json (with -v) for ${name}:"
    run_logged_expect 0 "${name}_diag_verbose" "${diag_cmd[@]}" || true
  fi
  return 1
}
"""


def _assert_run_payload(
    json_path: Path,
    *,
    expected_count: int,
    expected_arch: str | None = None,
    expected_endian: str | None = None,
    fp32_allowed_isas: set[str] | None = None,
    require_empty_stderr: Path | None = None,
) -> str:
    if require_empty_stderr is not None and require_empty_stderr.is_file():
        err_txt = require_empty_stderr.read_text(encoding="utf-8")
        if err_txt:
            raise AssertionError(f"Expected empty stderr in --json mode, got:\n{err_txt}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if expected_arch is not None and data["host"]["arch"] != expected_arch:
        raise AssertionError(f"arch={data['host']['arch']!r}, expected {expected_arch!r}")
    if expected_endian is not None and data["host"]["endianness"] != expected_endian:
        raise AssertionError(f"endian={data['host']['endianness']!r}, expected {expected_endian!r}")
    if data["summary"]["exit_code"] != 0:
        raise AssertionError(f"summary={data['summary']}")
    if data["summary"]["passed"] != expected_count or data["summary"]["failed"] != 0:
        raise AssertionError(f"summary={data['summary']}")
    for r in data["results"]:
        if r["status"] != "passed" or r["error_count"] != 0 or r["raw_ops"] <= 0:
            raise AssertionError(f"bad result: {r}")
    by_name = {r["name"]: r for r in data["results"]}
    if fp32_allowed_isas is not None:
        got = by_name["compute-fp32"]["isa"]
        if got not in fp32_allowed_isas:
            raise AssertionError(
                f"compute-fp32 isa={got!r}, expected one of {sorted(fp32_allowed_isas)}"
            )
    summary_str = ", ".join(f"{r['name']}:{r['isa']}" for r in data["results"])
    return f"passed={expected_count} ({summary_str})"


def _assert_neg_payload(json_path: Path, expected_status: str) -> str:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if data["summary"]["exit_code"] != 2:
        raise AssertionError(f"summary={data['summary']}")
    if len(data["results"]) != 1:
        raise AssertionError(f"results={data['results']}")
    res = data["results"][0]
    if res["status"] != expected_status or not res.get("reason"):
        raise AssertionError(f"unexpected result={res}")
    return f"{res['status']} — {res['reason']}"


def run_target_native(
    caps: HostCapabilities,
    logger: TargetLogger,
    duration: str,
) -> TargetResult:
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="cross-cpu-native-") as tmp:
        work = Path(tmp) / "work"
        copy_repo_clean(work)
        py = caps.host_python

        steps_ok = True
        for name, cmd in (
            ("01_compileall", [py, "-m", "compileall", "-q", "src", "tests"]),
            ("02_make_build", [caps.make_bin or "make", "build"]),
            ("03_unittest", [py, "-m", "unittest", "discover", "-s", "tests", "-v"]),
            ("04_info_verbose", [py, "-m", "src", "info", "-v"]),
        ):
            if not logger.run(name, cmd, cwd=work).passed:
                steps_ok = False
                break

        if steps_ok:
            info_rec = logger.run_cli_json_expect(0, "05_info_json", [py, "-m", "src", "info", "--json"], cwd=work)
            steps_ok = info_rec.passed
        if steps_ok:
            a1 = logger.record_assertion(
                "05_info_assert",
                lambda: _check_info_json(
                    logger.log_dir / "05_info_json.stdout.log",
                    require_empty_stderr=logger.log_dir / "05_info_json.stderr.log",
                ),
            )
            steps_ok = a1.passed

        if steps_ok:
            run_rec = logger.run_cli_json_expect(
                0,
                "06_run_json",
                [
                    py,
                    "-m",
                    "src",
                    "run",
                    "--test",
                    "compute-fp32,compute-int64,memory-pattern,coherency-atomic",
                    "--duration",
                    duration,
                    "--json",
                ],
                cwd=work,
            )
            steps_ok = run_rec.passed
        if steps_ok:
            a2 = logger.record_assertion(
                "06_run_assert",
                lambda: _assert_run_payload(
                    logger.log_dir / "06_run_json.stdout.log",
                    expected_count=4,
                    require_empty_stderr=logger.log_dir / "06_run_json.stderr.log",
                ),
            )
            steps_ok = a2.passed

        if steps_ok:
            all_rec = logger.run_cli_json_expect(
                0,
                "07_run_all_json",
                [py, "-m", "src", "run", "--test", "all", "--duration", "60ms", "--json"],
                cwd=work,
            )
            steps_ok = all_rec.passed
        if steps_ok:
            a3 = logger.record_assertion(
                "07_run_all_assert",
                lambda: _check_run_all_json(
                    logger.log_dir / "07_run_all_json.stdout.log",
                    require_empty_stderr=logger.log_dir / "07_run_all_json.stderr.log",
                ),
            )
            steps_ok = a3.passed

    return TargetResult(
        name="native",
        status="passed" if steps_ok else "failed",
        elapsed_s=time.monotonic() - t0,
        detail=f"{caps.os_name}/{caps.machine} native build + 39 tests + CLI",
        records=logger.records,
    )


def _check_info_json(
    json_path: Path,
    *,
    expected_arch: str | None = None,
    expected_endian: str | None = None,
    require_empty_stderr: Path | None = None,
) -> str:
    if require_empty_stderr is not None and require_empty_stderr.is_file():
        err_txt = require_empty_stderr.read_text(encoding="utf-8")
        if err_txt:
            raise AssertionError(f"Expected empty stderr in --json mode, got:\n{err_txt}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if data.get("command") != "info":
        raise AssertionError(f"unexpected command={data.get('command')!r}")
    if expected_arch is not None and data["native"]["arch_name"] != expected_arch:
        raise AssertionError(f"arch_name={data['native']['arch_name']!r}, expected {expected_arch!r}")
    if expected_endian is not None and data["native"]["endianness"] != expected_endian:
        raise AssertionError(f"endian={data['native']['endianness']!r}, expected {expected_endian!r}")
    return f"{data['native']['arch_name']} ({data['native']['endianness']}) — {data['native']['probe_summary']}"


def _check_run_all_json(json_path: Path, *, require_empty_stderr: Path | None = None) -> str:
    if require_empty_stderr is not None and require_empty_stderr.is_file():
        err_txt = require_empty_stderr.read_text(encoding="utf-8")
        if err_txt:
            raise AssertionError(f"Expected empty stderr in --json mode, got:\n{err_txt}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    s = data["summary"]
    if s["exit_code"] != 0 or s["failed"] != 0 or s["init_error"] != 0 or s["passed"] < 12:
        raise AssertionError(f"unexpected --test all summary={s}")
    return f"passed={s['passed']} unsupported={s['unsupported']} failed={s['failed']}"


def run_target_rosetta(
    caps: HostCapabilities,
    logger: TargetLogger,
    duration: str,
) -> TargetResult:
    t0 = time.monotonic()
    arch_bin = which_tool("arch", caps.env_path) or "/usr/bin/arch"
    py = caps.rosetta_python or caps.host_python

    with tempfile.TemporaryDirectory(prefix="cross-cpu-rosetta-") as tmp:
        work = Path(tmp) / "work"
        copy_repo_clean(work)

        ok = True
        if ok:
            ok = logger.run(
                "01_make_build_x86_64",
                [
                    arch_bin,
                    "-x86_64",
                    caps.make_bin or "make",
                    "build",
                    "ARCH=x86_64",
                    "CFLAGS=-arch x86_64",
                    "LDFLAGS=-arch x86_64",
                ],
                cwd=work,
            ).passed
        if ok:
            ok = logger.run("02_file_dylib", ["file", "src/_native.dylib"], cwd=work).passed
        if ok:
            ok = logger.run(
                "03_unittest_x86_64",
                [arch_bin, "-x86_64", py, "-m", "unittest", "discover", "-s", "tests", "-v"],
                cwd=work,
            ).passed
        if ok:
            ok = logger.run(
                "04_info_verbose",
                [arch_bin, "-x86_64", py, "-m", "src", "info", "-v"],
                cwd=work,
            ).passed
        if ok:
            ok = logger.run_cli_json_expect(
                0,
                "05_info_json",
                [arch_bin, "-x86_64", py, "-m", "src", "info", "--json"],
                cwd=work,
            ).passed
        if ok:
            ok = logger.record_assertion(
                "05_info_assert",
                lambda: _check_info_json(
                    logger.log_dir / "05_info_json.stdout.log",
                    expected_arch="x86_64",
                    expected_endian="little",
                    require_empty_stderr=logger.log_dir / "05_info_json.stderr.log",
                ),
            ).passed
        if ok:
            ok = logger.run_cli_json_expect(
                0,
                "06_run_json",
                [
                    arch_bin,
                    "-x86_64",
                    py,
                    "-m",
                    "src",
                    "run",
                    "--test",
                    "compute-fp32,compute-int64,memory-pattern,coherency-atomic",
                    "--duration",
                    duration,
                    "--json",
                ],
                cwd=work,
            ).passed
        if ok:
            ok = logger.record_assertion(
                "06_run_assert",
                lambda: _assert_run_payload(
                    logger.log_dir / "06_run_json.stdout.log",
                    expected_count=4,
                    expected_arch="x86_64",
                    expected_endian="little",
                    fp32_allowed_isas={"sse4", "avx2"},
                    require_empty_stderr=logger.log_dir / "06_run_json.stderr.log",
                ),
            ).passed

        # Negative ISA gates under Rosetta 2 (AVX-512 & AMX are never exposed on Rosetta 2;
        # AVX2 is absent on Apple Silicon Rosetta 2 unless macOS adds AVX2 translation).
        if ok:
            info_data = json.loads((logger.log_dir / "05_info_json.stdout.log").read_text(encoding="utf-8"))
            avail = {b["name"]: b["available"] for b in info_data["native"]["backends"]}
            for isa_name, wl in (("avx2", "compute-fp32"), ("avx512", "compute-fp32"), ("amx", "compute-matrix-bf16")):
                if avail.get(isa_name):
                    continue
                step = f"07_neg_rosetta_{isa_name}"
                ok = logger.run_cli_json_expect(
                    2,
                    step,
                    [
                        arch_bin,
                        "-x86_64",
                        py,
                        "-m",
                        "src",
                        "run",
                        "--isa",
                        isa_name,
                        "--test",
                        wl,
                        "--duration",
                        "100ms",
                        "--json",
                    ],
                    cwd=work,
                ).passed
                if not ok:
                    break
                ok = logger.record_assertion(
                    f"{step}_assert",
                    lambda p=logger.log_dir / f"{step}.stdout.log": _assert_neg_payload(p, "init_error"),
                ).passed
                if not ok:
                    break

    return TargetResult(
        name="rosetta",
        status="passed" if ok else "failed",
        elapsed_s=time.monotonic() - t0,
        detail="macOS Rosetta 2 x86_64 translation",
        records=logger.records,
    )


def run_target_x86_isa(
    caps: HostCapabilities,
    logger: TargetLogger,
    duration: str,
) -> TargetResult:
    t0 = time.monotonic()

    # Fast path: native Linux x86_64 host with qemu-x86_64 installed
    if caps.os_name == "Linux" and caps.machine in ("x86_64", "amd64") and caps.qemu_x86_64:
        return _run_x86_isa_native_qemu(caps, logger, duration, t0)

    # Container path (works on macOS Apple Silicon, Intel macOS, and Linux without host qemu-user)
    rt = caps.container_runtime
    assert rt is not None
    script = (
        "trap 'chmod -R a+rX /logs 2>/dev/null || true' EXIT\n"
        + _BASH_LOG_HELPERS
        + f"""
export DEBIAN_FRONTEND=noninteractive
run_logged "00_apt_install" bash -euo pipefail -c \
  'apt-get update -qq && apt-get install -y --no-install-recommends python3 gcc make libc6-dev qemu-user'

cp -a /repo /tmp/work
cd /tmp/work
rm -rf build dist ./*.egg-info
PY="/usr/bin/python3"

run_logged "00_qemu_version" qemu-x86_64 --version
run_logged "01_make_build" make build

for spec in "qemu64:scalar" "Nehalem:sse4" "Haswell:avx2" "max:avx2,avx512"; do
  cpu="${{spec%%:*}}"
  expected_isas="${{spec#*:}}"
  slug="${{cpu,,}}"

  run_logged "02_${{slug}}_info_verbose" qemu-x86_64 -cpu "${{cpu}}" "${{PY}}" -m src info -v
  run_cli_json_expect 0 "02_${{slug}}_info_json" qemu-x86_64 -cpu "${{cpu}}" "${{PY}}" -m src info --json
  run_logged "02_${{slug}}_info_assert" "${{PY}}" -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["command"] == "info", data
assert data["native"]["arch_name"] == "x86_64", data["native"]
assert data["native"]["endianness"] == "little", data["native"]
print("OK info:", data["native"]["probe_summary"])
' "/logs/02_${{slug}}_info_json.stdout.log"

  run_cli_json_expect 0 "03_${{slug}}_run_json" qemu-x86_64 -cpu "${{cpu}}" "${{PY}}" -m src run \
    --test compute-fp32,compute-int64,memory-pattern,coherency-atomic \
    --duration {shlex.quote(duration)} --json
  run_logged "03_${{slug}}_run_assert" "${{PY}}" -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
allowed = set(sys.argv[2].split(","))
assert data["summary"]["exit_code"] == 0, data["summary"]
assert data["summary"]["passed"] == 4, data["summary"]
assert data["summary"]["failed"] == 0, data["summary"]
by_name = {{r["name"]: r for r in data["results"]}}
for r in data["results"]:
    assert r["status"] == "passed", r
    assert r["error_count"] == 0, r
    assert r["raw_ops"] > 0, r
fp32_isa = by_name["compute-fp32"]["isa"]
assert fp32_isa in allowed, f"expected compute-fp32 ISA in {{sorted(allowed)}}, got {{fp32_isa!r}}"
print(f"OK run: compute-fp32 isa={{fp32_isa}}")
' "/logs/03_${{slug}}_run_json.stdout.log" "${{expected_isas}}"
done

run_cli_json_expect 0 "03_max_run_all_json" qemu-x86_64 -cpu "max" "${{PY}}" -m src run \
  --test all --duration 80ms --json
run_logged "03_max_run_all_assert" "${{PY}}" -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["summary"]["exit_code"] == 0, data["summary"]
assert data["summary"]["failed"] == 0, data["summary"]
assert data["summary"]["init_error"] == 0, data["summary"]
assert data["summary"]["passed"] >= 12, data["summary"]
print("OK --test all:", data["summary"])
' "/logs/03_max_run_all_json.stdout.log"

assert_neg_json() {{
  local log_name="$1"
  local expected_status="$2"
  run_logged "${{log_name}}_assert" "${{PY}}" -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
exp_status = sys.argv[2]
assert data["summary"]["exit_code"] == 2, data["summary"]
assert len(data["results"]) == 1, data["results"]
assert data["results"][0]["status"] == exp_status, data["results"][0]
assert data["results"][0]["reason"], data["results"][0]
print("OK negative gate:", data["results"][0]["status"], "-", data["results"][0]["reason"])
' "/logs/${{log_name}}.stdout.log" "${{expected_status}}"
}}

run_cli_json_expect 2 "04_neg_qemu64_sse4" qemu-x86_64 -cpu "qemu64" "${{PY}}" -m src run --isa sse4 --test compute-fp32 --duration 100ms --json
assert_neg_json "04_neg_qemu64_sse4" "init_error"
run_cli_json_expect 2 "04_neg_qemu64_avx2" qemu-x86_64 -cpu "qemu64" "${{PY}}" -m src run --isa avx2 --test compute-fp32 --duration 100ms --json
assert_neg_json "04_neg_qemu64_avx2" "init_error"
run_cli_json_expect 2 "04_neg_qemu64_crypto_aes" qemu-x86_64 -cpu "qemu64" "${{PY}}" -m src run --test crypto-aes --duration 100ms --json
assert_neg_json "04_neg_qemu64_crypto_aes" "unsupported"
run_cli_json_expect 2 "04_neg_nehalem_avx2" qemu-x86_64 -cpu "Nehalem" "${{PY}}" -m src run --isa avx2 --test compute-fp32 --duration 100ms --json
assert_neg_json "04_neg_nehalem_avx2" "init_error"
run_cli_json_expect 2 "04_neg_haswell_avx512" qemu-x86_64 -cpu "Haswell" "${{PY}}" -m src run --isa avx512 --test compute-fp32 --duration 100ms --json
assert_neg_json "04_neg_haswell_avx512" "init_error"
run_cli_json_expect 2 "04_neg_haswell_amx" qemu-x86_64 -cpu "Haswell" "${{PY}}" -m src run --isa amx --test compute-matrix-bf16 --duration 100ms --json
assert_neg_json "04_neg_haswell_amx" "init_error"
"""
    )

    rec = logger.run(
        "container_x86_isa",
        [
            rt,
            "run",
            "-i",
            "--rm",
            "--platform",
            "linux/amd64",
            "-v",
            f"{REPO_ROOT}:/repo:ro",
            "-v",
            f"{logger.log_dir}:/logs:rw",
            "debian:trixie-slim",
            "bash",
            "-euo",
            "pipefail",
        ],
        stdin_text=script,
    )
    logger.harvest_container_logs()
    return TargetResult(
        name="x86-isa",
        status="passed" if rec.passed else "failed",
        elapsed_s=time.monotonic() - t0,
        detail="x86_64 QEMU TCG per-ISA gating (qemu64/Nehalem/Haswell/max)",
        records=logger.records,
    )


def _run_x86_isa_native_qemu(
    caps: HostCapabilities,
    logger: TargetLogger,
    duration: str,
    t0: float,
) -> TargetResult:
    qemu = caps.qemu_x86_64 or "qemu-x86_64"
    py = "/usr/bin/python3" if Path("/usr/bin/python3").is_file() else caps.host_python
    with tempfile.TemporaryDirectory(prefix="cross-cpu-x86-isa-") as tmp:
        work = Path(tmp) / "work"
        copy_repo_clean(work)
        ok = logger.run("00_qemu_version", [qemu, "--version"], cwd=work).passed
        if ok:
            ok = logger.run("01_make_build", [caps.make_bin or "make", "build"], cwd=work).passed
        for cpu, allowed in (
            ("qemu64", {"scalar"}),
            ("Nehalem", {"sse4"}),
            ("Haswell", {"avx2"}),
            ("max", {"avx2", "avx512"}),
        ):
            if not ok:
                break
            slug = cpu.lower()
            ok = logger.run(
                f"02_{slug}_info_verbose", [qemu, "-cpu", cpu, py, "-m", "src", "info", "-v"], cwd=work
            ).passed
            if not ok:
                break
            ok = logger.run_cli_json_expect(
                0, f"02_{slug}_info_json", [qemu, "-cpu", cpu, py, "-m", "src", "info", "--json"], cwd=work
            ).passed
            if not ok:
                break
            ok = logger.record_assertion(
                f"02_{slug}_info_assert",
                lambda p=logger.log_dir / f"02_{slug}_info_json.stdout.log": _check_info_json(
                    p, expected_arch="x86_64", expected_endian="little"
                ),
            ).passed
            if not ok:
                break
            ok = logger.run_cli_json_expect(
                0,
                f"03_{slug}_run_json",
                [
                    qemu,
                    "-cpu",
                    cpu,
                    py,
                    "-m",
                    "src",
                    "run",
                    "--test",
                    "compute-fp32,compute-int64,memory-pattern,coherency-atomic",
                    "--duration",
                    duration,
                    "--json",
                ],
                cwd=work,
            ).passed
            if not ok:
                break
            ok = logger.record_assertion(
                f"03_{slug}_run_assert",
                lambda p=logger.log_dir / f"03_{slug}_run_json.stdout.log", a=allowed: _assert_run_payload(
                    p, expected_count=4, expected_arch="x86_64", expected_endian="little", fp32_allowed_isas=a
                ),
            ).passed
        if ok:
            neg_cases = (
                ("04_neg_qemu64_sse4", "qemu64", ["--isa", "sse4", "--test", "compute-fp32"], "init_error"),
                ("04_neg_qemu64_avx2", "qemu64", ["--isa", "avx2", "--test", "compute-fp32"], "init_error"),
                ("04_neg_qemu64_crypto_aes", "qemu64", ["--test", "crypto-aes"], "unsupported"),
                ("04_neg_nehalem_avx2", "Nehalem", ["--isa", "avx2", "--test", "compute-fp32"], "init_error"),
                ("04_neg_haswell_avx512", "Haswell", ["--isa", "avx512", "--test", "compute-fp32"], "init_error"),
                ("04_neg_haswell_amx", "Haswell", ["--isa", "amx", "--test", "compute-matrix-bf16"], "init_error"),
            )
            for step, cpu, extra_args, exp_status in neg_cases:
                ok = logger.run_cli_json_expect(
                    2,
                    step,
                    [qemu, "-cpu", cpu, py, "-m", "src", "run", *extra_args, "--duration", "100ms", "--json"],
                    cwd=work,
                ).passed
                if not ok:
                    break
                ok = logger.record_assertion(
                    f"{step}_assert",
                    lambda p=logger.log_dir / f"{step}.stdout.log", st=exp_status: _assert_neg_payload(p, st),
                ).passed
                if not ok:
                    break
    return TargetResult(
        name="x86-isa",
        status="passed" if ok else "failed",
        elapsed_s=time.monotonic() - t0,
        detail="native Linux qemu-x86_64 per-ISA gating",
        records=logger.records,
    )


def run_target_arm64_isa(
    caps: HostCapabilities,
    logger: TargetLogger,
    duration: str,
) -> TargetResult:
    t0 = time.monotonic()

    if caps.os_name == "Linux" and caps.machine in ("arm64", "aarch64") and caps.qemu_aarch64:
        return _run_arm64_isa_native_qemu(caps, logger, duration, t0)

    rt = caps.container_runtime
    assert rt is not None
    script = (
        "trap 'chmod -R a+rX /logs 2>/dev/null || true' EXIT\n"
        + _BASH_LOG_HELPERS
        + f"""
export DEBIAN_FRONTEND=noninteractive
run_logged "00_apt_install" bash -euo pipefail -c \
  'apt-get update -qq && apt-get install -y --no-install-recommends python3 gcc make libc6-dev qemu-user'

cp -a /repo /tmp/work
cd /tmp/work
rm -rf build dist ./*.egg-info
PY="/usr/bin/python3"

run_logged "00_qemu_version" qemu-aarch64 --version
run_logged "01_make_build" make build

for spec in \
  "cortex_a53:cortex-a53:neon" \
  "neon_only:max,sve=off,sme=off:neon" \
  "sve_on:max,sve=on,sme=off:sve_if_compiled"; do
  slug="${{spec%%:*}}"
  rest="${{spec#*:}}"
  cpu="${{rest%%:*}}"
  expected_mode="${{rest#*:}}"

  run_logged "02_${{slug}}_info_verbose" qemu-aarch64 -cpu "${{cpu}}" "${{PY}}" -m src info -v
  run_cli_json_expect 0 "02_${{slug}}_info_json" qemu-aarch64 -cpu "${{cpu}}" "${{PY}}" -m src info --json
  run_logged "02_${{slug}}_info_assert" "${{PY}}" -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["command"] == "info", data
assert data["native"]["arch_name"] == "arm64", data["native"]
assert data["native"]["endianness"] == "little", data["native"]
print("OK info:", data["native"]["probe_summary"])
' "/logs/02_${{slug}}_info_json.stdout.log"

  run_cli_json_expect 0 "03_${{slug}}_run_json" qemu-aarch64 -cpu "${{cpu}}" "${{PY}}" -m src run \
    --test compute-fp32,compute-int64,memory-pattern,coherency-atomic \
    --duration {shlex.quote(duration)} --json
  run_logged "03_${{slug}}_run_assert" "${{PY}}" -c '
import json, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))
run_data = json.load(open(sys.argv[2], encoding="utf-8"))
mode = sys.argv[3]
assert run_data["summary"]["exit_code"] == 0, run_data["summary"]
assert run_data["summary"]["passed"] == 4, run_data["summary"]
assert run_data["summary"]["failed"] == 0, run_data["summary"]
for r in run_data["results"]:
    assert r["status"] == "passed", r
    assert r["error_count"] == 0, r
    assert r["raw_ops"] > 0, r
backends = {{b["name"]: b for b in info["native"]["backends"]}}
expected_isa = ("sve" if backends.get("sve", {{}}).get("compiled_in") else "neon") if mode == "sve_if_compiled" else mode
by_name = {{r["name"]: r for r in run_data["results"]}}
got_isa = by_name["compute-fp32"]["isa"]
assert got_isa == expected_isa, f"expected compute-fp32 isa={{expected_isa!r}}, got {{got_isa!r}}"
print(f"OK run ({{mode}}): compute-fp32 isa={{got_isa}}")
' "/logs/02_${{slug}}_info_json.stdout.log" "/logs/03_${{slug}}_run_json.stdout.log" "${{expected_mode}}"
done

assert_neg_json() {{
  local log_name="$1"
  local expected_status="$2"
  run_logged "${{log_name}}_assert" "${{PY}}" -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
exp_status = sys.argv[2]
assert data["summary"]["exit_code"] == 2, data["summary"]
assert len(data["results"]) == 1, data["results"]
assert data["results"][0]["status"] == exp_status, data["results"][0]
assert data["results"][0]["reason"], data["results"][0]
print("OK negative gate:", data["results"][0]["status"], "-", data["results"][0]["reason"])
' "/logs/${{log_name}}.stdout.log" "${{expected_status}}"
}}

run_cli_json_expect 2 "04_neg_cortex_a53_fp16" qemu-aarch64 -cpu "cortex-a53" "${{PY}}" -m src run --test compute-fp16 --duration 100ms --json
assert_neg_json "04_neg_cortex_a53_fp16" "unsupported"
run_cli_json_expect 2 "04_neg_cortex_a53_sve" qemu-aarch64 -cpu "cortex-a53" "${{PY}}" -m src run --isa sve --test compute-fp32 --duration 100ms --json
assert_neg_json "04_neg_cortex_a53_sve" "init_error"
run_cli_json_expect 2 "04_neg_neon_only_sve" qemu-aarch64 -cpu "max,sve=off,sme=off" "${{PY}}" -m src run --isa sve --test compute-fp32 --duration 100ms --json
assert_neg_json "04_neg_neon_only_sve" "init_error"
run_cli_json_expect 2 "04_neg_neon_only_sme" qemu-aarch64 -cpu "max,sve=off,sme=off" "${{PY}}" -m src run --isa sme --test compute-matrix-fp32 --duration 100ms --json
assert_neg_json "04_neg_neon_only_sme" "init_error"
run_cli_json_expect 2 "04_neg_sve_on_sme" qemu-aarch64 -cpu "max,sve=on,sme=off" "${{PY}}" -m src run --isa sme --test compute-matrix-fp32 --duration 100ms --json
assert_neg_json "04_neg_sve_on_sme" "init_error"
"""
    )

    rec = logger.run(
        "container_arm64_isa",
        [
            rt,
            "run",
            "-i",
            "--rm",
            "--platform",
            "linux/arm64",
            "-v",
            f"{REPO_ROOT}:/repo:ro",
            "-v",
            f"{logger.log_dir}:/logs:rw",
            "debian:trixie-slim",
            "bash",
            "-euo",
            "pipefail",
        ],
        stdin_text=script,
    )
    logger.harvest_container_logs()
    return TargetResult(
        name="arm64-isa",
        status="passed" if rec.passed else "failed",
        elapsed_s=time.monotonic() - t0,
        detail="ARM64 QEMU TCG per-ISA gating (cortex-a53/neon_only/sve_on)",
        records=logger.records,
    )


def _run_arm64_isa_native_qemu(
    caps: HostCapabilities,
    logger: TargetLogger,
    duration: str,
    t0: float,
) -> TargetResult:
    qemu = caps.qemu_aarch64 or "qemu-aarch64"
    py = "/usr/bin/python3" if Path("/usr/bin/python3").is_file() else caps.host_python
    with tempfile.TemporaryDirectory(prefix="cross-cpu-arm64-isa-") as tmp:
        work = Path(tmp) / "work"
        copy_repo_clean(work)
        ok = logger.run("00_qemu_version", [qemu, "--version"], cwd=work).passed
        if ok:
            ok = logger.run("01_make_build", [caps.make_bin or "make", "build"], cwd=work).passed
        for slug, cpu, mode in (
            ("cortex_a53", "cortex-a53", "neon"),
            ("neon_only", "max,sve=off,sme=off", "neon"),
            ("sve_on", "max,sve=on,sme=off", "sve_if_compiled"),
        ):
            if not ok:
                break
            ok = logger.run(
                f"02_{slug}_info_verbose", [qemu, "-cpu", cpu, py, "-m", "src", "info", "-v"], cwd=work
            ).passed
            if not ok:
                break
            ok = logger.run_cli_json_expect(
                0, f"02_{slug}_info_json", [qemu, "-cpu", cpu, py, "-m", "src", "info", "--json"], cwd=work
            ).passed
            if not ok:
                break
            info_path = logger.log_dir / f"02_{slug}_info_json.stdout.log"
            ok = logger.record_assertion(
                f"02_{slug}_info_assert",
                lambda p=info_path: _check_info_json(p, expected_arch="arm64", expected_endian="little"),
            ).passed
            if not ok:
                break
            info_data = json.loads(info_path.read_text(encoding="utf-8"))
            backends = {b["name"]: b for b in info_data["native"]["backends"]}
            exp_isa = "sve" if (mode == "sve_if_compiled" and backends.get("sve", {}).get("compiled_in")) else "neon"
            ok = logger.run_cli_json_expect(
                0,
                f"03_{slug}_run_json",
                [
                    qemu,
                    "-cpu",
                    cpu,
                    py,
                    "-m",
                    "src",
                    "run",
                    "--test",
                    "compute-fp32,compute-int64,memory-pattern,coherency-atomic",
                    "--duration",
                    duration,
                    "--json",
                ],
                cwd=work,
            ).passed
            if not ok:
                break
            ok = logger.record_assertion(
                f"03_{slug}_run_assert",
                lambda p=logger.log_dir / f"03_{slug}_run_json.stdout.log", isa=exp_isa: _assert_run_payload(
                    p, expected_count=4, expected_arch="arm64", expected_endian="little", fp32_allowed_isas={isa}
                ),
            ).passed
        if ok:
            neg_cases = (
                ("04_neg_cortex_a53_fp16", "cortex-a53", ["--test", "compute-fp16"], "unsupported"),
                ("04_neg_cortex_a53_sve", "cortex-a53", ["--isa", "sve", "--test", "compute-fp32"], "init_error"),
                ("04_neg_neon_only_sve", "max,sve=off,sme=off", ["--isa", "sve", "--test", "compute-fp32"], "init_error"),
                ("04_neg_neon_only_sme", "max,sve=off,sme=off", ["--isa", "sme", "--test", "compute-matrix-fp32"], "init_error"),
                ("04_neg_sve_on_sme", "max,sve=on,sme=off", ["--isa", "sme", "--test", "compute-matrix-fp32"], "init_error"),
            )
            for step, cpu, extra_args, exp_status in neg_cases:
                ok = logger.run_cli_json_expect(
                    2,
                    step,
                    [qemu, "-cpu", cpu, py, "-m", "src", "run", *extra_args, "--duration", "100ms", "--json"],
                    cwd=work,
                ).passed
                if not ok:
                    break
                ok = logger.record_assertion(
                    f"{step}_assert",
                    lambda p=logger.log_dir / f"{step}.stdout.log", st=exp_status: _assert_neg_payload(p, st),
                ).passed
                if not ok:
                    break
    return TargetResult(
        name="arm64-isa",
        status="passed" if ok else "failed",
        elapsed_s=time.monotonic() - t0,
        detail="native Linux qemu-aarch64 per-ISA gating",
        records=logger.records,
    )


FOREIGN_ARCH_SPECS: dict[str, tuple[str, str, str]] = {
    "arch-arm64": ("linux/arm64", "arm64", "little"),
    "arch-riscv64": ("linux/riscv64", "riscv64", "little"),
    "arch-ppc64le": ("linux/ppc64le", "ppc64le", "little"),
    "arch-s390x": ("linux/s390x", "s390x", "big"),
}


def run_target_foreign_arch(
    target: str,
    caps: HostCapabilities,
    logger: TargetLogger,
    duration: str,
) -> TargetResult:
    t0 = time.monotonic()
    rt = caps.container_runtime
    assert rt is not None
    plat, expected_arch, expected_endian = FOREIGN_ARCH_SPECS[target]

    script = (
        "trap 'chmod -R a+rX /logs 2>/dev/null || true' EXIT\n"
        + _BASH_LOG_HELPERS
        + f"""
export DEBIAN_FRONTEND=noninteractive
run_logged "00_apt_install" bash -euo pipefail -c \
  'apt-get update -qq && apt-get install -y --no-install-recommends python3 python3-setuptools gcc make libc6-dev'

cp -a /repo /tmp/work
cd /tmp/work
rm -rf build dist ./*.egg-info

run_logged "01_env" uname -a
run_logged "02_cc_version" gcc --version
run_logged "03_compileall" python3 -m compileall -q src tests
run_logged "04_make_build" make build
run_logged "05_unittest" python3 -m unittest discover -s tests -v
run_logged "06_info_verbose" python3 -m src info -v
run_cli_json_expect 0 "07_info_json" python3 -m src info --json
run_logged "07_info_assert" python3 -c '
import json, os, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["command"] == "info", data
assert data["native"]["arch_name"] == os.environ["EXPECTED_ARCH"], data["native"]
assert data["native"]["endianness"] == os.environ["EXPECTED_ENDIAN"], data["native"]
print("OK info:", data["native"]["arch_name"], data["native"]["endianness"], data["native"]["probe_summary"])
' /logs/07_info_json.stdout.log

run_cli_json_expect 0 "08_run_json" python3 -m src run \
  --test compute-fp32,compute-int64,memory-pattern,coherency-atomic \
  --duration {shlex.quote(duration)} --json
run_logged "08_run_assert" python3 -c '
import json, os, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["host"]["arch"] == os.environ["EXPECTED_ARCH"], data["host"]
assert data["host"]["endianness"] == os.environ["EXPECTED_ENDIAN"], data["host"]
assert data["summary"]["exit_code"] == 0, data["summary"]
assert data["summary"]["passed"] == 4, data["summary"]
assert data["summary"]["failed"] == 0, data["summary"]
for r in data["results"]:
    assert r["status"] == "passed", r
    assert r["error_count"] == 0, r
    assert r["raw_ops"] > 0, r
print("OK run:", [(r["name"], r["isa"], round(r["rate"], 3)) for r in data["results"]])
' /logs/08_run_json.stdout.log
"""
    )

    rec = logger.run(
        f"container_{target}",
        [
            rt,
            "run",
            "-i",
            "--rm",
            "--platform",
            plat,
            "-v",
            f"{REPO_ROOT}:/repo:ro",
            "-v",
            f"{logger.log_dir}:/logs:rw",
            "-e",
            f"EXPECTED_ARCH={expected_arch}",
            "-e",
            f"EXPECTED_ENDIAN={expected_endian}",
            "debian:trixie-slim",
            "bash",
            "-euo",
            "pipefail",
        ],
        stdin_text=script,
        timeout_s=2700.0,
    )
    logger.harvest_container_logs()
    return TargetResult(
        name=target,
        status="passed" if rec.passed else "failed",
        elapsed_s=time.monotonic() - t0,
        detail=f"{plat} ({expected_endian}-endian)",
        records=logger.records,
    )


DISTRO_SPECS: dict[str, tuple[str, str, str]] = {
    "distro-alpine": (
        "alpine:latest",
        "apk add --no-cache python3 python3-dev py3-setuptools py3-wheel py3-build gcc clang musl-dev make bash",
        "-O2 -Wall -Wextra",
    ),
    "distro-debian": (
        "debian:trixie-slim",
        "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 python3-dev python3-setuptools python3-wheel python3-build gcc clang make libc6-dev",
        "-O2 -D_FORTIFY_SOURCE=3 -fstack-protector-strong -Werror=format-security",
    ),
    "distro-fedora": (
        "fedora:latest",
        "dnf install -y -q python3 python3-devel python3-setuptools python3-wheel python3-build gcc clang make redhat-rpm-config",
        "-O2 -D_FORTIFY_SOURCE=3 -fstack-protector-strong",
    ),
    "distro-arch": (
        "archlinux:latest",
        "pacman --disable-sandbox -Syu --noconfirm --needed python python-setuptools python-wheel python-build gcc clang make",
        "-O2",
    ),
}


def run_target_distro(
    target: str,
    caps: HostCapabilities,
    logger: TargetLogger,
    container_platform: str | None = None,
) -> TargetResult:
    t0 = time.monotonic()
    rt = caps.container_runtime
    assert rt is not None
    image, install_cmd, cflags = DISTRO_SPECS[target]

    script = r"""
trap 'chmod -R a+rX /logs 2>/dev/null || true' EXIT

run_logged() {
  _name="$1"; shift
  _out="/logs/${_name}.stdout.log"
  _err="/logs/${_name}.stderr.log"
  _rc=0
  printf '%s\n' "$*" > "/logs/${_name}.cmd"
  printf '0\n' > "/logs/${_name}.expected_rc"
  "$@" >"${_out}" 2>"${_err}" || _rc=$?
  printf '%s\n' "${_rc}" > "/logs/${_name}.rc"

  if [ "${_rc}" -eq 0 ]; then
    printf '[PASS] %s (rc=0)\n' "${_name}"
    return 0
  fi

  printf '%s\n' "================================================================"
  printf '[FAIL] %s\n' "${_name}"
  printf 'Command : %s\n' "$*"
  printf 'Exit    : %s (expected 0)\n' "${_rc}"
  printf '%s\n' "--- stdout (${_name}.stdout.log) ---"
  if [ -s "${_out}" ]; then cat "${_out}"; else printf '%s\n' "(empty)"; fi
  printf '%s\n' "--- stderr (${_name}.stderr.log) ---"
  if [ -s "${_err}" ]; then cat "${_err}"; else printf '%s\n' "(empty)"; fi
  printf '%s\n' "================================================================"
  return "${_rc}"
}

run_logged "00_install_deps" sh -eu -c "${INSTALL_CMD}"

cp -a /repo /tmp/work
cd /tmp/work
rm -rf build dist ./*.egg-info

run_logged "01_gcc_version" gcc --version
run_logged "02_build_gcc" env CFLAGS="${EXTRA_CFLAGS}" make build CC=gcc
run_logged "03_unittest_gcc" python3 -m unittest discover -s tests -v

make clean || true
run_logged "04_clang_version" clang --version
run_logged "05_build_clang" env CFLAGS="${EXTRA_CFLAGS}" make build CC=clang
run_logged "06_unittest_clang" python3 -m unittest discover -s tests -v

make clean || true
run_logged "07_wheel_offline" python3 -m build --wheel --no-isolation --outdir /tmp/dist .
run_logged "08_verify_wheel" python3 -c '
import glob, zipfile
wheels = glob.glob("/tmp/dist/cross_cpu-*.whl")
assert len(wheels) == 1, f"Expected 1 wheel, found: {wheels}"
with zipfile.ZipFile(wheels[0]) as zf:
    names = zf.namelist()
assert any("cross_cpu/_native" in n and n.endswith(".so") for n in names), names
print("OK wheel:", wheels[0], "entries:", len(names))
'
"""

    cmd = [rt, "run", "-i", "--rm"]
    if container_platform:
        cmd.extend(["--platform", container_platform])
    cmd.extend(
        [
            "-v",
            f"{REPO_ROOT}:/repo:ro",
            "-v",
            f"{logger.log_dir}:/logs:rw",
            "-e",
            f"INSTALL_CMD={install_cmd}",
            "-e",
            f"EXTRA_CFLAGS={cflags}",
            image,
            "sh",
            "-eu",
        ]
    )
    rec = logger.run(f"container_{target}", cmd, stdin_text=script, timeout_s=1800.0)
    logger.harvest_container_logs()
    return TargetResult(
        name=target,
        status="passed" if rec.passed else "failed",
        elapsed_s=time.monotonic() - t0,
        detail=f"{image} (GCC + Clang + offline wheel)",
        records=logger.records,
    )


def run_target_adversarial(
    caps: HostCapabilities,
    logger: TargetLogger,
    duration: str,
    container_platform: str | None = None,
) -> TargetResult:
    t0 = time.monotonic()
    rt = caps.container_runtime
    assert rt is not None

    ncpu = caps.container_ncpu
    if ncpu >= 3:
        cpuset = "0,2"
    elif ncpu == 2:
        cpuset = "1"
    else:
        cpuset = "0"

    with tempfile.TemporaryDirectory(prefix="cross-cpu-wheels-") as wheel_tmp:
        wheel_dir = Path(wheel_tmp)

        builder_script = (
            "trap 'chmod -R a+rX /wheels /logs 2>/dev/null || true' EXIT\n"
            + _BASH_LOG_HELPERS
            + """
export DEBIAN_FRONTEND=noninteractive
run_logged "00_builder_apt" bash -euo pipefail -c \
  'apt-get update -qq && apt-get install -y --no-install-recommends gcc libc6-dev'
run_logged "01_builder_pip_tools" python3 -m pip install --quiet --no-cache-dir setuptools wheel build

cp -a /repo /tmp/work
cd /tmp/work
rm -rf build dist ./*.egg-info
run_logged "02_builder_wheel" python3 -m build --wheel --no-isolation --outdir /wheels .
"""
        )

        build_cmd = [rt, "run", "-i", "--rm"]
        if container_platform:
            build_cmd.extend(["--platform", container_platform])
        build_cmd.extend(
            [
                "-v",
                f"{REPO_ROOT}:/repo:ro",
                "-v",
                f"{wheel_dir}:/wheels:rw",
                "-v",
                f"{logger.log_dir}:/logs:rw",
                "python:3.12-slim",
                "bash",
                "-euo",
                "pipefail",
            ]
        )
        build_rec = logger.run("00_wheel_builder", build_cmd, stdin_text=builder_script)
        logger.harvest_container_logs()
        if not build_rec.passed:
            return TargetResult(
                name="adversarial",
                status="failed",
                elapsed_s=time.monotonic() - t0,
                detail="failed to pre-build ABI-matched wheel in python:3.12-slim",
                records=logger.records,
            )

        adv_script = f"""
run_inner() {{
  local name="$1"; shift
  local out="/tmp/${{name}}.stdout.log"
  local err="/tmp/${{name}}.stderr.log"
  local irc=0
  "$@" >"${{out}}" 2>"${{err}}" || irc=$?
  if [[ "${{irc}}" -eq 0 ]]; then
    printf '[PASS] %s (rc=0)\n' "${{name}}"
    return 0
  fi
  printf '%s\n' "================================================================"
  printf '[FAIL] %s\n' "${{name}}"
  printf 'Command : %s\n' "$*"
  printf 'Exit    : %s (expected 0)\n' "${{irc}}"
  printf '%s\n' "--- stdout (${{name}}) ---"
  if [[ -s "${{out}}" ]]; then cat "${{out}}"; else printf '%s\n' "(empty)"; fi
  printf '%s\n' "--- stderr (${{name}}) ---"
  if [[ -s "${{err}}" ]]; then cat "${{err}}"; else printf '%s\n' "(empty)"; fi
  printf '%s\n' "================================================================"
  return "${{irc}}"
}}

assert_empty_stderr() {{
  local name="$1"
  if [[ -s "/tmp/${{name}}.stderr.log" ]]; then
    printf 'FAIL: expected empty stderr in --json mode for %s\n' "${{name}}" >&2
    cat "/tmp/${{name}}.stderr.log" >&2
    return 1
  fi
}}

run_inner "01_venv_create" python3 -m venv /tmp/venv
run_inner "02_pip_install_offline" /tmp/venv/bin/pip install --no-index --no-deps /wheels/*.whl
cd /tmp

run_inner "03_version" /tmp/venv/bin/cross-cpu --version
run_inner "04_info_verbose" /tmp/venv/bin/cross-cpu info -v

run_inner "05_list_json" /tmp/venv/bin/cross-cpu list --json
assert_empty_stderr "05_list_json"
run_inner "05_list_json_check" python3 -m json.tool /tmp/05_list_json.stdout.log

run_inner "06_info_json" /tmp/venv/bin/cross-cpu info --json
assert_empty_stderr "06_info_json"
run_inner "06_info_json_check" python3 -m json.tool /tmp/06_info_json.stdout.log

run_inner "07_run_cpuset_json" /tmp/venv/bin/cross-cpu run \
  --cores "${{CPUSET}}" \
  --test compute-fp32,compute-int64,cache-l1,memory-pattern,coherency-atomic \
  --duration {shlex.quote(duration)} --json
assert_empty_stderr "07_run_cpuset_json"
run_inner "07_run_cpuset_assert" python3 -c '
import json, os, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
expected_ids = sorted(int(x) for x in os.environ["CPUSET"].split(","))
assert data["config"]["selected_core_ids"] == expected_ids, data["config"]
assert data["summary"]["exit_code"] == 0, data["summary"]
assert data["summary"]["passed"] == 5, data["summary"]
assert data["summary"]["failed"] == 0, data["summary"]
print("OK cpuset run:", expected_ids, data["summary"])
' /tmp/07_run_cpuset_json.stdout.log

run_inner "08_run_all_json" /tmp/venv/bin/cross-cpu run \
  --cores all \
  --test compute-fp32,compute-int64,cache-l1,memory-pattern,coherency-atomic \
  --duration {shlex.quote(duration)} --json
assert_empty_stderr "08_run_all_json"
run_inner "08_run_all_assert" python3 -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["summary"]["exit_code"] == 0, data["summary"]
assert data["summary"]["passed"] == 5, data["summary"]
assert data["summary"]["failed"] == 0, data["summary"]
print("OK --cores all run:", data["summary"])
' /tmp/08_run_all_json.stdout.log
"""

        adv_cmd = [rt, "run", "-i", "--rm"]
        if container_platform:
            adv_cmd.extend(["--platform", container_platform])
        adv_cmd.extend(
            [
                "--network=none",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,exec,mode=1777,size=256m",
                "--user",
                "65534:65534",
                f"--cpuset-cpus={cpuset}",
                "-e",
                "HOME=/tmp",
                "-e",
                "XDG_CACHE_HOME=/tmp/cache",
                "-e",
                "TMPDIR=/tmp",
                "-e",
                "PIP_NO_CACHE_DIR=1",
                "-e",
                "PIP_DISABLE_PIP_VERSION_CHECK=1",
                "-e",
                f"CPUSET={cpuset}",
                "-v",
                f"{wheel_dir}:/wheels:ro",
                "python:3.12-slim",
                "bash",
                "-euo",
                "pipefail",
            ]
        )
        adv_rec = logger.run("03_adversarial_container", adv_cmd, stdin_text=adv_script)

    return TargetResult(
        name="adversarial",
        status="passed" if adv_rec.passed else "failed",
        elapsed_s=time.monotonic() - t0,
        detail=f"offline, read-only, UID 65534, cpuset={cpuset} (daemon ncpu={ncpu})",
        records=logger.records,
    )


def setup_binfmt(caps: HostCapabilities, log_dir: Path, verbose: bool, use_color: bool) -> bool:
    if not caps.container_daemon_ready or not caps.container_runtime:
        print(f"Cannot register binfmt_misc: {_container_hint(caps)}", file=sys.stderr)
        return False
    logger = TargetLogger("binfmt", log_dir / "binfmt", env_path=caps.env_path, verbose=verbose, use_color=use_color)
    rec = logger.run(
        "00_install_binfmt",
        [caps.container_runtime, "run", "--privileged", "--rm", "tonistiigi/binfmt", "--install", "all"],
    )
    return rec.passed


def write_markdown_report(log_root: Path, results: Sequence[TargetResult], caps: HostCapabilities) -> Path:
    report_path = log_root / "summary.md"
    lines: list[str] = [
        "# CrossCPU Local Environment Emulation Report",
        "",
        f"- **Timestamp:** {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"- **Host:** {caps.os_name} / {caps.machine} (Python {caps.host_python_version})",
        f"- **Container Runtime:** {caps.container_runtime_name or 'none'} "
        f"({'ready' if caps.container_daemon_ready else 'unavailable'})",
        f"- **Log Directory:** `{log_root}`",
        "",
        "## Target Summary",
        "",
        "| Target | Status | Duration | Steps | Detail |",
        "|---|---|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r.name}` | **{r.status.upper()}** | {r.elapsed_s:.2f}s | {len(r.records)} | {r.detail} |"
        )

    failed_targets = [r for r in results if r.status == "failed"]
    if failed_targets:
        lines.extend(["", "## Failure Details"])
        for r in failed_targets:
            lines.extend(["", f"### `{r.name}`"])
            bad_recs = [rec for rec in r.records if not rec.passed]
            if not bad_recs:
                lines.append(f"- {r.detail}")
            for rec in bad_recs:
                out_txt = rec.stdout_path.read_text(encoding="utf-8", errors="replace") if rec.stdout_path.is_file() else ""
                err_txt = rec.stderr_path.read_text(encoding="utf-8", errors="replace") if rec.stderr_path.is_file() else ""
                lines.extend(
                    [
                        "",
                        f"**`{rec.name}`** — exit `{decode_rc(rec.rc)}` (expected `{rec.expected_rc}`)",
                        "```text",
                        f"Command: {rec.cmd}",
                        "--- stdout (tail -n 200) ---",
                        _tail_text(out_txt, 200) or "(empty)",
                        "--- stderr (tail -n 200) ---",
                        _tail_text(err_txt, 200) or "(empty)",
                        "```",
                    ]
                )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def print_doctor(caps: HostCapabilities, use_color: bool) -> None:
    print("CrossCPU Host Emulation Capabilities")
    print("=" * 72)
    print(f"Host OS / Arch     : {caps.os_name} / {caps.machine}")
    print(f"Host Python        : {caps.host_python} ({caps.host_python_version})")
    print(f"Host C Toolchain   : cc={caps.cc_bin or 'missing'}, make={caps.make_bin or 'missing'}")
    if caps.os_name == "Darwin" and caps.machine in ("arm64", "aarch64"):
        ros = f"available ({caps.rosetta_python})" if caps.rosetta_available else "unavailable"
        print(f"macOS Rosetta 2    : {ros}")
    print(f"Host QEMU user-mode: qemu-x86_64={caps.qemu_x86_64 or 'none'}, qemu-aarch64={caps.qemu_aarch64 or 'none'}")
    if caps.container_daemon_ready:
        print(
            f"Container Runtime  : {caps.container_runtime_name} ({caps.container_runtime}) "
            f"[READY: {caps.container_arch}, {caps.container_ncpu} vCPU]"
        )
    else:
        print(f"Container Runtime  : {_container_hint(caps)}")

    print("\nAvailable Emulation Targets:")
    print(f"{'TARGET':<16} {'SUITE':<13} {'STATE':<10} DETAILS")
    print("-" * 72)
    target_to_suite = {t: s for s, ts in SUITE_TARGETS.items() for t in ts}
    for target in ALL_TARGETS:
        ready, note = target_readiness(target, caps)
        state_raw = "READY" if ready else "BLOCKED"
        badge = _color(state_raw, "32;1" if ready else "33", use_color)
        pad = len(badge) - len(state_raw)
        print(f"{target:<16} {target_to_suite[target]:<13} {badge:<{10 + pad}} {note}")
        print(f"{'':<16} {'':<13} {'':<10} -> {TARGET_DESCRIPTIONS[target]}")

    print("\nExamples:")
    print("  ./scripts/emulate.py --suite host              # Run native + Rosetta 2 (no Docker needed)")
    print("  ./scripts/emulate.py --suite available         # Run every target ready on this host")
    print("  ./scripts/emulate.py --target native,rosetta   # Run specific target(s)")
    print("  ./scripts/emulate.py --suite foreign-arch --setup-binfmt")
    print("  ./scripts/emulate.py --suite distros           # Alpine/Debian/Fedora/Arch + GCC & Clang")
    print("  ./scripts/emulate.py --suite all               # Full CI emulation matrix locally")


def resolve_targets(
    raw_targets: Sequence[str],
    raw_suites: Sequence[str],
    caps: HostCapabilities,
) -> list[str]:
    selected: list[str] = []

    def _add(t: str) -> None:
        if t not in selected:
            selected.append(t)

    for item in raw_suites:
        for piece in item.split(","):
            suite = piece.strip().lower()
            if not suite:
                continue
            if suite == "all":
                for t in ALL_TARGETS:
                    _add(t)
            elif suite == "available":
                for t in ALL_TARGETS:
                    ready, _ = target_readiness(t, caps)
                    if ready:
                        _add(t)
            elif suite in SUITE_TARGETS:
                for t in SUITE_TARGETS[suite]:
                    _add(t)
            else:
                valid = ", ".join((*SUITE_TARGETS.keys(), "available", "all"))
                raise ValueError(f"Unknown --suite {suite!r} (valid: {valid})")

    for item in raw_targets:
        for piece in item.split(","):
            t = piece.strip().lower()
            if not t:
                continue
            if t not in ALL_TARGETS:
                raise ValueError(f"Unknown --target {t!r} (valid: {', '.join(ALL_TARGETS)})")
            _add(t)

    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emulate.py",
        description="Emulate CrossCPU across architectures, ISAs, distros, and restricted containers on the host machine.",
    )
    parser.add_argument(
        "--list",
        "--doctor",
        dest="doctor",
        action="store_true",
        help="Inspect host emulation capabilities and list all targets with readiness status",
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        metavar="SUITE",
        help="Suite(s) to run (comma-separated or repeated): host, isa, foreign-arch, distros, adversarial, available, all",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="TARGET",
        help="Specific target(s) to run (comma-separated or repeated; run --list to see all names)",
    )
    parser.add_argument(
        "--runtime",
        choices=("auto", "docker", "podman"),
        default="auto",
        help="Container runtime for Linux/foreign-arch/distro/adversarial targets [default: auto]",
    )
    parser.add_argument(
        "--container-platform",
        default=None,
        metavar="PLAT",
        help="Optional --platform override for distro-* and adversarial targets (e.g. linux/amd64)",
    )
    parser.add_argument(
        "--setup-binfmt",
        action="store_true",
        help="Register multi-arch QEMU binfmt_misc handlers via tonistiigi/binfmt before container targets",
    )
    parser.add_argument(
        "--duration",
        default="150ms",
        metavar="TIME",
        help="Workload duration for CLI smoke runs [default: 150ms]",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        metavar="DIR",
        help="Directory for per-step logs and summary.md [default: /tmp/cross-cpu-emulation-<timestamp>]",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately after the first failed target",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which targets would run without executing them",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print stdout/stderr of passing steps in addition to failures",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    use_color = _supports_color()
    caps = detect_host_capabilities(args.runtime)

    if args.doctor or (not args.suite and not args.target):
        print_doctor(caps, use_color)
        return 0

    try:
        targets = resolve_targets(args.target, args.suite, caps)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not targets:
        print("No runnable targets matched the selection. Run with --doctor to inspect prerequisites.", file=sys.stderr)
        return 2

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_root = Path(args.log_dir) if args.log_dir else Path(tempfile.gettempdir()) / f"cross-cpu-emulation-{ts}"

    if args.dry_run:
        print(f"Dry run — log directory would be: {log_root}")
        for t in targets:
            ready, note = target_readiness(t, caps)
            state = "READY" if ready else "BLOCKED"
            print(f"  - {t:<16} [{state}] {note}")
        return 0

    log_root.mkdir(parents=True, exist_ok=True)
    print(f"CrossCPU Local Emulation Runner — logs: {log_root}")
    print(f"Selected targets ({len(targets)}): {', '.join(targets)}\n")

    if args.setup_binfmt:
        if not setup_binfmt(caps, log_root, args.verbose, use_color):
            return 2

    results: list[TargetResult] = []
    any_failed = False
    any_blocked = False

    try:
        for target in targets:
            ready, note = target_readiness(target, caps)
            print(f"==> Target: {target} ({note})")
            if not ready:
                badge = _color("SKIP", "33;1", use_color)
                print(f"  [{badge}] {target}: prerequisite missing — {note}\n")
                results.append(TargetResult(name=target, status="skipped", detail=note))
                any_blocked = True
                if args.fail_fast:
                    break
                continue

            logger = TargetLogger(
                target,
                log_root / target,
                env_path=caps.env_path,
                verbose=args.verbose,
                use_color=use_color,
            )

            if target == "native":
                res = run_target_native(caps, logger, args.duration)
            elif target == "rosetta":
                res = run_target_rosetta(caps, logger, args.duration)
            elif target == "x86-isa":
                res = run_target_x86_isa(caps, logger, args.duration)
            elif target == "arm64-isa":
                res = run_target_arm64_isa(caps, logger, args.duration)
            elif target in FOREIGN_ARCH_SPECS:
                res = run_target_foreign_arch(target, caps, logger, args.duration)
            elif target in DISTRO_SPECS:
                res = run_target_distro(target, caps, logger, args.container_platform)
            elif target == "adversarial":
                res = run_target_adversarial(caps, logger, args.duration, args.container_platform)
            else:
                res = TargetResult(name=target, status="failed", detail=f"unhandled target {target}")

            results.append(res)
            status_color = "32;1" if res.status == "passed" else "31;1"
            print(
                f"<== {target}: {_color(res.status.upper(), status_color, use_color)} "
                f"in {res.elapsed_s:.2f}s ({len(res.records)} step(s))\n"
            )
            if res.status == "failed":
                any_failed = True
                if args.fail_fast:
                    break
    except KeyboardInterrupt:
        print("\nInterrupted by user (SIGINT).", file=sys.stderr)
        any_failed = True

    report_path = write_markdown_report(log_root, results, caps)
    print("=" * 72)
    print(f"{'TARGET':<16} {'STATUS':<10} {'DURATION':>9}  DETAIL")
    print("-" * 72)
    for r in results:
        c = "32;1" if r.status == "passed" else ("33;1" if r.status == "skipped" else "31;1")
        st = r.status.upper()
        badge = _color(st, c, use_color)
        pad = len(badge) - len(st)
        print(f"{r.name:<16} {badge:<{10 + pad}} {r.elapsed_s:>8.2f}s  {r.detail}")
    print("-" * 72)
    print(f"Full logs & report: {report_path}")

    if any_failed:
        return 1
    if any_blocked:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
