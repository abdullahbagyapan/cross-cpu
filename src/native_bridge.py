# SPDX-License-Identifier: MIT
"""ABI-safe ctypes bridge to the compiled C11 CrossCPU native shared library."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Sequence

from .models import ISABackend, NativeCapabilities

CROSS_CPU_ABI_VERSION = 1
CROSS_CPU_MAX_ISAS = 16
CROSS_CPU_MAX_NAME = 32
CROSS_CPU_MAX_REASON = 256
CROSS_CPU_MAX_CORES = 1024

# Native status codes matching native/include/cross_cpu_abi.h
STATUS_OK = 0
STATUS_UNSUPPORTED = 1
STATUS_INIT_ERROR = 2
STATUS_COMPUTE_ERROR = 3
STATUS_STOPPED = 4

# Canonical ISA name -> native enum ID
ISA_NAME_TO_ID: dict[str, int] = {
    "auto": 0,
    "scalar": 1,
    "sse4": 2,
    "avx2": 3,
    "avx512": 4,
    "amx": 5,
    "neon": 6,
    "sve": 7,
    "sme": 8,
    "rvv": 9,
    "vsx": 10,
    "vx": 11,
}

ISA_ID_TO_NAME: dict[int, str] = {v: k for k, v in ISA_NAME_TO_ID.items()}


class NativeBridgeError(RuntimeError):
    """Raised when the C11 shared library cannot be located, loaded, or initialized."""


class _CISAInfo(ctypes.Structure):
    _fields_ = [
        ("isa_id", ctypes.c_uint32),
        ("priority", ctypes.c_uint32),
        ("available", ctypes.c_uint32),
        ("compiled_in", ctypes.c_uint32),
        ("workload_mask", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("name", ctypes.c_char * CROSS_CPU_MAX_NAME),
        ("detail", ctypes.c_char * CROSS_CPU_MAX_REASON),
    ]


class _CCaps(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("arch_id", ctypes.c_uint32),
        ("endianness", ctypes.c_uint32),
        ("isa_count", ctypes.c_uint32),
        ("raw_probe", ctypes.c_uint64 * 4),
        ("feature_flags", ctypes.c_uint64),
        ("arch_name", ctypes.c_char * CROSS_CPU_MAX_NAME),
        ("probe_summary", ctypes.c_char * CROSS_CPU_MAX_REASON),
        ("isas", _CISAInfo * CROSS_CPU_MAX_ISAS),
    ]


class _CRunConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("workload_id", ctypes.c_uint32),
        ("requested_isa", ctypes.c_uint32),
        ("num_threads", ctypes.c_uint32),
        ("duration_ns", ctypes.c_uint64),
        ("batch_ns", ctypes.c_uint64),
        ("buffer_bytes", ctypes.c_uint64),
        ("seed", ctypes.c_uint64),
        ("qos_class", ctypes.c_uint32),
        ("inject_fault", ctypes.c_uint32),
        ("core_ids", ctypes.c_int32 * CROSS_CPU_MAX_CORES),
    ]


class _CRunResult(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("status", ctypes.c_int32),
        ("selected_isa", ctypes.c_uint32),
        ("threads_executed", ctypes.c_uint32),
        ("total_ops", ctypes.c_uint64),
        ("elapsed_ns", ctypes.c_uint64),
        ("error_count", ctypes.c_uint64),
        ("iterations", ctypes.c_uint64),
        ("expected_sample", ctypes.c_double),
        ("observed_sample", ctypes.c_double),
        ("failed_thread", ctypes.c_uint32),
        ("failed_core", ctypes.c_uint32),
        ("isa_name", ctypes.c_char * CROSS_CPU_MAX_NAME),
        ("reason", ctypes.c_char * CROSS_CPU_MAX_REASON),
    ]


def _decode_cstr(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()


def find_native_library(explicit_path: str | Path | None = None) -> Path:
    """Locate the compiled ``_native`` shared library (.so or .dylib)."""
    if explicit_path is not None:
        p = Path(explicit_path).expanduser().resolve()
        if p.is_file():
            return p
        raise NativeBridgeError(f"Configured native library not found: {p}")

    env_path = os.environ.get("CROSS_CPU_NATIVE_LIB")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_file():
            return p
        raise NativeBridgeError(f"CROSS_CPU_NATIVE_LIB does not exist: {p}")

    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent

    # Prefer platform-specific extension and ABI-tagged setuptools outputs
    ext_order = (".dylib", ".so") if sys.platform == "darwin" else (".so", ".dylib")

    search_dirs = [
        pkg_dir,
        repo_root / "build",
        repo_root / "native" / "build",
    ]

    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for ext in ext_order:
            direct = directory / f"_native{ext}"
            if direct.is_file():
                return direct
        # Match setuptools ABI-tagged binaries (e.g. _native.cpython-312-x86_64-linux-gnu.so)
        for ext in ext_order:
            candidates = sorted(directory.glob(f"_native.*{ext}"))
            if candidates:
                return candidates[0]

    raise NativeBridgeError(
        "Compiled C11 native library (_native.so/_native.dylib) not found. "
        "Run 'make build' or install the package via 'pip install --no-build-isolation --no-deps .'."
    )


class NativeLibrary:
    """Safe Python wrapper around the CrossCPU C11 native shared library."""

    def __init__(self, library_path: str | Path | None = None) -> None:
        self.path = find_native_library(library_path)
        try:
            self._lib = ctypes.CDLL(str(self.path))
        except OSError as exc:
            raise NativeBridgeError(
                f"Failed to load native library {self.path}: {exc}"
            ) from exc

        self._bind_symbols()

        abi_ver = int(self._lib.cross_cpu_abi_version())
        if abi_ver != CROSS_CPU_ABI_VERSION:
            raise NativeBridgeError(
                f"Native ABI version mismatch: expected {CROSS_CPU_ABI_VERSION}, got {abi_ver}"
            )

        rc = int(self._lib.cross_cpu_init())
        if rc != STATUS_OK:
            raise NativeBridgeError(
                f"cross_cpu_init() failed with native status {rc}"
            )

    def _bind_symbols(self) -> None:
        lib = self._lib

        lib.cross_cpu_abi_version.argtypes = []
        lib.cross_cpu_abi_version.restype = ctypes.c_uint32

        lib.cross_cpu_init.argtypes = []
        lib.cross_cpu_init.restype = ctypes.c_int32

        lib.cross_cpu_get_caps.argtypes = [ctypes.POINTER(_CCaps)]
        lib.cross_cpu_get_caps.restype = ctypes.c_int32

        lib.cross_cpu_resolve_isa.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.cross_cpu_resolve_isa.restype = ctypes.c_int32

        lib.cross_cpu_run_workload.argtypes = [
            ctypes.POINTER(_CRunConfig),
            ctypes.POINTER(_CRunResult),
        ]
        lib.cross_cpu_run_workload.restype = ctypes.c_int32

        lib.cross_cpu_request_stop.argtypes = []
        lib.cross_cpu_request_stop.restype = None

        lib.cross_cpu_reset_stop.argtypes = []
        lib.cross_cpu_reset_stop.restype = None

        lib.cross_cpu_get_progress_ops.argtypes = []
        lib.cross_cpu_get_progress_ops.restype = ctypes.c_uint64

    def get_caps(self) -> NativeCapabilities:
        """Query detected host architecture and registered ISA backends."""
        raw = _CCaps()
        rc = int(self._lib.cross_cpu_get_caps(ctypes.byref(raw)))
        if rc != STATUS_OK:
            raise NativeBridgeError(f"cross_cpu_get_caps() failed with status {rc}")

        backends: list[ISABackend] = []
        count = min(int(raw.isa_count), CROSS_CPU_MAX_ISAS)
        for idx in range(count):
            entry = raw.isas[idx]
            backends.append(
                ISABackend(
                    isa_id=int(entry.isa_id),
                    name=_decode_cstr(bytes(entry.name)),
                    priority=int(entry.priority),
                    available=bool(entry.available),
                    compiled_in=bool(entry.compiled_in),
                    workload_mask=int(entry.workload_mask),
                    detail=_decode_cstr(bytes(entry.detail)),
                )
            )

        return NativeCapabilities(
            abi_version=int(raw.abi_version),
            arch_id=int(raw.arch_id),
            arch_name=_decode_cstr(bytes(raw.arch_name)),
            endianness="big" if int(raw.endianness) == 1 else "little",
            raw_probe=(
                int(raw.raw_probe[0]),
                int(raw.raw_probe[1]),
                int(raw.raw_probe[2]),
                int(raw.raw_probe[3]),
            ),
            feature_flags=int(raw.feature_flags),
            probe_summary=_decode_cstr(bytes(raw.probe_summary)),
            backends=tuple(backends),
        )

    def resolve_isa(
        self, workload_id: int, requested_isa: str = "auto"
    ) -> tuple[int, str, str]:
        """Resolve native ISA for a workload; return ``(status, isa_name, reason)``."""
        isa_id = ISA_NAME_TO_ID.get(requested_isa.lower())
        if isa_id is None:
            return STATUS_INIT_ERROR, "none", f"Unknown ISA name: {requested_isa!r}"
        out_isa = ctypes.c_uint32(0)
        reason_buf = ctypes.create_string_buffer(CROSS_CPU_MAX_REASON)
        rc = int(
            self._lib.cross_cpu_resolve_isa(
                ctypes.c_uint32(workload_id),
                ctypes.c_uint32(isa_id),
                ctypes.byref(out_isa),
                reason_buf,
                ctypes.c_size_t(CROSS_CPU_MAX_REASON),
            )
        )
        isa_name = ISA_ID_TO_NAME.get(int(out_isa.value), "none")
        return rc, isa_name, _decode_cstr(reason_buf.raw)

    def run_workload(
        self,
        *,
        workload_id: int,
        requested_isa: str,
        core_ids: Sequence[int],
        duration_ns: int,
        batch_ns: int = 100_000_000,
        buffer_bytes: int = 0,
        seed: int = 0xC055C90012345678,
        qos_class: int = 0,
        inject_fault: bool = False,
    ) -> dict[str, int | float | str | None]:
        """Execute a C11 workload across ``core_ids`` and return raw result fields."""
        isa_id = ISA_NAME_TO_ID.get(requested_isa.lower())
        if isa_id is None:
            return {
                "status": STATUS_INIT_ERROR,
                "selected_isa": 0,
                "isa_name": "none",
                "threads_executed": 0,
                "total_ops": 0,
                "elapsed_ns": 0,
                "error_count": 0,
                "iterations": 0,
                "expected_sample": None,
                "observed_sample": None,
                "failed_thread": None,
                "failed_core": None,
                "reason": f"Unknown ISA name: {requested_isa!r}",
            }

        threads = min(len(core_ids), CROSS_CPU_MAX_CORES)
        cfg = _CRunConfig()
        cfg.abi_version = CROSS_CPU_ABI_VERSION
        cfg.workload_id = workload_id
        cfg.requested_isa = isa_id
        cfg.num_threads = threads
        cfg.duration_ns = max(0, int(duration_ns))
        cfg.batch_ns = max(1_000_000, int(batch_ns))
        cfg.buffer_bytes = max(0, int(buffer_bytes))
        cfg.seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        cfg.qos_class = int(qos_class)
        cfg.inject_fault = 1 if inject_fault else 0
        for idx in range(threads):
            cfg.core_ids[idx] = int(core_ids[idx])

        res = _CRunResult()
        self._lib.cross_cpu_run_workload(ctypes.byref(cfg), ctypes.byref(res))

        failed_thread = None if int(res.failed_thread) == 0xFFFFFFFF else int(res.failed_thread)
        failed_core = None if int(res.failed_core) == 0xFFFFFFFF else int(res.failed_core)

        return {
            "status": int(res.status),
            "selected_isa": int(res.selected_isa),
            "isa_name": _decode_cstr(bytes(res.isa_name)),
            "threads_executed": int(res.threads_executed),
            "total_ops": int(res.total_ops),
            "elapsed_ns": int(res.elapsed_ns),
            "error_count": int(res.error_count),
            "iterations": int(res.iterations),
            "expected_sample": float(res.expected_sample) if int(res.error_count) > 0 else None,
            "observed_sample": float(res.observed_sample) if int(res.error_count) > 0 else None,
            "failed_thread": failed_thread,
            "failed_core": failed_core,
            "reason": _decode_cstr(bytes(res.reason)),
        }

    def request_stop(self) -> None:
        """Signal C11 worker threads to terminate cleanly after their current slice."""
        self._lib.cross_cpu_request_stop()

    def reset_stop(self) -> None:
        """Clear the C11 atomic stop flag before starting a run."""
        self._lib.cross_cpu_reset_stop()

    def get_progress_ops(self) -> int:
        """Read cumulative ops/bytes completed by the active C11 workload."""
        return int(self._lib.cross_cpu_get_progress_ops())
