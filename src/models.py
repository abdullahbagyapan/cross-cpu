# SPDX-License-Identifier: MIT
"""Structured dataclasses shared across topology, native bridge, telemetry, and CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CoreInfo:
    """Description of a single OS logical CPU thread."""

    logical_id: int
    physical_core_id: int
    socket_id: int = 0
    numa_node: int = 0
    core_type: str = "standard"  # "p-core", "e-core", or "standard"
    capacity: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "logical_id": self.logical_id,
            "physical_core_id": self.physical_core_id,
            "socket_id": self.socket_id,
            "numa_node": self.numa_node,
            "core_type": self.core_type,
        }
        if self.capacity is not None:
            data["capacity"] = self.capacity
        return data


@dataclass(frozen=True)
class CacheInfo:
    """CPU cache hierarchy sizes in bytes (0 when absent/unknown)."""

    l1i_bytes: int = 0
    l1d_bytes: int = 0
    l2_bytes: int = 0
    l3_bytes: int = 0
    line_size_bytes: int = 64
    l2_shared_cpus: int = 1
    l3_shared_cpus: int = 1
    discovered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "l1i_bytes": self.l1i_bytes,
            "l1d_bytes": self.l1d_bytes,
            "l2_bytes": self.l2_bytes,
            "l3_bytes": self.l3_bytes,
            "line_size_bytes": self.line_size_bytes,
            "l2_shared_cpus": self.l2_shared_cpus,
            "l3_shared_cpus": self.l3_shared_cpus,
            "discovered": self.discovered,
        }


@dataclass(frozen=True)
class CPUTopology:
    """Discovered host CPU, NUMA, cache, and hybrid-core topology."""

    os_name: str
    machine: str
    model_name: str
    sockets: int
    physical_cores: int
    logical_cores: int
    numa_nodes: tuple[int, ...]
    is_hybrid: bool
    p_core_ids: tuple[int, ...]
    e_core_ids: tuple[int, ...]
    memory_bytes: int
    caches: CacheInfo
    cores: tuple[CoreInfo, ...]
    pinning_supported: bool
    pinning_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "os": self.os_name,
            "machine": self.machine,
            "model_name": self.model_name,
            "sockets": self.sockets,
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "numa_nodes": list(self.numa_nodes),
            "is_hybrid": self.is_hybrid,
            "p_core_ids": list(self.p_core_ids),
            "e_core_ids": list(self.e_core_ids),
            "memory_bytes": self.memory_bytes,
            "caches": self.caches.to_dict(),
            "pinning_supported": self.pinning_supported,
            "pinning_note": self.pinning_note,
            "cores": [c.to_dict() for c in self.cores],
        }


@dataclass(frozen=True)
class ISABackend:
    """Host capability metadata for a compiled/detected native ISA backend."""

    isa_id: int
    name: str
    priority: int
    available: bool
    compiled_in: bool
    workload_mask: int
    detail: str

    def supports_workload_id(self, workload_id: int) -> bool:
        return self.available and bool(self.workload_mask & (1 << workload_id))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeCapabilities:
    """Host architecture probe summary returned by the C11 shared library."""

    abi_version: int
    arch_id: int
    arch_name: str
    endianness: str
    raw_probe: tuple[int, int, int, int]
    feature_flags: int
    probe_summary: str
    backends: tuple[ISABackend, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "abi_version": self.abi_version,
            "arch_id": self.arch_id,
            "arch_name": self.arch_name,
            "endianness": self.endianness,
            "raw_probe": [hex(v) for v in self.raw_probe],
            "feature_flags": hex(self.feature_flags),
            "probe_summary": self.probe_summary,
            "backends": [b.to_dict() for b in self.backends],
        }


@dataclass(frozen=True)
class TelemetrySample:
    """A single sanitized 1 Hz telemetry snapshot (only valid metrics included)."""

    timestamp_unix: float
    elapsed_s: float
    metrics: dict[str, float | int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        clean = {k: v for k, v in self.metrics.items() if v is not None}
        return {
            "timestamp": round(self.timestamp_unix, 3),
            "elapsed_s": round(self.elapsed_s, 3),
            "metrics": clean,
        }


@dataclass(frozen=True)
class WorkloadSpec:
    """Canonical definition of a CrossCPU stress & SDC verification workload."""

    workload_id: int
    name: str
    category: str
    rate_unit: str
    total_unit: str
    description: str
    hardware_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkloadResult:
    """Execution and verification result for a single workload run."""

    name: str
    category: str
    status: str  # "passed", "failed", "unsupported", "init_error", "stopped"
    isa: str
    threads: int
    duration_s: float
    rate: float
    rate_unit: str
    total_work: float
    total_unit: str
    raw_ops: int
    iterations: int
    error_count: int
    exit_code: int
    reason: str = ""
    expected_sample: float | None = None
    observed_sample: float | None = None
    failed_thread: int | None = None
    failed_core: int | None = None
    buffer_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "isa": self.isa,
            "threads": self.threads,
            "duration_s": round(self.duration_s, 6),
            "rate": round(self.rate, 4),
            "rate_unit": self.rate_unit,
            "total_work": round(self.total_work, 6),
            "total_unit": self.total_unit,
            "raw_ops": self.raw_ops,
            "iterations": self.iterations,
            "error_count": self.error_count,
            "exit_code": self.exit_code,
        }
        if self.reason:
            out["reason"] = self.reason
        if self.buffer_bytes is not None and self.buffer_bytes > 0:
            out["buffer_bytes"] = self.buffer_bytes
        if self.error_count > 0:
            if self.expected_sample is not None:
                out["expected_sample"] = self.expected_sample
            if self.observed_sample is not None:
                out["observed_sample"] = self.observed_sample
            if self.failed_thread is not None:
                out["failed_thread"] = self.failed_thread
            if self.failed_core is not None:
                out["failed_core"] = self.failed_core
        return out
