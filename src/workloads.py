# SPDX-License-Identifier: MIT
"""Canonical workload registry, fixed SI scaling (10^9), and topology-aware buffer sizing."""

from __future__ import annotations

from .models import CPUTopology, WorkloadSpec

SI_GIGA = 1_000_000_000.0

WORKLOAD_SPECS: tuple[WorkloadSpec, ...] = (
    # Integer (GOP/s, GOP)
    WorkloadSpec(
        workload_id=0,
        name="compute-int64",
        category="integer",
        rate_unit="GOP/s",
        total_unit="GOP",
        description="64-bit unsigned modular arithmetic with bit-exact analytical verification",
    ),
    WorkloadSpec(
        workload_id=1,
        name="compute-int32",
        category="integer",
        rate_unit="GOP/s",
        total_unit="GOP",
        description="32-bit unsigned modular arithmetic with bit-exact analytical verification",
    ),
    WorkloadSpec(
        workload_id=2,
        name="compute-int16",
        category="integer",
        rate_unit="GOP/s",
        total_unit="GOP",
        description="16-bit unsigned modular arithmetic with bit-exact analytical verification",
    ),
    WorkloadSpec(
        workload_id=3,
        name="compute-int8",
        category="integer",
        rate_unit="GOP/s",
        total_unit="GOP",
        description="8-bit unsigned modular arithmetic with bit-exact analytical verification",
    ),
    WorkloadSpec(
        workload_id=4,
        name="compute-vnni",
        category="integer",
        rate_unit="GOP/s",
        total_unit="GOP",
        description="Hardware integer dot-product / VNNI (x86 AVX-VNNI/AVX512-VNNI or ARM DotProd/I8MM)",
        hardware_only=True,
    ),
    # Floating-Point & ML (GFLOP/s, GFLOP)
    WorkloadSpec(
        workload_id=5,
        name="compute-fp64",
        category="floating-point",
        rate_unit="GFLOP/s",
        total_unit="GFLOP",
        description="IEEE-754 FP64 contractive FMA recurrence with symmetry and ULP verification",
    ),
    WorkloadSpec(
        workload_id=6,
        name="compute-fp32",
        category="floating-point",
        rate_unit="GFLOP/s",
        total_unit="GFLOP",
        description="IEEE-754 FP32 contractive FMA recurrence with symmetry and ULP verification",
    ),
    WorkloadSpec(
        workload_id=7,
        name="compute-fp16",
        category="floating-point",
        rate_unit="GFLOP/s",
        total_unit="GFLOP",
        description="Native hardware IEEE-754 FP16 FMA recurrence (no scalar upcasting)",
        hardware_only=True,
    ),
    WorkloadSpec(
        workload_id=8,
        name="compute-bf16",
        category="floating-point",
        rate_unit="GFLOP/s",
        total_unit="GFLOP",
        description="Native hardware BFloat16 dot-product / FMA recurrence (no scalar upcasting)",
        hardware_only=True,
    ),
    # Matrix Engines (GFLOP/s / GOP/s)
    WorkloadSpec(
        workload_id=9,
        name="compute-matrix-fp32",
        category="matrix",
        rate_unit="GFLOP/s",
        total_unit="GFLOP",
        description="Hardware FP32 matrix tile multiply (ARM SME) with analytical tile verification",
        hardware_only=True,
    ),
    WorkloadSpec(
        workload_id=10,
        name="compute-matrix-bf16",
        category="matrix",
        rate_unit="GFLOP/s",
        total_unit="GFLOP",
        description="Hardware BF16 matrix tile multiply (Intel AMX / ARM SME) with analytical verification",
        hardware_only=True,
    ),
    WorkloadSpec(
        workload_id=11,
        name="compute-matrix-int8",
        category="matrix",
        rate_unit="GOP/s",
        total_unit="GOP",
        description="Hardware INT8 matrix tile multiply (Intel AMX / ARM SME) with bit-exact verification",
        hardware_only=True,
    ),
    # Hardware Crypto (GB/s, GB)
    WorkloadSpec(
        workload_id=12,
        name="crypto-aes",
        category="crypto",
        rate_unit="GB/s",
        total_unit="GB",
        description="Hardware AES-128 encryption continuously verified against NIST FIPS 197 KAT",
        hardware_only=True,
    ),
    WorkloadSpec(
        workload_id=13,
        name="crypto-sha256",
        category="crypto",
        rate_unit="GB/s",
        total_unit="GB",
        description="Hardware SHA-256 compression continuously verified against NIST FIPS 180-4 KAT",
        hardware_only=True,
    ),
    # Cache Tiers (GB/s, GB)
    WorkloadSpec(
        workload_id=14,
        name="cache-l1",
        category="cache",
        rate_unit="GB/s",
        total_unit="GB",
        description="L1 data cache (~50% L1d) read/write throughput and bit-exact verification",
    ),
    WorkloadSpec(
        workload_id=15,
        name="cache-l2",
        category="cache",
        rate_unit="GB/s",
        total_unit="GB",
        description="L2 cache (~50% L2) read/write throughput and bit-exact verification",
    ),
    WorkloadSpec(
        workload_id=16,
        name="cache-l3",
        category="cache",
        rate_unit="GB/s",
        total_unit="GB",
        description="Shared L3/LLC (~50% L3) read/write throughput and bit-exact verification",
    ),
    # RAM & Coherency
    WorkloadSpec(
        workload_id=17,
        name="bandwidth-ram",
        category="memory",
        rate_unit="GB/s",
        total_unit="GB",
        description="Main memory bandwidth (>=4x LLC) using non-temporal stores where available",
    ),
    WorkloadSpec(
        workload_id=18,
        name="memory-pattern",
        category="memory",
        rate_unit="GB/s",
        total_unit="GB",
        description="Adversarial DRAM bit-pattern sweep (checkerboard, walking-1/0) with cache flush",
    ),
    WorkloadSpec(
        workload_id=19,
        name="memory-sweep",
        category="memory",
        rate_unit="GB/s",
        total_unit="GB",
        description="SplitMix64 address-hashed DRAM sweep with cache flush/eviction verification",
    ),
    WorkloadSpec(
        workload_id=20,
        name="coherency-atomic",
        category="coherency",
        rate_unit="GOP/s",
        total_unit="GOP",
        description="Multi-core C11 <stdatomic.h> cache-line contention with exact global count check",
    ),
)

WORKLOAD_BY_NAME: dict[str, WorkloadSpec] = {spec.name: spec for spec in WORKLOAD_SPECS}
WORKLOAD_BY_ID: dict[int, WorkloadSpec] = {spec.workload_id: spec for spec in WORKLOAD_SPECS}
ALL_WORKLOAD_NAMES: tuple[str, ...] = tuple(spec.name for spec in WORKLOAD_SPECS)


def _align_128(value: int, minimum: int = 4096) -> int:
    n = max(minimum, int(value))
    rem = n % 128
    return n if rem == 0 else n + (128 - rem)


def compute_workload_buffer_bytes(
    spec: WorkloadSpec,
    topology: CPUTopology,
    num_threads: int,
) -> tuple[int, str | None]:
    """
    Compute per-thread working-set bytes dynamically from discovered topology.

    Returns ``(buffer_bytes, unsupported_reason)``.
    """
    threads = max(1, int(num_threads))
    caches = topology.caches
    mem_cap_total = (
        max(32 * 1024 * 1024, topology.memory_bytes // 4)
        if topology.memory_bytes > 0
        else 256 * 1024 * 1024
    )
    mem_cap_per_thread = max(4 * 1024 * 1024, mem_cap_total // threads)

    if spec.name == "cache-l1":
        l1d = caches.l1d_bytes if caches.l1d_bytes > 0 else 32 * 1024
        return _align_128(l1d // 2, minimum=4096), None

    if spec.name == "cache-l2":
        l2 = caches.l2_bytes if caches.l2_bytes > 0 else 512 * 1024
        sharing = max(1, min(threads, caches.l2_shared_cpus))
        return _align_128((l2 // 2) // sharing, minimum=16 * 1024), None

    if spec.name == "cache-l3":
        if caches.discovered and caches.l3_bytes <= 0:
            return (
                0,
                "No L3 cache tier reported by CPU topology (last-level cache is L2)",
            )
        l3 = caches.l3_bytes if caches.l3_bytes > 0 else 8 * 1024 * 1024
        sharing = max(1, min(threads, caches.l3_shared_cpus))
        per_thread = min(mem_cap_per_thread, (l3 // 2) // sharing)
        return _align_128(per_thread, minimum=64 * 1024), None

    if spec.name == "bandwidth-ram":
        llc = caches.l3_bytes or caches.l2_bytes or (8 * 1024 * 1024)
        # Ensure aggregate working set across all threads is >= 4x LLC
        target_total = max(4 * llc, 32 * 1024 * 1024)
        per_thread = max(4 * 1024 * 1024, (target_total + threads - 1) // threads)
        per_thread = min(per_thread, 64 * 1024 * 1024, mem_cap_per_thread)
        return _align_128(per_thread, minimum=4 * 1024 * 1024), None

    if spec.name in ("memory-pattern", "memory-sweep"):
        llc = caches.l3_bytes or caches.l2_bytes or (4 * 1024 * 1024)
        # Large enough to exceed caches while finishing cache-flushed sweeps cleanly in ms runs
        per_thread = max(2 * 1024 * 1024, min(llc, 8 * 1024 * 1024))
        per_thread = min(per_thread, mem_cap_per_thread)
        return _align_128(per_thread, minimum=1024 * 1024), None

    return 0, None


def scale_work_metrics(raw_ops: int, elapsed_ns: int) -> tuple[float, float]:
    """
    Convert raw ops/FLOPs/bytes and elapsed nanoseconds into fixed SI 10^9 units.

    Returns ``(rate_giga_per_sec, total_giga)``. Never auto-scales units.
    """
    total_giga = float(raw_ops) / SI_GIGA
    if elapsed_ns <= 0:
        return 0.0, total_giga
    duration_s = float(elapsed_ns) / SI_GIGA
    rate = total_giga / duration_s if duration_s > 0.0 else 0.0
    return rate, total_giga
