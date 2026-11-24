# SPDX-License-Identifier: MIT
"""Host CPU, NUMA, cache hierarchy, and hybrid-core topology discovery."""

from __future__ import annotations

import ctypes
import difflib
import os
import platform
import re
import sys
from pathlib import Path

from .models import CacheInfo, CoreInfo, CPUTopology

CORE_PRESETS = ("all", "physical", "p-cores", "e-cores")


class TopologyUsageError(ValueError):
    """Raised when --cores or --numa has invalid syntax or unknown preset name."""


class TopologyUnavailableError(RuntimeError):
    """Raised when a syntactically valid --cores or --numa selection is unavailable."""


def parse_cpu_list(spec: str) -> list[int]:
    """Parse Linux-style CPU list notation (e.g. ``'0-3,6,8-9'``) into sorted unique IDs."""
    text = spec.strip()
    if not text:
        return []
    result: set[int] = set()
    for token in text.split(","):
        part = token.strip()
        if not part:
            raise TopologyUsageError(f"Empty element in CPU list: {spec!r}")
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2 or not bounds[0].isdigit() or not bounds[1].isdigit():
                raise TopologyUsageError(f"Invalid CPU range {part!r} in {spec!r}")
            start = int(bounds[0])
            end = int(bounds[1])
            if start > end:
                raise TopologyUsageError(
                    f"Inverted CPU range {start}-{end} in {spec!r}"
                )
            if end - start > 65536:
                raise TopologyUsageError(f"CPU range {part!r} exceeds reasonable bounds")
            for cpu_id in range(start, end + 1):
                result.add(cpu_id)
        else:
            if not part.isdigit():
                raise TopologyUsageError(f"Invalid CPU ID {part!r} in {spec!r}")
            result.add(int(part))
    return sorted(result)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    text = _read_text(path)
    if text is None or not text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        return None


def _parse_cache_size(raw: str | None) -> int:
    """Parse Linux sysfs cache size strings like ``'32K'``, ``'1024K'``, ``'16M'``."""
    if not raw:
        return 0
    m = re.fullmatch(r"(\d+)\s*([KMGkmg]?)[Bb]?", raw.strip())
    if not m:
        return 0
    value = int(m.group(1))
    suffix = m.group(2).upper()
    mult = {"": 1, "K": 1024, "M": 1024 * 1024, "G": 1024 * 1024 * 1024}[suffix]
    return value * mult


def _parse_proc_cpuinfo_model(proc_root: Path) -> str:
    content = _read_text(proc_root / "cpuinfo")
    if not content:
        return platform.processor() or platform.machine() or "Unknown CPU"
    for key in ("model name", "Model", "cpu", "Processor", "Hardware", "machine"):
        for line in content.splitlines():
            if ":" not in line:
                continue
            lhs, rhs = line.split(":", 1)
            if lhs.strip() == key and rhs.strip():
                return rhs.strip()
    return platform.processor() or platform.machine() or "Unknown CPU"


def _parse_proc_meminfo_bytes(proc_root: Path) -> int:
    content = _read_text(proc_root / "meminfo")
    if not content:
        return 0
    for line in content.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return 0


def discover_linux_topology(
    *,
    sysfs_root: Path = Path("/sys"),
    proc_root: Path = Path("/proc"),
    machine: str | None = None,
) -> CPUTopology:
    """Discover CPU, NUMA, cache hierarchy, and hybrid topology from Linux sysfs."""
    cpu_root = sysfs_root / "devices" / "system" / "cpu"
    node_root = sysfs_root / "devices" / "system" / "node"

    # Determine online logical CPUs
    online_str = _read_text(cpu_root / "online")
    cpu_ids: list[int] = []
    if online_str:
        try:
            cpu_ids = parse_cpu_list(online_str)
        except TopologyUsageError:
            cpu_ids = []

    if not cpu_ids and cpu_root.is_dir():
        found: list[int] = []
        for entry in cpu_root.iterdir():
            m = re.fullmatch(r"cpu(\d+)", entry.name)
            if m and entry.is_dir():
                cid = int(m.group(1))
                online_flag = _read_int(entry / "online")
                if online_flag is None or online_flag == 1:
                    found.append(cid)
        cpu_ids = sorted(found)

    if not cpu_ids:
        fallback_count = os.cpu_count() or 1
        cpu_ids = list(range(fallback_count))

    # Map logical CPU -> NUMA node
    cpu_to_numa: dict[int, int] = {}
    numa_nodes_set: set[int] = set()
    if node_root.is_dir():
        for entry in sorted(node_root.iterdir(), key=lambda p: p.name):
            m = re.fullmatch(r"node(\d+)", entry.name)
            if not m or not entry.is_dir():
                continue
            nid = int(m.group(1))
            cpulist = _read_text(entry / "cpulist")
            if cpulist:
                try:
                    members = parse_cpu_list(cpulist)
                except TopologyUsageError:
                    members = []
                if members:
                    numa_nodes_set.add(nid)
                    for cid in members:
                        cpu_to_numa[cid] = nid
    if not numa_nodes_set:
        numa_nodes_set.add(0)

    # Hybrid detection: x86 cpu_core / cpu_atom first
    x86_p_set: set[int] = set()
    x86_e_set: set[int] = set()
    core_cpus_txt = _read_text(sysfs_root / "devices" / "cpu_core" / "cpus")
    atom_cpus_txt = _read_text(sysfs_root / "devices" / "cpu_atom" / "cpus")
    if core_cpus_txt and atom_cpus_txt:
        try:
            x86_p_set = set(parse_cpu_list(core_cpus_txt))
            x86_e_set = set(parse_cpu_list(atom_cpus_txt))
        except TopologyUsageError:
            x86_p_set.clear()
            x86_e_set.clear()

    # ARM big.LITTLE / asymmetric capacity detection via cpu*/cpu_capacity
    capacities: dict[int, int] = {}
    for cid in cpu_ids:
        cap = _read_int(cpu_root / f"cpu{cid}" / "cpu_capacity")
        if cap is not None and cap > 0:
            capacities[cid] = cap

    arm_hybrid = False
    max_cap = 0
    if not (x86_p_set and x86_e_set) and len(capacities) == len(cpu_ids):
        unique_caps = sorted(set(capacities.values()))
        if len(unique_caps) >= 2:
            arm_hybrid = True
            max_cap = unique_caps[-1]

    # Per-CPU core/socket mapping
    cores: list[CoreInfo] = []
    sockets_set: set[int] = set()
    physical_keys: set[tuple[int, int]] = set()
    p_core_ids: list[int] = []
    e_core_ids: list[int] = []

    for cid in cpu_ids:
        topo_dir = cpu_root / f"cpu{cid}" / "topology"
        socket_id = _read_int(topo_dir / "physical_package_id")
        if socket_id is None or socket_id < 0:
            socket_id = 0
        core_id = _read_int(topo_dir / "core_id")
        if core_id is None or core_id < 0:
            # Fallback to lowest thread sibling or logical id
            siblings_txt = _read_text(topo_dir / "thread_siblings_list")
            if siblings_txt:
                try:
                    sibs = parse_cpu_list(siblings_txt)
                    core_id = sibs[0] if sibs else cid
                except TopologyUsageError:
                    core_id = cid
            else:
                core_id = cid

        numa_node = cpu_to_numa.get(cid, 0)
        cap = capacities.get(cid)
        if x86_p_set and x86_e_set:
            if cid in x86_p_set:
                ctype = "p-core"
                p_core_ids.append(cid)
            elif cid in x86_e_set:
                ctype = "e-core"
                e_core_ids.append(cid)
            else:
                ctype = "standard"
        elif arm_hybrid and cap is not None:
            if cap == max_cap:
                ctype = "p-core"
                p_core_ids.append(cid)
            else:
                ctype = "e-core"
                e_core_ids.append(cid)
        else:
            ctype = "standard"

        sockets_set.add(socket_id)
        physical_keys.add((socket_id, core_id))
        cores.append(
            CoreInfo(
                logical_id=cid,
                physical_core_id=core_id,
                socket_id=socket_id,
                numa_node=numa_node,
                core_type=ctype,
                capacity=cap,
            )
        )

    # Cache hierarchy discovery (prefer a P-core or cpu0)
    probe_cpu = p_core_ids[0] if p_core_ids else cpu_ids[0]
    caches = _discover_linux_caches(cpu_root / f"cpu{probe_cpu}" / "cache")

    is_hybrid = bool(p_core_ids and e_core_ids)
    return CPUTopology(
        os_name="Linux",
        machine=machine or platform.machine(),
        model_name=_parse_proc_cpuinfo_model(proc_root),
        sockets=max(1, len(sockets_set)),
        physical_cores=max(1, len(physical_keys)),
        logical_cores=len(cores),
        numa_nodes=tuple(sorted(numa_nodes_set)),
        is_hybrid=is_hybrid,
        p_core_ids=tuple(p_core_ids),
        e_core_ids=tuple(e_core_ids),
        memory_bytes=_parse_proc_meminfo_bytes(proc_root),
        caches=caches,
        cores=tuple(cores),
        pinning_supported=True,
        pinning_note="Linux hard core affinity pinning via sched_setaffinity() with first-touch NUMA allocation",
    )


def _discover_linux_caches(cache_dir: Path) -> CacheInfo:
    if not cache_dir.is_dir():
        return CacheInfo(discovered=False)

    l1i = 0
    l1d = 0
    l2 = 0
    l3 = 0
    line_size = 64
    l2_shared = 1
    l3_shared = 1
    found_any = False

    for entry in sorted(cache_dir.iterdir(), key=lambda p: p.name):
        if not entry.name.startswith("index") or not entry.is_dir():
            continue
        level = _read_int(entry / "level")
        ctype = (_read_text(entry / "type") or "").lower()
        size = _parse_cache_size(_read_text(entry / "size"))
        coherency = _read_int(entry / "coherency_line_size")
        if coherency and coherency > 0:
            line_size = coherency
        shared_txt = _read_text(entry / "shared_cpu_list")
        shared_count = 1
        if shared_txt:
            try:
                shared_count = max(1, len(parse_cpu_list(shared_txt)))
            except TopologyUsageError:
                shared_count = 1
        if size <= 0 or level is None:
            continue
        found_any = True
        if level == 1 and ctype == "instruction":
            l1i = max(l1i, size)
        elif level == 1 and ctype in ("data", "unified"):
            l1d = max(l1d, size)
        elif level == 2:
            l2 = max(l2, size)
            l2_shared = max(l2_shared, shared_count)
        elif level == 3:
            l3 = max(l3, size)
            l3_shared = max(l3_shared, shared_count)

    return CacheInfo(
        l1i_bytes=l1i,
        l1d_bytes=l1d,
        l2_bytes=l2,
        l3_bytes=l3,
        line_size_bytes=line_size,
        l2_shared_cpus=l2_shared,
        l3_shared_cpus=l3_shared,
        discovered=found_any,
    )


def _macos_sysctl_int(libc: ctypes.CDLL, name: str) -> int | None:
    val = ctypes.c_int64(0)
    size = ctypes.c_size_t(ctypes.sizeof(val))
    rc = libc.sysctlbyname(name.encode("ascii"), ctypes.byref(val), ctypes.byref(size), None, 0)
    if rc != 0:
        return None
    if size.value == 4:
        val32 = ctypes.c_int32(0)
        size32 = ctypes.c_size_t(4)
        if (
            libc.sysctlbyname(
                name.encode("ascii"), ctypes.byref(val32), ctypes.byref(size32), None, 0
            )
            == 0
        ):
            return int(val32.value)
    return int(val.value)


def _macos_sysctl_str(libc: ctypes.CDLL, name: str) -> str | None:
    size = ctypes.c_size_t(0)
    rc = libc.sysctlbyname(name.encode("ascii"), None, ctypes.byref(size), None, 0)
    if rc != 0 or size.value == 0:
        return None
    buf = ctypes.create_string_buffer(size.value)
    rc = libc.sysctlbyname(name.encode("ascii"), buf, ctypes.byref(size), None, 0)
    if rc != 0:
        return None
    return buf.value.decode("utf-8", errors="replace").strip()


def discover_macos_topology(
    sysctl_values: dict[str, int | str] | None = None,
    *,
    machine: str | None = None,
) -> CPUTopology:
    """Discover CPU, cache, and P-core/E-core performance-level topology on macOS."""
    libc: ctypes.CDLL | None = None
    if sysctl_values is None:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        libc.sysctlbyname.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        libc.sysctlbyname.restype = ctypes.c_int

    def get_int(key: str) -> int | None:
        if sysctl_values is not None:
            val = sysctl_values.get(key)
            return int(val) if isinstance(val, int) else None
        assert libc is not None
        return _macos_sysctl_int(libc, key)

    def get_str(key: str) -> str | None:
        if sysctl_values is not None:
            val = sysctl_values.get(key)
            return str(val) if isinstance(val, str) else None
        assert libc is not None
        return _macos_sysctl_str(libc, key)

    model = get_str("machdep.cpu.brand_string") or platform.machine() or "Apple Silicon"
    logical_total = get_int("hw.logicalcpu") or os.cpu_count() or 1
    physical_total = get_int("hw.physicalcpu") or logical_total
    mem_bytes = get_int("hw.memsize") or 0
    line_size = get_int("hw.cachelinesize") or 64
    nperflevels = get_int("hw.nperflevels") or 1

    cores: list[CoreInfo] = []
    p_core_ids: list[int] = []
    e_core_ids: list[int] = []

    if nperflevels >= 2:
        p_logical = get_int("hw.perflevel0.logicalcpu") or 0
        e_logical = get_int("hw.perflevel1.logicalcpu") or 0
        if p_logical + e_logical == logical_total and p_logical > 0 and e_logical > 0:
            for idx in range(p_logical):
                p_core_ids.append(idx)
                cores.append(
                    CoreInfo(
                        logical_id=idx,
                        physical_core_id=idx,
                        socket_id=0,
                        numa_node=0,
                        core_type="p-core",
                    )
                )
            for offset in range(e_logical):
                cid = p_logical + offset
                e_core_ids.append(cid)
                cores.append(
                    CoreInfo(
                        logical_id=cid,
                        physical_core_id=cid,
                        socket_id=0,
                        numa_node=0,
                        core_type="e-core",
                    )
                )

    if not cores:
        smt_factor = max(1, logical_total // max(1, physical_total))
        for idx in range(logical_total):
            cores.append(
                CoreInfo(
                    logical_id=idx,
                    physical_core_id=idx // smt_factor,
                    socket_id=0,
                    numa_node=0,
                    core_type="standard",
                )
            )

    l1i = get_int("hw.perflevel0.l1icachesize") or get_int("hw.l1icachesize") or 0
    l1d = get_int("hw.perflevel0.l1dcachesize") or get_int("hw.l1dcachesize") or 0
    l2 = get_int("hw.perflevel0.l2cachesize") or get_int("hw.l2cachesize") or 0
    l3 = (get_int("hw.perflevel0.l3cachesize") or get_int("hw.perflevel1.l3cachesize") or get_int("hw.l3cachesize") or 0)
    cpus_per_l2 = get_int("hw.perflevel0.cpusperl2") or 1
    cpus_per_l3 = (get_int("hw.perflevel0.cpusperl3") or get_int("hw.perflevel1.cpusperl3") or (logical_total if l3 > 0 else 1))

    caches = CacheInfo(
        l1i_bytes=l1i,
        l1d_bytes=l1d,
        l2_bytes=l2,
        l3_bytes=l3,
        line_size_bytes=line_size,
        l2_shared_cpus=max(1, cpus_per_l2),
        l3_shared_cpus=max(1, cpus_per_l3),
        discovered=(l1d > 0 or l2 > 0 or l3 > 0),
    )

    return CPUTopology(
        os_name="macOS",
        machine=machine or platform.machine(),
        model_name=model,
        sockets=1,
        physical_cores=physical_total,
        logical_cores=len(cores),
        numa_nodes=(0,),
        is_hybrid=bool(p_core_ids and e_core_ids),
        p_core_ids=tuple(p_core_ids),
        e_core_ids=tuple(e_core_ids),
        memory_bytes=mem_bytes,
        caches=caches,
        cores=tuple(cores),
        pinning_supported=False,
        pinning_note=(
            "macOS does not expose hard user-space CPU core pinning; "
            "CrossCPU uses pthread_set_qos_class_self_np() "
            "(QOS_CLASS_USER_INTERACTIVE for p-cores, QOS_CLASS_BACKGROUND for e-cores)"
        ),
    )


def discover_generic_topology():
    count = os.cpu_count() or 1
    cores = tuple(
        CoreInfo(logical_id=i, physical_core_id=i, socket_id=0, numa_node=0)
        for i in range(count)
    )
    return CPUTopology(
        os_name=platform.system() or sys.platform,
        machine=platform.machine(),
        model_name=platform.processor() or platform.machine() or "Generic POSIX CPU",
        sockets=1,
        physical_cores=count,
        logical_cores=count,
        numa_nodes=(0,),
        is_hybrid=False,
        p_core_ids=(),
        e_core_ids=(),
        memory_bytes=0,
        caches=CacheInfo(discovered=False),
        cores=cores,
        pinning_supported=False,
        pinning_note="Generic POSIX host; OS scheduler manages thread placement",
    )


def discover_topology() -> CPUTopology:
    """Discover host CPU topology for Linux, macOS, or generic POSIX fallback."""
    if sys.platform.startswith("linux"):
        return discover_linux_topology()
    elif sys.platform.startswith("darwin"):
        return discover_macos_topology()
    else:
        return discover_generic_topology()
    

def select_cores(
    topology: CPUTopology,
    cores_spec: str = "all",
    numa_spec: str = "all",
) -> tuple[list[int], int, list[str]]:
    """
    Resolve ``--cores`` and ``--numa`` against discovered host topology.

    Returns ``(selected_logical_core_ids, qos_hint, notes)``.
    Raises ``TopologyUsageError`` on bad syntax/unknown preset and
    ``TopologyUnavailableError`` when a valid request matches zero host CPUs.
    """
    notes: list[str] = []
    spec = cores_spec.strip().lower()
    numa = numa_spec.strip().lower()

    # Validate --numa
    target_numa: int | None = None
    if numa != "all":
        if not numa.isdigit():
            raise TopologyUsageError(
                f"Invalid --numa value {numa_spec!r}; expected 'all' or a non-negative integer node ID"
            )
        target_numa = int(numa)
        if target_numa not in topology.numa_nodes:
            avail = ", ".join(str(n) for n in topology.numa_nodes)
            raise TopologyUnavailableError(
                f"NUMA node {target_numa} is not available on this host (available nodes: {avail})"
            )

    core_by_id = {c.logical_id: c for c in topology.cores}
    qos_hint = 0  # CROSS_CPU_QOS_DEFAULT

    if spec == "all":
        selected = [c.logical_id for c in topology.cores]
    elif spec == "physical":
        seen_phys: set[tuple[int, int]] = set()
        selected = []
        for c in topology.cores:
            key = (c.socket_id, c.physical_core_id)
            if key not in seen_phys:
                seen_phys.add(key)
                selected.append(c.logical_id)
    elif spec == "p-cores":
        if not topology.is_hybrid or not topology.p_core_ids:
            raise TopologyUnavailableError(
                "Requested --cores p-cores, but this host does not expose hybrid P-core/E-core clusters"
            )
        selected = list(topology.p_core_ids)
        qos_hint = 1  # QOS_CLASS_USER_INTERACTIVE
    elif spec == "e-cores":
        if not topology.is_hybrid or not topology.e_core_ids:
            raise TopologyUnavailableError(
                "Requested --cores e-cores, but this host does not expose hybrid P-core/E-core clusters"
            )
        selected = list(topology.e_core_ids)
        qos_hint = 2  # QOS_CLASS_BACKGROUND
    else:
        # Check whether user typed a close preset typo (e.g. "p-core", "phys")
        if any(ch.isalpha() for ch in spec):
            matches = difflib.get_close_matches(spec, CORE_PRESETS, n=1, cutoff=0.5)
            hint = f" Did you mean '{matches[0]}'?" if matches else ""
            presets = ", ".join(CORE_PRESETS)
            raise TopologyUsageError(
                f"Unknown --cores preset {cores_spec!r} (valid presets: {presets}, or CPU list like 0-3,6).{hint}"
            )
        requested = parse_cpu_list(spec)
        if not requested:
            raise TopologyUsageError("Empty --cores list")
        missing = [cid for cid in requested if cid not in core_by_id]
        if missing:
            max_id = max(core_by_id.keys())
            raise TopologyUnavailableError(
                f"Requested CPU ID(s) {missing} are not online on this host (online range: 0..{max_id})"
            )
        selected = requested

    if target_numa is not None:
        selected = [cid for cid in selected if core_by_id[cid].numa_node == target_numa]
        if not selected:
            raise TopologyUnavailableError(
                f"No cores match --cores {cores_spec} on NUMA node {target_numa}"
            )

    if not topology.pinning_supported:
        notes.append(topology.pinning_note)

    return selected, qos_hint, notes
