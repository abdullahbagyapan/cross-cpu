# SPDX-License-Identifier: MIT
"""Offline unit tests for Linux sysfs and macOS sysctl topology discovery and core selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.topology import (
    TopologyUnavailableError,
    TopologyUsageError,
    discover_linux_topology,
    discover_macos_topology,
    discover_topology,
    parse_cpu_list,
    select_cores,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestParseCPUList(unittest.TestCase):
    def test_valid_lists(self) -> None:
        self.assertEqual(parse_cpu_list("0"), [0])
        self.assertEqual(parse_cpu_list("0-3,6,8-9"), [0, 1, 2, 3, 6, 8, 9])
        self.assertEqual(parse_cpu_list("3,1,2,1"), [1, 2, 3])

    def test_invalid_lists(self) -> None:
        for bad in ("5-2", "0,,2", "-1", "a-b", "1-2-3"):
            with self.assertRaises(TopologyUsageError, msg=f"Should fail: {bad!r}"):
                parse_cpu_list(bad)


class TestLinuxTopologyDiscovery(unittest.TestCase):
    def test_multi_socket_numa_and_caches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs = root / "sys"
            proc = root / "proc"

            _write(sysfs / "devices/system/cpu/online", "0-7")
            _write(sysfs / "devices/system/node/node0/cpulist", "0-3")
            _write(sysfs / "devices/system/node/node1/cpulist", "4-7")
            _write(proc / "cpuinfo", "model name\t: Mock Dual-Socket EPYC\n")
            _write(proc / "meminfo", "MemTotal:       67108864 kB\n")

            # Socket 0: CPUs 0,2 share core 0; CPUs 1,3 share core 1
            # Socket 1: CPUs 4,6 share core 0; CPUs 5,7 share core 1
            for cid in range(8):
                sock = 0 if cid < 4 else 1
                core = cid % 2
                cdir = sysfs / f"devices/system/cpu/cpu{cid}"
                _write(cdir / "topology/physical_package_id", str(sock))
                _write(cdir / "topology/core_id", str(core))

            c0 = sysfs / "devices/system/cpu/cpu0/cache"
            _write(c0 / "index0/level", "1")
            _write(c0 / "index0/type", "Data")
            _write(c0 / "index0/size", "32K")
            _write(c0 / "index0/coherency_line_size", "64")
            _write(c0 / "index0/shared_cpu_list", "0,2")

            _write(c0 / "index1/level", "1")
            _write(c0 / "index1/type", "Instruction")
            _write(c0 / "index1/size", "32K")

            _write(c0 / "index2/level", "2")
            _write(c0 / "index2/type", "Unified")
            _write(c0 / "index2/size", "1024K")
            _write(c0 / "index2/shared_cpu_list", "0,2")

            _write(c0 / "index3/level", "3")
            _write(c0 / "index3/type", "Unified")
            _write(c0 / "index3/size", "32M")
            _write(c0 / "index3/shared_cpu_list", "0-3")

            topo = discover_linux_topology(sysfs_root=sysfs, proc_root=proc, machine="x86_64")
            self.assertEqual(topo.sockets, 2)
            self.assertEqual(topo.physical_cores, 4)
            self.assertEqual(topo.logical_cores, 8)
            self.assertEqual(topo.numa_nodes, (0, 1))
            self.assertFalse(topo.is_hybrid)
            self.assertEqual(topo.caches.l1d_bytes, 32 * 1024)
            self.assertEqual(topo.caches.l2_bytes, 1024 * 1024)
            self.assertEqual(topo.caches.l3_bytes, 32 * 1024 * 1024)
            self.assertTrue(topo.pinning_supported)

            # Selection checks
            phys_ids, _, _ = select_cores(topo, "physical", "all")
            self.assertEqual(phys_ids, [0, 1, 4, 5])

            n1_ids, _, _ = select_cores(topo, "all", "1")
            self.assertEqual(n1_ids, [4, 5, 6, 7])

            with self.assertRaises(TopologyUnavailableError):
                select_cores(topo, "all", "2")

            with self.assertRaises(TopologyUnavailableError):
                select_cores(topo, "p-cores", "all")

    def test_linux_x86_hybrid_and_arm_capacity_hybrid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # x86 hybrid via cpu_core / cpu_atom
            sys_x86 = root / "sys_x86"
            proc_x86 = root / "proc_x86"
            _write(sys_x86 / "devices/system/cpu/online", "0-5")
            _write(sys_x86 / "devices/cpu_core/cpus", "0-3")
            _write(sys_x86 / "devices/cpu_atom/cpus", "4-5")
            _write(proc_x86 / "cpuinfo", "model name : Mock Hybrid x86\n")
            topo_x86 = discover_linux_topology(sysfs_root=sys_x86, proc_root=proc_x86, machine="x86_64")
            self.assertTrue(topo_x86.is_hybrid)
            self.assertEqual(topo_x86.p_core_ids, (0, 1, 2, 3))
            self.assertEqual(topo_x86.e_core_ids, (4, 5))

            p_ids, _, _ = select_cores(topo_x86, "p-cores", "all")
            e_ids, _, _ = select_cores(topo_x86, "e-cores", "all")
            self.assertEqual(p_ids, [0, 1, 2, 3])
            self.assertEqual(e_ids, [4, 5])

            # ARM big.LITTLE via cpu*/cpu_capacity
            sys_arm = root / "sys_arm"
            proc_arm = root / "proc_arm"
            _write(sys_arm / "devices/system/cpu/online", "0-3")
            for cid, cap in ((0, 512), (1, 512), (2, 1024), (3, 1024)):
                _write(sys_arm / f"devices/system/cpu/cpu{cid}/cpu_capacity", str(cap))
            _write(proc_arm / "cpuinfo", "Model : Mock ARM big.LITTLE\n")
            topo_arm = discover_linux_topology(sysfs_root=sys_arm, proc_root=proc_arm, machine="aarch64")
            self.assertTrue(topo_arm.is_hybrid)
            self.assertEqual(topo_arm.p_core_ids, (2, 3))
            self.assertEqual(topo_arm.e_core_ids, (0, 1))


class TestMacOSTopologyDiscovery(unittest.TestCase):
    def test_mocked_apple_silicon_perflevels(self) -> None:
        mock_sysctl: dict[str, int | str] = {
            "machdep.cpu.brand_string": "Apple M-Mock",
            "hw.logicalcpu": 10,
            "hw.physicalcpu": 10,
            "hw.memsize": 16 * 1024**3,
            "hw.cachelinesize": 128,
            "hw.nperflevels": 2,
            "hw.perflevel0.logicalcpu": 6,
            "hw.perflevel1.logicalcpu": 4,
            "hw.perflevel0.l1icachesize": 196608,
            "hw.perflevel0.l1dcachesize": 131072,
            "hw.perflevel0.l2cachesize": 16 * 1024 * 1024,
            "hw.perflevel0.cpusperl2": 6,
        }
        topo = discover_macos_topology(mock_sysctl, machine="arm64")
        self.assertTrue(topo.is_hybrid)
        self.assertEqual(topo.p_core_ids, (0, 1, 2, 3, 4, 5))
        self.assertEqual(topo.e_core_ids, (6, 7, 8, 9))
        self.assertFalse(topo.pinning_supported)
        self.assertIn("QOS_CLASS", topo.pinning_note)

        p_ids, p_qos, notes = select_cores(topo, "p-cores", "all")
        self.assertEqual(p_ids, [0, 1, 2, 3, 4, 5])
        self.assertEqual(p_qos, 1)
        self.assertTrue(len(notes) >= 1)

        e_ids, e_qos, _ = select_cores(topo, "e-cores", "all")
        self.assertEqual(e_ids, [6, 7, 8, 9])
        self.assertEqual(e_qos, 2)

    def test_live_host_topology(self) -> None:
        topo = discover_topology()
        self.assertGreaterEqual(topo.logical_cores, 1)
        self.assertGreaterEqual(topo.physical_cores, 1)
        self.assertGreaterEqual(len(topo.numa_nodes), 1)


if __name__ == "__main__":
    unittest.main()
