# SPDX-License-Identifier: MIT
"""Offline tests for workload registry, buffer sizing, real native execution, and SDC fault injection."""

from __future__ import annotations

import io
import json
import threading
import time
import unittest

from src.cli import main
from src.exit_codes import ExitCode
from src.models import CacheInfo, CoreInfo, CPUTopology
from src.native_bridge import STATUS_OK, STATUS_STOPPED, NativeLibrary
from src.topology import discover_topology
from src.workloads import (
    ALL_WORKLOAD_NAMES,
    WORKLOAD_BY_NAME,
    WORKLOAD_SPECS,
    compute_workload_buffer_bytes,
    scale_work_metrics,
)


class TestWorkloadRegistryAndSizing(unittest.TestCase):
    def test_canonical_workloads_and_fixed_units(self) -> None:
        self.assertEqual(len(WORKLOAD_SPECS), 21)
        self.assertEqual(len(set(ALL_WORKLOAD_NAMES)), 21)

        valid_pairs = {
            ("GOP/s", "GOP"),
            ("GFLOP/s", "GFLOP"),
            ("GB/s", "GB"),
        }
        for spec in WORKLOAD_SPECS:
            self.assertIn((spec.rate_unit, spec.total_unit), valid_pairs)

        # Verify SI 10^9 scaling (never binary 2^30 or auto-scaled M/T)
        rate, total = scale_work_metrics(2_000_000_000, 500_000_000)
        self.assertAlmostEqual(total, 2.0)
        self.assertAlmostEqual(rate, 4.0)

    def test_topology_aware_cache_and_ram_buffer_sizing(self) -> None:
        topo_with_l3 = CPUTopology(
            os_name="Linux",
            machine="x86_64",
            model_name="Mock",
            sockets=1,
            physical_cores=4,
            logical_cores=4,
            numa_nodes=(0,),
            is_hybrid=False,
            p_core_ids=(),
            e_core_ids=(),
            memory_bytes=16 * 1024**3,
            caches=CacheInfo(
                l1i_bytes=32 * 1024,
                l1d_bytes=32 * 1024,
                l2_bytes=512 * 1024,
                l3_bytes=16 * 1024 * 1024,
                line_size_bytes=64,
                l2_shared_cpus=1,
                l3_shared_cpus=4,
                discovered=True,
            ),
            cores=tuple(CoreInfo(i, i) for i in range(4)),
            pinning_supported=True,
            pinning_note="mock",
        )
        l1_buf, err1 = compute_workload_buffer_bytes(WORKLOAD_BY_NAME["cache-l1"], topo_with_l3, 4)
        l2_buf, err2 = compute_workload_buffer_bytes(WORKLOAD_BY_NAME["cache-l2"], topo_with_l3, 4)
        l3_buf, err3 = compute_workload_buffer_bytes(WORKLOAD_BY_NAME["cache-l3"], topo_with_l3, 4)
        ram_buf, err4 = compute_workload_buffer_bytes(
            WORKLOAD_BY_NAME["bandwidth-ram"], topo_with_l3, 4
        )
        self.assertIsNone(err1)
        self.assertIsNone(err2)
        self.assertIsNone(err3)
        self.assertIsNone(err4)
        self.assertEqual(l1_buf, 16 * 1024)
        self.assertEqual(l2_buf, 256 * 1024)
        self.assertEqual(l3_buf, 2 * 1024 * 1024)
        self.assertGreaterEqual(ram_buf * 4, 4 * topo_with_l3.caches.l3_bytes)

        # Host without L3 reports cache-l3 cleanly as unsupported
        topo_no_l3 = CPUTopology(
            os_name="macOS",
            machine="arm64",
            model_name="MockNoL3",
            sockets=1,
            physical_cores=4,
            logical_cores=4,
            numa_nodes=(0,),
            is_hybrid=False,
            p_core_ids=(),
            e_core_ids=(),
            memory_bytes=16 * 1024**3,
            caches=CacheInfo(
                l1d_bytes=64 * 1024,
                l2_bytes=4 * 1024 * 1024,
                l3_bytes=0,
                discovered=True,
            ),
            cores=tuple(CoreInfo(i, i) for i in range(4)),
            pinning_supported=False,
            pinning_note="mock",
        )
        buf, reason = compute_workload_buffer_bytes(WORKLOAD_BY_NAME["cache-l3"], topo_no_l3, 4)
        self.assertEqual(buf, 0)
        self.assertIsNotNone(reason)

        # Single-core runs must still exceed LLC for bandwidth-ram, memory-pattern, and memory-sweep
        for wl_name in ("bandwidth-ram", "memory-pattern", "memory-sweep"):
            one_core_buf, err = compute_workload_buffer_bytes(
                WORKLOAD_BY_NAME[wl_name], topo_with_l3, 1
            )
            self.assertIsNone(err)
            self.assertGreater(
                one_core_buf,
                topo_with_l3.caches.l3_bytes,
                msg=f"{wl_name} 1-core buffer must exceed L3 LLC",
            )

        # Low-RAM container safety: aggregate allocation across 16 threads must stay <= 25% total / 50% avail
        topo_low_ram = CPUTopology(
            os_name="Linux",
            machine="x86_64",
            model_name="MockLowRAM",
            sockets=1,
            physical_cores=16,
            logical_cores=16,
            numa_nodes=(0,),
            is_hybrid=False,
            p_core_ids=(),
            e_core_ids=(),
            memory_bytes=64 * 1024 * 1024,
            caches=CacheInfo(l1d_bytes=32 * 1024, l2_bytes=256 * 1024, l3_bytes=16 * 1024 * 1024, discovered=True),
            cores=tuple(CoreInfo(i, i) for i in range(16)),
            pinning_supported=True,
            pinning_note="mock",
        )
        for wl_name in ("bandwidth-ram", "memory-pattern", "memory-sweep"):
            low_buf, err = compute_workload_buffer_bytes(
                WORKLOAD_BY_NAME[wl_name],
                topo_low_ram,
                16,
                available_memory_bytes=32 * 1024 * 1024,
            )
            self.assertIsNone(err)
            self.assertLessEqual(low_buf * 16, 16 * 1024 * 1024)


class TestNativeExecutionAndFaultInjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = NativeLibrary()
        cls.topo = discover_topology()

    def test_real_native_workloads_pass_with_zero_errors(self) -> None:
        tests = [
            "compute-int64",
            "compute-int32",
            "compute-fp64",
            "compute-fp32",
            "cache-l1",
            "bandwidth-ram",
            "memory-pattern",
            "memory-sweep",
            "coherency-atomic",
        ]
        # Also include hardware crypto/fp16/bf16/vnni whenever natively supported on this host
        for optional in ("compute-vnni", "compute-fp16", "compute-bf16", "crypto-aes", "crypto-sha256"):
            st, _, _ = self.lib.resolve_isa(WORKLOAD_BY_NAME[optional].workload_id, "auto")
            if st == STATUS_OK:
                tests.append(optional)

        out = io.StringIO()
        err = io.StringIO()
        rc = main(
            ["run", "--test", ",".join(tests), "--duration", "60ms", "--json"],
            stdout=out,
            stderr=err,
            native_library=self.lib,
            topology_override=self.topo,
        )
        self.assertEqual(rc, int(ExitCode.SUCCESS), msg=f"stderr={err.getvalue()}")
        self.assertEqual(err.getvalue(), "")
        payload = json.loads(out.getvalue())
        self.assertNotIn("edac_ce_delta", payload["summary"])
        self.assertNotIn("edac_ue_delta", payload["summary"])
        self.assertEqual(payload["summary"]["passed"], len(tests))
        self.assertEqual(payload["summary"]["failed"], 0)
        for res in payload["results"]:
            self.assertEqual(res["status"], "passed")
            self.assertEqual(res["error_count"], 0)
            self.assertGreater(res["raw_ops"], 0)

    def test_fault_injection_triggers_compute_error_exit_code_3(self) -> None:
        fault_targets = [
            "compute-int64",
            "compute-fp32",
            "bandwidth-ram",
            "memory-pattern",
            "memory-sweep",
            "coherency-atomic",
        ]
        for crypto_wl in ("crypto-aes", "crypto-sha256"):
            st, _, _ = self.lib.resolve_isa(WORKLOAD_BY_NAME[crypto_wl].workload_id, "auto")
            if st == STATUS_OK:
                fault_targets.append(crypto_wl)

        for wl_name in fault_targets:
            out = io.StringIO()
            err = io.StringIO()
            rc = main(
                ["run", "--cores", "0", "--test", wl_name, "--duration", "60ms", "--json"],
                stdout=out,
                stderr=err,
                native_library=self.lib,
                topology_override=self.topo,
                inject_fault_workloads={wl_name},
            )
            self.assertEqual(
                rc,
                int(ExitCode.COMPUTE_ERROR),
                msg=f"Fault injection on {wl_name} must exit 3 (got {rc}, out={out.getvalue()})",
            )
            self.assertEqual(err.getvalue(), "")
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["results"][0]["status"], "failed")
            self.assertGreater(payload["results"][0]["error_count"], 0)
            self.assertEqual(payload["summary"]["exit_code"], int(ExitCode.COMPUTE_ERROR))

        # Also verify scalar bandwidth-ram (with cache flush) catches bit errors
        out_s = io.StringIO()
        err_s = io.StringIO()
        rc_s = main(
            [
                "run",
                "--cores",
                "0",
                "--isa",
                "scalar",
                "--test",
                "bandwidth-ram",
                "--duration",
                "60ms",
                "--json",
            ],
            stdout=out_s,
            stderr=err_s,
            native_library=self.lib,
            topology_override=self.topo,
            inject_fault_workloads={"bandwidth-ram"},
        )
        self.assertEqual(rc_s, int(ExitCode.COMPUTE_ERROR))
        self.assertEqual(err_s.getvalue(), "")

    def test_test_all_skips_unsupported_without_init_error(self) -> None:
        out = io.StringIO()
        err = io.StringIO()
        rc = main(
            ["run", "--cores", "0", "--isa", "scalar", "--test", "all", "--duration", "20ms", "--json"],
            stdout=out,
            stderr=err,
            native_library=self.lib,
            topology_override=self.topo,
        )
        self.assertEqual(rc, int(ExitCode.SUCCESS), msg=f"out={out.getvalue()} err={err.getvalue()}")
        self.assertEqual(err.getvalue(), "")
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["summary"]["exit_code"], int(ExitCode.SUCCESS))
        self.assertGreaterEqual(payload["summary"]["unsupported"], 8)
        self.assertGreaterEqual(payload["summary"]["passed"], 12)
        for res in payload["results"]:
            if res["status"] == "unsupported":
                self.assertTrue(len(res.get("reason", "")) > 0)

    def test_atomic_stop_flag_cancels_workload_cleanly(self) -> None:
        self.lib.reset_stop()
        box: list[dict[str, int | float | str | None]] = []

        def _run() -> None:
            box.append(
                self.lib.run_workload(
                    workload_id=WORKLOAD_BY_NAME["compute-fp32"].workload_id,
                    requested_isa="auto",
                    core_ids=[0],
                    duration_ns=5_000_000_000,
                    batch_ns=50_000_000,
                )
            )

        t0 = time.monotonic()
        th = threading.Thread(target=_run)
        th.start()
        time.sleep(0.05)
        self.lib.request_stop()
        th.join(timeout=2.0)
        elapsed = time.monotonic() - t0
        self.lib.reset_stop()

        self.assertFalse(th.is_alive(), msg="Worker thread did not join promptly on stop")
        self.assertLess(elapsed, 1.5)
        self.assertEqual(len(box), 1)
        self.assertEqual(box[0]["status"], STATUS_STOPPED)
        self.assertEqual(box[0]["error_count"], 0)

    def test_explicit_unsupported_workload_exits_init_error_2(self) -> None:
        # Force scalar on compute-fp16 -> guaranteed unsupported on every architecture
        out = io.StringIO()
        err = io.StringIO()
        rc = main(
            ["run", "--isa", "scalar", "--test", "compute-fp16", "--duration", "50ms", "--json"],
            stdout=out,
            stderr=err,
            native_library=self.lib,
            topology_override=self.topo,
        )
        self.assertEqual(rc, int(ExitCode.INIT_ERROR))
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["results"][0]["status"], "unsupported")
        self.assertTrue(len(payload["results"][0]["reason"]) > 0)
        self.assertEqual(err.getvalue(), "")

        # In text mode, unsupported reason is still emitted to stderr
        out_txt = io.StringIO()
        err_txt = io.StringIO()
        rc_txt = main(
            ["run", "--isa", "scalar", "--test", "compute-fp16", "--duration", "50ms"],
            stdout=out_txt,
            stderr=err_txt,
            native_library=self.lib,
            topology_override=self.topo,
        )
        self.assertEqual(rc_txt, int(ExitCode.INIT_ERROR))
        self.assertIn("unsupported", err_txt.getvalue())

    def test_warmup_flag_runs_cleanly(self) -> None:
        out = io.StringIO()
        err = io.StringIO()
        rc = main(
            [
                "run",
                "--test",
                "compute-fp32",
                "--warmup",
                "40ms",
                "--duration",
                "60ms",
                "--json",
            ],
            stdout=out,
            stderr=err,
            native_library=self.lib,
            topology_override=self.topo,
        )
        self.assertEqual(rc, int(ExitCode.SUCCESS))
        payload = json.loads(out.getvalue())
        self.assertAlmostEqual(payload["config"]["warmup_s"], 0.04)
        self.assertEqual(payload["results"][0]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
