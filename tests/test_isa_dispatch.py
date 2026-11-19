# SPDX-License-Identifier: MIT
"""Tests for strict architecture isolation, ISA priority dispatch, and zero fake emulation."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from src.native_bridge import (
    CROSS_CPU_ABI_VERSION,
    STATUS_INIT_ERROR,
    STATUS_OK,
    STATUS_UNSUPPORTED,
    NativeLibrary,
)
from src.workloads import WORKLOAD_BY_NAME

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestArchitectureIsolation(unittest.TestCase):
    def test_native_core_has_zero_arch_ifdefs(self) -> None:
        core_dir = REPO_ROOT / "native" / "core"
        arch_macro_re = re.compile(
            r"(__x86_64__|__aarch64__|__arm64__|__riscv|__powerpc__|__ppc64__|__s390x__|__i386__|__arm__)"
        )
        c_files = sorted(core_dir.glob("*.c"))
        self.assertTrue(len(c_files) >= 2)
        for path in c_files:
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(
                arch_macro_re.search(text),
                msg=f"Architecture-specific macro found in {path}",
            )

    def test_arch_directory_separation_layout(self) -> None:
        arch_dir = REPO_ROOT / "native" / "arch"
        for rel in (
            "common/isa_scalar.c",
            "common/cache_flush_generic.c",
            "x86_64/probe_x86.c",
            "x86_64/isa_sse4.c",
            "x86_64/isa_avx2.c",
            "x86_64/isa_avx512.c",
            "x86_64/isa_amx.c",
            "arm64/probe_arm64.c",
            "arm64/isa_neon.c",
            "arm64/isa_sve.c",
            "arm64/isa_sme.c",
            "other/riscv64/probe_riscv64.c",
            "other/riscv64/isa_rvv.c",
            "other/ppc64/probe_ppc64.c",
            "other/ppc64/isa_vsx.c",
            "other/s390x/probe_s390x.c",
            "other/s390x/isa_vx.c",
            "other/generic/probe_generic.c",
        ):
            self.assertTrue((arch_dir / rel).is_file(), msg=f"Missing {rel}")


class TestISADispatchAndZeroFakeEmulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = NativeLibrary()
        cls.caps = cls.lib.get_caps()

    def test_caps_priority_descending_and_scalar_present(self) -> None:
        self.assertEqual(self.caps.abi_version, CROSS_CPU_ABI_VERSION)
        self.assertIn(self.caps.endianness, ("little", "big"))
        priorities = [b.priority for b in self.caps.backends]
        self.assertEqual(priorities, sorted(priorities, reverse=True))

        by_name = {b.name: b for b in self.caps.backends}
        self.assertIn("scalar", by_name)
        self.assertTrue(by_name["scalar"].available)
        self.assertEqual(by_name["scalar"].priority, 20)

    def test_zero_fake_emulation_for_hardware_only_workloads_under_scalar(self) -> None:
        hardware_only = [
            "compute-vnni",
            "compute-fp16",
            "compute-bf16",
            "compute-matrix-fp32",
            "compute-matrix-bf16",
            "compute-matrix-int8",
            "crypto-aes",
            "crypto-sha256",
        ]
        for wl_name in hardware_only:
            wl = WORKLOAD_BY_NAME[wl_name]
            status, isa_name, reason = self.lib.resolve_isa(wl.workload_id, "scalar")
            self.assertEqual(
                status,
                STATUS_UNSUPPORTED,
                msg=f"{wl_name} must be unsupported under scalar (no fake emulation)",
            )
            self.assertEqual(isa_name, "none")
            self.assertTrue(len(reason) > 0, msg=f"Missing explicit reason for {wl_name}")

    def test_unavailable_forced_isa_returns_init_error(self) -> None:
        # Pick an ISA foreign to the current host architecture
        foreign_isa = "avx512" if self.caps.arch_name != "x86_64" else "sme"
        wl = WORKLOAD_BY_NAME["compute-fp32"]
        status, isa_name, reason = self.lib.resolve_isa(wl.workload_id, foreign_isa)
        self.assertEqual(status, STATUS_INIT_ERROR)
        self.assertEqual(isa_name, "none")
        self.assertTrue(len(reason) > 0)

    def test_auto_selects_highest_priority_available_backend(self) -> None:
        wl = WORKLOAD_BY_NAME["compute-fp32"]
        status, isa_name, _ = self.lib.resolve_isa(wl.workload_id, "auto")
        self.assertEqual(status, STATUS_OK)
        expected = next(
            b.name for b in self.caps.backends if b.supports_workload_id(wl.workload_id)
        )
        self.assertEqual(isa_name, expected)


if __name__ == "__main__":
    unittest.main()
