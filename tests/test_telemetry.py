# SPDX-License-Identifier: MIT
"""Offline tests for filtered 1 Hz telemetry sanitization, RAPL resilience, and EDAC HARDWARE_ERROR."""

from __future__ import annotations

import io
import json
import math
import tempfile
import time
import unittest
from pathlib import Path

from src.cli import main
from src.exit_codes import ExitCode
from src.models import TelemetrySample
from src.native_bridge import NativeLibrary
from src.telemetry import (
    LinuxTelemetryReader,
    TelemetryReader,
    TelemetrySampler,
    format_telemetry_line,
    sanitize_metrics,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class _EmptyReader(TelemetryReader):
    def prime(self) -> None:
        return

    def read(self) -> dict[str, float | int]:
        return sanitize_metrics(
            {
                "cpu_util_pct": float("nan"),
                "freq_mhz": 0.0,
                "power_w": -5.0,
                "fan_rpm": -100,
                "temp_c": float("inf"),
            }
        )


class _MockEdacFaultReader(TelemetryReader):
    """Simulates a host where EDAC uncorrectable ECC errors increment mid-run."""

    def __init__(self, ue_increment: int = 1) -> None:
        self._ue_increment = ue_increment
        self._emitted = False

    def prime(self) -> None:
        self._emitted = False

    def read(self) -> dict[str, float | int]:
        if not self._emitted:
            self._emitted = True
            return sanitize_metrics({"cpu_util_pct": 85.0, "edac_ue_delta": self._ue_increment})
        return sanitize_metrics({"cpu_util_pct": 88.0})

    def read_edac_deltas(self) -> tuple[int, int]:
        if not self._emitted:
            self._emitted = True
            return 0, self._ue_increment
        return 0, 0


class TestTelemetrySanitizationAndSampler(unittest.TestCase):
    def test_sanitize_drops_all_invalid_values(self) -> None:
        raw = {
            "cpu_util_pct": 140.0,  # out of range (> 100)
            "freq_mhz": 0.0,        # 0 MHz invalid
            "temp_c": math.nan,     # NaN invalid
            "power_w": -12.5,       # negative invalid
            "fan_rpm": -1,          # negative invalid
            "throttle_delta": -3,   # negative invalid
            "edac_ce_delta": None,  # None invalid
            "edac_ue_delta": float("inf"),
        }
        cleaned = sanitize_metrics(raw)
        self.assertEqual(cleaned, {})

    def test_sanitize_keeps_valid_metrics_and_never_emits_na_or_null(self) -> None:
        raw = {
            "cpu_util_pct": 92.456,
            "freq_mhz": 3450.2,
            "temp_c": 67.8,
            "power_w": 42.1,
            "fan_rpm": 1800,
            "throttle_delta": 0,
            "edac_ce_delta": None,
        }
        cleaned = sanitize_metrics(raw)
        self.assertNotIn("edac_ce_delta", cleaned)
        for v in cleaned.values():
            self.assertIsNotNone(v)

        sample = TelemetrySample(timestamp_unix=1000.0, elapsed_s=1.0, metrics=cleaned)
        as_dict = sample.to_dict()
        self.assertNotIn("null", json.dumps(as_dict["metrics"]))
        line = format_telemetry_line(sample)
        self.assertNotIn("N/A", line)
        self.assertIn("cpu=92.5%", line)
        self.assertIn("freq=3450MHz", line)
        self.assertIn("temp=67.8C", line)
        self.assertIn("pwr=42.1W", line)

    def test_sampler_suppresses_empty_ticks_and_stops_instantly(self) -> None:
        seen: list[TelemetrySample] = []
        sampler = TelemetrySampler(reader=_EmptyReader(), interval_s=1.0, on_sample=seen.append)
        t0 = time.monotonic()
        sampler.start()
        time.sleep(0.05)
        sampler.stop()
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 0.4, msg="Sampler blocked instead of waking via Event.wait()")
        self.assertEqual(seen, [])
        self.assertEqual(sampler.samples, [])

    def test_linux_reader_mock_sysfs_and_rapl_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc = root / "proc"
            sysfs = root / "sys"

            _write(proc / "stat", "cpu  100 0 50 850 0 0 0 0 0 0\n")
            _write(sysfs / "devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "3200000\n")
            _write(sysfs / "class/hwmon/hwmon0/temp1_input", "54500\n")
            _write(sysfs / "class/hwmon/hwmon0/fan1_input", "1500\n")
            _write(sysfs / "devices/system/cpu/cpu0/thermal_throttle/core_throttle_count", "2\n")
            _write(sysfs / "devices/system/edac/mc/mc0/ce_count", "1\n")
            _write(sysfs / "devices/system/edac/mc/mc0/ue_count", "0\n")

            # Create an unreadable RAPL energy_uj file to verify PermissionError resilience
            rapl_file = sysfs / "class/powercap/intel-rapl:0/energy_uj"
            _write(rapl_file, "1000000\n")
            rapl_file.chmod(0o000)

            try:
                reader = LinuxTelemetryReader(proc_root=proc, sysfs_root=sysfs)
                reader.prime()

                _write(proc / "stat", "cpu  180 0 70 900 0 0 0 0 0 0\n")
                _write(sysfs / "devices/system/cpu/cpu0/thermal_throttle/core_throttle_count", "4\n")
                _write(sysfs / "devices/system/edac/mc/mc0/ce_count", "3\n")
                _write(sysfs / "devices/system/edac/mc/mc0/ue_count", "2\n")

                metrics = reader.read()
                self.assertIn("cpu_util_pct", metrics)
                self.assertAlmostEqual(float(metrics["cpu_util_pct"]), 66.67, places=1)
                self.assertAlmostEqual(float(metrics["freq_mhz"]), 3200.0)
                self.assertAlmostEqual(float(metrics["temp_c"]), 54.5)
                self.assertEqual(metrics["fan_rpm"], 1500)
                self.assertEqual(metrics["throttle_delta"], 2)
                self.assertEqual(metrics["edac_ce_delta"], 2)
                self.assertEqual(metrics["edac_ue_delta"], 2)
                # RAPL was unreadable -> omitted cleanly, never crashed or null
                self.assertNotIn("power_w", metrics)
            finally:
                rapl_file.chmod(0o644)

    def test_mocked_edac_ue_triggers_hardware_error_exit_code_4(self) -> None:
        lib = NativeLibrary()
        out = io.StringIO()
        err = io.StringIO()
        rc = main(
            ["run", "--test", "compute-fp32", "--duration", "80ms", "--json"],
            stdout=out,
            stderr=err,
            native_library=lib,
            telemetry_reader=_MockEdacFaultReader(ue_increment=2),
        )
        self.assertEqual(rc, int(ExitCode.HARDWARE_ERROR))
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["summary"]["edac_ue_delta"], 2)
        self.assertEqual(payload["summary"]["exit_code"], int(ExitCode.HARDWARE_ERROR))
        self.assertEqual(payload["summary"]["exit_name"], "HARDWARE_ERROR")
        self.assertIn("hardware_error", err.getvalue())


if __name__ == "__main__":
    unittest.main()
