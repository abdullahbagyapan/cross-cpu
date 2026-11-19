# SPDX-License-Identifier: MIT
"""Sanitized Linux and macOS telemetry readers using Python stdlib only."""

from __future__ import annotations

import abc
import ctypes
import math
import sys
import time
from pathlib import Path
from typing import Any

# macOS Mach constants (from <mach/processor_info.h>)
_PROCESSOR_CPU_LOAD_INFO = 2
_CPU_STATE_USER = 0
_CPU_STATE_SYSTEM = 1
_CPU_STATE_IDLE = 2
_CPU_STATE_NICE = 3
_CPU_STATE_MAX = 4


def sanitize_float(
    value: Any,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    strictly_positive: bool = False,
) -> float | None:
    """Normalize None, NaN, Inf, and out-of-range numeric values to None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    if strictly_positive and num <= 0.0:
        return None
    if min_value is not None and num < min_value:
        return None
    if max_value is not None and num > max_value:
        return None
    return num


def sanitize_nonneg_int(value: Any) -> int | None:
    """Normalize None, NaN, Inf, and negative counts to None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float):
        if not math.isfinite(value) or value < 0.0:
            return None
        return int(value)
    try:
        num = int(value)
    except (TypeError, ValueError):
        return None
    if num < 0:
        return None
    return num


def sanitize_metrics(raw: dict[str, Any]) -> dict[str, float | int]:
    """Filter raw sensor readings, omitting every invalid or unusable metric."""
    cleaned: dict[str, float | int] = {}

    if "cpu_util_pct" in raw:
        v = sanitize_float(raw["cpu_util_pct"], min_value=0.0, max_value=100.0)
        if v is not None:
            cleaned["cpu_util_pct"] = round(v, 2)

    if "freq_mhz" in raw:
        v = sanitize_float(raw["freq_mhz"], strictly_positive=True, max_value=20_000.0)
        if v is not None:
            cleaned["freq_mhz"] = round(v, 1)

    if "temp_c" in raw:
        v = sanitize_float(raw["temp_c"], min_value=-40.0, max_value=180.0)
        if v is not None:
            cleaned["temp_c"] = round(v, 1)

    if "power_w" in raw:
        v = sanitize_float(raw["power_w"], min_value=0.0, max_value=5000.0)
        if v is not None:
            cleaned["power_w"] = round(v, 2)

    if "fan_rpm" in raw:
        v = sanitize_float(raw["fan_rpm"], min_value=0.0, max_value=50_000.0)
        if v is not None and v >= 0.0:
            cleaned["fan_rpm"] = int(round(v))

    for count_key in ("throttle_delta", "edac_ce_delta", "edac_ue_delta"):
        if count_key in raw:
            iv = sanitize_nonneg_int(raw[count_key])
            if iv is not None:
                cleaned[count_key] = iv

    return cleaned


class TelemetryReader(abc.ABC):
    """Abstract platform telemetry reader."""

    @abc.abstractmethod
    def prime(self) -> None:
        """Capture baseline counters before timed stress execution starts."""

    @abc.abstractmethod
    def read(self) -> dict[str, float | int]:
        """Return a sanitized dictionary of usable metrics (empty if none usable)."""

    def read_edac_deltas(self) -> tuple[int, int]:
        """Return ``(ce_delta, ue_delta)`` since the last poll; default (0, 0)."""
        return 0, 0


class NullTelemetryReader(TelemetryReader):
    """Fallback reader for platforms with no accessible hardware telemetry."""

    def prime(self) -> None:
        return

    def read(self) -> dict[str, float | int]:
        return {}


class LinuxTelemetryReader(TelemetryReader):
    """Sanitized Linux telemetry reader (/proc/stat, hwmon, cpufreq, RAPL, throttle, EDAC)."""

    def __init__(
        self,
        *,
        proc_root: Path = Path("/proc"),
        sysfs_root: Path = Path("/sys"),
    ) -> None:
        self._proc_root = proc_root
        self._sysfs_root = sysfs_root
        self._prev_cpu: tuple[int, int] | None = None
        self._prev_rapl: dict[Path, tuple[int, float, int]] = {}
        self._prev_throttle: int | None = None
        self._prev_edac_ce: int | None = None
        self._prev_edac_ue: int | None = None

    @staticmethod
    def _read_int_file(path: Path) -> int | None:
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                return None
            return int(text)
        except (OSError, PermissionError, ValueError):
            return None

    def _read_proc_stat(self) -> tuple[int, int] | None:
        try:
            with (self._proc_root / "stat").open("r", encoding="utf-8", errors="replace") as fh:
                first = fh.readline()
        except OSError:
            return None
        parts = first.split()
        if len(parts) < 5 or parts[0] != "cpu":
            return None
        try:
            nums = [int(p) for p in parts[1:]]
        except ValueError:
            return None
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = sum(nums)
        busy = total - idle
        return busy, total

    def _read_cpufreq_mhz(self) -> float | None:
        cpu_dir = self._sysfs_root / "devices" / "system" / "cpu"
        if not cpu_dir.is_dir():
            return None
        khz_vals: list[int] = []
        for entry in sorted(cpu_dir.glob("cpu[0-9]*/cpufreq/scaling_cur_freq")):
            val = self._read_int_file(entry)
            if val is not None and val > 0:
                khz_vals.append(val)
        if not khz_vals:
            return None
        return (sum(khz_vals) / len(khz_vals)) / 1000.0

    def _read_hwmon(self) -> tuple[float | None, int | None]:
        hwmon_root = self._sysfs_root / "class" / "hwmon"
        if not hwmon_root.is_dir():
            return None, None
        temps_c: list[float] = []
        fans_rpm: list[int] = []
        for hw in sorted(hwmon_root.glob("hwmon*")):
            if not hw.is_dir():
                continue
            for tf in sorted(hw.glob("temp*_input")):
                mdeg = self._read_int_file(tf)
                if mdeg is not None:
                    c = mdeg / 1000.0
                    if -40.0 <= c <= 180.0:
                        temps_c.append(c)
            for ff in sorted(hw.glob("fan*_input")):
                rpm = self._read_int_file(ff)
                if rpm is not None and rpm > 0:
                    fans_rpm.append(rpm)
        temp_peak = max(temps_c) if temps_c else None
        fan_peak = max(fans_rpm) if fans_rpm else None
        return temp_peak, fan_peak

    def _read_rapl_watts(self, now_s: float) -> float | None:
        rapl_root = self._sysfs_root / "class" / "powercap"
        if not rapl_root.is_dir():
            return None

        total_w = 0.0
        any_valid = False
        # Top-level package domains only (intel-rapl:0, intel-rapl:1, not subzones intel-rapl:0:0)
        for domain in sorted(rapl_root.glob("intel-rapl:[0-9]*")):
            if domain.name.count(":") != 1 or not domain.is_dir():
                continue
            energy_file = domain / "energy_uj"
            energy_uj = self._read_int_file(energy_file)
            if energy_uj is None or energy_uj < 0:
                continue
            max_range = self._read_int_file(domain / "max_energy_range_uj") or 0
            prev = self._prev_rapl.get(energy_file)
            self._prev_rapl[energy_file] = (energy_uj, now_s, max_range)
            if prev is None:
                continue
            prev_uj, prev_t, prev_max = prev
            dt = now_s - prev_t
            if dt <= 0.0:
                continue
            delta_uj = energy_uj - prev_uj
            if delta_uj < 0 and prev_max > 0:
                delta_uj += prev_max
            if delta_uj >= 0:
                total_w += (delta_uj / 1_000_000.0) / dt
                any_valid = True
        return total_w if any_valid else None

    def _read_throttle_total(self) -> int | None:
        cpu_dir = self._sysfs_root / "devices" / "system" / "cpu"
        if not cpu_dir.is_dir():
            return None
        total = 0
        found = False
        for path in cpu_dir.glob("cpu[0-9]*/thermal_throttle/*_throttle_count"):
            val = self._read_int_file(path)
            if val is not None and val >= 0:
                total += val
                found = True
        return total if found else None

    def _read_edac_totals(self) -> tuple[int | None, int | None]:
        edac_dir = self._sysfs_root / "devices" / "system" / "edac" / "mc"
        if not edac_dir.is_dir():
            return None, None
        ce_total = 0
        ue_total = 0
        ce_found = False
        ue_found = False
        for mc in sorted(edac_dir.glob("mc[0-9]*")):
            if not mc.is_dir():
                continue
            ce = self._read_int_file(mc / "ce_count")
            if ce is not None and ce >= 0:
                ce_total += ce
                ce_found = True
            ue = self._read_int_file(mc / "ue_count")
            if ue is not None and ue >= 0:
                ue_total += ue
                ue_found = True
        return (ce_total if ce_found else None), (ue_total if ue_found else None)

    def prime(self) -> None:
        now_s = time.monotonic()
        self._prev_cpu = self._read_proc_stat()
        self._read_rapl_watts(now_s)
        self._prev_throttle = self._read_throttle_total()
        self._prev_edac_ce, self._prev_edac_ue = self._read_edac_totals()

    def read_edac_deltas(self) -> tuple[int, int]:
        ce_now, ue_now = self._read_edac_totals()
        ce_delta = 0
        ue_delta = 0
        if ce_now is not None:
            if self._prev_edac_ce is not None and ce_now >= self._prev_edac_ce:
                ce_delta = ce_now - self._prev_edac_ce
            self._prev_edac_ce = ce_now
        if ue_now is not None:
            if self._prev_edac_ue is not None and ue_now >= self._prev_edac_ue:
                ue_delta = ue_now - self._prev_edac_ue
            self._prev_edac_ue = ue_now
        return ce_delta, ue_delta

    def read(self) -> dict[str, float | int]:
        now_s = time.monotonic()
        raw: dict[str, Any] = {}

        cpu_now = self._read_proc_stat()
        if cpu_now is not None:
            if self._prev_cpu is not None:
                db = cpu_now[0] - self._prev_cpu[0]
                dt = cpu_now[1] - self._prev_cpu[1]
                if dt > 0 and db >= 0:
                    raw["cpu_util_pct"] = (db / dt) * 100.0
            self._prev_cpu = cpu_now

        freq = self._read_cpufreq_mhz()
        if freq is not None:
            raw["freq_mhz"] = freq

        temp_c, fan_rpm = self._read_hwmon()
        if temp_c is not None:
            raw["temp_c"] = temp_c
        if fan_rpm is not None:
            raw["fan_rpm"] = fan_rpm

        power_w = self._read_rapl_watts(now_s)
        if power_w is not None:
            raw["power_w"] = power_w

        th_now = self._read_throttle_total()
        if th_now is not None:
            if self._prev_throttle is not None and th_now >= self._prev_throttle:
                raw["throttle_delta"] = th_now - self._prev_throttle
            self._prev_throttle = th_now

        ce_now, ue_now = self._read_edac_totals()
        if ce_now is not None:
            if self._prev_edac_ce is not None and ce_now >= self._prev_edac_ce:
                raw["edac_ce_delta"] = ce_now - self._prev_edac_ce
            self._prev_edac_ce = ce_now
        if ue_now is not None:
            if self._prev_edac_ue is not None and ue_now >= self._prev_edac_ue:
                raw["edac_ue_delta"] = ue_now - self._prev_edac_ue
            self._prev_edac_ue = ue_now

        return sanitize_metrics(raw)


class MacOSTelemetryReader(TelemetryReader):
    """Unprivileged macOS Mach host_processor_info telemetry reader (zero sudo)."""

    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = None
        self._prev_ticks: tuple[int, int] | None = None
        try:
            lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            lib.mach_host_self.argtypes = []
            lib.mach_host_self.restype = ctypes.c_uint32
            lib.host_processor_info.argtypes = [
                ctypes.c_uint32,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
                ctypes.POINTER(ctypes.c_uint32),
            ]
            lib.host_processor_info.restype = ctypes.c_int
            lib.vm_deallocate.argtypes = [
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            lib.vm_deallocate.restype = ctypes.c_int
            self._lib = lib
        except OSError:
            self._lib = None

    def _read_mach_ticks(self) -> tuple[int, int] | None:
        if self._lib is None:
            return None
        ncpu = ctypes.c_uint32(0)
        info = ctypes.POINTER(ctypes.c_int32)()
        count = ctypes.c_uint32(0)
        rc = int(
            self._lib.host_processor_info(
                self._lib.mach_host_self(),
                _PROCESSOR_CPU_LOAD_INFO,
                ctypes.byref(ncpu),
                ctypes.byref(info),
                ctypes.byref(count),
            )
        )
        if rc != 0 or not info:
            return None
        try:
            busy = 0
            total = 0
            for idx in range(int(ncpu.value)):
                base = idx * _CPU_STATE_MAX
                user = int(info[base + _CPU_STATE_USER])
                sys_t = int(info[base + _CPU_STATE_SYSTEM])
                idle = int(info[base + _CPU_STATE_IDLE])
                nice = int(info[base + _CPU_STATE_NICE])
                busy += user + sys_t + nice
                total += user + sys_t + nice + idle
            return busy, total
        finally:
            try:
                task = ctypes.c_uint32.in_dll(self._lib, "mach_task_self_").value
                self._lib.vm_deallocate(
                    task,
                    ctypes.cast(info, ctypes.c_void_p),
                    int(count.value) * ctypes.sizeof(ctypes.c_int32),
                )
            except Exception:
                pass

    def prime(self) -> None:
        self._prev_ticks = self._read_mach_ticks()

    def read(self) -> dict[str, float | int]:
        now_ticks = self._read_mach_ticks()
        if now_ticks is None:
            return {}
        raw: dict[str, Any] = {}
        if self._prev_ticks is not None:
            db = now_ticks[0] - self._prev_ticks[0]
            dt = now_ticks[1] - self._prev_ticks[1]
            if dt > 0 and db >= 0:
                raw["cpu_util_pct"] = (db / dt) * 100.0
        self._prev_ticks = now_ticks
        return sanitize_metrics(raw)


def create_platform_reader() -> TelemetryReader:
    """Instantiate the appropriate telemetry reader for the current OS."""
    if sys.platform.startswith("linux"):
        return LinuxTelemetryReader()
    if sys.platform == "darwin":
        return MacOSTelemetryReader()
    return NullTelemetryReader()
