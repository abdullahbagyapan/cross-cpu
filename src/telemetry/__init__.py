# SPDX-License-Identifier: MIT
"""Drift-compensated 1 Hz filtered telemetry sampler and formatter."""

from __future__ import annotations

import threading
import time
from typing import Callable

from ..models import TelemetrySample
from .readers import (
    LinuxTelemetryReader,
    MacOSTelemetryReader,
    NullTelemetryReader,
    TelemetryReader,
    create_platform_reader,
    sanitize_metrics,
)

__all__ = [
    "LinuxTelemetryReader",
    "MacOSTelemetryReader",
    "NullTelemetryReader",
    "TelemetryReader",
    "TelemetrySampler",
    "create_platform_reader",
    "format_telemetry_line",
    "sanitize_metrics",
]


def format_telemetry_line(sample: TelemetrySample) -> str:
    """Render a concise single-line text telemetry summary, omitting missing sensors."""
    parts: list[str] = []
    m = sample.metrics
    if "cpu_util_pct" in m:
        parts.append(f"cpu={m['cpu_util_pct']:.1f}%")
    if "freq_mhz" in m:
        parts.append(f"freq={m['freq_mhz']:.0f}MHz")
    if "temp_c" in m:
        parts.append(f"temp={m['temp_c']:.1f}C")
    if "power_w" in m:
        parts.append(f"pwr={m['power_w']:.1f}W")
    if "fan_rpm" in m:
        parts.append(f"fan={int(m['fan_rpm'])}RPM")
    if "throttle_delta" in m and int(m["throttle_delta"]) > 0:
        parts.append(f"throttle=+{int(m['throttle_delta'])}")
    if "edac_ce_delta" in m and int(m["edac_ce_delta"]) > 0:
        parts.append(f"ecc_ce=+{int(m['edac_ce_delta'])}")
    if "edac_ue_delta" in m and int(m["edac_ue_delta"]) > 0:
        parts.append(f"ECC_UE=+{int(m['edac_ue_delta'])}")
    return " ".join(parts)


class TelemetrySampler:
    """
    Drift-compensated 1 Hz background telemetry sampler.

    Uses ``threading.Event.wait()`` for instant stop wakeup on sub-second runs
    or ``SIGINT``, strips unusable metrics, and suppresses ticks with zero
    valid sensors.
    """

    def __init__(
        self,
        reader: TelemetryReader | None = None,
        *,
        interval_s: float = 1.0,
        on_sample: Callable[[TelemetrySample], None] | None = None,
    ) -> None:
        self._reader = reader if reader is not None else create_platform_reader()
        self._interval_s = max(0.05, float(interval_s))
        self._on_sample = on_sample
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._samples: list[TelemetrySample] = []
        self._start_mono = 0.0
        self._edac_ce_delta_total = 0
        self._edac_ue_delta_total = 0

    @property
    def samples(self) -> list[TelemetrySample]:
        with self._lock:
            return list(self._samples)

    @property
    def edac_ce_delta_total(self) -> int:
        with self._lock:
            return self._edac_ce_delta_total

    @property
    def edac_ue_delta_total(self) -> int:
        with self._lock:
            return self._edac_ue_delta_total

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        with self._lock:
            self._samples.clear()
            self._edac_ce_delta_total = 0
            self._edac_ue_delta_total = 0
        self._reader.prime()
        self._start_mono = time.monotonic()
        self._thread = threading.Thread(
            target=self._loop,
            name="cross-cpu-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # Final EDAC poll ensures even sub-second runs detect ECC faults
        ce_d, ue_d = self._reader.read_edac_deltas()
        if ce_d > 0 or ue_d > 0:
            with self._lock:
                self._edac_ce_delta_total += ce_d
                self._edac_ue_delta_total += ue_d

    def _record_metrics(self, metrics: dict[str, float | int]) -> None:
        clean = sanitize_metrics(metrics)
        ce_d = int(clean.get("edac_ce_delta", 0))
        ue_d = int(clean.get("edac_ue_delta", 0))
        if ce_d > 0 or ue_d > 0:
            with self._lock:
                self._edac_ce_delta_total += ce_d
                self._edac_ue_delta_total += ue_d

        # Suppress ticks completely when zero sensors are usable
        if not clean:
            return

        sample = TelemetrySample(
            timestamp_unix=time.time(),
            elapsed_s=max(0.0, time.monotonic() - self._start_mono),
            metrics=clean,
        )
        with self._lock:
            self._samples.append(sample)

        if self._on_sample is not None:
            try:
                self._on_sample(sample)
            except Exception:
                pass

    def _loop(self) -> None:
        step = 1
        while not self._stop_event.is_set():
            target = self._start_mono + (step * self._interval_s)
            wait_s = max(0.0, target - time.monotonic())
            if self._stop_event.wait(timeout=wait_s):
                break
            try:
                metrics = self._reader.read()
            except Exception:
                metrics = {}
            self._record_metrics(metrics)
            step += 1
