# SPDX-License-Identifier: MIT
"""Workload execution engine with warmup, signal-to-C-atomic stop bridge, and 1 Hz telemetry."""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .exit_codes import ExitCode, aggregate_exit_codes
from .models import CPUTopology, NativeCapabilities, TelemetrySample, WorkloadResult, WorkloadSpec
from .native_bridge import (
    STATUS_COMPUTE_ERROR,
    STATUS_INIT_ERROR,
    STATUS_OK,
    STATUS_STOPPED,
    STATUS_UNSUPPORTED,
    NativeLibrary,
)
from .telemetry import TelemetryReader, TelemetrySampler
from .workloads import compute_workload_buffer_bytes, scale_work_metrics


@dataclass
class RunReport:
    """Aggregated outcome of executing one or more workloads."""

    results: list[WorkloadResult] = field(default_factory=list)
    telemetry: list[TelemetrySample] = field(default_factory=list)
    edac_ce_delta: int = 0
    edac_ue_delta: int = 0
    interrupted: bool = False
    hardware_fault_reason: str = ""
    exit_code: ExitCode = ExitCode.SUCCESS


ProgressCallback = Callable[[int, int, str, float, float, Optional[TelemetrySample]], None]
ResultCallback = Callable[[int, int, WorkloadResult], None]
DiagCallback = Callable[[str], None]


def format_duration_progress(elapsed_s: float, total_s: float) -> str:
    """Format progress timing in ms for sub-second runs (e.g. ``250ms/500ms``) and s otherwise."""
    clamped = min(max(0.0, elapsed_s), total_s) if total_s > 0.0 else max(0.0, elapsed_s)
    if total_s < 1.0:
        cur_ms = int(round(clamped * 1000.0))
        tot_ms = max(1, int(round(total_s * 1000.0)))
        return f"{cur_ms}ms/{tot_ms}ms"
    if total_s < 10.0 and (total_s % 1.0) != 0.0:
        return f"{clamped:.1f}s/{total_s:.1f}s"
    return f"{int(clamped)}s/{int(round(total_s))}s"


class WorkloadRunner:
    """Orchestrates warmup, timed execution, signal cancellation, and filtered 1 Hz telemetry."""

    def __init__(
        self,
        native: NativeLibrary,
        topology: CPUTopology,
        caps: NativeCapabilities,
        *,
        telemetry_reader: TelemetryReader | None = None,
        telemetry_interval_s: float = 1.0,
    ) -> None:
        self.native = native
        self.topology = topology
        self.caps = caps
        self._telemetry_reader = telemetry_reader
        self._telemetry_interval_s = telemetry_interval_s

    def run(
        self,
        workloads: Sequence[WorkloadSpec],
        *,
        requested_isa: str = "auto",
        core_ids: Sequence[int],
        qos_class: int = 0,
        duration_s: float = 10.0,
        warmup_s: float = 0.0,
        explicit_tests: bool = False,
        inject_fault_workloads: set[str] | None = None,
        on_progress: ProgressCallback | None = None,
        on_result: ResultCallback | None = None,
        on_diag: DiagCallback | None = None,
    ) -> RunReport:
        report = RunReport()
        codes: list[ExitCode] = [ExitCode.SUCCESS]
        stop_event = threading.Event()
        latest_sample_box: list[TelemetrySample | None] = [None]

        # Validate forced ISA availability upfront before launching anything
        if requested_isa != "auto":
            by_name = {b.name: b for b in self.caps.backends}
            backend = by_name.get(requested_isa)
            if backend is None or not backend.available:
                detail = (
                    f": {backend.detail}"
                    if backend is not None and backend.detail
                    else f" on host architecture '{self.caps.arch_name}'"
                )
                reason = f"Requested ISA '{requested_isa}' is not available{detail}"
                if on_diag:
                    on_diag(reason)
                for idx, spec in enumerate(workloads, start=1):
                    res = WorkloadResult(
                        name=spec.name,
                        category=spec.category,
                        status="init_error",
                        isa="none",
                        threads=0,
                        duration_s=0.0,
                        rate=0.0,
                        rate_unit=spec.rate_unit,
                        total_work=0.0,
                        total_unit=spec.total_unit,
                        raw_ops=0,
                        iterations=0,
                        error_count=0,
                        exit_code=int(ExitCode.INIT_ERROR),
                        reason=reason,
                    )
                    report.results.append(res)
                    if on_result:
                        on_result(idx, len(workloads), res)
                report.exit_code = ExitCode.INIT_ERROR
                return report

        def _handle_signal(_signum: int, _frame: object) -> None:
            report.interrupted = True
            stop_event.set()
            self.native.request_stop()

        installed_signals = False
        prev_sigint = None
        prev_sigterm = None
        if threading.current_thread() is threading.main_thread():
            prev_sigint = signal.getsignal(signal.SIGINT)
            prev_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)
            installed_signals = True

        def _on_sample(sample: TelemetrySample) -> None:
            latest_sample_box[0] = sample

        sampler = TelemetrySampler(
            reader=self._telemetry_reader,
            interval_s=self._telemetry_interval_s,
            on_sample=_on_sample,
        )

        self.native.reset_stop()
        sampler.start()
        try:
            total_count = len(workloads)
            duration_ns = max(1_000_000, int(round(duration_s * 1_000_000_000.0)))
            warmup_ns = int(round(warmup_s * 1_000_000_000.0)) if warmup_s > 0.0 else 0
            batch_ns = min(100_000_000, max(5_000_000, duration_ns // 4))

            for idx, spec in enumerate(workloads, start=1):
                if stop_event.is_set():
                    break

                buf_bytes, buf_unsupported = compute_workload_buffer_bytes(
                    spec, self.topology, len(core_ids)
                )
                if buf_unsupported is not None:
                    code = ExitCode.INIT_ERROR if explicit_tests else ExitCode.SUCCESS
                    codes.append(code)
                    res = WorkloadResult(
                        name=spec.name,
                        category=spec.category,
                        status="unsupported",
                        isa="none",
                        threads=0,
                        duration_s=0.0,
                        rate=0.0,
                        rate_unit=spec.rate_unit,
                        total_work=0.0,
                        total_unit=spec.total_unit,
                        raw_ops=0,
                        iterations=0,
                        error_count=0,
                        exit_code=int(code),
                        reason=buf_unsupported,
                    )
                    report.results.append(res)
                    if on_result:
                        on_result(idx, total_count, res)
                    continue

                resolve_status, resolved_isa, resolve_reason = self.native.resolve_isa(
                    spec.workload_id, requested_isa
                )
                if resolve_status != STATUS_OK:
                    if resolve_status == STATUS_INIT_ERROR:
                        status_str = "init_error"
                        code = ExitCode.INIT_ERROR
                    else:
                        status_str = "unsupported"
                        code = ExitCode.INIT_ERROR if explicit_tests else ExitCode.SUCCESS
                    codes.append(code)
                    res = WorkloadResult(
                        name=spec.name,
                        category=spec.category,
                        status=status_str,
                        isa="none",
                        threads=0,
                        duration_s=0.0,
                        rate=0.0,
                        rate_unit=spec.rate_unit,
                        total_work=0.0,
                        total_unit=spec.total_unit,
                        raw_ops=0,
                        iterations=0,
                        error_count=0,
                        exit_code=int(code),
                        reason=resolve_reason,
                    )
                    report.results.append(res)
                    if on_result:
                        on_result(idx, total_count, res)
                    continue

                if on_diag:
                    buf_note = f" buf={buf_bytes}B/thread" if buf_bytes > 0 else ""
                    on_diag(
                        f"Dispatching {spec.name}: isa={resolved_isa} threads={len(core_ids)}{buf_note}"
                    )

                inject = bool(inject_fault_workloads and spec.name in inject_fault_workloads)

                # Optional warmup phase (still verifies SDC if hardware corrupts during warmup)
                if warmup_ns > 0 and not stop_event.is_set():
                    warm_raw = self._execute_native_with_progress(
                        spec=spec,
                        requested_isa=requested_isa,
                        core_ids=core_ids,
                        duration_ns=warmup_ns,
                        batch_ns=min(100_000_000, max(5_000_000, warmup_ns // 4)),
                        buffer_bytes=buf_bytes,
                        qos_class=qos_class,
                        inject_fault=False,
                        idx=idx,
                        total_count=total_count,
                        label=f"{spec.name} (warmup)",
                        target_s=warmup_s,
                        stop_event=stop_event,
                        latest_sample_box=latest_sample_box,
                        on_progress=on_progress,
                    )
                    if warm_raw["status"] == STATUS_COMPUTE_ERROR:
                        res = self._raw_to_result(
                            spec, warm_raw, buf_bytes, explicit_tests=explicit_tests
                        )
                        codes.append(ExitCode(res.exit_code))
                        report.results.append(res)
                        if on_result:
                            on_result(idx, total_count, res)
                        continue

                if stop_event.is_set():
                    break

                raw = self._execute_native_with_progress(
                    spec=spec,
                    requested_isa=requested_isa,
                    core_ids=core_ids,
                    duration_ns=duration_ns,
                    batch_ns=batch_ns,
                    buffer_bytes=buf_bytes,
                    qos_class=qos_class,
                    inject_fault=inject,
                    idx=idx,
                    total_count=total_count,
                    label=spec.name,
                    target_s=duration_s,
                    stop_event=stop_event,
                    latest_sample_box=latest_sample_box,
                    on_progress=on_progress,
                )
                res = self._raw_to_result(spec, raw, buf_bytes, explicit_tests=explicit_tests)
                codes.append(ExitCode(res.exit_code))
                report.results.append(res)
                if on_result:
                    on_result(idx, total_count, res)
        finally:
            sampler.stop()
            self.native.reset_stop()
            if installed_signals:
                signal.signal(signal.SIGINT, prev_sigint)
                signal.signal(signal.SIGTERM, prev_sigterm)

        report.telemetry = sampler.samples
        report.edac_ce_delta = sampler.edac_ce_delta_total
        report.edac_ue_delta = sampler.edac_ue_delta_total

        if report.edac_ue_delta > 0:
            report.hardware_fault_reason = (
                f"EDAC reported +{report.edac_ue_delta} uncorrectable ECC memory error(s) "
                "during stress run"
            )
            codes.append(ExitCode.HARDWARE_ERROR)

        report.exit_code = aggregate_exit_codes(codes)
        return report

    def _execute_native_with_progress(
        self,
        *,
        spec: WorkloadSpec,
        requested_isa: str,
        core_ids: Sequence[int],
        duration_ns: int,
        batch_ns: int,
        buffer_bytes: int,
        qos_class: int,
        inject_fault: bool,
        idx: int,
        total_count: int,
        label: str,
        target_s: float,
        stop_event: threading.Event,
        latest_sample_box: list[TelemetrySample | None],
        on_progress: ProgressCallback | None,
    ) -> dict[str, int | float | str | None]:
        result_box: list[dict[str, int | float | str | None]] = []
        done_event = threading.Event()

        def _worker() -> None:
            try:
                raw = self.native.run_workload(
                    workload_id=spec.workload_id,
                    requested_isa=requested_isa,
                    core_ids=core_ids,
                    duration_ns=duration_ns,
                    batch_ns=batch_ns,
                    buffer_bytes=buffer_bytes,
                    qos_class=qos_class,
                    inject_fault=inject_fault,
                )
                result_box.append(raw)
            finally:
                done_event.set()

        t_start = time.monotonic()
        thread = threading.Thread(
            target=_worker, name=f"cross-cpu-wl-{spec.workload_id}", daemon=True
        )
        thread.start()

        poll_s = 0.05 if target_s < 1.0 else 0.25
        while not done_event.wait(timeout=poll_s):
            if on_progress is not None:
                elapsed = time.monotonic() - t_start
                on_progress(idx, total_count, label, elapsed, target_s, latest_sample_box[0])
            if stop_event.is_set():
                self.native.request_stop()

        thread.join()
        if on_progress is not None:
            on_progress(idx, total_count, label, target_s, target_s, latest_sample_box[0])

        if result_box:
            return result_box[0]
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
            "reason": "Native execution thread terminated without returning a result",
        }

    @staticmethod
    def _raw_to_result(
        spec: WorkloadSpec,
        raw: dict[str, int | float | str | None],
        buffer_bytes: int,
        *,
        explicit_tests: bool,
    ) -> WorkloadResult:
        status_code = int(raw["status"] or 0)
        elapsed_ns = int(raw["elapsed_ns"] or 0)
        raw_ops = int(raw["total_ops"] or 0)
        rate, total_work = scale_work_metrics(raw_ops, elapsed_ns)
        duration_s = float(elapsed_ns) / 1_000_000_000.0 if elapsed_ns > 0 else 0.0

        if status_code == STATUS_OK:
            status_str = "passed"
            exit_code = ExitCode.SUCCESS
        elif status_code == STATUS_COMPUTE_ERROR:
            status_str = "failed"
            exit_code = ExitCode.COMPUTE_ERROR
        elif status_code == STATUS_UNSUPPORTED:
            status_str = "unsupported"
            exit_code = ExitCode.INIT_ERROR if explicit_tests else ExitCode.SUCCESS
        elif status_code == STATUS_STOPPED:
            status_str = "stopped"
            exit_code = ExitCode.SUCCESS
        else:
            status_str = "init_error"
            exit_code = ExitCode.INIT_ERROR

        expected_s = raw["expected_sample"]
        observed_s = raw["observed_sample"]
        f_thread = raw["failed_thread"]
        f_core = raw["failed_core"]

        return WorkloadResult(
            name=spec.name,
            category=spec.category,
            status=status_str,
            isa=str(raw["isa_name"] or "none"),
            threads=int(raw["threads_executed"] or 0),
            duration_s=duration_s,
            rate=rate,
            rate_unit=spec.rate_unit,
            total_work=total_work,
            total_unit=spec.total_unit,
            raw_ops=raw_ops,
            iterations=int(raw["iterations"] or 0),
            error_count=int(raw["error_count"] or 0),
            exit_code=int(exit_code),
            reason=str(raw["reason"] or ""),
            expected_sample=float(expected_s) if expected_s is not None else None,
            observed_sample=float(observed_s) if observed_s is not None else None,
            failed_thread=int(f_thread) if f_thread is not None else None,
            failed_core=int(f_core) if f_core is not None else None,
            buffer_bytes=buffer_bytes if buffer_bytes > 0 else None,
        )
