# 1 Hz Filtered Telemetry & EDAC Monitoring

`CrossCPU` implements a lightweight, zero-dependency 2-file telemetry subsystem under `src/telemetry/`:

- `src/telemetry/__init__.py` — Drift-compensated 1 Hz `TelemetrySampler` and single-line text formatter.
- `src/telemetry/readers.py` — Sanitized Linux (`sysfs`/`procfs`/`hwmon`/`RAPL`/`EDAC`) and macOS (Mach `host_processor_info`) readers.

## Drift-Compensated Sampler & Instant Stop (`threading.Event.wait`)

`TelemetrySampler` computes each tick deadline from a monotonic start timestamp (`start_mono + step * interval_s`) and sleeps using `threading.Event.wait(timeout)`. This guarantees:

1. **Zero Cumulative Drift:** Long multi-hour stress runs maintain exact 1 Hz cadence.
2. **Instant Stop on `ms` Runs & `SIGINT`:** Calling `sampler.stop()` sets the `Event`, waking the background thread in `< 1 ms` rather than blocking for up to 1 second.

## Platform Readers

### Linux (`LinuxTelemetryReader`)

| Metric Key | Source Path | Description |
| --- | --- | --- |
| `cpu_util_pct` | `/proc/stat` | Aggregate CPU busy percentage (`0.0..100.0`). |
| `freq_mhz` | `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq` | Mean online core frequency in MHz (`> 0`). |
| `temp_c` | `/sys/class/hwmon/hwmon*/temp*_input` | Peak core/package temperature in °C (`-40..180`). |
| `fan_rpm` | `/sys/class/hwmon/hwmon*/fan*_input` | Peak cooling fan speed in RPM (`> 0`). |
| `power_w` | `/sys/class/powercap/intel-rapl:[0-9]*/energy_uj` | Package power in Watts computed from energy counter deltas (handles `max_energy_range_uj` wrap and catches `PermissionError` gracefully on hardened kernels). |
| `throttle_delta` | `/sys/devices/system/cpu/cpu*/thermal_throttle/*_throttle_count` | Thermal throttle event count increment during the interval. |
| `edac_ce_delta` | `/sys/devices/system/edac/mc/mc*/ce_count` | Correctable ECC memory error delta. |
| `edac_ue_delta` | `/sys/devices/system/edac/mc/mc*/ue_count` | Uncorrectable ECC memory error delta. Any `edac_ue_delta > 0` escalates the process exit code to `ExitCode.HARDWARE_ERROR (4)`. |

### macOS (`MacOSTelemetryReader`)

- Queries Mach `host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, ...)` via `ctypes` (`libSystem.B.dylib`) and deallocates kernel buffers cleanly via `vm_deallocate()` without requiring `sudo`.

## Sanitization & Suppression Contract

All raw sensor values pass through `sanitize_metrics()`:

1. `None`, `NaN`, `Inf`, `-Inf`, negative power/RPM/counts, `0` MHz frequencies, and percentages outside `[0.0, 100.0]` are normalized to `None` and **omitted**.
2. `text` mode **never** prints `N/A` placeholders.
3. `json` mode **never** emits `null` keys inside `"metrics"`.
4. If a tick yields zero valid sensors, the tick is **completely suppressed**.
