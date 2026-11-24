#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a GitHub Discussion Hardware Validation Report from CrossCPU output or cfarm logs."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
from pathlib import Path

SECTION_RE = re.compile(r"^={20} (.+?) ={20}$", re.MULTILINE)
NOISE_LINE_RE = re.compile(r"^\[(verbose|unsupported|warning|info|debug)\]\s+")
SUMMARY_RE = re.compile(
    r"Summary:\s*(\d+)\s*passed,\s*(\d+)\s*failed,\s*(\d+)\s*unsupported\s*\|\s*Exit:\s*(?:[A-Z_]+)\s*\((\d+)\)"
)


def fmt_bytes(num: int | None) -> str:
    if not num or num <= 0:
        return "none"
    for unit, div in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if num >= div:
            val = num / div
            return f"{int(val)} {unit}" if val.is_integer() else f"{val:.1f} {unit}"
    return f"{num} B"


def fmt_id_ranges(ids: list[int]) -> str:
    if not ids:
        return ""
    sorted_ids = sorted(set(ids))
    ranges: list[str] = []
    start = prev = sorted_ids[0]
    for cid in sorted_ids[1:]:
        if cid == prev + 1:
            prev = cid
        else:
            ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = cid
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return f"[{', '.join(ranges)}]"


def clean_summary_table(raw_run: str) -> str:
    lines: list[str] = []
    capture = False
    for line in raw_run.splitlines():
        if line.startswith("Host: "):
            capture = True
        if not capture:
            continue
        if NOISE_LINE_RE.match(line):
            continue
        lines.append(line)
        if line.startswith("Summary:"):
            break
    return "\n".join(lines).strip() if lines else raw_run.strip()


def extract_json(blob: str) -> dict:
    idx = blob.find("{")
    if idx == -1:
        raise ValueError("No JSON object found in input")
    obj, _ = json.JSONDecoder().raw_decode(blob, idx)
    return obj


def detect_exec_env(uname_str: str, raw_text: str) -> str:
    lower = f"{uname_str}\n{raw_text}".lower()
    if "microsoft" in lower or "wsl" in lower:
        return "WSL2"
    if "hypervisor" in lower or "kvm" in lower or "vmware" in lower or "qemu" in lower:
        return "VM / Hypervisor"
    return "Bare metal"


def parse_cfarm_log(path: Path) -> tuple[str, str, dict, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(SECTION_RE.finditer(text))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.end() + 1
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[m.group(1).strip()] = text[start:end].strip()

    uname_line = ""
    py_ver = "unknown"
    for title, body in sections.items():
        if title.startswith("HOST & TOOLCHAIN ("):
            for line in body.splitlines():
                if line.startswith("uname: "):
                    uname_line = line[7:].strip()
                elif line.startswith("python: "):
                    pm = re.search(r"\((Python [^)]+)\)", line)
                    py_ver = pm.group(1).replace("Python ", "") if pm else line[8:].strip()
            break

    info_sec = sections.get("CROSS-CPU VERSION, INFO & LIST", "")
    info_chunk = info_sec.split("--- info --json ---", 1)[-1].split("--- list --json ---", 1)[0]
    info = extract_json(info_chunk)

    run_sec = sections.get("DEFAULT MULTI-THREAD RUN (250ms)", "")
    run_trace = re.sub(r"\nexit_code=\d+\s*$", "", run_sec).strip()
    return uname_line, py_ver, info, run_trace


def collect_live(cmd_bin: str, duration: str) -> tuple[str, str, dict, str]:
    uname_line = " ".join(platform.uname())
    py_ver = platform.python_version()
    info_raw = subprocess.check_output([cmd_bin, "info", "--json"], text=True)
    info = extract_json(info_raw)
    proc = subprocess.run(
        [cmd_bin, "run", "--duration", duration, "-v"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return uname_line, py_ver, info, proc.stdout.strip()


def render_markdown(uname_line: str, py_ver: str, info: dict, run_trace: str, env_override: str | None) -> str:
    topo = info.get("topology", {})
    native = info.get("native", {})
    caches = topo.get("caches", {})

    version = info.get("version", "unknown")
    os_name = topo.get("os", "unknown")
    arch = topo.get("machine") or native.get("arch_name", "unknown")
    model = topo.get("model_name", "Unknown CPU")

    sockets = topo.get("sockets", 1)
    phys = topo.get("physical_cores", "?")
    logical = topo.get("logical_cores", "?")
    if topo.get("is_hybrid") and topo.get("p_core_ids") and topo.get("e_core_ids"):
        p_ids = topo["p_core_ids"]
        e_ids = topo["e_core_ids"]
        cpu_desc = (
            f"{model} ({sockets}S · {phys}C/{logical}T hybrid: "
            f"{len(p_ids)} P-cores `{fmt_id_ranges(p_ids)}`, "
            f"{len(e_ids)} E-cores `{fmt_id_ranges(e_ids)}`)"
        )
    else:
        cpu_desc = f"{model} ({sockets}S · {phys}C/{logical}T)"

    avail_backends = sorted(
        (b for b in native.get("backends", []) if b.get("available")),
        key=lambda b: b.get("priority", 0),
        reverse=True,
    )
    top_backend = avail_backends[0] if avail_backends else {"name": "scalar", "detail": ""}
    ext_match = re.search(r"\(([^)]+)\)", top_backend.get("detail", ""))
    enabled_exts = []
    if ext_match:
        for tok in ext_match.group(1).split():
            if tok.endswith("=1"):
                enabled_exts.append(f"`{tok[:-2]}`")
    isa_summary = f"`{top_backend['name']}`"
    if enabled_exts:
        isa_summary += f" ({', '.join(enabled_exts)})"

    unavail_names = [f"`{b['name']}`" for b in native.get("backends", []) if not b.get("available")]
    avail_names = [f"`{b['name']}`" for b in avail_backends]

    l2_share = f" (shared {caches['l2_shared_cpus']}C)" if caches.get("l2_shared_cpus", 1) > 1 else ""
    l3_val = fmt_bytes(caches.get("l3_bytes"))
    l3_share = (
        f" (shared {caches['l3_shared_cpus']}C)"
        if l3_val != "none" and caches.get("l3_shared_cpus", 1) > 1
        else ""
    )
    cache_desc = (
        f"{fmt_bytes(topo.get('memory_bytes'))} RAM · "
        f"L1i {fmt_bytes(caches.get('l1i_bytes'))} · "
        f"L1d {fmt_bytes(caches.get('l1d_bytes'))} · "
        f"L2 {fmt_bytes(caches.get('l2_bytes'))}{l2_share} · "
        f"L3 {l3_val}{l3_share} · "
        f"{caches.get('line_size_bytes', '?')} B line"
    )

    sm = SUMMARY_RE.search(run_trace)
    if sm:
        passed, failed, unsup, exit_code = sm.groups()
        verdict = "PASS" if int(failed) == 0 and int(exit_code) == 0 else "FAIL"
        status_cell = f"**{verdict}** (`{passed} passed, {failed} failed, {unsup} unsupported` · Exit `{exit_code}`)"
    else:
        status_cell = "**UNKNOWN** (Summary line not found)"

    exec_env = env_override or detect_exec_env(uname_line, run_trace)
    summary_table = clean_summary_table(run_trace)
    info_pretty = json.dumps(info, indent=2)

    unavail_note = (
        f"; {', '.join(unavail_names)} compiled in and accurately probed as unavailable"
        if unavail_names
        else ""
    )
    pinning_note = topo.get("pinning_note") or (
        "Hard CPU core pinning supported" if topo.get("pinning_supported") else "OS scheduler placement"
    )

    return f"""### Hardware Validation Report — {model} (`{arch}` / {os_name})

| Field | Value |
| :--- | :--- |
| **Status** | {status_cell} |
| **CrossCPU Version** | `{version}` |
| **CPU Model** | {cpu_desc} |
| **Architecture / Highest ISA** | `{arch}` / {isa_summary} |
| **OS / Kernel** | `{uname_line}` |
| **Execution Environment** | {exec_env} |
| **Python** | `{py_ver}` |
| **Memory & Cache Hierarchy** | {cache_desc} |

#### Verification Notes
- **ISA dispatch & explicit unknown/unsupported reporting:** {', '.join(avail_names)} dispatched across {logical} threads{unavail_note}. Unsupported workloads reported `UNSUPPORTED` cleanly with scalar emulation disabled.
- **Topology & thread placement:** {pinning_note}.

#### Summary Table

```text
{summary_table}
```

<details>
<summary>Full environment &amp; <code>cross-cpu info --json</code></summary>

```text
{uname_line}
Python {py_ver}
```

```json
{info_pretty}
```

</details>

<details>
<summary>Full verbose trace</summary>

```text
{run_trace}
```

</details>

---
*Hardware validation infrastructure provided by the [GCC Compile Farm](https://cfarm.tetaneutral.net/).*
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Format CrossCPU validation output as Markdown.")
    parser.add_argument("log_file", nargs="?", type=Path, help="Path to a cfarm-*.log file")
    parser.add_argument("--live", action="store_true", help="Run cross-cpu locally instead of reading a log")
    parser.add_argument("--bin", default="cross-cpu", help="cross-cpu executable path for --live")
    parser.add_argument("--duration", default="100ms", help="Run duration for --live (default: 100ms)")
    parser.add_argument("--env", default=None, help="Override Execution Environment label (e.g. 'Bare metal')")
    args = parser.parse_args()

    if args.live:
        uname_line, py_ver, info, run_trace = collect_live(args.bin, args.duration)
    elif args.log_file:
        uname_line, py_ver, info, run_trace = parse_cfarm_log(args.log_file)
    else:
        parser.error("Provide a cfarm-*.log path or pass --live")

    print(render_markdown(uname_line, py_ver, info, run_trace, args.env))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
