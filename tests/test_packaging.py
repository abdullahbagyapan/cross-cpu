# SPDX-License-Identifier: MIT
"""Offline packaging, SPDX header, completion parity, and sdist completeness tests."""

from __future__ import annotations

import ast
import importlib.util
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from src.cli import ALL_ISA_NAMES, SUBCOMMANDS
from src.topology import CORE_PRESETS
from src.workloads import ALL_WORKLOAD_NAMES

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestPackagingAndHygiene(unittest.TestCase):
    def test_spdx_headers_across_source_files(self) -> None:
        checked = 0
        # Python, shell, Makefile, TOML, YAML
        for pattern in (
            "src/**/*.py",
            "tests/**/*.py",
            "completions/*",
            "setup.py",
            "pyproject.toml",
            "Makefile",
            "MANIFEST.in",
            "cross-cpu",
            ".github/workflows/*.yml",
        ):
            for path in sorted(REPO_ROOT.glob(pattern)):
                if not path.is_file():
                    continue
                lines = path.read_text(encoding="utf-8").splitlines()
                self.assertTrue(len(lines) >= 1, msg=f"Empty file: {path}")
                if path.name == "_cross-cpu":
                    self.assertEqual(lines[0], "#compdef cross-cpu")
                    self.assertIn("SPDX-License-Identifier: MIT", lines[1])
                elif lines[0].startswith("#!"):
                    self.assertIn("SPDX-License-Identifier: MIT", lines[1])
                else:
                    self.assertIn("SPDX-License-Identifier: MIT", lines[0], msg=str(path))
                checked += 1

        # C11 sources and headers
        for pattern in ("native/**/*.c", "native/**/*.h"):
            for path in sorted(REPO_ROOT.glob(pattern)):
                first_line = path.read_text(encoding="utf-8").splitlines()[0]
                self.assertEqual(first_line, "// SPDX-License-Identifier: MIT", msg=str(path))
                checked += 1

        self.assertGreaterEqual(checked, 40)

    def test_src_relative_imports_only_and_zero_third_party_deps(self) -> None:
        stdlib_names = set(getattr(sys, "stdlib_module_names", ()))
        for path in sorted((REPO_ROOT / "src").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".", 1)[0]
                        self.assertNotEqual(
                            top, "cross_cpu", msg=f"Absolute cross_cpu import in {path}"
                        )
                        if stdlib_names:
                            self.assertIn(
                                top,
                                stdlib_names,
                                msg=f"Non-stdlib runtime import {top!r} in {path}",
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        top = node.module.split(".", 1)[0]
                        self.assertNotEqual(
                            top, "cross_cpu", msg=f"Absolute cross_cpu import in {path}"
                        )
                        self.assertNotEqual(
                            top, "src", msg=f"Absolute src import in {path}"
                        )
                        if stdlib_names:
                            self.assertIn(
                                top,
                                stdlib_names,
                                msg=f"Non-stdlib runtime import {top!r} in {path}",
                            )

    def test_completion_parity_and_bash_syntax(self) -> None:
        bash_file = REPO_ROOT / "completions" / "cross-cpu.bash"
        zsh_file = REPO_ROOT / "completions" / "_cross-cpu"
        fish_file = REPO_ROOT / "completions" / "cross-cpu.fish"

        bash_txt = bash_file.read_text(encoding="utf-8")
        zsh_txt = zsh_file.read_text(encoding="utf-8")
        fish_txt = fish_file.read_text(encoding="utf-8")

        for label, txt in (("bash", bash_txt), ("zsh", zsh_txt), ("fish", fish_txt)):
            for cmd in SUBCOMMANDS:
                self.assertIn(cmd, txt, msg=f"{label} completion missing command {cmd}")
            for wl in ALL_WORKLOAD_NAMES:
                self.assertIn(wl, txt, msg=f"{label} completion missing workload {wl}")
            for isa in ALL_ISA_NAMES:
                self.assertIn(isa, txt, msg=f"{label} completion missing ISA {isa}")
            for preset in CORE_PRESETS:
                self.assertIn(preset, txt, msg=f"{label} completion missing core preset {preset}")
            for flag in (
                "--cores",
                "--numa",
                "--isa",
                "--test",
                "--duration",
                "--warmup",
                "--format",
                "--json",
                "--verbose",
                "--no-banner",
            ):
                flag_key = flag.lstrip("-") if label == "fish" else flag
                self.assertIn(flag_key, txt, msg=f"{label} completion missing flag {flag}")

        bash_bin = shutil.which("bash")
        if bash_bin:
            proc = subprocess.run(
                [bash_bin, "-n", str(bash_file)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"bash -n failed: {proc.stderr}")

    def test_pyproject_package_mapping_and_zero_dependencies(self) -> None:
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('dependencies = []', pyproject)
        self.assertIn('"cross_cpu" = "src"', pyproject)
        self.assertIn('cross-cpu = "cross_cpu.cli:main"', pyproject)

    def test_sdist_includes_all_arch_directories_offline(self) -> None:
        if importlib.util.find_spec("setuptools") is None:
            self.skipTest("setuptools not installed in current test interpreter")

        with tempfile.TemporaryDirectory(prefix="cross-cpu-sdist-") as tmp:
            dist_dir = Path(tmp) / "dist"
            egg_dir = Path(tmp) / "egg"
            egg_dir.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [
                    sys.executable,
                    "setup.py",
                    "egg_info",
                    "--egg-base",
                    str(egg_dir),
                    "sdist",
                    "--dist-dir",
                    str(dist_dir),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                proc.returncode, 0, msg=f"sdist failed:\n{proc.stdout}\n{proc.stderr}"
            )
            tarballs = list(dist_dir.glob("*.tar.gz"))
            self.assertEqual(len(tarballs), 1)

            with tarfile.open(tarballs[0], "r:gz") as tf:
                names = tf.getnames()

            required = [
                "LICENSE",
                "Makefile",
                "pyproject.toml",
                "setup.py",
                "MANIFEST.in",
                "native/include/cross_cpu_abi.h",
                "native/include/cross_cpu_arch.h",
                "native/core/worker_pool.c",
                "native/core/dispatch.c",
                "native/arch/common/isa_scalar.c",
                "native/arch/common/cache_flush_generic.c",
                "native/arch/x86_64/probe_x86.c",
                "native/arch/x86_64/isa_sse4.c",
                "native/arch/x86_64/isa_avx2.c",
                "native/arch/x86_64/isa_avx512.c",
                "native/arch/x86_64/isa_amx.c",
                "native/arch/arm64/probe_arm64.c",
                "native/arch/arm64/isa_neon.c",
                "native/arch/arm64/isa_sve.c",
                "native/arch/arm64/isa_sme.c",
                "native/arch/other/riscv64/probe_riscv64.c",
                "native/arch/other/riscv64/isa_rvv.c",
                "native/arch/other/ppc64/probe_ppc64.c",
                "native/arch/other/ppc64/isa_vsx.c",
                "native/arch/other/s390x/probe_s390x.c",
                "native/arch/other/s390x/isa_vx.c",
                "native/arch/other/generic/probe_generic.c",
                "src/cli.py",
                "src/telemetry/readers.py",
                "completions/cross-cpu.bash",
                "completions/_cross-cpu",
                "completions/cross-cpu.fish",
                "tests/test_packaging.py",
            ]
            # Verify docs/changelog when present in workspace
            for optional_doc in ("docs/cross-cpu.1", "CHANGELOG.md", "README.md"):
                if (REPO_ROOT / optional_doc).is_file():
                    required.append(optional_doc)

            for req in required:
                self.assertTrue(
                    any(re.search(rf"/{re.escape(req)}$", n) for n in names),
                    msg=f"Missing {req} in sdist tarball",
                )
            # Confirm no compiled binary artifacts leaked into sdist
            for n in names:
                self.assertFalse(
                    n.endswith((".so", ".dylib", ".o", ".pyc")),
                    msg=f"Compiled artifact leaked into sdist: {n}",
                )


if __name__ == "__main__":
    unittest.main()
