# SPDX-License-Identifier: MIT
"""Unit tests for CrossCPU CLI parsing, timing, typo hints, banner guards, and stream hygiene."""

from __future__ import annotations

import io
import json
import unittest

from src.banner import BANNER_LINES, render_banner, should_show_banner, supports_color
from src.cli import CLIUsageError, main, parse_duration
from src.exit_codes import ExitCode
from src.runner import format_duration_progress


class _FakeTTY(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class TestDurationAndProgress(unittest.TestCase):
    def test_parse_duration_suffixes(self) -> None:
        self.assertAlmostEqual(parse_duration("500ms"), 0.5)
        self.assertAlmostEqual(parse_duration("1.5s"), 1.5)
        self.assertAlmostEqual(parse_duration("5m"), 300.0)
        self.assertAlmostEqual(parse_duration("1h"), 3600.0)
        self.assertAlmostEqual(parse_duration("250MS"), 0.25)

    def test_parse_duration_zero_and_negative_rules(self) -> None:
        with self.assertRaises(CLIUsageError):
            parse_duration("0s", allow_zero=False)
        with self.assertRaises(CLIUsageError):
            parse_duration("0ms", allow_zero=False)
        with self.assertRaises(CLIUsageError):
            parse_duration("-5s", allow_zero=False)
        self.assertEqual(parse_duration("0s", allow_zero=True), 0.0)
        self.assertEqual(parse_duration("0ms", allow_zero=True), 0.0)
        with self.assertRaises(CLIUsageError):
            parse_duration("-1ms", allow_zero=True)

    def test_parse_duration_invalid_inputs(self) -> None:
        for bad in ("10", "abc", "5x", "ms", "", "10s extra"):
            with self.assertRaises(CLIUsageError, msg=f"Should reject {bad!r}"):
                parse_duration(bad)

    def test_subsecond_progress_formatted_in_milliseconds(self) -> None:
        self.assertEqual(format_duration_progress(0.25, 0.5), "250ms/500ms")
        self.assertNotEqual(format_duration_progress(0.25, 0.5), "0s/0s")
        self.assertEqual(format_duration_progress(18.2, 60.0), "18s/60s")


class TestBannerGuards(unittest.TestCase):
    def test_banner_width_under_80_cols(self) -> None:
        for line in BANNER_LINES:
            self.assertLess(len(line), 80)
        for line in render_banner(use_color=False).splitlines():
            self.assertLess(len(line), 80)

    def test_banner_and_color_guards(self) -> None:
        tty = _FakeTTY(True)
        pipe = _FakeTTY(False)

        self.assertTrue(should_show_banner(tty, env={"TERM": "xterm-256color"}))
        self.assertFalse(should_show_banner(tty, no_banner=True, env={"TERM": "xterm"}))
        self.assertFalse(should_show_banner(tty, json_mode=True, env={"TERM": "xterm"}))
        self.assertFalse(should_show_banner(pipe, env={"TERM": "xterm"}))
        self.assertFalse(should_show_banner(tty, env={"TERM": "dumb"}))

        self.assertTrue(supports_color(tty, env={"TERM": "xterm-256color"}))
        self.assertFalse(supports_color(tty, env={"NO_COLOR": "1", "TERM": "xterm"}))
        self.assertFalse(supports_color(tty, env={"NO_COLOR": "", "TERM": "xterm"}))
        self.assertFalse(supports_color(tty, env={"TERM": "dumb"}))
        self.assertFalse(supports_color(pipe, env={"TERM": "xterm"}))


class TestCLIBehavior(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        out = io.StringIO()
        err = io.StringIO()
        rc = main(argv, stdout=out, stderr=err)
        return rc, out.getvalue(), err.getvalue()

    def test_help_and_version_stdout(self) -> None:
        rc, out, err = self._run_cli(["--version"])
        self.assertEqual(rc, int(ExitCode.SUCCESS))
        self.assertIn("cross-cpu", out)
        self.assertEqual(err, "")

        rc, out, err = self._run_cli(["--help"])
        self.assertEqual(rc, int(ExitCode.SUCCESS))
        self.assertIn("Examples:", out)
        self.assertEqual(err, "")

    def test_usage_errors_return_exit_code_1_with_difflib_hints(self) -> None:
        # Unknown subcommand typo -> exit 1 + hint
        rc, out, err = self._run_cli(["lst"])
        self.assertEqual(rc, int(ExitCode.USAGE_ERROR))
        self.assertEqual(out, "")
        self.assertIn("list", err)

        # Unknown flag -> exit 1 (not argparse default 2)
        rc, out, err = self._run_cli(["run", "--nonexistent-flag"])
        self.assertEqual(rc, int(ExitCode.USAGE_ERROR))
        self.assertEqual(out, "")
        self.assertIn("Usage error:", err)

        # Unknown workload typo -> exit 1 + hint
        rc, out, err = self._run_cli(["run", "--test", "compute-fp33", "--duration", "100ms"])
        self.assertEqual(rc, int(ExitCode.USAGE_ERROR))
        self.assertIn("compute-fp32", err)

        # Unknown ISA typo -> exit 1 + hint
        rc, out, err = self._run_cli(["run", "--isa", "avx22", "--duration", "100ms"])
        self.assertEqual(rc, int(ExitCode.USAGE_ERROR))
        self.assertIn("avx2", err)

        # Unknown cores preset typo -> exit 1 + hint
        rc, out, err = self._run_cli(["run", "--cores", "p-core", "--duration", "100ms"])
        self.assertEqual(rc, int(ExitCode.USAGE_ERROR))
        self.assertIn("p-cores", err)

        # Invalid duration <= 0 -> exit 1
        rc, out, err = self._run_cli(["run", "--duration", "0s"])
        self.assertEqual(rc, int(ExitCode.USAGE_ERROR))

        # Conflicting --format text --json -> exit 1
        rc, out, err = self._run_cli(["list", "--format", "text", "--json"])
        self.assertEqual(rc, int(ExitCode.USAGE_ERROR))
        self.assertIn("Conflicting", err)

    def test_json_commands_emit_clean_stdout_and_verbose_stderr(self) -> None:
        rc, out, err = self._run_cli(["list", "--json", "-v"])
        self.assertEqual(rc, int(ExitCode.SUCCESS))
        data = json.loads(out)
        self.assertEqual(data["command"], "list")
        self.assertEqual(len(data["workloads"]), 21)
        self.assertIn("[verbose]", err)

        rc, out, err = self._run_cli(["info", "--json"])
        self.assertEqual(rc, int(ExitCode.SUCCESS))
        info = json.loads(out)
        self.assertEqual(info["command"], "info")
        self.assertIn("topology", info)
        self.assertIn("native", info)
        self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
