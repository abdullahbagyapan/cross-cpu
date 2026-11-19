# SPDX-License-Identifier: MIT
"""Unit tests for severity-ordered ExitCode aggregation."""

from __future__ import annotations

import unittest

from src.exit_codes import ExitCode, aggregate_exit_codes


class TestExitCodes(unittest.TestCase):
    def test_enum_values_and_ordering(self) -> None:
        self.assertEqual(int(ExitCode.SUCCESS), 0)
        self.assertEqual(int(ExitCode.USAGE_ERROR), 1)
        self.assertEqual(int(ExitCode.INIT_ERROR), 2)
        self.assertEqual(int(ExitCode.COMPUTE_ERROR), 3)
        self.assertEqual(int(ExitCode.HARDWARE_ERROR), 4)
        self.assertLess(ExitCode.SUCCESS, ExitCode.USAGE_ERROR)
        self.assertLess(ExitCode.USAGE_ERROR, ExitCode.INIT_ERROR)
        self.assertLess(ExitCode.INIT_ERROR, ExitCode.COMPUTE_ERROR)
        self.assertLess(ExitCode.COMPUTE_ERROR, ExitCode.HARDWARE_ERROR)

    def test_aggregation_uses_maximum_severity(self) -> None:
        self.assertEqual(aggregate_exit_codes([]), ExitCode.SUCCESS)
        self.assertEqual(aggregate_exit_codes([ExitCode.SUCCESS]), ExitCode.SUCCESS)
        self.assertEqual(
            aggregate_exit_codes([ExitCode.SUCCESS, ExitCode.INIT_ERROR, ExitCode.USAGE_ERROR]),
            ExitCode.INIT_ERROR,
        )
        self.assertEqual(
            aggregate_exit_codes([ExitCode.INIT_ERROR, ExitCode.COMPUTE_ERROR, ExitCode.SUCCESS]),
            ExitCode.COMPUTE_ERROR,
        )
        self.assertEqual(
            aggregate_exit_codes(
                [ExitCode.COMPUTE_ERROR, ExitCode.HARDWARE_ERROR, ExitCode.USAGE_ERROR]
            ),
            ExitCode.HARDWARE_ERROR,
        )


if __name__ == "__main__":
    unittest.main()
