# SPDX-License-Identifier: MIT
"""Severity-ordered process exit codes for CrossCPU."""

from __future__ import annotations

from enum import IntEnum
from typing import Iterable


class ExitCode(IntEnum):
    """Severity-ordered exit codes; combine multiple outcomes with ``max()``."""

    SUCCESS = 0
    USAGE_ERROR = 1
    INIT_ERROR = 2
    COMPUTE_ERROR = 3
    HARDWARE_ERROR = 4


def aggregate_exit_codes(codes: Iterable[int | ExitCode]) -> ExitCode:
    """Return the highest-severity exit code across ``codes``."""
    highest = int(ExitCode.SUCCESS)
    for code in codes:
        val = int(code)
        if val > highest:
            highest = val
    if highest > int(ExitCode.HARDWARE_ERROR):
        highest = int(ExitCode.HARDWARE_ERROR)
    if highest < int(ExitCode.SUCCESS):
        highest = int(ExitCode.SUCCESS)
    return ExitCode(highest)
