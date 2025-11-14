#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Command execution utilities."""

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("papagai.cmd")


def run_command(
    cmd: list[str], cwd: Optional[Path] = None, check: bool = True
) -> subprocess.CompletedProcess:
    """
    Run a command and return the result.

    Args:
        cmd: Command and arguments as a list of strings
        cwd: Optional working directory for the command
        check: If True, raise CalledProcessError on non-zero exit

    Returns:
        CompletedProcess instance with stdout, stderr, and return code

    Raises:
        subprocess.CalledProcessError: If check is True and command fails
    """
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )
