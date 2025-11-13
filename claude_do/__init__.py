#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""claude-do: A CLI tool to run Claude on git worktrees with automated commits."""

__version__ = "0.1.0"

from .cli import claude_do

__all__ = ["claude_do"]
