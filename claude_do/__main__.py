#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Entry point for running claude-do as a module: python -m claude_do"""

import sys

from .cli import claude_do

if __name__ == "__main__":
    sys.exit(claude_do())
