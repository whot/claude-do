#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Git worktree management for papagai."""

import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

from .cmd import run_command

BRANCH_PREFIX = "papagai"


@dataclass
class Worktree:
    """
    Git worktree context manager for automated branch creation and cleanup.

    Attributes:
        worktree_dir: Path to the worktree directory
        branch: Name of the created branch
        repo_dir: Path to the repository root
    """

    worktree_dir: Path
    branch: str
    repo_dir: Path

    @classmethod
    def from_branch(
        cls, repo_dir: Path, base_branch: str, branch_prefix: str | None = None
    ) -> Self:
        """
        Create a new review branch using git worktree.

        Args:
            repo_dir: Path to the repository root
            base_branch: Branch to base the new branch on

        Returns:
            Worktree instance

        Raises:
            subprocess.CalledProcessError: If git worktree creation fails
        """
        assert base_branch is not None
        rand = str(uuid.uuid4()).split("-")[0]
        date = datetime.now().strftime("%Y-%m-%d")
        branch_prefix = branch_prefix or ""
        branch = f"{branch_prefix}{base_branch}-{date}-{rand}"
        worktree_dir = repo_dir / branch

        run_command(
            [
                "git",
                "worktree",
                "add",
                "--quiet",
                "-b",
                branch,
                str(worktree_dir),
                base_branch,
            ],
            cwd=repo_dir,
        )

        return cls(worktree_dir=worktree_dir, branch=branch, repo_dir=repo_dir)

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and cleanup the worktree."""
        self._cleanup()

    def _cleanup(self) -> None:
        """Clean up the worktree and any empty parent directories."""
        try:
            try:
                run_command(
                    ["git", "diff", "--quiet", "--exit-code"],
                    cwd=self.worktree_dir,
                    check=True,
                )
            except subprocess.SubprocessError:
                print("Changes still present in worktree, refusing to clean up.")
                print("To clean up manually, run:")
                print(f"  $ git worktree remove --force {self.branch}")
                return

            run_command(
                ["git", "worktree", "remove", "--force", str(self.branch)],
                cwd=self.repo_dir,
                check=False,
            )

            # Remove the worktree directory if it still exists
            if self.worktree_dir.exists():
                shutil.rmtree(self.worktree_dir, ignore_errors=True)

            # Remove empty parent directories up to repo_dir
            current = self.worktree_dir.parent
            while current != self.repo_dir and current.is_relative_to(self.repo_dir):
                try:
                    if current.exists() and not any(current.iterdir()):
                        current.rmdir()
                        current = current.parent
                    else:
                        break
                except OSError:
                    # Directory not empty or other error, stop cleanup
                    break
        except Exception as e:
            print(f"Warning during cleanup: {e}", file=sys.stderr)
