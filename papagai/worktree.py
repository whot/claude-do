#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Git worktree management for papagai."""

import logging
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

from .cmd import run_command

logger = logging.getLogger("papagai.worktree")

BRANCH_PREFIX = "papagai"
LATEST_BRANCH = f"{BRANCH_PREFIX}/latest"


def repoint_latest_branch(repo_dir: Path, branch: str) -> None:
    """
    Update the papagai/latest branch to point to the specified branch.

    Removes the papagai/latest branch if it exists, then creates it
    pointing to the same commit as the specified branch.

    Args:
        repo_dir: Path to the repository root
        branch: Branch name to point papagai/latest to
    """
    try:
        run_command(
            ["git", "branch", "-f", LATEST_BRANCH, branch],
            cwd=repo_dir,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to update {LATEST_BRANCH}: {e}", file=sys.stderr)


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
        date = datetime.now().strftime("%Y%m%d-%H%M")
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

            repoint_latest_branch(self.repo_dir, self.branch)

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


@dataclass
class WorktreeOverlayFs(Worktree):
    """
    Git worktree using overlay filesystem (fuse-overlayfs).

    This class creates a copy-on-write worktree using fuse-overlayfs,
    where the original repository is the read-only lower layer and
    modifications are stored in an upper layer.

    Attributes:
        worktree_dir: Path to the mounted overlay directory
        branch: Name of the created branch
        repo_dir: Path to the repository root
        overlay_base_dir: Path to the overlay filesystem base directory
        mount_dir: Path to the mounted overlay filesystem
    """

    overlay_base_dir: Path | None = None
    mount_dir: Path | None = None

    def umount(self, check: bool = False):
        run_command(["fusermount", "-u", str(self.mount_dir)], check=check)

    @classmethod
    def is_supported(cls) -> bool:
        """
        Check if fuse-overlayfs is available on the system.

        Returns:
            True if fuse-overlayfs command is available, False otherwise
        """
        return shutil.which("fuse-overlayfs") is not None

    @classmethod
    def from_branch(
        cls, repo_dir: Path, base_branch: str, branch_prefix: str | None = None
    ) -> Self:
        """
        Create a new overlay worktree using fuse-overlayfs.

        Directory structure:
        - $XDG_CACHE_HOME/papagai/<project>/<branch>-<date>-<uuid>/
           - upperdir/
           - workdir/  - fuse-overlayfs workdir
           - mounted/  - mount point

        The repo_dir is the read-only lower layer with fuse-overlayfs.

        Args:
            repo_dir: Path to the repository root
            base_branch: Branch to base the new branch on
            branch_prefix: Optional prefix for the branch name

        Returns:
            WorktreeOverlayFs instance with mounted overlay filesystem

        Raises:
            subprocess.CalledProcessError: If git or mount operations fail
            RuntimeError: If fuse-overlayfs is not available
        """
        assert base_branch is not None

        # Generate unique directory name using same scheme as Worktree
        rand = str(uuid.uuid4()).split("-")[0]
        date = datetime.now().strftime("%Y%m%d-%H%M")
        # Skip the branch prefix here so we don't nest directories too much
        branch = f"{base_branch}-{date}-{rand}"

        xdg_cache_home = (
            Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "papagai"
        )
        overlay_base_dir = xdg_cache_home / repo_dir.name / branch
        overlay_base_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Setting up overlayfs in {overlay_base_dir}")

        upperdir = overlay_base_dir / "upperdir"
        workdir = overlay_base_dir / "workdir"
        mount_dir = overlay_base_dir / "mounted"

        upperdir.mkdir(exist_ok=True)
        workdir.mkdir(exist_ok=True)
        mount_dir.mkdir(exist_ok=True)

        # Now add the branch prefix
        branch = f"{branch_prefix or ''}{branch}"

        try:
            run_command(
                [
                    "fuse-overlayfs",
                    "-o",
                    f"lowerdir={repo_dir},upperdir={upperdir},workdir={workdir}",
                    str(mount_dir),
                ]
            )
        except subprocess.CalledProcessError as e:
            # Cleanup directories if mount fails
            shutil.rmtree(overlay_base_dir, ignore_errors=True)
            raise RuntimeError(
                f"Failed to mount overlay filesystem. Is fuse-overlayfs installed? Error: {e}"
            ) from e

        # Create a new git branch in the mounted directory
        try:
            run_command(
                ["git", "checkout", "-fb", branch, base_branch],
                cwd=mount_dir,
            )
        except subprocess.CalledProcessError as e:
            # Cleanup on failure
            run_command(["fusermount", "-u", str(mount_dir)], check=False)
            shutil.rmtree(overlay_base_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to create git branch: {e}") from e

        return cls(
            worktree_dir=mount_dir,
            branch=branch,
            repo_dir=repo_dir,
            overlay_base_dir=overlay_base_dir,
            mount_dir=mount_dir,
        )

    def _cleanup(self) -> None:
        """Clean up the overlay filesystem and directories."""
        try:
            # Check for uncommitted changes
            try:
                run_command(
                    ["git", "diff", "--quiet", "--exit-code"],
                    cwd=self.worktree_dir,
                    check=True,
                )
            except subprocess.SubprocessError:
                print(
                    f"Changes still present in worktree {self.worktree_dir}, refusing to clean up."
                )
                print("To clean up manually, run:")
                print(f"  $ fusermount -u {self.mount_dir}")
                print(f"  $ rm -rf {self.overlay_base_dir}")
                return

            # Pull the branch from the overlay into the main repository
            # before unmounting and cleaning up
            try:
                run_command(
                    [
                        "git",
                        "fetch",
                        str(self.mount_dir),
                        f"{self.branch}:{self.branch}",
                    ],
                    cwd=self.repo_dir,
                    check=True,
                )
                # Verify the branch exists in the main repository
                run_command(
                    ["git", "rev-parse", "--verify", self.branch],
                    cwd=self.repo_dir,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(
                    f"Warning: Failed to pull branch {self.branch} from overlay: {e}",
                    file=sys.stderr,
                )
                print("To clean up manually, run:")
                print(f"  $ git fetch {self.mount_dir} {self.branch}:{self.branch}")
                print(f"  $ fusermount -u {self.mount_dir}")
                print(f"  $ rm -rf {self.overlay_base_dir}")
                return

            repoint_latest_branch(self.repo_dir, self.branch)

            # Unmount the overlay filesystem
            if self.mount_dir and self.mount_dir.exists():
                try:
                    self.umount(check=True)
                except subprocess.CalledProcessError as e:
                    print(
                        f"Warning: Failed to unmount {self.mount_dir}: {e}",
                        file=sys.stderr,
                    )
                    print(
                        f"You may need to manually unmount: fusermount -u {self.mount_dir}",
                        file=sys.stderr,
                    )
                    return

            # Remove the overlay base directory
            if self.overlay_base_dir and self.overlay_base_dir.exists():
                shutil.rmtree(self.overlay_base_dir, ignore_errors=True)

        except Exception as e:
            print(f"Warning during cleanup: {e}", file=sys.stderr)
