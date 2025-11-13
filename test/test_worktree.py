#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for worktree management utilities."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from papagai.worktree import BRANCH_PREFIX, Worktree, WorktreeOverlayFs


@pytest.fixture
def mock_git_repo(tmp_path):
    """Create a mock git repository directory."""
    repo_dir = tmp_path / "test-repo"
    repo_dir.mkdir()
    return repo_dir


@pytest.fixture
def mock_worktree(mock_git_repo):
    """Create a mock Worktree instance."""
    worktree_dir = mock_git_repo / "papagai" / "main-2025-01-01-abc123"
    branch = "papagai/main-2025-01-01-abc123"
    return Worktree(
        worktree_dir=worktree_dir,
        branch=branch,
        repo_dir=mock_git_repo,
    )


class TestWorktreeDataclass:
    """Tests for Worktree dataclass structure."""

    def test_worktree_initialization(self, mock_git_repo):
        """Test Worktree can be initialized with required fields."""
        worktree_dir = mock_git_repo / "test-worktree"
        branch = "papagai/test-branch"

        worktree = Worktree(
            worktree_dir=worktree_dir,
            branch=branch,
            repo_dir=mock_git_repo,
        )

        assert worktree.worktree_dir == worktree_dir
        assert worktree.branch == branch
        assert worktree.repo_dir == mock_git_repo

    def test_worktree_attributes_are_paths(self, mock_worktree):
        """Test that worktree_dir and repo_dir are Path objects."""
        assert isinstance(mock_worktree.worktree_dir, Path)
        assert isinstance(mock_worktree.repo_dir, Path)
        assert isinstance(mock_worktree.branch, str)


class TestFromBranch:
    """Tests for Worktree.from_branch() classmethod."""

    @pytest.mark.parametrize(
        "base_branch", ["main", "develop", "feature/test", "v1.0.0"]
    )
    def test_from_branch_creates_worktree(self, mock_git_repo, base_branch):
        """Test from_branch creates a worktree for different base branches."""
        with patch("papagai.worktree.run_command") as mock_run:
            mock_run.return_value = MagicMock()

            worktree = Worktree.from_branch(
                mock_git_repo, base_branch, branch_prefix=f"{BRANCH_PREFIX}/"
            )

            # Check worktree attributes
            assert worktree.repo_dir == mock_git_repo
            assert worktree.branch.startswith(f"{BRANCH_PREFIX}/{base_branch}")
            assert str(worktree.worktree_dir).startswith(str(mock_git_repo))

            # Verify git command was called
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0][0] == "git"
            assert call_args[0][0][1] == "worktree"
            assert call_args[0][0][2] == "add"
            assert base_branch in call_args[0][0]

    def test_from_branch_creates_unique_branches(self, mock_git_repo):
        """Test from_branch creates unique branch names on each call."""
        with patch("papagai.worktree.run_command"):
            worktree1 = Worktree.from_branch(mock_git_repo, "main")
            worktree2 = Worktree.from_branch(mock_git_repo, "main")

            assert worktree1.branch != worktree2.branch

    def test_from_branch_branch_name_format(self, mock_git_repo):
        """Test that branch names follow the expected format."""
        with patch("papagai.worktree.run_command"):
            worktree = Worktree.from_branch(
                mock_git_repo, "main", branch_prefix=f"{BRANCH_PREFIX}/"
            )

            # Branch should be: papagai/main-YYYY-MM-DD-XXXXXXXX
            parts = worktree.branch.split("/")
            assert len(parts) == 2
            assert parts[0] == BRANCH_PREFIX

            # Second part should be: main-YYYY-MM-DD-XXXXXXXX
            branch_parts = parts[1].split("-")
            assert branch_parts[0] == "main"
            assert len(branch_parts) >= 4  # base-YYYY-MM-DD-uuid

    def test_from_branch_git_command_parameters(self, mock_git_repo):
        """Test that git worktree command is called with correct parameters."""
        with patch("papagai.worktree.run_command") as mock_run:
            worktree = Worktree.from_branch(mock_git_repo, "develop")

            call_args = mock_run.call_args
            git_cmd = call_args[0][0]

            assert git_cmd[0] == "git"
            assert git_cmd[1] == "worktree"
            assert git_cmd[2] == "add"
            assert "--quiet" in git_cmd
            assert "-b" in git_cmd
            assert worktree.branch in git_cmd
            assert str(worktree.worktree_dir) in git_cmd
            assert "develop" in git_cmd

            # Check cwd parameter
            assert call_args[1]["cwd"] == mock_git_repo

    def test_from_branch_raises_on_git_error(self, mock_git_repo):
        """Test from_branch raises CalledProcessError when git fails."""
        with patch("papagai.worktree.run_command") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")

            with pytest.raises(subprocess.CalledProcessError):
                Worktree.from_branch(mock_git_repo, "main")


class TestContextManager:
    """Tests for Worktree context manager functionality."""

    def test_context_manager_enter(self, mock_worktree):
        """Test __enter__ returns the Worktree instance."""
        result = mock_worktree.__enter__()
        assert result is mock_worktree

    def test_context_manager_exit_calls_cleanup(self, mock_worktree):
        """Test __exit__ calls _cleanup method."""
        with patch.object(mock_worktree, "_cleanup") as mock_cleanup:
            mock_worktree.__exit__(None, None, None)
            mock_cleanup.assert_called_once()

    def test_context_manager_with_statement(self, mock_git_repo):
        """Test Worktree works correctly in with statement."""
        with patch("papagai.worktree.run_command"):
            worktree = Worktree.from_branch(mock_git_repo, "main")

        with patch.object(worktree, "_cleanup") as mock_cleanup:
            with worktree as wt:
                assert wt is worktree
            mock_cleanup.assert_called_once()

    def test_context_manager_cleanup_on_exception(self, mock_git_repo):
        """Test cleanup is called even when exception occurs in with block."""
        with patch("papagai.worktree.run_command"):
            worktree = Worktree.from_branch(mock_git_repo, "main")

        with patch.object(worktree, "_cleanup") as mock_cleanup:
            try:
                with worktree:
                    raise ValueError("Test exception")
            except ValueError:
                pass
            mock_cleanup.assert_called_once()


class TestCleanup:
    """Tests for Worktree._cleanup() method."""

    def test_cleanup_removes_clean_worktree(self, mock_worktree):
        """Test cleanup removes worktree with no uncommitted changes."""
        # Create the worktree directory
        mock_worktree.worktree_dir.mkdir(parents=True)

        with patch("papagai.worktree.run_command") as mock_run:
            # Mock git diff to succeed (no changes)
            mock_run.return_value = MagicMock()

            mock_worktree._cleanup()

            # Should call git diff and git worktree remove
            assert mock_run.call_count == 2
            calls = mock_run.call_args_list

            # First call: git diff --quiet --exit-code
            assert calls[0][0][0][0] == "git"
            assert calls[0][0][0][1] == "diff"
            assert "--quiet" in calls[0][0][0]

            # Second call: git worktree remove
            assert calls[1][0][0][0] == "git"
            assert calls[1][0][0][1] == "worktree"
            assert calls[1][0][0][2] == "remove"

    def test_cleanup_refuses_with_uncommitted_changes(self, mock_worktree, capsys):
        """Test cleanup refuses to remove worktree with uncommitted changes."""
        mock_worktree.worktree_dir.mkdir(parents=True)

        with patch("papagai.worktree.run_command") as mock_run:
            # Mock git diff to fail (changes present)
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")

            mock_worktree._cleanup()

            # Should only call git diff once, then return
            assert mock_run.call_count == 1

            # Check warning message
            captured = capsys.readouterr()
            assert "Changes still present in worktree" in captured.out
            assert "refusing to clean up" in captured.out
            assert mock_worktree.branch in captured.out

    def test_cleanup_removes_worktree_directory(self, mock_worktree):
        """Test cleanup removes worktree directory if it exists."""
        # Create worktree directory with a file
        mock_worktree.worktree_dir.mkdir(parents=True)
        test_file = mock_worktree.worktree_dir / "test.txt"
        test_file.write_text("test content")

        with patch("papagai.worktree.run_command") as mock_run:
            mock_run.return_value = MagicMock()

            mock_worktree._cleanup()

            # Directory should be removed
            assert not mock_worktree.worktree_dir.exists()

    def test_cleanup_removes_empty_parent_directories(self, mock_worktree):
        """Test cleanup removes empty parent directories up to repo_dir."""
        # Create nested directory structure
        nested_dir = mock_worktree.repo_dir / "a" / "b" / "c"
        nested_dir.mkdir(parents=True)

        # Update worktree to use nested directory
        mock_worktree.worktree_dir = nested_dir

        with patch("papagai.worktree.run_command") as mock_run:
            mock_run.return_value = MagicMock()

            mock_worktree._cleanup()

            # All empty parent directories should be removed
            assert not (mock_worktree.repo_dir / "a").exists()

    def test_cleanup_preserves_non_empty_parent_directories(self, mock_worktree):
        """Test cleanup preserves parent directories that contain other files."""
        # Create nested directory structure
        parent_dir = mock_worktree.repo_dir / "parent"
        parent_dir.mkdir()

        # Add a file in parent directory
        other_file = parent_dir / "other.txt"
        other_file.write_text("other content")

        # Create worktree dir inside parent
        worktree_dir = parent_dir / "worktree"
        worktree_dir.mkdir()
        mock_worktree.worktree_dir = worktree_dir

        with patch("papagai.worktree.run_command") as mock_run:
            mock_run.return_value = MagicMock()

            mock_worktree._cleanup()

            # Parent directory should still exist (not empty)
            assert parent_dir.exists()
            assert other_file.exists()

    def test_cleanup_handles_exceptions_gracefully(self, mock_worktree, capsys):
        """Test cleanup handles exceptions without crashing."""
        with patch("papagai.worktree.run_command") as mock_run:
            mock_run.side_effect = Exception("Unexpected error")

            # Should not raise, just print warning
            mock_worktree._cleanup()

            captured = capsys.readouterr()
            assert "Warning during cleanup" in captured.err

    @pytest.mark.parametrize("check_value", [True, False])
    def test_cleanup_git_worktree_remove_check_parameter(
        self, mock_worktree, check_value
    ):
        """Test that git worktree remove is called with check=False."""
        mock_worktree.worktree_dir.mkdir(parents=True)

        with patch("papagai.worktree.run_command") as mock_run:
            mock_run.return_value = MagicMock()

            mock_worktree._cleanup()

            # Find the git worktree remove call
            calls = mock_run.call_args_list
            remove_call = [c for c in calls if c[0][0][2] == "remove"][0]

            # check should be False
            assert remove_call[1]["check"] is False


class TestIntegration:
    """Integration tests for Worktree."""

    def test_full_workflow_with_context_manager(self, mock_git_repo):
        """Test complete workflow: create, use, cleanup."""
        with patch("papagai.worktree.run_command") as mock_run:
            mock_run.return_value = MagicMock()

            with Worktree.from_branch(
                mock_git_repo, "main", branch_prefix=f"{BRANCH_PREFIX}/"
            ) as worktree:
                # Verify worktree was created
                assert worktree.branch.startswith(f"{BRANCH_PREFIX}/main")
                assert worktree.repo_dir == mock_git_repo

            # Verify cleanup was called (git diff + git worktree remove)
            assert mock_run.call_count >= 2


class TestWorktreeOverlayFsDataclass:
    """Tests for WorktreeOverlayFs dataclass structure."""

    def test_overlay_fs_initialization(self, mock_git_repo, tmp_path):
        """Test WorktreeOverlayFs can be initialized with all required fields."""
        worktree_dir = tmp_path / "mounted"
        branch = "papagai/test-branch"
        overlay_base_dir = tmp_path / "overlay"
        mount_dir = tmp_path / "mounted"

        overlay_fs = WorktreeOverlayFs(
            worktree_dir=worktree_dir,
            branch=branch,
            repo_dir=mock_git_repo,
            overlay_base_dir=overlay_base_dir,
            mount_dir=mount_dir,
        )

        assert overlay_fs.worktree_dir == worktree_dir
        assert overlay_fs.branch == branch
        assert overlay_fs.repo_dir == mock_git_repo
        assert overlay_fs.overlay_base_dir == overlay_base_dir
        assert overlay_fs.mount_dir == mount_dir

    def test_overlay_fs_inherits_from_worktree(self):
        """Test WorktreeOverlayFs is a subclass of Worktree."""
        assert issubclass(WorktreeOverlayFs, Worktree)

    def test_overlay_fs_optional_fields_default_none(self, mock_git_repo, tmp_path):
        """Test overlay_base_dir and mount_dir default to None."""
        overlay_fs = WorktreeOverlayFs(
            worktree_dir=tmp_path / "test",
            branch="test-branch",
            repo_dir=mock_git_repo,
        )

        assert overlay_fs.overlay_base_dir is None
        assert overlay_fs.mount_dir is None


class TestOverlayFsFromBranch:
    """Tests for WorktreeOverlayFs.from_branch() classmethod."""

    @patch.dict(os.environ, {"XDG_CACHE_HOME": "/tmp/test-cache"})
    def test_from_branch_creates_cache_directory_structure(self, mock_git_repo):
        """Test from_branch creates proper directory structure in cache."""
        with patch("papagai.worktree.run_command") as mock_run:
            mock_run.return_value = MagicMock()

            overlay_fs = WorktreeOverlayFs.from_branch(
                mock_git_repo, "main", branch_prefix=f"{BRANCH_PREFIX}/"
            )

            # Check directory structure was created
            assert overlay_fs.overlay_base_dir.parent.name == "test-repo"
            assert str(overlay_fs.overlay_base_dir).startswith(
                "/tmp/test-cache/papagai/"
            )

    @patch.dict(os.environ, {}, clear=True)
    def test_from_branch_uses_home_cache_when_xdg_not_set(self, mock_git_repo):
        """Test from_branch falls back to ~/.cache when XDG_CACHE_HOME not set."""
        # Remove XDG_CACHE_HOME if it exists
        os.environ.pop("XDG_CACHE_HOME", None)

        with patch("papagai.worktree.run_command") as mock_run:
            mock_run.return_value = MagicMock()

            overlay_fs = WorktreeOverlayFs.from_branch(mock_git_repo, "main")

            # Should use ~/.cache
            expected_prefix = str(Path.home() / ".cache" / "papagai")
            assert str(overlay_fs.overlay_base_dir).startswith(expected_prefix)

    def test_from_branch_creates_overlay_subdirectories(self, mock_git_repo, tmp_path):
        """Test from_branch creates upperdir, workdir, and mounted subdirectories."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            with patch("papagai.worktree.run_command") as mock_run:
                mock_run.return_value = MagicMock()

                overlay_fs = WorktreeOverlayFs.from_branch(mock_git_repo, "main")

                # Check subdirectories were created
                assert (overlay_fs.overlay_base_dir / "upperdir").exists()
                assert (overlay_fs.overlay_base_dir / "workdir").exists()
                assert (overlay_fs.overlay_base_dir / "mounted").exists()

    def test_from_branch_mounts_with_fuse_overlayfs(self, mock_git_repo, tmp_path):
        """Test from_branch calls fuse-overlayfs with correct parameters."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            with patch("papagai.worktree.run_command") as mock_run:
                mock_run.return_value = MagicMock()

                overlay_fs = WorktreeOverlayFs.from_branch(mock_git_repo, "main")

                # Find the fuse-overlayfs call
                fuse_calls = [
                    c for c in mock_run.call_args_list if c[0][0][0] == "fuse-overlayfs"
                ]
                assert len(fuse_calls) == 1

                fuse_cmd = fuse_calls[0][0][0]
                assert fuse_cmd[0] == "fuse-overlayfs"
                assert fuse_cmd[1] == "-o"

                # Check mount options
                mount_opts = fuse_cmd[2]
                assert f"lowerdir={mock_git_repo}" in mount_opts
                assert "upperdir=" in mount_opts
                assert "workdir=" in mount_opts

                # Check mount point
                assert fuse_cmd[3] == str(overlay_fs.mount_dir)

    def test_from_branch_creates_git_branch_in_mount(self, mock_git_repo, tmp_path):
        """Test from_branch creates a git branch in the mounted directory."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            with patch("papagai.worktree.run_command") as mock_run:
                mock_run.return_value = MagicMock()

                overlay_fs = WorktreeOverlayFs.from_branch(
                    mock_git_repo, "develop", branch_prefix=f"{BRANCH_PREFIX}/"
                )

                # Find the git checkout call
                git_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "git"]
                assert len(git_calls) == 1

                git_cmd = git_calls[0][0][0]
                assert git_cmd[0] == "git"
                assert git_cmd[1] == "checkout"
                assert git_cmd[2] == "-fb"
                assert git_cmd[3] == overlay_fs.branch
                assert git_cmd[4] == "develop"

                # Check cwd is the mount directory
                assert git_calls[0][1]["cwd"] == overlay_fs.mount_dir

    def test_from_branch_sets_worktree_dir_to_mounted(self, mock_git_repo, tmp_path):
        """Test from_branch sets worktree_dir to the mounted directory."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            with patch("papagai.worktree.run_command") as mock_run:
                mock_run.return_value = MagicMock()

                overlay_fs = WorktreeOverlayFs.from_branch(mock_git_repo, "main")

                assert overlay_fs.worktree_dir == overlay_fs.mount_dir
                assert overlay_fs.worktree_dir.name == "mounted"

    def test_from_branch_uses_same_naming_scheme_as_worktree(
        self, mock_git_repo, tmp_path
    ):
        """Test from_branch generates branch names using the same scheme as Worktree."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            with patch("papagai.worktree.run_command") as mock_run:
                mock_run.return_value = MagicMock()

                overlay_fs = WorktreeOverlayFs.from_branch(
                    mock_git_repo, "main", branch_prefix=f"{BRANCH_PREFIX}/"
                )

                # Branch should be: papagai/main-YYYY-MM-DD-XXXXXXXX
                parts = overlay_fs.branch.split("/")
                assert len(parts) == 2
                assert parts[0] == BRANCH_PREFIX

                # Second part should be: main-YYYY-MM-DD-uuid
                branch_parts = parts[1].split("-")
                assert branch_parts[0] == "main"
                assert len(branch_parts) >= 4

    def test_from_branch_cleanup_on_mount_failure(self, mock_git_repo, tmp_path):
        """Test from_branch cleans up directories if mount fails."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            with patch("papagai.worktree.run_command") as mock_run:
                # Make fuse-overlayfs fail
                mock_run.side_effect = subprocess.CalledProcessError(
                    1, "fuse-overlayfs"
                )

                with pytest.raises(
                    RuntimeError, match="Failed to mount overlay filesystem"
                ):
                    WorktreeOverlayFs.from_branch(mock_git_repo, "main")

                # Directory should be cleaned up
                papagai_dir = tmp_path / "papagai" / "test-repo"
                if papagai_dir.exists():
                    # If directory exists, it should be empty
                    assert len(list(papagai_dir.iterdir())) == 0

    def test_from_branch_cleanup_on_git_branch_failure(self, mock_git_repo, tmp_path):
        """Test from_branch unmounts and cleans up if git branch creation fails."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            with patch("papagai.worktree.run_command") as mock_run:
                # Make git checkout fail, but fuse-overlayfs succeed
                def run_side_effect(cmd, **kwargs):
                    if cmd[0] == "git":
                        raise subprocess.CalledProcessError(1, "git")
                    return MagicMock()

                mock_run.side_effect = run_side_effect

                with pytest.raises(RuntimeError, match="Failed to create git branch"):
                    WorktreeOverlayFs.from_branch(mock_git_repo, "main")

                # Should have attempted to unmount
                unmount_calls = [
                    c for c in mock_run.call_args_list if c[0][0][0] == "fusermount"
                ]
                assert len(unmount_calls) == 1
                assert unmount_calls[0][0][0][1] == "-u"

    def test_from_branch_creates_unique_branches(self, mock_git_repo, tmp_path):
        """Test from_branch creates unique branch names on each call."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            with patch("papagai.worktree.run_command") as mock_run:
                mock_run.return_value = MagicMock()

                overlay_fs1 = WorktreeOverlayFs.from_branch(mock_git_repo, "main")
                overlay_fs2 = WorktreeOverlayFs.from_branch(mock_git_repo, "main")

                assert overlay_fs1.branch != overlay_fs2.branch
                assert overlay_fs1.overlay_base_dir != overlay_fs2.overlay_base_dir


class TestOverlayFsCleanup:
    """Tests for WorktreeOverlayFs._cleanup() method."""

    def test_cleanup_unmounts_overlay_filesystem(self, mock_git_repo, tmp_path):
        """Test cleanup unmounts the overlay filesystem."""
        overlay_fs = WorktreeOverlayFs(
            worktree_dir=tmp_path / "mounted",
            branch="test-branch",
            repo_dir=mock_git_repo,
            overlay_base_dir=tmp_path / "overlay",
            mount_dir=tmp_path / "mounted",
        )
        overlay_fs.mount_dir.mkdir(parents=True)

        with patch("papagai.worktree.run_command") as mock_run:
            mock_run.return_value = MagicMock()

            overlay_fs._cleanup()

            # Find the fusermount call
            unmount_calls = [
                c for c in mock_run.call_args_list if c[0][0][0] == "fusermount"
            ]
            assert len(unmount_calls) == 1
            assert unmount_calls[0][0][0] == [
                "fusermount",
                "-u",
                str(overlay_fs.mount_dir),
            ]

    def test_cleanup_removes_overlay_base_directory(self, mock_git_repo, tmp_path):
        """Test cleanup removes the entire overlay base directory."""
        overlay_base = tmp_path / "overlay"
        overlay_base.mkdir(parents=True)
        mount_dir = overlay_base / "mounted"
        mount_dir.mkdir()

        # Create some files in the overlay directory
        (overlay_base / "upperdir").mkdir()
        (overlay_base / "workdir").mkdir()
        (overlay_base / "upperdir" / "test.txt").write_text("test")

        overlay_fs = WorktreeOverlayFs(
            worktree_dir=mount_dir,
            branch="test-branch",
            repo_dir=mock_git_repo,
            overlay_base_dir=overlay_base,
            mount_dir=mount_dir,
        )

        with patch("papagai.worktree.run_command") as mock_run:
            mock_run.return_value = MagicMock()

            overlay_fs._cleanup()

            # Directory should be removed
            assert not overlay_base.exists()

    def test_cleanup_refuses_with_uncommitted_changes(
        self, mock_git_repo, tmp_path, capsys
    ):
        """Test cleanup refuses to unmount with uncommitted changes."""
        overlay_base = tmp_path / "overlay"
        mount_dir = overlay_base / "mounted"
        mount_dir.mkdir(parents=True)

        overlay_fs = WorktreeOverlayFs(
            worktree_dir=mount_dir,
            branch="test-branch",
            repo_dir=mock_git_repo,
            overlay_base_dir=overlay_base,
            mount_dir=mount_dir,
        )

        with patch("papagai.worktree.run_command") as mock_run:
            # Make git diff fail (changes present)
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")

            overlay_fs._cleanup()

            # Should only call git diff, not fusermount
            assert mock_run.call_count == 1
            unmount_calls = [
                c for c in mock_run.call_args_list if c[0][0][0] == "fusermount"
            ]
            assert len(unmount_calls) == 0

            # Check warning message
            captured = capsys.readouterr()
            assert "Changes still present in worktree" in captured.out
            assert "fusermount -u" in captured.out

    def test_cleanup_handles_unmount_failure_gracefully(
        self, mock_git_repo, tmp_path, capsys
    ):
        """Test cleanup handles unmount failures gracefully."""
        overlay_base = tmp_path / "overlay"
        mount_dir = overlay_base / "mounted"
        mount_dir.mkdir(parents=True)

        overlay_fs = WorktreeOverlayFs(
            worktree_dir=mount_dir,
            branch="test-branch",
            repo_dir=mock_git_repo,
            overlay_base_dir=overlay_base,
            mount_dir=mount_dir,
        )

        with patch("papagai.worktree.run_command") as mock_run:

            def run_side_effect(cmd, **kwargs):
                if cmd[0] == "fusermount":
                    raise subprocess.CalledProcessError(1, "fusermount")
                return MagicMock()

            mock_run.side_effect = run_side_effect

            overlay_fs._cleanup()

            # Check warning message
            captured = capsys.readouterr()
            assert "Failed to unmount" in captured.err
            assert "manually unmount" in captured.err

    def test_cleanup_handles_exceptions_gracefully(
        self, mock_git_repo, tmp_path, capsys
    ):
        """Test cleanup handles exceptions without crashing."""
        overlay_fs = WorktreeOverlayFs(
            worktree_dir=tmp_path / "mounted",
            branch="test-branch",
            repo_dir=mock_git_repo,
            overlay_base_dir=tmp_path / "overlay",
            mount_dir=tmp_path / "mounted",
        )

        with patch("papagai.worktree.run_command") as mock_run:
            mock_run.side_effect = Exception("Unexpected error")

            # Should not raise, just print warning
            overlay_fs._cleanup()

            captured = capsys.readouterr()
            assert "Warning during cleanup" in captured.err


class TestOverlayFsContextManager:
    """Tests for WorktreeOverlayFs context manager functionality."""

    def test_context_manager_calls_cleanup_on_exit(self, mock_git_repo, tmp_path):
        """Test context manager calls cleanup on exit."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            with patch("papagai.worktree.run_command") as mock_run:
                mock_run.return_value = MagicMock()

                overlay_fs = WorktreeOverlayFs.from_branch(mock_git_repo, "main")

            with patch.object(overlay_fs, "_cleanup") as mock_cleanup:
                with overlay_fs as wt:
                    assert wt is overlay_fs
                mock_cleanup.assert_called_once()

    def test_context_manager_cleanup_on_exception(self, mock_git_repo, tmp_path):
        """Test cleanup is called even when exception occurs in with block."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            with patch("papagai.worktree.run_command") as mock_run:
                mock_run.return_value = MagicMock()

                overlay_fs = WorktreeOverlayFs.from_branch(mock_git_repo, "main")

        with patch.object(overlay_fs, "_cleanup") as mock_cleanup:
            try:
                with overlay_fs:
                    raise ValueError("Test exception")
            except ValueError:
                pass
            mock_cleanup.assert_called_once()


class TestOverlayFsIntegration:
    """Integration tests for WorktreeOverlayFs."""

    def test_full_workflow_with_context_manager(self, mock_git_repo, tmp_path):
        """Test complete workflow: create, use, cleanup."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            with patch("papagai.worktree.run_command") as mock_run:
                mock_run.return_value = MagicMock()

                with WorktreeOverlayFs.from_branch(
                    mock_git_repo, "main", branch_prefix=f"{BRANCH_PREFIX}/"
                ) as overlay_fs:
                    # Verify overlay was created
                    assert overlay_fs.branch.startswith(f"{BRANCH_PREFIX}/main")
                    assert overlay_fs.repo_dir == mock_git_repo
                    assert overlay_fs.worktree_dir == overlay_fs.mount_dir
                    assert overlay_fs.overlay_base_dir is not None

                # Verify mount and unmount were called
                mount_calls = [
                    c for c in mock_run.call_args_list if c[0][0][0] == "fuse-overlayfs"
                ]
                unmount_calls = [
                    c for c in mock_run.call_args_list if c[0][0][0] == "fusermount"
                ]
                assert len(mount_calls) == 1
                assert len(unmount_calls) == 1
