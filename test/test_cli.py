#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for CLI utility functions."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from click.testing import CliRunner

from claude_do.cli import (
    get_branch,
    purge_branches,
    claude_do,
    BRANCH_PREFIX,
)


@pytest.fixture
def mock_repo(tmp_path):
    """Create a mock git repository directory."""
    repo_dir = tmp_path / "test-repo"
    repo_dir.mkdir()
    return repo_dir


class TestGetBranch:
    """Tests for get_branch() function."""

    def test_get_branch_default_head(self, mock_repo):
        """Test get_branch returns branch name for HEAD."""
        with patch("claude_do.cli.run_command") as mock_run:
            mock_run.return_value = MagicMock(stdout="main\n")

            branch = get_branch(mock_repo)

            assert branch == "main"
            mock_run.assert_called_once_with(
                ["git", "rev-parse", "--abbrev-ref", "--verify", "HEAD"],
                cwd=mock_repo,
            )

    @pytest.mark.parametrize(
        "ref,expected_branch",
        [
            ("HEAD", "main"),
            ("main", "main"),
            ("develop", "develop"),
            ("feature/test", "feature/test"),
            ("v1.0.0", "v1.0.0"),
        ],
    )
    def test_get_branch_with_different_refs(self, mock_repo, ref, expected_branch):
        """Test get_branch with various ref types."""
        with patch("claude_do.cli.run_command") as mock_run:
            mock_run.return_value = MagicMock(stdout=f"{expected_branch}\n")

            branch = get_branch(mock_repo, ref)

            assert branch == expected_branch
            mock_run.assert_called_once_with(
                ["git", "rev-parse", "--abbrev-ref", "--verify", ref],
                cwd=mock_repo,
            )

    def test_get_branch_strips_whitespace(self, mock_repo):
        """Test get_branch strips leading/trailing whitespace."""
        with patch("claude_do.cli.run_command") as mock_run:
            mock_run.return_value = MagicMock(stdout="  main  \n\n")

            branch = get_branch(mock_repo)

            assert branch == "main"

    def test_get_branch_raises_on_invalid_ref(self, mock_repo):
        """Test get_branch raises CalledProcessError for invalid ref."""
        with patch("claude_do.cli.run_command") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")

            with pytest.raises(subprocess.CalledProcessError):
                get_branch(mock_repo, "nonexistent-branch")

    def test_get_branch_raises_on_non_git_repo(self, mock_repo):
        """Test get_branch raises CalledProcessError for non-git directory."""
        with patch("claude_do.cli.run_command") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(128, "git")

            with pytest.raises(subprocess.CalledProcessError):
                get_branch(mock_repo)

    def test_get_branch_uses_correct_cwd(self, mock_repo):
        """Test get_branch uses the provided repo_dir as cwd."""
        with patch("claude_do.cli.run_command") as mock_run:
            mock_run.return_value = MagicMock(stdout="main\n")

            get_branch(mock_repo)

            call_args = mock_run.call_args
            assert call_args[1]["cwd"] == mock_repo


class TestPurgeDoneBranches:
    """Tests for purge_branches() function."""

    def test_purge_no_branches(self, mock_repo, capsys):
        """Test purge when no claude-do branches exist."""
        with patch("claude_do.cli.run_command") as mock_run:
            # Mock git branch list returning empty
            mock_run.return_value = MagicMock(stdout="\n")

            purge_branches(mock_repo)

            # Should only call git branch list, not git branch -D
            assert mock_run.call_count == 1
            call_args = mock_run.call_args_list[0]
            assert call_args[0][0][0] == "git"
            assert call_args[0][0][1] == "branch"
            assert f"{BRANCH_PREFIX}/*" in call_args[0][0]

            # No output expected
            captured = capsys.readouterr()
            assert "Deleting branch:" not in captured.out

    def test_purge_single_branch(self, mock_repo, capsys):
        """Test purge with one claude-do branch."""
        with patch("claude_do.cli.run_command") as mock_run:
            branch_name = f"{BRANCH_PREFIX}/main-2025-01-01-abc123"
            mock_run.return_value = MagicMock(stdout=f"{branch_name}\n")

            purge_branches(mock_repo)

            # Should call git branch list, then git branch -D
            assert mock_run.call_count == 2

            # First call: list branches
            list_call = mock_run.call_args_list[0]
            assert list_call[0][0][0] == "git"
            assert list_call[0][0][1] == "branch"
            assert "--list" in list_call[0][0]

            # Second call: delete branch
            delete_call = mock_run.call_args_list[1]
            assert delete_call[0][0] == ["git", "branch", "-D", branch_name]
            assert delete_call[1]["cwd"] == mock_repo

            # Check output message
            captured = capsys.readouterr()
            assert f"Deleting branch: {branch_name}" in captured.out

    def test_purge_multiple_branches(self, mock_repo, capsys):
        """Test purge with multiple claude-do branches."""
        with patch("claude_do.cli.run_command") as mock_run:
            branches = [
                f"{BRANCH_PREFIX}/main-2025-01-01-abc123",
                f"{BRANCH_PREFIX}/develop-2025-01-02-def456",
                f"{BRANCH_PREFIX}/feature-2025-01-03-ghi789",
            ]
            mock_run.return_value = MagicMock(stdout="\n".join(branches) + "\n")

            purge_branches(mock_repo)

            # Should call git branch list once, then git branch -D for each branch
            assert mock_run.call_count == 4

            # Verify each branch was deleted
            delete_calls = mock_run.call_args_list[1:]
            for i, branch in enumerate(branches):
                assert delete_calls[i][0][0] == ["git", "branch", "-D", branch]

            # Check output messages
            captured = capsys.readouterr()
            for branch in branches:
                assert f"Deleting branch: {branch}" in captured.out

    def test_purge_skips_empty_lines(self, mock_repo):
        """Test purge skips empty lines in git output."""
        with patch("claude_do.cli.run_command") as mock_run:
            branch_name = f"{BRANCH_PREFIX}/main-2025-01-01-abc123"
            # Output with empty lines
            mock_run.return_value = MagicMock(stdout=f"\n{branch_name}\n\n")

            purge_branches(mock_repo)

            # Should only delete the one non-empty branch
            assert mock_run.call_count == 2
            delete_call = mock_run.call_args_list[1]
            assert delete_call[0][0] == ["git", "branch", "-D", branch_name]

    def test_purge_uses_correct_branch_prefix(self, mock_repo):
        """Test purge uses the correct BRANCH_PREFIX."""
        with patch("claude_do.cli.run_command") as mock_run:
            mock_run.return_value = MagicMock(stdout="\n")

            purge_branches(mock_repo)

            # Verify the branch list command uses BRANCH_PREFIX
            call_args = mock_run.call_args_list[0]
            git_cmd = call_args[0][0]
            assert f"{BRANCH_PREFIX}/*" in git_cmd

    def test_purge_git_command_format(self, mock_repo):
        """Test purge calls git with correct command format."""
        with patch("claude_do.cli.run_command") as mock_run:
            mock_run.return_value = MagicMock(stdout="\n")

            purge_branches(mock_repo)

            # Verify git branch command format
            call_args = mock_run.call_args_list[0]
            git_cmd = call_args[0][0]
            assert git_cmd[0] == "git"
            assert git_cmd[1] == "branch"
            assert "--format=%(refname:short)" in git_cmd
            assert "--list" in git_cmd

    def test_purge_uses_correct_cwd(self, mock_repo):
        """Test purge uses the provided repo_dir as cwd."""
        with patch("claude_do.cli.run_command") as mock_run:
            branch_name = f"{BRANCH_PREFIX}/main-2025-01-01-abc123"
            mock_run.return_value = MagicMock(stdout=f"{branch_name}\n")

            purge_branches(mock_repo)

            # All calls should use the repo_dir as cwd
            for call_args in mock_run.call_args_list:
                assert call_args[1]["cwd"] == mock_repo

    def test_purge_handles_branch_with_slashes(self, mock_repo, capsys):
        """Test purge handles branches with slashes in name."""
        with patch("claude_do.cli.run_command") as mock_run:
            branch_name = f"{BRANCH_PREFIX}/feature/test-2025-01-01-abc123"
            mock_run.return_value = MagicMock(stdout=f"{branch_name}\n")

            purge_branches(mock_repo)

            # Should delete the branch
            assert mock_run.call_count == 2
            delete_call = mock_run.call_args_list[1]
            assert delete_call[0][0] == ["git", "branch", "-D", branch_name]

            captured = capsys.readouterr()
            assert f"Deleting branch: {branch_name}" in captured.out


class TestIntegration:
    """Integration tests for CLI functions."""

    def test_get_branch_and_purge_workflow(self, mock_repo):
        """Test workflow of getting current branch and purging old branches."""
        with patch("claude_do.cli.run_command") as mock_run:
            # First call: get_branch
            # Second call: purge list branches
            # Third call: purge delete branch
            branch_to_delete = f"{BRANCH_PREFIX}/main-2025-01-01-abc123"
            mock_run.side_effect = [
                MagicMock(stdout="main\n"),  # get_branch
                MagicMock(stdout=f"{branch_to_delete}\n"),  # purge list
                MagicMock(stdout=""),  # purge delete
            ]

            # Get current branch
            current = get_branch(mock_repo)
            assert current == "main"

            # Purge old branches
            purge_branches(mock_repo)

            # Verify all commands were called
            assert mock_run.call_count == 3

    def test_functions_work_with_different_repo_paths(self, tmp_path):
        """Test functions work correctly with different repository paths."""
        repo1 = tmp_path / "repo1"
        repo2 = tmp_path / "repo2"
        repo1.mkdir()
        repo2.mkdir()

        with patch("claude_do.cli.run_command") as mock_run:
            mock_run.return_value = MagicMock(stdout="main\n")

            # Test with first repo
            branch1 = get_branch(repo1)
            assert branch1 == "main"
            assert mock_run.call_args[1]["cwd"] == repo1

            # Test with second repo
            branch2 = get_branch(repo2)
            assert branch2 == "main"
            assert mock_run.call_args[1]["cwd"] == repo2


class TestCLICommands:
    """Tests for CLI commands using CliRunner."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def mock_instructions_file(self, tmp_path):
        """Create a temporary instructions file."""
        instructions = tmp_path / "instructions.md"
        instructions.write_text(
            """---
description: Test task
tools: Bash(test:*)
---

Do something interesting.
"""
        )
        return instructions

    def test_main_help(self, runner):
        """Test main command --help."""
        result = runner.invoke(claude_do, ["--help"])
        assert result.exit_code == 0
        assert "Claude-do: Automate code changes with Claude AI" in result.output
        assert "do" in result.output
        assert "purge" in result.output
        assert "task" in result.output
        assert "review" in result.output

    def test_main_dry_run_flag(self, runner):
        """Test --dry-run flag is recognized."""
        result = runner.invoke(claude_do, ["--dry-run", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output


class TestDoCommand:
    """Tests for the 'do' command."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def mock_instructions_file(self, tmp_path):
        """Create a temporary instructions file."""
        instructions = tmp_path / "instructions.md"
        instructions.write_text(
            """---
description: Test task
tools: Bash(test:*)
---

Do something interesting.
"""
        )
        return instructions

    def test_do_help(self, runner):
        """Test 'do' command --help."""
        result = runner.invoke(claude_do, ["do", "--help"])
        assert result.exit_code == 0
        assert "Tell Claude to do something on a work tree" in result.output
        assert "--base-branch" in result.output
        assert "--instructions" in result.output

    def test_do_with_instructions_file(self, runner, mock_instructions_file):
        """Test 'do' command with instructions file."""
        with patch("claude_do.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(
                claude_do, ["do", "--instructions", str(mock_instructions_file)]
            )

            # Should call claude_do with the instructions
            mock_claude_run.assert_called_once()
            assert result.exit_code == 0

    def test_do_with_nonexistent_instructions_file(self, runner, tmp_path):
        """Test 'do' command with non-existent instructions file."""
        nonexistent = tmp_path / "nonexistent.md"

        result = runner.invoke(claude_do, ["do", "--instructions", str(nonexistent)])

        # Click returns exit code 2 for validation errors
        assert result.exit_code == 2
        assert "does not exist" in result.output

    def test_do_with_base_branch(self, runner, mock_instructions_file):
        """Test 'do' command with custom base branch."""
        with patch("claude_do.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(
                claude_do,
                [
                    "do",
                    "--base-branch",
                    "develop",
                    "--instructions",
                    str(mock_instructions_file),
                ],
            )

            # Should call claude_do with develop as base_branch
            mock_claude_run.assert_called_once()
            call_kwargs = mock_claude_run.call_args
            assert call_kwargs[1]["base_branch"] == "develop"
            assert result.exit_code == 0

    def test_do_with_stdin_input(self, runner):
        """Test 'do' command with stdin input."""
        with patch("claude_do.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(
                claude_do, ["do"], input="Fix all the bugs\n", catch_exceptions=False
            )

            # Should call claude_do with stdin instructions
            mock_claude_run.assert_called_once()
            assert result.exit_code == 0

    def test_do_with_empty_stdin(self, runner):
        """Test 'do' command with empty stdin."""
        result = runner.invoke(claude_do, ["do"], input="")

        # Command shows error message
        assert "Empty instructions" in result.output

    def test_do_with_dry_run(self, runner, mock_instructions_file):
        """Test 'do' command with --dry-run flag."""
        with patch("claude_do.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(
                claude_do,
                ["--dry-run", "do", "--instructions", str(mock_instructions_file)],
            )

            # Should call claude_do with dry_run=True
            mock_claude_run.assert_called_once()
            call_kwargs = mock_claude_run.call_args
            assert call_kwargs[1]["dry_run"] is True
            assert result.exit_code == 0


class TestPurgeCommand:
    """Tests for the 'purge' command."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner instance."""
        return CliRunner()

    def test_purge_help(self, runner):
        """Test 'purge' command --help."""
        result = runner.invoke(claude_do, ["purge", "--help"])
        assert result.exit_code == 0
        assert "Delete all existing claude-do branches" in result.output

    def test_purge_success(self, runner):
        """Test 'purge' command succeeds."""
        with patch("claude_do.cli.purge_branches") as mock_purge:
            result = runner.invoke(claude_do, ["purge"])

            mock_purge.assert_called_once()
            assert result.exit_code == 0

    def test_purge_with_git_error(self, runner):
        """Test 'purge' command handles git errors."""
        with patch("claude_do.cli.purge_branches") as mock_purge:
            mock_purge.side_effect = subprocess.CalledProcessError(1, "git")

            result = runner.invoke(claude_do, ["purge"])

            # Command catches exception and shows error message
            assert "Error purging done branches" in result.output


class TestTaskCommand:
    """Tests for the 'task' command."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner instance."""
        return CliRunner()

    def test_task_help(self, runner):
        """Test 'task' command --help."""
        result = runner.invoke(claude_do, ["task", "--help"])
        assert result.exit_code == 0
        assert "Run a pre-written task" in result.output
        assert "--list" in result.output
        assert "--base-branch" in result.output

    def test_task_list(self, runner):
        """Test 'task --list' shows available tasks."""
        result = runner.invoke(claude_do, ["task", "--list"])
        assert result.exit_code == 0
        # Should show at least some tasks
        assert len(result.output) > 0

    def test_task_without_args(self, runner):
        """Test 'task' without arguments shows error."""
        result = runner.invoke(claude_do, ["task"])
        # Command shows error message
        assert "Error: missing task name" in result.output

    def test_task_with_valid_task(self, runner):
        """Test 'task' with a valid task name."""
        with patch("claude_do.cli.claude_run") as mock_claude_run:
            with patch("claude_do.cli.get_builtin_tasks_dir") as mock_get_dir:
                # Create a mock instructions directory
                mock_dir = MagicMock()
                mock_get_dir.return_value = mock_dir

                # Mock the task file
                mock_task_file = MagicMock()
                mock_task_file.exists.return_value = True
                mock_dir.__truediv__.return_value = mock_task_file

                with patch(
                    "claude_do.cli.MarkdownInstructions.from_file"
                ) as mock_from_file:
                    mock_instructions = MagicMock()
                    mock_from_file.return_value = mock_instructions
                    mock_claude_run.return_value = 0

                    result = runner.invoke(
                        claude_do, ["task", "generic/review"], catch_exceptions=False
                    )

                    mock_claude_run.assert_called_once()
                    assert result.exit_code == 0

    def test_task_with_nonexistent_task(self, runner):
        """Test 'task' with non-existent task."""
        with patch("claude_do.cli.get_builtin_tasks_dir") as mock_get_dir:
            # Create a mock instructions directory
            mock_dir = MagicMock()
            mock_get_dir.return_value = mock_dir

            # Mock the task file as non-existent
            mock_task_file = MagicMock()
            mock_task_file.exists.return_value = False
            mock_dir.__truediv__.return_value = mock_task_file

            result = runner.invoke(claude_do, ["task", "nonexistent/task"])

            # Command shows error message
            assert "Task 'nonexistent/task' not found" in result.output

    def test_task_with_base_branch(self, runner):
        """Test 'task' with custom base branch."""
        with patch("claude_do.cli.claude_run") as mock_claude_run:
            with patch("claude_do.cli.get_builtin_tasks_dir") as mock_get_dir:
                # Create a mock instructions directory
                mock_dir = MagicMock()
                mock_get_dir.return_value = mock_dir

                # Mock the task file
                mock_task_file = MagicMock()
                mock_task_file.exists.return_value = True
                mock_dir.__truediv__.return_value = mock_task_file

                with patch(
                    "claude_do.cli.MarkdownInstructions.from_file"
                ) as mock_from_file:
                    mock_instructions = MagicMock()
                    mock_from_file.return_value = mock_instructions
                    mock_claude_run.return_value = 0

                    result = runner.invoke(
                        claude_do,
                        ["task", "--base-branch", "develop", "generic/review"],
                    )

                    # Should call claude_do with develop as base_branch
                    mock_claude_run.assert_called_once()
                    call_kwargs = mock_claude_run.call_args
                    assert call_kwargs[1]["base_branch"] == "develop"
                    assert result.exit_code == 0

    def test_task_with_dry_run(self, runner):
        """Test 'task' with --dry-run flag."""
        with patch("claude_do.cli.claude_run") as mock_claude_run:
            with patch("claude_do.cli.get_builtin_tasks_dir") as mock_get_dir:
                # Create a mock instructions directory
                mock_dir = MagicMock()
                mock_get_dir.return_value = mock_dir

                # Mock the task file
                mock_task_file = MagicMock()
                mock_task_file.exists.return_value = True
                mock_dir.__truediv__.return_value = mock_task_file

                with patch(
                    "claude_do.cli.MarkdownInstructions.from_file"
                ) as mock_from_file:
                    mock_instructions = MagicMock()
                    mock_from_file.return_value = mock_instructions
                    mock_claude_run.return_value = 0

                    result = runner.invoke(
                        claude_do, ["--dry-run", "task", "generic/review"]
                    )

                    # Should call claude_do with dry_run=True
                    mock_claude_run.assert_called_once()
                    call_kwargs = mock_claude_run.call_args
                    assert call_kwargs[1]["dry_run"] is True
                    assert result.exit_code == 0


class TestReviewCommand:
    """Tests for the 'review' command."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner instance."""
        return CliRunner()

    def test_review_help(self, runner):
        """Test 'review' command --help."""
        result = runner.invoke(claude_do, ["review", "--help"])
        assert result.exit_code == 0
        assert "Run a code review on the current branch" in result.output
        assert "generic/review" in result.output
        assert "--base-branch" in result.output

    def test_review_success(self, runner):
        """Test 'review' command succeeds."""
        with patch("claude_do.cli.claude_run") as mock_claude_run:
            with patch("claude_do.cli.get_builtin_tasks_dir") as mock_get_dir:
                # Create a mock instructions directory
                mock_dir = MagicMock()
                mock_get_dir.return_value = mock_dir

                # Mock the review task file
                mock_task_file = MagicMock()
                mock_task_file.exists.return_value = True
                mock_dir.__truediv__.return_value = mock_task_file

                with patch(
                    "claude_do.cli.MarkdownInstructions.from_file"
                ) as mock_from_file:
                    mock_instructions = MagicMock()
                    mock_from_file.return_value = mock_instructions
                    mock_claude_run.return_value = 0

                    result = runner.invoke(claude_do, ["review"])

                    mock_claude_run.assert_called_once()
                    assert result.exit_code == 0

    def test_review_with_base_branch(self, runner):
        """Test 'review' command with custom base branch."""
        with patch("claude_do.cli.claude_run") as mock_claude_run:
            with patch("claude_do.cli.get_builtin_tasks_dir") as mock_get_dir:
                # Create a mock instructions directory
                mock_dir = MagicMock()
                mock_get_dir.return_value = mock_dir

                # Mock the review task file
                mock_task_file = MagicMock()
                mock_task_file.exists.return_value = True
                mock_dir.__truediv__.return_value = mock_task_file

                with patch(
                    "claude_do.cli.MarkdownInstructions.from_file"
                ) as mock_from_file:
                    mock_instructions = MagicMock()
                    mock_from_file.return_value = mock_instructions
                    mock_claude_run.return_value = 0

                    result = runner.invoke(
                        claude_do, ["review", "--base-branch", "develop"]
                    )

                    # Should call claude_do with develop as base_branch
                    mock_claude_run.assert_called_once()
                    call_kwargs = mock_claude_run.call_args
                    assert call_kwargs[1]["base_branch"] == "develop"
                    assert result.exit_code == 0

    def test_review_with_dry_run(self, runner):
        """Test 'review' command with --dry-run flag."""
        with patch("claude_do.cli.claude_run") as mock_claude_run:
            with patch("claude_do.cli.get_builtin_tasks_dir") as mock_get_dir:
                # Create a mock instructions directory
                mock_dir = MagicMock()
                mock_get_dir.return_value = mock_dir

                # Mock the review task file
                mock_task_file = MagicMock()
                mock_task_file.exists.return_value = True
                mock_dir.__truediv__.return_value = mock_task_file

                with patch(
                    "claude_do.cli.MarkdownInstructions.from_file"
                ) as mock_from_file:
                    mock_instructions = MagicMock()
                    mock_from_file.return_value = mock_instructions
                    mock_claude_run.return_value = 0

                    result = runner.invoke(claude_do, ["--dry-run", "review"])

                    # Should call claude_do with dry_run=True
                    mock_claude_run.assert_called_once()
                    call_kwargs = mock_claude_run.call_args
                    assert call_kwargs[1]["dry_run"] is True
                    assert result.exit_code == 0

    def test_review_missing_task_file(self, runner, tmp_path):
        """Test 'review' command when review.md doesn't exist."""
        # Create a fake instructions directory without the review file
        fake_instructions_dir = tmp_path / "instructions"
        fake_instructions_dir.mkdir()
        generic_dir = fake_instructions_dir / "generic"
        generic_dir.mkdir()
        # Note: NOT creating review.md file

        with patch("claude_do.cli.get_builtin_tasks_dir") as mock_get_dir:
            mock_get_dir.return_value = fake_instructions_dir

            result = runner.invoke(claude_do, ["review"])

            # Command shows error message
            assert "Review task not found" in result.output

    def test_review_equivalent_to_task(self, runner):
        """Test 'review' calls same code as 'task generic/review'."""
        with patch("claude_do.cli.claude_run") as mock_claude_run:
            with patch("claude_do.cli.get_builtin_tasks_dir") as mock_get_dir:
                # Create a mock instructions directory
                mock_dir = MagicMock()
                mock_get_dir.return_value = mock_dir

                # Mock the review task file
                mock_task_file = MagicMock()
                mock_task_file.exists.return_value = True
                mock_dir.__truediv__.return_value = mock_task_file

                with patch(
                    "claude_do.cli.MarkdownInstructions.from_file"
                ) as mock_from_file:
                    mock_instructions = MagicMock()
                    mock_from_file.return_value = mock_instructions
                    mock_claude_run.return_value = 0

                    # Run review command
                    result1 = runner.invoke(
                        claude_do, ["review", "--base-branch", "main"]
                    )
                    review_call = mock_claude_run.call_args

                    # Reset mock
                    mock_claude_run.reset_mock()

                    # Run task command
                    result2 = runner.invoke(
                        claude_do, ["task", "--base-branch", "main", "generic/review"]
                    )
                    task_call = mock_claude_run.call_args

                    # Both should call claude_do with same arguments
                    assert result1.exit_code == 0
                    assert result2.exit_code == 0
                    assert review_call[1]["base_branch"] == task_call[1]["base_branch"]
                    assert review_call[1]["dry_run"] == task_call[1]["dry_run"]
