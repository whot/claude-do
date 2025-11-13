#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for CLI utility functions."""

import logging
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from papagai.cli import (
    get_branch,
    purge_branches,
    papagai,
    BRANCH_PREFIX,
)

logger = logging.getLogger("papagai.test")


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
        with patch("papagai.cli.run_command") as mock_run:
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
        with patch("papagai.cli.run_command") as mock_run:
            mock_run.return_value = MagicMock(stdout=f"{expected_branch}\n")

            branch = get_branch(mock_repo, ref)

            assert branch == expected_branch
            mock_run.assert_called_once_with(
                ["git", "rev-parse", "--abbrev-ref", "--verify", ref],
                cwd=mock_repo,
            )

    def test_get_branch_strips_whitespace(self, mock_repo):
        """Test get_branch strips leading/trailing whitespace."""
        with patch("papagai.cli.run_command") as mock_run:
            mock_run.return_value = MagicMock(stdout="  main  \n\n")

            branch = get_branch(mock_repo)

            assert branch == "main"

    def test_get_branch_raises_on_invalid_ref(self, mock_repo):
        """Test get_branch raises CalledProcessError for invalid ref."""
        with patch("papagai.cli.run_command") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")

            with pytest.raises(subprocess.CalledProcessError):
                get_branch(mock_repo, "nonexistent-branch")

    def test_get_branch_raises_on_non_git_repo(self, mock_repo):
        """Test get_branch raises CalledProcessError for non-git directory."""
        with patch("papagai.cli.run_command") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(128, "git")

            with pytest.raises(subprocess.CalledProcessError):
                get_branch(mock_repo)

    def test_get_branch_uses_correct_cwd(self, mock_repo):
        """Test get_branch uses the provided repo_dir as cwd."""
        with patch("papagai.cli.run_command") as mock_run:
            mock_run.return_value = MagicMock(stdout="main\n")

            get_branch(mock_repo)

            call_args = mock_run.call_args
            assert call_args[1]["cwd"] == mock_repo


class TestPurgeDoneBranches:
    """Tests for purge_branches() function."""

    def test_purge_no_branches(self, mock_repo, capsys):
        """Test purge when no papagai branches exist."""
        with patch("papagai.cli.run_command") as mock_run:
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
        """Test purge with one papagai branch."""
        with patch("papagai.cli.run_command") as mock_run:
            branch_name = f"{BRANCH_PREFIX}/main-20250101-1200-abc123"
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
        """Test purge with multiple papagai branches."""
        with patch("papagai.cli.run_command") as mock_run:
            branches = [
                f"{BRANCH_PREFIX}/main-20250101-1200-abc123",
                f"{BRANCH_PREFIX}/develop-20250102-1300-def456",
                f"{BRANCH_PREFIX}/feature-20250103-1400-ghi789",
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
        with patch("papagai.cli.run_command") as mock_run:
            branch_name = f"{BRANCH_PREFIX}/main-20250101-1200-abc123"
            # Output with empty lines
            mock_run.return_value = MagicMock(stdout=f"\n{branch_name}\n\n")

            purge_branches(mock_repo)

            # Should only delete the one non-empty branch
            assert mock_run.call_count == 2
            delete_call = mock_run.call_args_list[1]
            assert delete_call[0][0] == ["git", "branch", "-D", branch_name]

    def test_purge_uses_correct_branch_prefix(self, mock_repo):
        """Test purge uses the correct BRANCH_PREFIX."""
        with patch("papagai.cli.run_command") as mock_run:
            mock_run.return_value = MagicMock(stdout="\n")

            purge_branches(mock_repo)

            # Verify the branch list command uses BRANCH_PREFIX
            call_args = mock_run.call_args_list[0]
            git_cmd = call_args[0][0]
            assert f"{BRANCH_PREFIX}/*" in git_cmd

    def test_purge_git_command_format(self, mock_repo):
        """Test purge calls git with correct command format."""
        with patch("papagai.cli.run_command") as mock_run:
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
        with patch("papagai.cli.run_command") as mock_run:
            branch_name = f"{BRANCH_PREFIX}/main-20250101-1200-abc123"
            mock_run.return_value = MagicMock(stdout=f"{branch_name}\n")

            purge_branches(mock_repo)

            # All calls should use the repo_dir as cwd
            for call_args in mock_run.call_args_list:
                assert call_args[1]["cwd"] == mock_repo

    def test_purge_handles_branch_with_slashes(self, mock_repo, capsys):
        """Test purge handles branches with slashes in name."""
        with patch("papagai.cli.run_command") as mock_run:
            branch_name = f"{BRANCH_PREFIX}/feature/test-20250101-1200-abc123"
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
        with patch("papagai.cli.run_command") as mock_run:
            # First call: get_branch
            # Second call: purge list branches
            # Third call: purge delete branch
            branch_to_delete = f"{BRANCH_PREFIX}/main-20250101-1200-abc123"
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

        with patch("papagai.cli.run_command") as mock_run:
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
        result = runner.invoke(papagai, ["--help"])
        assert result.exit_code == 0
        assert "Papagai: Automate code changes with Claude AI" in result.output
        assert "do" in result.output
        assert "purge" in result.output
        assert "task" in result.output
        assert "review" in result.output

    def test_main_dry_run_flag(self, runner):
        """Test --dry-run flag is recognized."""
        result = runner.invoke(papagai, ["--dry-run", "--help"])
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
        result = runner.invoke(papagai, ["do", "--help"])
        assert result.exit_code == 0
        assert (
            "Tell Claude to do something non-code related on a work tree"
            in result.output
        )
        assert "--base-branch" in result.output
        assert "INSTRUCTIONS_FILE" in result.output

    def test_do_with_instructions_file(self, runner, mock_instructions_file):
        """Test 'do' command with instructions file."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(papagai, ["do", str(mock_instructions_file)])

            # Should call claude_run with the instructions
            mock_claude_run.assert_called_once()
            assert result.exit_code == 0

    def test_do_with_nonexistent_instructions_file(self, runner, tmp_path):
        """Test 'do' command with non-existent instructions file."""
        nonexistent = tmp_path / "nonexistent.md"

        result = runner.invoke(papagai, ["do", str(nonexistent)])

        # Click returns exit code 2 for validation errors
        assert result.exit_code == 2
        assert "does not exist" in result.output

    def test_do_with_base_branch(self, runner, mock_instructions_file):
        """Test 'do' command with custom base branch."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(
                papagai,
                [
                    "do",
                    "--base-branch",
                    "develop",
                    str(mock_instructions_file),
                ],
            )

            # Should call claude_run with develop as base_branch
            mock_claude_run.assert_called_once()
            call_kwargs = mock_claude_run.call_args
            assert call_kwargs[1]["base_branch"] == "develop"
            assert result.exit_code == 0

    def test_do_with_stdin_input(self, runner):
        """Test 'do' command with stdin input."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(
                papagai, ["do"], input="Fix all the bugs\n", catch_exceptions=False
            )

            # Should call claude_run with stdin instructions
            mock_claude_run.assert_called_once()
            assert result.exit_code == 0

    def test_do_with_empty_stdin(self, runner):
        """Test 'do' command with empty stdin."""
        result = runner.invoke(papagai, ["do"], input="")

        # Command shows error message
        assert "Empty instructions" in result.output

    def test_do_with_dry_run(self, runner, mock_instructions_file):
        """Test 'do' command with --dry-run flag."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(
                papagai,
                ["--dry-run", "do", str(mock_instructions_file)],
            )

            # Should call claude_run with dry_run=True
            mock_claude_run.assert_called_once()
            call_kwargs = mock_claude_run.call_args
            assert call_kwargs[1]["dry_run"] is True
            assert result.exit_code == 0


class TestCodeCommand:
    """Tests for the 'code' command."""

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

    def test_code_help(self, runner):
        """Test 'code' command --help."""
        result = runner.invoke(papagai, ["code", "--help"])
        assert result.exit_code == 0
        assert "Tell Claude to code something on a work tree" in result.output
        assert "--base-branch" in result.output
        assert "INSTRUCTIONS_FILE" in result.output

    def test_code_with_instructions_file(self, runner, mock_instructions_file):
        """Test 'code' command with instructions file."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(papagai, ["code", str(mock_instructions_file)])

            # Should call claude_run with the instructions
            mock_claude_run.assert_called_once()
            assert result.exit_code == 0

    def test_code_with_nonexistent_instructions_file(self, runner, tmp_path):
        """Test 'code' command with non-existent instructions file."""
        nonexistent = tmp_path / "nonexistent.md"

        result = runner.invoke(papagai, ["code", str(nonexistent)])

        # Click returns exit code 2 for validation errors
        assert result.exit_code == 2
        assert "does not exist" in result.output

    def test_code_with_base_branch(self, runner, mock_instructions_file):
        """Test 'code' command with custom base branch."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(
                papagai,
                [
                    "code",
                    "--base-branch",
                    "develop",
                    str(mock_instructions_file),
                ],
            )

            # Should call claude_run with develop as base_branch
            mock_claude_run.assert_called_once()
            call_kwargs = mock_claude_run.call_args
            assert call_kwargs[1]["base_branch"] == "develop"
            assert result.exit_code == 0

    def test_code_with_stdin_input(self, runner):
        """Test 'code' command with stdin input."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(
                papagai, ["code"], input="Fix all the bugs\n", catch_exceptions=False
            )

            # Should call claude_run with stdin instructions
            mock_claude_run.assert_called_once()
            assert result.exit_code == 0

    def test_code_with_empty_stdin(self, runner):
        """Test 'code' command with empty stdin."""
        result = runner.invoke(papagai, ["code"], input="")

        # Command shows error message
        assert "Empty instructions" in result.output

    def test_code_with_dry_run(self, runner, mock_instructions_file):
        """Test 'code' command with --dry-run flag."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(
                papagai,
                ["--dry-run", "code", str(mock_instructions_file)],
            )

            # Should call claude_run with dry_run=True
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
        result = runner.invoke(papagai, ["purge", "--help"])
        assert result.exit_code == 0
        assert "Delete all existing papagai branches" in result.output

    def test_purge_success(self, runner):
        """Test 'purge' command succeeds."""
        with patch("papagai.cli.purge_branches") as mock_purge:
            result = runner.invoke(papagai, ["purge"])

            mock_purge.assert_called_once()
            assert result.exit_code == 0

    def test_purge_with_git_error(self, runner):
        """Test 'purge' command handles git errors."""
        with patch("papagai.cli.purge_branches") as mock_purge:
            mock_purge.side_effect = subprocess.CalledProcessError(1, "git")

            result = runner.invoke(papagai, ["purge"])

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
        result = runner.invoke(papagai, ["task", "--help"])
        assert result.exit_code == 0
        assert "Run a pre-written task" in result.output
        assert "--list" in result.output
        assert "--base-branch" in result.output

    def test_task_list(self, runner):
        """Test 'task --list' shows available tasks."""
        result = runner.invoke(papagai, ["task", "--list"])
        assert result.exit_code == 0
        # Should show at least some tasks
        assert len(result.output) > 0

    def test_task_without_args(self, runner):
        """Test 'task' without arguments shows error."""
        result = runner.invoke(papagai, ["task"])
        # Command shows error message
        assert "Error: missing task name" in result.output

    def test_task_with_valid_task(self, runner):
        """Test 'task' with a valid task name."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            with patch("papagai.cli.get_builtin_tasks_dir") as mock_get_dir:
                # Create a mock instructions directory
                mock_dir = MagicMock()
                mock_get_dir.return_value = mock_dir

                # Mock the task file
                mock_task_file = MagicMock()
                mock_task_file.exists.return_value = True
                mock_dir.__truediv__.return_value = mock_task_file

                with patch(
                    "papagai.cli.MarkdownInstructions.from_file"
                ) as mock_from_file:
                    mock_instructions = MagicMock()
                    mock_from_file.return_value = mock_instructions
                    mock_claude_run.return_value = 0

                    result = runner.invoke(
                        papagai, ["task", "generic/review"], catch_exceptions=False
                    )

                    mock_claude_run.assert_called_once()
                    assert result.exit_code == 0

    def test_task_with_nonexistent_task(self, runner):
        """Test 'task' with non-existent task."""
        with patch("papagai.cli.get_builtin_tasks_dir") as mock_get_dir:
            # Create a mock instructions directory
            mock_dir = MagicMock()
            mock_get_dir.return_value = mock_dir

            # Mock the task file as non-existent
            mock_task_file = MagicMock()
            mock_task_file.exists.return_value = False
            mock_dir.__truediv__.return_value = mock_task_file

            result = runner.invoke(papagai, ["task", "nonexistent/task"])

            # Command shows error message
            assert "Task 'nonexistent/task' not found" in result.output

    def test_task_with_base_branch(self, runner):
        """Test 'task' with custom base branch."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            with patch("papagai.cli.get_builtin_tasks_dir") as mock_get_dir:
                # Create a mock instructions directory
                mock_dir = MagicMock()
                mock_get_dir.return_value = mock_dir

                # Mock the task file
                mock_task_file = MagicMock()
                mock_task_file.exists.return_value = True
                mock_dir.__truediv__.return_value = mock_task_file

                with patch(
                    "papagai.cli.MarkdownInstructions.from_file"
                ) as mock_from_file:
                    mock_instructions = MagicMock()
                    mock_from_file.return_value = mock_instructions
                    mock_claude_run.return_value = 0

                    result = runner.invoke(
                        papagai,
                        ["task", "--base-branch", "develop", "generic/review"],
                    )

                    # Should call claude_run with develop as base_branch
                    mock_claude_run.assert_called_once()
                    call_kwargs = mock_claude_run.call_args
                    assert call_kwargs[1]["base_branch"] == "develop"
                    assert result.exit_code == 0

    def test_task_with_dry_run(self, runner):
        """Test 'task' with --dry-run flag."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            with patch("papagai.cli.get_builtin_tasks_dir") as mock_get_dir:
                # Create a mock instructions directory
                mock_dir = MagicMock()
                mock_get_dir.return_value = mock_dir

                # Mock the task file
                mock_task_file = MagicMock()
                mock_task_file.exists.return_value = True
                mock_dir.__truediv__.return_value = mock_task_file

                with patch(
                    "papagai.cli.MarkdownInstructions.from_file"
                ) as mock_from_file:
                    mock_instructions = MagicMock()
                    mock_from_file.return_value = mock_instructions
                    mock_claude_run.return_value = 0

                    result = runner.invoke(
                        papagai, ["--dry-run", "task", "generic/review"]
                    )

                    # Should call claude_run with dry_run=True
                    mock_claude_run.assert_called_once()
                    call_kwargs = mock_claude_run.call_args
                    assert call_kwargs[1]["dry_run"] is True
                    assert result.exit_code == 0


class TestIsolationOption:
    """Tests for the --isolation option across do, code, and review commands."""

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

    @pytest.mark.parametrize("command", ["do", "code", "review"])
    def test_isolation_option_in_help(self, runner, command):
        """Test --isolation option appears in help for do, code, and review commands."""
        result = runner.invoke(papagai, [command, "--help"])
        assert result.exit_code == 0
        assert "--isolation" in result.output
        assert "auto" in result.output
        assert "worktree" in result.output
        assert "overlayfs" in result.output

    @pytest.mark.parametrize("command", ["do", "code"])
    @pytest.mark.parametrize(
        "isolation_value",
        ["auto", "worktree", "overlayfs"],
    )
    def test_isolation_option_passed_to_claude_run(
        self, runner, command, isolation_value, mock_instructions_file
    ):
        """Test --isolation option is correctly passed to claude_run for do and code commands."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(
                papagai,
                [
                    command,
                    "--isolation",
                    isolation_value,
                    str(mock_instructions_file),
                ],
            )

            # Should call claude_run with the correct isolation value
            mock_claude_run.assert_called_once()
            call_kwargs = mock_claude_run.call_args
            from papagai.cli import Isolation

            assert call_kwargs[1]["isolation"] == Isolation(isolation_value)
            assert result.exit_code == 0

    @pytest.mark.parametrize(
        "isolation_value",
        ["auto", "worktree", "overlayfs"],
    )
    def test_isolation_option_passed_to_claude_run_review(
        self, runner, isolation_value
    ):
        """Test --isolation option is correctly passed to claude_run for review command."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            with patch("papagai.cli.get_builtin_tasks_dir") as mock_get_dir:
                # Create a mock instructions directory
                mock_dir = MagicMock()
                mock_get_dir.return_value = mock_dir

                # Mock the review task file
                mock_task_file = MagicMock()
                mock_task_file.exists.return_value = True
                mock_dir.__truediv__.return_value = mock_task_file

                with patch(
                    "papagai.cli.MarkdownInstructions.from_file"
                ) as mock_from_file:
                    mock_instructions = MagicMock()
                    mock_from_file.return_value = mock_instructions
                    mock_claude_run.return_value = 0

                    result = runner.invoke(
                        papagai,
                        ["review", "--isolation", isolation_value],
                    )

                    # Should call claude_run with the correct isolation value
                    mock_claude_run.assert_called_once()
                    call_kwargs = mock_claude_run.call_args
                    from papagai.cli import Isolation

                    assert call_kwargs[1]["isolation"] == Isolation(isolation_value)
                    assert result.exit_code == 0

    @pytest.mark.parametrize("command", ["do", "code", "review"])
    def test_isolation_default_is_auto(self, runner, command, mock_instructions_file):
        """Test that default isolation mode is 'auto' for all commands."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            if command == "review":
                with patch("papagai.cli.get_builtin_tasks_dir") as mock_get_dir:
                    # Create a mock instructions directory
                    mock_dir = MagicMock()
                    mock_get_dir.return_value = mock_dir

                    # Mock the review task file
                    mock_task_file = MagicMock()
                    mock_task_file.exists.return_value = True
                    mock_dir.__truediv__.return_value = mock_task_file

                    with patch(
                        "papagai.cli.MarkdownInstructions.from_file"
                    ) as mock_from_file:
                        mock_instructions = MagicMock()
                        mock_from_file.return_value = mock_instructions
                        mock_claude_run.return_value = 0

                        result = runner.invoke(papagai, [command])
            else:
                mock_claude_run.return_value = 0
                result = runner.invoke(
                    papagai,
                    [command, str(mock_instructions_file)],
                )

            # Should call claude_run with isolation=Isolation.AUTO
            mock_claude_run.assert_called_once()
            call_kwargs = mock_claude_run.call_args
            from papagai.cli import Isolation

            assert call_kwargs[1]["isolation"] == Isolation.AUTO
            assert result.exit_code == 0

    @pytest.mark.parametrize("command", ["do", "code", "review"])
    def test_isolation_with_invalid_value(
        self, runner, command, mock_instructions_file
    ):
        """Test that invalid isolation values are rejected."""
        cmd_args = [command, "--isolation", "invalid"]

        if command in ["do", "code"]:
            cmd_args.append(str(mock_instructions_file))

        result = runner.invoke(papagai, cmd_args)

        # Click should reject the invalid choice
        assert result.exit_code == 2
        assert "Invalid value" in result.output or "invalid" in result.output.lower()


class TestReviewCommand:
    """Tests for the 'review' command."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner instance."""
        return CliRunner()

    def test_review_help(self, runner):
        """Test 'review' command --help."""
        result = runner.invoke(papagai, ["review", "--help"])
        assert result.exit_code == 0
        assert "Run a code review on the current branch" in result.output
        assert "--base-branch" in result.output

    def test_review_success(self, runner):
        """Test 'review' command succeeds."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            with patch("papagai.cli.get_builtin_tasks_dir") as mock_get_dir:
                # Create a mock instructions directory
                mock_dir = MagicMock()
                mock_get_dir.return_value = mock_dir

                # Mock the review task file
                mock_task_file = MagicMock()
                mock_task_file.exists.return_value = True
                mock_dir.__truediv__.return_value = mock_task_file

                with patch(
                    "papagai.cli.MarkdownInstructions.from_file"
                ) as mock_from_file:
                    mock_instructions = MagicMock()
                    mock_from_file.return_value = mock_instructions
                    mock_claude_run.return_value = 0

                    result = runner.invoke(papagai, ["review"])

                    mock_claude_run.assert_called_once()
                    assert result.exit_code == 0

    def test_review_with_base_branch(self, runner):
        """Test 'review' command with custom base branch."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            with patch("papagai.cli.get_builtin_tasks_dir") as mock_get_dir:
                # Create a mock instructions directory
                mock_dir = MagicMock()
                mock_get_dir.return_value = mock_dir

                # Mock the review task file
                mock_task_file = MagicMock()
                mock_task_file.exists.return_value = True
                mock_dir.__truediv__.return_value = mock_task_file

                with patch(
                    "papagai.cli.MarkdownInstructions.from_file"
                ) as mock_from_file:
                    mock_instructions = MagicMock()
                    mock_from_file.return_value = mock_instructions
                    mock_claude_run.return_value = 0

                    result = runner.invoke(
                        papagai, ["review", "--base-branch", "develop"]
                    )

                    # Should call claude_run with develop as base_branch
                    mock_claude_run.assert_called_once()
                    call_kwargs = mock_claude_run.call_args
                    assert call_kwargs[1]["base_branch"] == "develop"
                    assert result.exit_code == 0

    def test_review_with_dry_run(self, runner):
        """Test 'review' command with --dry-run flag."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            with patch("papagai.cli.get_builtin_tasks_dir") as mock_get_dir:
                # Create a mock instructions directory
                mock_dir = MagicMock()
                mock_get_dir.return_value = mock_dir

                # Mock the review task file
                mock_task_file = MagicMock()
                mock_task_file.exists.return_value = True
                mock_dir.__truediv__.return_value = mock_task_file

                with patch(
                    "papagai.cli.MarkdownInstructions.from_file"
                ) as mock_from_file:
                    mock_instructions = MagicMock()
                    mock_from_file.return_value = mock_instructions
                    mock_claude_run.return_value = 0

                    result = runner.invoke(papagai, ["--dry-run", "review"])

                    # Should call claude_run with dry_run=True
                    mock_claude_run.assert_called_once()
                    call_kwargs = mock_claude_run.call_args
                    assert call_kwargs[1]["dry_run"] is True
                    assert result.exit_code == 0

    def test_review_missing_task_file(self, runner, tmp_path):
        """Test 'review' command when review.md doesn't exist."""
        # Create a fake primers directory without the review file
        fake_primers_dir = tmp_path / "primers"
        fake_primers_dir.mkdir()
        # Note: NOT creating review.md file

        with patch("papagai.cli.get_builtin_primers_dir") as mock_get_dir:
            mock_get_dir.return_value = fake_primers_dir

            result = runner.invoke(papagai, ["review"])

            # Command shows error message
            assert "Review task not found" in result.output

    def test_review_loads_from_primers(self, runner):
        """Test 'review' command loads review.md from primers directory."""
        with patch("papagai.cli.claude_run") as mock_claude_run:
            with patch("papagai.cli.get_builtin_primers_dir") as mock_get_dir:
                # Create a mock primers directory
                mock_dir = MagicMock()
                mock_get_dir.return_value = mock_dir

                # Mock the review primer file
                mock_task_file = MagicMock()
                mock_task_file.exists.return_value = True
                mock_dir.__truediv__.return_value = mock_task_file

                with patch(
                    "papagai.cli.MarkdownInstructions.from_file"
                ) as mock_from_file:
                    mock_instructions = MagicMock()
                    mock_from_file.return_value = mock_instructions
                    mock_claude_run.return_value = 0

                    # Run review command
                    result = runner.invoke(papagai, ["review", "--base-branch", "main"])

                    # Verify it loaded from primers
                    assert result.exit_code == 0
                    mock_get_dir.assert_called_once()
                    mock_claude_run.assert_called_once()
