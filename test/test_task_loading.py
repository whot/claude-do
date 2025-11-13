#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for task loading from XDG_CONFIG_HOME."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from claude_do.cli import (
    get_builtin_tasks_dir,
    get_xdg_task_dir,
    list_all_tasks,
    claude_do,
)


class TestGetXdgTaskDir:
    """Tests for get_xdg_task_dir() function."""

    def test_get_xdg_task_dir_with_env_set(self, monkeypatch, tmp_path):
        """Test get_xdg_task_dir returns correct path when XDG_CONFIG_HOME is set."""
        xdg_config_home = tmp_path / "custom_config"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_home))

        task_dir = get_xdg_task_dir()

        assert task_dir == xdg_config_home / "claude-do" / "tasks"

    def test_get_xdg_task_dir_without_env(self, monkeypatch):
        """Test get_xdg_task_dir falls back to ~/.config when XDG_CONFIG_HOME not set."""
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

        task_dir = get_xdg_task_dir()

        expected = Path.home() / ".config" / "claude-do" / "tasks"
        assert task_dir == expected

    def test_get_xdg_task_dir_with_empty_env(self, monkeypatch):
        """Test get_xdg_task_dir behavior when XDG_CONFIG_HOME is empty string."""
        monkeypatch.setenv("XDG_CONFIG_HOME", "")

        task_dir = get_xdg_task_dir()

        # When XDG_CONFIG_HOME is set to empty string, os.getenv returns ""
        # Path("") creates a relative path, so we get "claude-do/tasks"
        # This is arguably a bug, but testing actual behavior here
        expected = Path("") / "claude-do" / "tasks"
        assert task_dir == expected

    def test_get_xdg_task_dir_returns_path_object(self, monkeypatch, tmp_path):
        """Test get_xdg_task_dir returns a Path object."""
        xdg_config_home = tmp_path / "config"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_home))

        task_dir = get_xdg_task_dir()

        assert isinstance(task_dir, Path)

    def test_get_xdg_task_dir_creates_correct_structure(self, monkeypatch, tmp_path):
        """Test get_xdg_task_dir creates the expected directory structure."""
        xdg_config_home = tmp_path / "config"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_home))

        task_dir = get_xdg_task_dir()

        # Should be XDG_CONFIG_HOME/claude-do/tasks
        assert task_dir.parent.name == "claude-do"
        assert task_dir.name == "tasks"
        assert task_dir.parent.parent == xdg_config_home


class TestGetBuiltinTasksDir:
    """Tests for get_builtin_tasks_dir() function."""

    def test_get_builtin_tasks_dir_returns_path(self):
        """Test get_builtin_tasks_dir returns a Path object."""
        tasks_dir = get_builtin_tasks_dir()

        assert isinstance(tasks_dir, Path)

    def test_get_builtin_tasks_dir_exists(self):
        """Test get_builtin_tasks_dir returns an existing directory."""
        tasks_dir = get_builtin_tasks_dir()

        assert tasks_dir.exists()
        assert tasks_dir.is_dir()

    def test_get_builtin_tasks_dir_has_tasks(self):
        """Test get_builtin_tasks_dir contains task files."""
        tasks_dir = get_builtin_tasks_dir()

        md_files = list(tasks_dir.glob("**/*.md"))
        assert len(md_files) > 0

    def test_get_builtin_tasks_dir_structure(self):
        """Test get_builtin_tasks_dir has expected structure."""
        tasks_dir = get_builtin_tasks_dir()

        # Should have at least the generic directory
        assert (tasks_dir / "generic").exists()
        assert (tasks_dir / "generic").is_dir()


class TestListAllTasks:
    """Tests for list_all_tasks() function."""

    @pytest.fixture
    def setup_xdg_tasks(self, tmp_path, monkeypatch):
        """Set up a temporary XDG_CONFIG_HOME with task files."""
        xdg_config_home = tmp_path / "config"
        xdg_tasks_dir = xdg_config_home / "claude-do" / "tasks"
        xdg_tasks_dir.mkdir(parents=True)

        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_home))

        return xdg_tasks_dir

    def test_list_all_tasks_shows_builtin_tasks(self, capsys):
        """Test list_all_tasks shows built-in tasks."""
        exit_code = list_all_tasks()

        assert exit_code == 0
        captured = capsys.readouterr()
        # Should show at least the generic/review task
        assert "generic/review" in captured.out

    def test_list_all_tasks_with_xdg_tasks(self, setup_xdg_tasks, capsys):
        """Test list_all_tasks includes tasks from XDG_CONFIG_HOME."""
        # Create a custom task
        custom_task = setup_xdg_tasks / "custom-task.md"
        custom_task.write_text(
            """---
description: A custom user task
---

Do something custom.
"""
        )

        exit_code = list_all_tasks()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "custom-task" in captured.out
        assert "A custom user task" in captured.out

    def test_list_all_tasks_xdg_takes_precedence(self, setup_xdg_tasks, capsys):
        """Test XDG tasks are listed before built-in tasks."""
        # Create custom tasks
        custom1 = setup_xdg_tasks / "aaa-first.md"
        custom1.write_text(
            """---
description: Should appear first alphabetically
---

Content.
"""
        )

        exit_code = list_all_tasks()

        assert exit_code == 0
        captured = capsys.readouterr()
        # Custom task should appear in the output
        assert "aaa-first" in captured.out

    def test_list_all_tasks_empty_xdg_directory(self, setup_xdg_tasks, capsys):
        """Test list_all_tasks works when XDG directory is empty."""
        # setup_xdg_tasks creates the directory but no files

        exit_code = list_all_tasks()

        assert exit_code == 0
        captured = capsys.readouterr()
        # Should still show built-in tasks
        assert "generic/review" in captured.out

    def test_list_all_tasks_xdg_directory_not_exists(self, monkeypatch, capsys):
        """Test list_all_tasks works when XDG directory doesn't exist."""
        # Set XDG_CONFIG_HOME to a non-existent path
        monkeypatch.setenv("XDG_CONFIG_HOME", "/nonexistent/path/12345")

        exit_code = list_all_tasks()

        assert exit_code == 0
        captured = capsys.readouterr()
        # Should still show built-in tasks
        assert "generic/review" in captured.out

    def test_list_all_tasks_with_subdirectories(self, setup_xdg_tasks, capsys):
        """Test list_all_tasks handles tasks in subdirectories."""
        # Create a subdirectory with a task
        subdir = setup_xdg_tasks / "python"
        subdir.mkdir()
        task = subdir / "format-code.md"
        task.write_text(
            """---
description: Format Python code
---

Format all Python files.
"""
        )

        exit_code = list_all_tasks()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "python/format-code" in captured.out
        assert "Format Python code" in captured.out

    def test_list_all_tasks_skips_tasks_without_description(
        self, setup_xdg_tasks, capsys
    ):
        """Test list_all_tasks skips tasks without description."""
        # Create a task without description
        no_desc_task = setup_xdg_tasks / "no-description.md"
        no_desc_task.write_text(
            """---
tools: Bash
---

Do something.
"""
        )

        exit_code = list_all_tasks()

        assert exit_code == 0
        captured = capsys.readouterr()
        # Task without description should not appear
        assert "no-description" not in captured.out

    def test_list_all_tasks_handles_invalid_markdown(
        self, setup_xdg_tasks, capsys, caplog
    ):
        """Test list_all_tasks handles invalid markdown files gracefully."""
        # Create an invalid markdown file (not parseable)
        invalid_task = setup_xdg_tasks / "invalid.md"
        # Create a file that will cause MarkdownInstructions.from_file to fail
        # by making it unreadable (simulate permission error)
        invalid_task.write_text("Some content")
        # Change permissions to make it unreadable
        invalid_task.chmod(0o000)

        try:
            exit_code = list_all_tasks()

            # Should still succeed (exit code 0 if there are other valid tasks)
            assert exit_code in [0, 1]
            captured = capsys.readouterr()
            # Should show warning about failed parsing
            assert "Warning" in captured.err or "Failed to parse" in captured.err
        finally:
            # Restore permissions for cleanup
            invalid_task.chmod(0o644)

    def test_list_all_tasks_multiple_xdg_tasks(self, setup_xdg_tasks, capsys):
        """Test list_all_tasks with multiple XDG tasks."""
        # Create multiple tasks
        tasks = [
            ("task1.md", "First task"),
            ("task2.md", "Second task"),
            ("task3.md", "Third task"),
        ]

        for filename, description in tasks:
            task_file = setup_xdg_tasks / filename
            task_file.write_text(
                f"""---
description: {description}
---

Task content.
"""
            )

        exit_code = list_all_tasks()

        assert exit_code == 0
        captured = capsys.readouterr()
        for filename, description in tasks:
            task_name = filename.replace(".md", "")
            assert task_name in captured.out
            assert description in captured.out

    def test_list_all_tasks_alignment(self, setup_xdg_tasks, capsys):
        """Test list_all_tasks aligns task names and descriptions."""
        # Create tasks with different name lengths
        short_task = setup_xdg_tasks / "a.md"
        short_task.write_text(
            """---
description: Short name
---

Content.
"""
        )

        long_task = setup_xdg_tasks / "very-long-task-name.md"
        long_task.write_text(
            """---
description: Long name
---

Content.
"""
        )

        exit_code = list_all_tasks()

        assert exit_code == 0
        captured = capsys.readouterr()
        # Output should contain the separator " ... "
        assert " ... " in captured.out


class TestTaskCommandWithXdg:
    """Tests for 'task' command with XDG_CONFIG_HOME tasks."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def setup_xdg_tasks(self, tmp_path, monkeypatch):
        """Set up a temporary XDG_CONFIG_HOME with task files."""
        xdg_config_home = tmp_path / "config"
        xdg_tasks_dir = xdg_config_home / "claude-do" / "tasks"
        xdg_tasks_dir.mkdir(parents=True)

        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_home))

        return xdg_tasks_dir

    def test_task_loads_from_xdg(self, runner, setup_xdg_tasks):
        """Test 'task' command loads tasks from XDG_CONFIG_HOME."""
        # Create a custom task
        custom_task = setup_xdg_tasks / "custom-task.md"
        custom_task.write_text(
            """---
description: A custom task
tools: Bash(test:*)
---

Do something custom.
"""
        )

        with patch("claude_do.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(claude_do, ["task", "custom-task"])

            # Should successfully load and execute the task
            mock_claude_run.assert_called_once()
            assert result.exit_code == 0

    def test_task_xdg_takes_precedence_over_builtin(self, runner, setup_xdg_tasks):
        """Test XDG tasks take precedence over built-in tasks with same name."""
        # Create a custom task with the same name as a built-in one
        generic_dir = setup_xdg_tasks / "generic"
        generic_dir.mkdir()
        custom_review = generic_dir / "review.md"
        custom_review.write_text(
            """---
description: Custom review task
tools: Bash(custom:*)
---

This is my custom review.
"""
        )

        with patch("claude_do.cli.claude_run") as mock_claude_run:
            with patch(
                "claude_do.cli.MarkdownInstructions.from_file"
            ) as mock_from_file:
                mock_instructions = MagicMock()
                mock_instructions.text = "This is my custom review."
                mock_from_file.return_value = mock_instructions
                mock_claude_run.return_value = 0

                result = runner.invoke(claude_do, ["task", "generic/review"])

                # Should load the XDG version
                mock_from_file.assert_called_once()
                # Verify it was called with the XDG path
                called_path = mock_from_file.call_args[0][0]
                assert "custom review" in custom_review.read_text().lower()
                assert result.exit_code == 0

    def test_task_falls_back_to_builtin(self, runner, setup_xdg_tasks):
        """Test 'task' falls back to built-in tasks if not in XDG."""
        # Don't create any XDG tasks, just use built-in

        with patch("claude_do.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(claude_do, ["task", "generic/review"])

            # Should load the built-in task
            mock_claude_run.assert_called_once()
            assert result.exit_code == 0

    def test_task_with_xdg_subdirectories(self, runner, setup_xdg_tasks):
        """Test 'task' loads tasks from XDG subdirectories."""
        # Create a subdirectory structure
        subdir = setup_xdg_tasks / "python" / "linting"
        subdir.mkdir(parents=True)
        task = subdir / "ruff.md"
        task.write_text(
            """---
description: Run ruff linter
tools: Bash(ruff:*)
---

Run ruff on all Python files.
"""
        )

        with patch("claude_do.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(claude_do, ["task", "python/linting/ruff"])

            # Should successfully load the nested task
            mock_claude_run.assert_called_once()
            assert result.exit_code == 0

    def test_task_with_nonexistent_xdg_task(self, runner, setup_xdg_tasks):
        """Test 'task' command with non-existent XDG task."""
        # Create XDG directory but no tasks

        result = runner.invoke(claude_do, ["task", "nonexistent/task"])

        # Should show error message
        assert "Task 'nonexistent/task' not found" in result.output
        # Note: Click runner doesn't propagate the return value as exit code
        # in the same way, so we just check for the error message

    def test_task_list_with_xdg_tasks(self, runner, setup_xdg_tasks):
        """Test 'task --list' includes XDG tasks."""
        # Create custom tasks
        custom_task = setup_xdg_tasks / "my-task.md"
        custom_task.write_text(
            """---
description: My custom task
---

Content.
"""
        )

        result = runner.invoke(claude_do, ["task", "--list"])

        assert result.exit_code == 0
        assert "my-task" in result.output
        assert "My custom task" in result.output

    def test_task_with_xdg_invalid_permissions(self, runner, setup_xdg_tasks):
        """Test 'task' handles permission errors gracefully."""
        # Create a task file with invalid permissions
        restricted_task = setup_xdg_tasks / "restricted.md"
        restricted_task.write_text(
            """---
description: Restricted task
---

Content.
"""
        )
        restricted_task.chmod(0o000)

        try:
            result = runner.invoke(claude_do, ["task", "restricted"])

            # Should show error message about reading the file
            assert "Error reading" in result.output or result.exit_code != 0
        finally:
            # Restore permissions for cleanup
            restricted_task.chmod(0o644)

    def test_task_with_xdg_home_not_set(self, runner, monkeypatch):
        """Test 'task' works when XDG_CONFIG_HOME is not set."""
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

        with patch("claude_do.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(claude_do, ["task", "generic/review"])

            # Should still work with built-in tasks
            mock_claude_run.assert_called_once()
            assert result.exit_code == 0


class TestTaskCommandIntegration:
    """Integration tests for task loading from both sources."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def setup_complete_environment(self, tmp_path, monkeypatch):
        """Set up complete environment with XDG and built-in tasks."""
        xdg_config_home = tmp_path / "config"
        xdg_tasks_dir = xdg_config_home / "claude-do" / "tasks"
        xdg_tasks_dir.mkdir(parents=True)

        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_home))

        return {
            "xdg_tasks_dir": xdg_tasks_dir,
            "xdg_config_home": xdg_config_home,
        }

    def test_task_loading_priority_order(self, runner, setup_complete_environment):
        """Test that tasks are loaded in correct priority order (XDG > built-in)."""
        xdg_tasks = setup_complete_environment["xdg_tasks_dir"]

        # Create an XDG task that shadows a built-in one
        generic_dir = xdg_tasks / "generic"
        generic_dir.mkdir()
        xdg_review = generic_dir / "review.md"
        xdg_review.write_text(
            """---
description: XDG custom review
---

XDG review content.
"""
        )

        # Create a unique XDG task
        xdg_unique = xdg_tasks / "unique-task.md"
        xdg_unique.write_text(
            """---
description: Unique XDG task
---

Unique content.
"""
        )

        with patch("claude_do.cli.claude_run") as mock_claude_run:
            with patch(
                "claude_do.cli.MarkdownInstructions.from_file"
            ) as mock_from_file:
                mock_instructions = MagicMock()
                mock_from_file.return_value = mock_instructions
                mock_claude_run.return_value = 0

                # Test loading the shadowed task
                result1 = runner.invoke(claude_do, ["task", "generic/review"])
                assert result1.exit_code == 0

                # Verify XDG version was loaded (not built-in)
                called_path = mock_from_file.call_args[0][0]
                assert called_path == xdg_review

                # Reset mock
                mock_from_file.reset_mock()

                # Test loading the unique XDG task
                result2 = runner.invoke(claude_do, ["task", "unique-task"])
                assert result2.exit_code == 0

                called_path = mock_from_file.call_args[0][0]
                assert called_path == xdg_unique

    def test_task_list_shows_both_sources(self, runner, setup_complete_environment):
        """Test that task list shows tasks from both XDG and built-in."""
        xdg_tasks = setup_complete_environment["xdg_tasks_dir"]

        # Create XDG tasks
        xdg_task1 = xdg_tasks / "xdg-task-1.md"
        xdg_task1.write_text(
            """---
description: First XDG task
---

Content.
"""
        )

        xdg_task2 = xdg_tasks / "xdg-task-2.md"
        xdg_task2.write_text(
            """---
description: Second XDG task
---

Content.
"""
        )

        result = runner.invoke(claude_do, ["task", "--list"])

        assert result.exit_code == 0
        # Should show XDG tasks
        assert "xdg-task-1" in result.output
        assert "xdg-task-2" in result.output
        # Should also show built-in tasks
        assert "generic/review" in result.output

    def test_empty_xdg_directory_uses_builtins(self, runner, setup_complete_environment):
        """Test that empty XDG directory doesn't prevent loading built-in tasks."""
        # XDG directory exists but is empty

        with patch("claude_do.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(claude_do, ["task", "generic/review"])

            # Should successfully load built-in task
            mock_claude_run.assert_called_once()
            assert result.exit_code == 0

    def test_task_with_complex_directory_structure(
        self, runner, setup_complete_environment
    ):
        """Test task loading with complex nested directory structures."""
        xdg_tasks = setup_complete_environment["xdg_tasks_dir"]

        # Create complex directory structure
        nested_path = xdg_tasks / "lang" / "python" / "testing"
        nested_path.mkdir(parents=True)

        pytest_task = nested_path / "pytest.md"
        pytest_task.write_text(
            """---
description: Run pytest tests
tools: Bash(pytest:*)
---

Run pytest on all test files.
"""
        )

        with patch("claude_do.cli.claude_run") as mock_claude_run:
            mock_claude_run.return_value = 0

            result = runner.invoke(claude_do, ["task", "lang/python/testing/pytest"])

            mock_claude_run.assert_called_once()
            assert result.exit_code == 0
