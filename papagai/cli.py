#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

import enum
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click

try:
    import rich.logging

    handler = rich.logging.RichHandler(rich_tracebacks=True)
except ModuleNotFoundError:
    handler = logging.StreamHandler()


from .cmd import run_command
from .markdown import MarkdownInstructions
from .worktree import Worktree, WorktreeOverlayFs, BRANCH_PREFIX


logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[handler],
)
logger = logging.getLogger("papagai")


@dataclass
class Context:
    dry_run: bool = False


ALLOWED_TOOLS = [
    "Glob",
    "Grep",
    "Read",
    "Bash(git status)",
    "Bash(git diff:*)",
    "Bash(git log:*)",
    "Bash(git show:*)",
    "Bash(git add:*)",
    "Bash(git commit:*)",
    "Bash(uv :*)",
    "Bash(pytest3 :*)",
    "Edit(./**)",
    "Write(./**)",
]

PROMPT_SUFFIX = """

# Important

Any changes to this repository should be commited into git following git best
practices:
- use descriptive subject lines
- group logical changes together
"""


class Isolation(enum.StrEnum):
    AUTO = "auto"
    WORKTREE = "worktree"
    OVERLAYFS = "overlayfs"


def get_builtin_tasks_dir() -> Path:
    package_dir = Path(__file__).parent
    return package_dir / "tasks"


def get_builtin_primers_dir() -> Path:
    package_dir = Path(__file__).parent
    return package_dir / "primers"


def get_xdg_task_dir() -> Path:
    return (
        Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
        / "papagai"
        / "tasks"
    )


def get_branch(repo_dir: Path, ref: str = "HEAD") -> str:
    """
    Get the branch name for a given ref (commit-ish).

    Args:
        repo_dir: Path to git repository
        ref: Git ref (branch name, HEAD, etc.). Default: HEAD

    Returns:
        The branch name

    Raises:
        subprocess.CalledProcessError if ref doesn't exist or not a git repo
    """
    result = run_command(
        ["git", "rev-parse", "--abbrev-ref", "--verify", ref],
        cwd=repo_dir,
    )
    return result.stdout.strip()


def purge_branches(repo_dir: Path) -> None:
    result = run_command(
        ["git", "branch", "--format=%(refname:short)", "--list", f"{BRANCH_PREFIX}/*"],
        cwd=repo_dir,
    )
    branches = result.stdout.strip().split("\n")
    for branch in [b for b in branches if b]:
        click.echo(f"Deleting branch: {branch}")
        run_command(["git", "branch", "-D", branch], cwd=repo_dir)


def run_claude(
    worktree_dir: Path, instructions: str, dry_run: bool, allowed_tools: list[str] = []
) -> None:
    """Run the Claude review agent."""

    cmd = [
        "claude",
        "--allowed-tools",
        " ".join(allowed_tools),
        "-p",
        instructions,
    ]

    if dry_run:
        click.echo("Would execute command:")
        click.echo(f"  cd {shlex.quote(str(worktree_dir))}")
        click.echo(f"  {' '.join(shlex.quote(arg) for arg in cmd)}")
        return

    click.secho(
        "Claude is pondering, contemplating, mulling, puzzling, meditating, etc.",
        fg="blue",
    )

    try:
        result = run_command(cmd, cwd=worktree_dir)
        if result.stdout:
            click.echo(result.stdout)
        if result.stderr:
            click.secho(result.stderr, err=True, fg="red")
    except subprocess.CalledProcessError as e:
        click.secho(f"Error running claude: {e}", err=True, fg="red")
        raise


def claude_run(
    base_branch: str,
    instructions: MarkdownInstructions,
    dry_run: bool,
    branch_prefix: str = "",
    isolation: Isolation = Isolation.AUTO,
):
    # Resolve repository directory
    repo_dir = Path.cwd().resolve()
    if not repo_dir.is_dir():
        click.secho(f"Error: {repo_dir} is not a directory", err=True, fg="red")
        return 1

    try:
        branch = get_branch(repo_dir, base_branch)
    except subprocess.CalledProcessError:
        click.secho(
            f"Error: Unable to find branch {base_branch} in this repo",
            err=True,
            fg="red",
        )
        return 1

    worktree_class = Worktree
    if isolation in [Isolation.AUTO, Isolation.OVERLAYFS]:
        if WorktreeOverlayFs.is_supported():
            worktree_class = WorktreeOverlayFs
        elif isolation != Isolation.AUTO:
            click.secho(
                "Error: fuse-overlayfs is not available. Please install it or use --isolation=worktree",
                err=True,
                fg="red",
            )
            return 1
        else:
            worktree_class = Worktree
        if isolation == Isolation.AUTO:
            logger.debug(f"Using isolation {worktree_class.__name__}")
    elif isolation == Isolation.WORKTREE:
        worktree_class = Worktree
    else:
        raise NotImplementedError(f"Error: Invalid isolation mode {isolation}")

    allowed_tools = ALLOWED_TOOLS + instructions.tools
    try:
        with worktree_class.from_branch(
            repo_dir, branch, branch_prefix=f"{BRANCH_PREFIX}/{branch_prefix}"
        ) as worktree:
            click.secho(
                f"Working in branch {worktree.branch} (based off {branch})", bold=True
            )

            assert instructions.text
            insts = instructions.text.replace("{BRANCH}", branch)
            insts = insts.replace("{WORKTREE_BRANCH}", worktree.branch)
            insts = f"{insts}\n{PROMPT_SUFFIX}"

            run_claude(worktree.worktree_dir, insts, dry_run, allowed_tools)
            click.secho(
                f"My work here is done. Check out branch {worktree.branch}", bold=True
            )

        return 0
    except AssertionError as e:
        raise e
    except Exception as e:
        click.secho(f"Error: {e}", err=True, fg="red")
        return 1


def list_all_tasks() -> int:
    """
    List all available tasks from the instructions directory.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Find the instructions directory from the installed package
    builtin_tasks_dir = get_builtin_tasks_dir()
    if not builtin_tasks_dir.exists() or not builtin_tasks_dir.is_dir():
        click.secho(
            f"Error: Tasks directory not found at {builtin_tasks_dir}",
            err=True,
            fg="red",
        )
        return 1

    all_dirs = [builtin_tasks_dir]
    xdg_dir = get_xdg_task_dir()
    if xdg_dir.exists():
        all_dirs = [xdg_dir] + all_dirs

    @dataclass
    class Task:
        name: str
        description: str

    tasks: list[Task] = []
    for dir in all_dirs:
        md_files = sorted(dir.glob("**/*.md"))
        if not md_files:
            continue

        for md_file in md_files:
            try:
                md = MarkdownInstructions.from_file(md_file)
                if md.description:
                    # Get relative path from instructions directory without .md extension
                    rel_path = md_file.relative_to(dir)
                    task_name = str(rel_path.with_suffix(""))
                    tasks.append(Task(task_name, md.description))
                else:
                    click.secho(
                        "Found task file {md_file} but it doesn't have a description"
                    )
            except Exception as e:
                click.secho(
                    f"Warning: Failed to parse {md_file}: {e}", err=True, fg="red"
                )

    if not tasks:
        click.secho("No tasks with descriptions found.", err=True, fg="red")
        return 1

    # Calculate the maximum task name length for alignment
    max_name_len = max(len(task.name) for task in tasks)

    # Print tasks with aligned descriptions
    for task in tasks:
        click.echo(f"{task.name:<{max_name_len}} ... {task.description}")

    return 0


@click.group()
@click.option("-v", "--verbose", count=True, help="increase verbosity")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the claude command that would be executed without running it",
)
@click.pass_context
def papagai(ctx, dry_run: bool, verbose: int):
    """Papagai: Automate code changes with Claude AI on git worktrees."""

    verbose_levels = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    logger.setLevel(verbose_levels.get(verbose, 0))
    logger.debug(f"Verbose level set {logger.getEffectiveLevel()}")
    # Store context object for subcommands
    ctx.obj = Context(dry_run=dry_run)


@papagai.command("do")
@click.argument(
    "instructions_file",
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
@click.option(
    "--base-branch",
    default="HEAD",
    help="Branch to base the work on (default: current branch)",
)
@click.option(
    "--isolation",
    type=click.Choice(["auto", "worktree", "overlayfs"], case_sensitive=False),
    default="auto",
    help="Worktree isolation mode: auto (try overlayfs, fall back to worktree), worktree (git worktree), or overlayfs (fuse-overlayfs)",
)
@click.pass_context
def cmd_do(
    ctx,
    instructions_file: Optional[Path],
    base_branch: str,
    isolation: str,
) -> int:
    """
    Tell Claude to do something non-code related on a work tree.

    This is the command for non-coding related tasks, and the instructions
    should include priming Claude for the task at hand.

    See papagai code for programming tasks.
    """

    if instructions_file is not None:
        try:
            instructions = MarkdownInstructions.from_file(instructions_file)
        except (FileNotFoundError, PermissionError) as e:
            click.secho(f"Error reading instructions file: {e}", err=True, fg="red")
            return 1
    else:
        if sys.stdin.isatty():
            click.secho(
                "Please tell me what you want me to do (Ctrl+D to complete)", bold=True
            )
        instructions = MarkdownInstructions(text=sys.stdin.read())

    if not instructions.text:
        click.secho(
            "Empty instructions. That's it, I can't work under these conditions!",
            err=True,
            fg="red",
        )
        return 1

    return claude_run(
        base_branch=base_branch,
        instructions=instructions,
        dry_run=ctx.obj.dry_run,
        isolation=Isolation(isolation),
    )


@papagai.command("code")
@click.argument(
    "instructions_file",
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
@click.option(
    "--base-branch",
    default="HEAD",
    help="Branch to base the work on (default: current branch)",
)
@click.option(
    "--isolation",
    type=click.Choice(["auto", "worktree", "overlayfs"], case_sensitive=False),
    default="auto",
    help="Worktree isolation mode: auto (try overlayfs, fall back to worktree), worktree (git worktree), or overlayfs (fuse-overlayfs)",
)
@click.pass_context
def cmd_code(
    ctx,
    instructions_file: Optional[Path],
    base_branch: str,
    isolation: str,
) -> int:
    """
    Tell Claude to code something on a work tree.

    This command primes Claude to be software developer with an automatically
    inserted prompt prefix. The provided instructions thus only need to focus on
    the actual code.

    See papagai code for coding tasks.
    """

    if instructions_file is not None:
        try:
            instructions = MarkdownInstructions.from_file(instructions_file)
        except (FileNotFoundError, PermissionError) as e:
            click.secho(f"Error reading instructions file: {e}", err=True, fg="red")
            return 1
    else:
        if sys.stdin.isatty():
            click.secho(
                "Please tell me what you want me to do (Ctrl+D to complete)", bold=True
            )
        instructions = MarkdownInstructions(text=sys.stdin.read())

    if not instructions.text:
        click.secho(
            "Empty instructions. That's it, I can't work under these conditions!",
            err=True,
            fg="red",
        )
        return 1

    primer = MarkdownInstructions.from_file(get_builtin_primers_dir() / "code.md")
    instructions = primer.combine(instructions)

    return claude_run(
        base_branch=base_branch,
        instructions=instructions,
        dry_run=ctx.obj.dry_run,
        isolation=Isolation(isolation),
    )


@papagai.command("purge")
def cmd_purge() -> int:
    """Delete all existing papagai branches."""
    repo_dir = Path.cwd().resolve()
    if not repo_dir.is_dir():
        click.secho(f"Error: {repo_dir} is not a directory", err=True, fg="red")
        return 1

    try:
        purge_branches(repo_dir)
    except subprocess.CalledProcessError as e:
        click.secho(f"Error purging done branches: {e}", err=True, fg="red")
        return 1
    return 0


@papagai.command("task")
@click.option(
    "--list",
    "list_tasks",
    is_flag=True,
    help="List all available tasks",
)
@click.option(
    "--base-branch",
    default="HEAD",
    help="Branch to base the work on (default: current branch)",
)
@click.argument("task_name", required=False)
@click.pass_context
def cmd_task(
    ctx,
    list_tasks: bool,
    base_branch: str,
    task_name: Optional[str],
) -> int:
    """
    Run a pre-written task, either from the built-in list or from tasks in
    XDG_CONFIG_HOME/papagai/tasks/**/*.md.

    Use --list to see all available tasks.
    """
    if list_tasks:
        return list_all_tasks()

    if not task_name:
        click.secho(
            "Error: missing task name. Available tasks:",
            err=True,
            fg="red",
        )
        return list_all_tasks()

    # Resolve repository directory
    repo_dir = Path.cwd().resolve()
    if not repo_dir.is_dir():
        click.secho(f"Error: {repo_dir} is not a directory", err=True, fg="red")
        return 1

    # Resolve task to instructions file from installed package
    task_files = [
        get_xdg_task_dir() / f"{task_name}.md",
        get_builtin_tasks_dir() / f"{task_name}.md",
    ]

    task_file = next((f for f in task_files if f.exists()), None)
    if not task_file:
        click.secho(
            f"Error: Task '{task_name}' not found",
            err=True,
            fg="red",
        )
        click.secho(
            "Run 'papagai task --list' to see available tasks",
            err=True,
            fg="red",
        )
        return 1
    try:
        instructions = MarkdownInstructions.from_file(task_file)
    except (FileNotFoundError, PermissionError) as e:
        click.secho(f"Error reading instructions file: {e}", err=True, fg="red")
        return 1

    return claude_run(
        base_branch=base_branch,
        instructions=instructions,
        dry_run=ctx.obj.dry_run,
    )


@papagai.command("review")
@click.option(
    "--base-branch",
    default="HEAD",
    help="Branch to base the work on (default: current branch)",
)
@click.option(
    "--isolation",
    type=click.Choice(["auto", "worktree", "overlayfs"], case_sensitive=False),
    default="auto",
    help="Worktree isolation mode: auto (try overlayfs, fall back to worktree), worktree (git worktree), or overlayfs (fuse-overlayfs)",
)
@click.pass_context
def cmd_review(
    ctx,
    base_branch: str,
    isolation: str,
) -> int:
    """
    Run a code review on the current branch.
    """
    # Resolve repository directory
    repo_dir = Path.cwd().resolve()
    if not repo_dir.is_dir():
        click.secho(f"Error: {repo_dir} is not a directory", err=True, fg="red")
        return 1

    primers_dir = get_builtin_primers_dir()
    review_task_file = primers_dir / "review.md"

    if not review_task_file.exists():
        click.secho(
            f"Error: Review task not found at {review_task_file}. This is a bug",
            err=True,
            fg="red",
        )
        return 1

    try:
        instructions = MarkdownInstructions.from_file(review_task_file)
    except (FileNotFoundError, PermissionError) as e:
        click.secho(f"Error reading review instructions: {e}", err=True, fg="red")
        return 1

    return claude_run(
        base_branch=base_branch,
        instructions=instructions,
        dry_run=ctx.obj.dry_run,
        branch_prefix="review-",
        isolation=Isolation(isolation),
    )


if __name__ == "__main__":
    sys.exit(papagai())
