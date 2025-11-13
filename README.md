# claude-do

`claude-do` is a commandline utility to have [Claude](https://claude.ai/) go off and do something
in a git repository. Any changes made by claude are done in a worktree, allowing multiple tasks
to work simultaneously.


## Installation

```console
$ pip install https://github.com/whot/claude-do
```

If running from the git repository, use [uv](https://github.com/astral-sh/uv):

```console
$ uv run claude-do
```

## Usage

The primary subcommand is `do` to get Claude to do something.

```console
# Instructions via a file
$ echo "Update all .format() strings with f-strings" > instructions.md
$ claude-do do --instructions instructions.md

$ claude-do do
Please tell me what you want me to do (Ctrl+D to complete)
Update all .format() strings with f-strings
Working in claude-do/main-2025-11-12-7be3946e (based off main)
[...]
My work here is done. Check out branch claude-do/main-2025-11-12-7be3946e
```

The instructions should be in Markdown. If there is a frontmatter key
`"tools"` it is extracted and passed to Claude as the set of allowed tools.
For example:
```markdown
---
tools: Bash(uv :*)
---
Update all .format() strings with f-strings
```


## Pre-written tasks

The `claude-do task` command runs pre-written tasks. These are read from
`$XDG_CONFIG_HOME/claude-do/tasks/**/*.md` and must look like this:

```
---
description: some description
---
You are a very smart LLM. Blah blah.
```

Additionally `claude-do` ships with built-in tasks. These are tasks that were
(somewhat) successfully used elsewhere and might be useful for other
repos.
```console
$ claude-do task --list
[...]
python/update-to-3.9                ... update a Python code base to Python 3.9+
$ claude-do task python/update-to-3.9
```
The `--list` command will also list any tasks in `XDG_CONFIG_HOME`.

### Variable substitution in task files

- `{BRANCH}` is replaced with the original branch
- `{WORKTREE_BRANCH}` is replaced with the current working branch

### Code Review

The `review` command is a convenience shortcut for running a code review:
```console
$ claude-do review
```

This command will review all commits on the current branch, providing inline
feedback and creating fixup commits for any issues found.

### Cleanup

`claude-do` works on branches (and in workdirs), if you want to get rid yourself
of any leftover branches use:
```console
$ claude-do purge
```


# Acknowledgements

`claude-do` is motivated by and based on
[claude-review-agent](https://github.com/swick/claude-dev-tools/).

Most of `claude-do` (especially the tests!) were written using Claude.
