# VINO Agent Rules

## Workflow
You are authorized to proactively explore, inspect, edit, and test files to solve problems, unless restricted below. There is only one active branch: main.

You must never:
- create branches or worktrees
- commit, push, merge, or alter the git state
- delete data, outputs, logs, checkpoints, reports, or pretrained files
- run destructive commands such as rm -rf, rmdir, git reset --hard, git clean -fdx, or sudo

## Protected directories
Never edit, delete, move, or overwrite anything inside:
- data/
- outputs/
- logs/
- pretrained/
- reports/
- wandb/
- checkpoints/

## Required behavior
Before editing:
1. Explain the planned change briefly.
2. Touch only relevant files.

After editing:
1. Run focused tests to verify your changes.
2. Summarize changed files.
3. Tell the user exactly what changed.
4. Stop and wait for human review (via `git diff`). Do not attempt to commit.