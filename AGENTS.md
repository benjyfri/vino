# VINO Agent Rules

## Workflow
There is only one active branch: main.

Agents may inspect, edit, and test files only when explicitly asked.

Agents must never:
- create branches
- create worktrees
- commit
- push
- run sbatch
- delete data, outputs, logs, checkpoints, reports, or pretrained files
- run destructive commands such as rm -rf, git reset --hard, git clean -fdx, sudo, scancel

The human user is the only one allowed to:
- commit
- push
- merge
- submit SLURM jobs
- approve large experiment runs

## Protected directories
Never edit, delete, move, or overwrite:
- data/
- outputs/
- logs/
- pretrained/
- reports/
- wandb/
- checkpoints/

## Allowed work
Agents may edit:
- vino/
- tests/
- scripts/
- configs/
- sbatch/
- .ai/

Agents may run:
- pytest
- python scripts
- bash .ai/scripts/repo_health.sh
- bash .ai/scripts/run_tests_fast.sh
- bash .ai/scripts/preview_sbatch.sh

## Required behavior
Before editing:
1. Explain the planned change briefly.
2. Touch only relevant files.

After editing:
1. Run focused tests.
2. Summarize changed files.
3. Tell the user exactly what changed.
4. Do not commit.

## Human review
The user will inspect:

git diff

and will decide whether to commit.
