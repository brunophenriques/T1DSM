# Contributing Guidelines

## 1. Branching Strategy

The `main` branch contains the stable and reviewed version of the project.

- Each task should be developed in a separate branch.
- Do not commit or push directly to `main`.
- Use descriptive branch names:
  - `feature/task-name` — New implementations or features.
  - `experiment/task-name` — Experiments and evaluations.
  - `fix/task-name` — Bug fixes.
  - `docs/task-name` — Documentation updates.

Before starting new work, update your local `main` branch and create a new branch from it.

## 2. Commit Conventions

All commits must have clear and descriptive messages.

Use the following prefixes:

- `feat:` — New features or implementations.
- `fix:` — Bug fixes.
- `docs:` — Documentation changes.
- `test:` — Tests.
- `refactor:` — Code restructuring without changing functionality.
- `exp:` — Experiments and evaluation.

Example: `feat: implement adaptive rule pruning`

Commit regularly, preferably after completing a meaningful change.

## 3. Pull Requests

All changes must be integrated into `main` through Pull Requests (PRs).

- Open a PR when a task is ready for review.
- Describe what was changed and why.
- Link the relevant task or issue when applicable.
- At least one other team member should review the PR before merging.
- Resolve conflicts and address relevant feedback before merging.

## 4. Code and Experiment Standards

- Preserve the original UnimNeuron baseline for comparison.
- Keep experimental modifications separate from the baseline whenever possible.
- Document important methodological decisions.
- Use explicit random seeds for reproducible experiments.
- Record dependencies, configurations, datasets and execution commands.
- Add appropriate tests for modified components.
- Avoid committing unnecessary generated files, local environments, or credentials.

## 5. Contribution Tracking

All team members are responsible for documenting their work in `CONTRIBUTIONS.md`.

Each contribution should include:

- Date.
- Team member.
- Description of completed work.
- Supporting evidence, such as a Pull Request, commit, experiment, or document.

Contributions may include implementation, experiments, literature review, analysis, documentation and scientific writing.

Entries must reflect work actually completed.

## 6. Team Responsibilities

- Communicate significant changes to the team.
- Avoid making conflicting modifications without coordination.
- Review teammates' work when requested.
- Ensure that every member understands the methodology, code and results.
- Keep the repository organized and reproducible.