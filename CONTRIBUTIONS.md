# Team Contribution Log

This document records the individual contributions of team members to the Data Stream Mining Research Project at NOVA IMS.

Its purpose is to ensure transparency, accountability and recognition of individual work.

## Team Members

| Member | GitHub Username |
|---|---|
| Bruno Henriques | @brunophenriques |
| Diogo Padilha | @diogonpadilha |
| João Pinho | @pinho-1 |
| Maria Inês | @mariainessantos |
| Luis Soeiro | @luisrafaelsoeiro |

## Contribution History

| Date | Member | Contribution | Evidence |
|---|---|---|---|
| 2026-11-09 | Bruno Henriques | Set up the team collaboration workflow, including branching strategy, commit conventions, Pull Request guidelines, and contribution tracking. | [PR #1](https://github.com/brunophenriques/T1DSM/pull/1) [PR #2](https://github.com/brunophenriques/T1DSM/pull/2) |
| 2026-11-19 | Maria Inês | Implemented the feature-relevance module on branch `mines`: read-only extraction and checkpoint logging of the model's global and per-rule feature weights, tie-aware stability and validity metrics, a synthetic stream with ground-truth relevance and drift, and a config- and seed-driven runner. Added tests for non-intrusive logging, absence of label leakage, and equivalence with the baseline prequential loop. Developed with AI assistance (Claude); code reviewed and tested locally. | [relevance/](https://github.com/brunophenriques/T1DSM/tree/mines/relevance) [run_relevance.py](https://github.com/brunophenriques/T1DSM/blob/mines/run_relevance.py) [configs/](https://github.com/brunophenriques/T1DSM/tree/mines/configs) [tests/](https://github.com/brunophenriques/T1DSM/tree/mines/tests) [PR #3](https://github.com/brunophenriques/T1DSM/pull/3) |
| 2026-11-22 | Maria Inês | Completed Workflow 1 (SQ1, stability of the relevance weights G) on branch `mines`: stability metrics (top-k Jaccard, Kendall τ-b, weighted τ, Spearman, top-k switch rate, per-feature SD) and a statistics module (bootstrap and block-bootstrap CIs, Wilcoxon with rank-biserial, Holm, exact shuffle p-value) with unit tests; Elec2 leak-free preprocessing and a shuffled control; runs on S3, Elec2 and 30 shuffled copies, with and without `date`; the SQ1 row of Table 1 and Figure 2a; the SQ1 protocol and a documented notebook. Result: on Elec2 the top-3 of G is less stable than in shuffled copies (Jaccard@200 0.70 vs 0.88 ± 0.02 without `date`, 0.65 vs 0.77 ± 0.03 with it; exact p = 0.032 in both, the minimum for 30 shuffles), while it is perfectly stable on stationary synthetic streams. Developed with AI assistance (Claude); code reviewed and tested locally. | [wf1_stability/](https://github.com/brunophenriques/T1DSM/tree/mines/wf1_stability) [PROTOCOL_SQ1.md](https://github.com/brunophenriques/T1DSM/blob/mines/wf1_stability/PROTOCOL_SQ1.md) [table1_sq1.md](https://github.com/brunophenriques/T1DSM/blob/mines/wf1_stability/table1_sq1.md) [relevance/stability.py](https://github.com/brunophenriques/T1DSM/blob/mines/relevance/stability.py) [relevance/stats.py](https://github.com/brunophenriques/T1DSM/blob/mines/relevance/stats.py) [tests/test_stability.py](https://github.com/brunophenriques/T1DSM/blob/mines/tests/test_stability.py) [sq1_stability.ipynb](https://github.com/brunophenriques/T1DSM/blob/mines/notebooks/sq1_stability.ipynb) |


## Guidelines

- Update this log whenever a meaningful task is completed.
- Include contributions involving research, coding, experiments, analysis, documentation and writing.
- Provide supporting evidence whenever possible.
- Do not record planned tasks as completed contributions.
- All team members are responsible for keeping their entries accurate.