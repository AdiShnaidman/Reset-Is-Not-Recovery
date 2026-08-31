# R7 Evidence Full-14 Report

## Inputs

- Existing recovery hierarchy raw rows from `runs/recovery_hierarchy/recovery_hierarchy_n500` and `runs/recovery_hierarchy_expanded_family_v1/raw`.
- Existing R7 trusted-evidence raw rows from `runs/groundlm_evidence_recovery_v1`.
- Accepted n=500 item subsets from `runs/hysteresis_model_suite_v1/subsets`.
- R7 evidence generated automatically by the existing rule: dataset reference/explanation/evidence field when available, otherwise deterministic constructed gold evidence from the saved question and benchmark-correct option text.
- No manual evidence was written.

## Completion

- Complete pairs: 14/14
- Failed or incomplete pairs: 0/14

## Full-14 R0 vs R7

- R0 accuracy: 0.368
- R7 accuracy: 0.929
- R0 mean p(w_i): 0.404
- R7 mean p(w_i): 0.045
- Mean p(w_i) change: -0.359
- R0 hysteresis gap: 0.303
- R7 hysteresis gap: -0.055
- Gap change: -0.359
- R0 wrong-following: 41.2%
- R7 wrong-following: 4.3%
- R0 contamination: 27.0%
- R7 contamination: 1.8%

## Dataset Deltas

| Dataset | R0 acc. | R7 acc. | R0 p(w_i) | R7 p(w_i) | R0 wrong-follow | R7 wrong-follow |
|---|---:|---:|---:|---:|---:|---:|
| mmlu_pro | 0.221 | 0.903 | 0.518 | 0.060 | 53.2% | 5.7% |
| truthfulqa_mc | 0.515 | 0.956 | 0.290 | 0.031 | 29.1% | 3.0% |

## Runtime Notes

Each model-dataset pair was scored with the existing `groundlm_evidence_recovery_v1.py` runner. Gemma chat-template compatibility uses the existing system-prompt folding path in that runner.
