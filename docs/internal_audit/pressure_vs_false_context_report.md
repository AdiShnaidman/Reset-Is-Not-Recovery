# Pressure-vs-False-Context Full14 Report

This is a protocol-preserving expansion over the available model-dataset pairs. It reuses the accepted subsets and the saved ordinary-reset pressure-context rows, and only scores the new false-context reset conditions.

## Completion

- Model-dataset-condition rows: 70
- Conditions: clean_no_mention, neutral_wrong_mention, explicit_false_wrong_mention, quoted_false_claim, user_pressure
- Aggregate item-condition rows: 35000

## Aggregate Summary

| condition | n | clean p(w) | reset p(w) | gap | reset follows w (%) | lock-in (%) |
|---|---:|---:|---:|---:|---:|---:|
| Clean reset | 7000 | 0.101 | 0.099 | -0.002 | 9.9 | -- |
| Neutral mention | 7000 | 0.101 | 0.265 | 0.165 | 27.7 | -- |
| Explicit false mention | 7000 | 0.101 | 0.173 | 0.072 | 17.9 | -- |
| Quoted false claim | 7000 | 0.101 | 0.147 | 0.046 | 15.1 | -- |
| User pressure | 7000 | 0.101 | 0.404 | 0.303 | 41.2 | 48.0 |

## Dataset Summary

### mmlu_pro

| condition | n | gap | reset follows w (%) |
|---|---:|---:|---:|
| Clean reset | 3500 | -0.002 | 8.5 |
| Neutral mention | 3500 | 0.206 | 31.3 |
| Explicit false mention | 3500 | 0.041 | 13.4 |
| Quoted false claim | 3500 | 0.067 | 16.0 |
| User pressure | 3500 | 0.431 | 53.2 |

### truthfulqa_mc

| condition | n | gap | reset follows w (%) |
|---|---:|---:|---:|
| Clean reset | 3500 | -0.002 | 11.3 |
| Neutral mention | 3500 | 0.123 | 24.1 |
| Explicit false mention | 3500 | 0.103 | 22.3 |
| Quoted false claim | 3500 | 0.026 | 14.1 |
| User pressure | 3500 | 0.175 | 29.1 |

## Artifacts

- `pressure_vs_false_context_item_rows.csv`
- `pressure_vs_false_context_pair_summary.csv`
- `pressure_vs_false_context_model_dataset_detail.csv`
- `pressure_vs_false_context_dataset_summary.csv`
- `pressure_vs_false_context_aggregate_summary.csv`
- `pressure_vs_false_context_table.tex`
- `pressure_vs_false_context_artifact_manifest.md`
