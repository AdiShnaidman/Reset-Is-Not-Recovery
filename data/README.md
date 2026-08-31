# Data Metadata

This directory contains metadata needed to reproduce the paper's reported item selections and advocated wrong-answer choices without relying on private run directories.

## Contents

- `item_subsets/mmlu_pro_n500_seed42.csv`
- `item_subsets/truthfulqa_mc_n500_seed42.csv`
- `advocated_wrong_answers/mmlu_pro_advocated_wrong_seed42.csv`
- `advocated_wrong_answers/truthfulqa_mc_advocated_wrong_seed42.csv`
- `evidence_metadata/r7_pair_completion_status.csv`

## Selection Rule

The released main experiments use `seed=42` and `max_items=500` per dataset. The item metadata records stable question IDs, original/source row IDs where available, correct answer label/text, and the advocated wrong-answer label/text.

## Dataset Access

Fetch source datasets through their public providers and terms:

- TruthfulQA multiple choice: `truthfulqa/truthful_qa`
- MMLU-Pro: `TIGER-Lab/MMLU-Pro`

The release preserves the item IDs and answer metadata needed for reproducibility. It does not include model weights, private caches, credentials, or local run logs.
