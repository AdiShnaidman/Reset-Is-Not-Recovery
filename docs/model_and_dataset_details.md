# Model And Dataset Details

## Models In Released Full14 Analyses

- `qwen25_15b`: Qwen2.5-1.5B-Instruct
- `qwen25_7b`: Qwen2.5-7B-Instruct
- `qwen25_14b`: Qwen2.5-14B-Instruct
- `mistral7b_v03`: Mistral-7B-Instruct-v0.3
- `llama31_8b`: Llama-3.1-8B-Instruct
- `gemma2_2b`: Gemma-2-2B-IT
- `gemma2_9b`: Gemma-2-9B-IT

No 27B, 32B, 70B, or 72B model artifacts are included in the released table set.

## Datasets

- TruthfulQA multiple choice, public dataset handle `truthfulqa/truthful_qa`.
- MMLU-Pro, public dataset handle `TIGER-Lab/MMLU-Pro`.

Each main dataset subset uses `n=500` and `seed=42`. The exact released item IDs and answer metadata are in `data/item_subsets/` and `data/advocated_wrong_answers/`.

## Scoring Protocol

The original model runs use option-label likelihood scoring over displayed answer options. The scorer handles bare and space-prefixed option-letter tokenizations, normalizes probabilities over the displayed option labels, and reports entropy and max-probability diagnostics where relevant. Gemma chat templates use the documented system-role fallback that folds the system prompt into the final user message.
