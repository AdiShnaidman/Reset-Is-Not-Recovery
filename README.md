# Reset Is Not Recovery

**Evaluating Recoverability from False Conversational Context via Sycophancy Hysteresis**

Adi Shnaidman

Accepted at **GroundLM, an EMNLP 2026 Workshop**.

**Paper:** [arXiv](https://arxiv.org/abs/2609.33672)  
**Project page:** [Project Page](https://adishnaidman.github.io/projects/reset-is-not-recovery/)  
**Citation:** [BibTeX](#citation)

---

## Main Finding

An explicit instruction to *reset* a conversation does not necessarily restore an LLM to its original behavior after the user has applied false conversational pressure.

We study this phenomenon as **sycophancy hysteresis**: behavioral effects of false conversational context that persist even after the original pressure is withdrawn, contradicted, or explicitly reset.

Across controlled factual-dialogue experiments, we find a sharp distinction between:

- **history-preserving recovery operations**, which often fail to restore the model's clean behavior; and
- **removing the pressure-bearing conversational history**, which produces substantially stronger recovery.

This suggests that correcting the visible instruction is not always equivalent to neutralizing the influence of the conversational state that produced the error.

---

## Why This Matters

Most evaluations of LLM sycophancy ask whether a model follows incorrect user pressure.

This work asks a different question:

**What happens after the pressure stops?**

For reliable multi-turn systems, detecting an initial failure is only part of the problem. A model may continue to behave differently even after receiving a correction, reset instruction, or new evidence.

The evaluation therefore separates:

1. susceptibility to false conversational pressure;
2. persistence after that pressure;
3. recoverability under different interventions; and
4. restoration of the original clean behavior.

The repository provides a reusable evaluation scaffold for studying these post-pressure dynamics in factual dialogue.

---

## Repository

This repository contains the reproducibility artifacts for:

**Reset Is Not Recovery: Evaluating Recoverability from False Conversational Context via Sycophancy Hysteresis**

The released artifacts include prompt templates, item subsets, advocated wrong-answer selections, aggregate tables, analysis code, and reproducibility checks for the reported metrics.

This is a reproducibility repository rather than a polished general-purpose Python package. The lightweight modules in `src/` document reusable pieces of the protocol, while `src/original_research_scripts/` preserves the research scripts used to build the released aggregates.

## Included Artifacts

- Prompt templates for clean questions, pressure turns, R0-R7 recovery operations, false-context controls, and relabel diagnostics.
- Dataset metadata with exact item IDs, source row IDs where available, correct answers, and advocated wrong-answer selections.
- Aggregate CSVs and LaTeX tables for main and appendix results.
- Public table reproduction scripts that regenerate stable table CSVs from included saved aggregate artifacts.
- Source manifests, checksums, and protocol audit notes.

## Not Included

- Model weights or tokenizer/model caches.
- API keys, tokens, credentials, hostnames, local cluster paths, or private infrastructure logs.
- Full private run directories and broken exploratory runs.
- Fresh model-inference outputs beyond the saved aggregate and item-level artifacts needed for the reported metrics.

## Installation

For table reproduction from saved artifacts, Python 3.10+ and the standard library are sufficient. To reuse the scoring scaffold for new model runs, install the optional dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

The experiments use public benchmark datasets through their standard providers:

- TruthfulQA multiple choice: `truthfulqa/truthful_qa`
- MMLU-Pro: `TIGER-Lab/MMLU-Pro`

The released metadata in `data/item_subsets/` and `data/advocated_wrong_answers/` records the exact `n=500`, `seed=42` item selections and advocated wrong-answer choices used by the paper. See `data/README.md` for details.

## Reproduce Main Tables

The scripts below regenerate the public table CSVs from included aggregate artifacts and verify exact byte-level equality with the expected release CSVs.

```bash
scripts/reproduce_table2_pressure_ramp.sh
scripts/reproduce_table3_controls.sh
scripts/reproduce_table4_recovery_counts.sh
scripts/reproduce_table5_truncation_boundary.sh
scripts/reproduce_table6_r7_evidence.sh
scripts/reproduce_appendix_tables.sh
```

Run everything with:

```bash
scripts/reproduce_all_tables.sh
```

Expected outputs:

- `results/main_tables/table2_pressure_ramp.csv`
- `results/main_tables/table3_controls.csv`
- `results/main_tables/table4_recovery_counts.csv`
- `results/main_tables/table5_truncation_boundary.csv`
- `results/main_tables/table6_r7_evidence.csv`
- `results/appendix_tables/pressure_vs_false_context_aggregate_summary.csv`
- `results/appendix_tables/wrong_answer_plausibility_aggregate_tertile.csv`
- `results/appendix_tables/non_history_clearing_relabel_restoration_metrics.csv`
- `results/appendix_tables/non_history_clearing_relabel_outcome_metrics.csv`
- `results/appendix_tables/non_history_clearing_relabel_semantic_letter.csv`

## Runtime

Regenerating tables from saved artifacts should take seconds on a laptop. Fresh model inference is intentionally not part of the default release workflow and can require GPU hardware. See `docs/expected_runtime.md`.

## Reproducibility Checklist

See `docs/reproducibility_checklist.md` for commands run, expected output paths, and known limitations.

## Citation

If you use this work, please cite:

```bibtex
@article{shnaidman2026reset,
  title={Reset Is Not Recovery: Evaluating Recoverability from False Conversational Context via Sycophancy Hysteresis},
  author={Shnaidman, Adi},
  journal={arXiv preprint arXiv:2609.33672},
  year={2026}
}
```

The repository also includes machine-readable citation metadata in [`CITATION.cff`](CITATION.cff).

## License

This repository is released under the MIT License. See `LICENSE`.
