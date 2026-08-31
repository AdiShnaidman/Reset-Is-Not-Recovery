# Artifact Notes

The release is organized around behavior-preserving reproducibility from saved artifacts. Default reproduction scripts regenerate paper table CSVs from included aggregate CSVs and verify exact equality against expected release outputs.

## Main Result Roots

- `results/main_tables/`: stable public-facing CSVs for main paper tables.
- `results/raw_aggregates/`: aggregate artifacts used to create the stable table CSVs.
- `results/appendix_tables/`: appendix-facing CSVs and LaTeX table artifacts.
- `prompts/`: prompt templates and control prompts.
- `data/`: item subsets and advocated wrong-answer selections.
- `docs/internal_audit/`: source manifests, checksum reports, and protocol audit notes.

The advocated-wrong plausibility appendix table is included as an aggregate reproduction target in `results/appendix_tables/wrong_answer_plausibility_aggregate_tertile.csv`, with by-pair source rows under `results/raw_aggregates/wrong_answer_plausibility/`.

The non-history-clearing relabel diagnostics reported in the appendix are included as compact CSV and LaTeX artifacts under `results/appendix_tables/`, with source rows under `results/raw_aggregates/non_history_clearing_relabel/`.

## Provenance Caveats

The pressure-vs-false-context full14 artifact set uses newly scored false-context controls and imports the `user_pressure` baseline from saved recovery-hierarchy ordinary-reset rows. The protocol audit in `docs/internal_audit/protocol_uniformity_audit.md` documents the uniformity check and this provenance caveat.

The original eight pressure-vs-false-context pairs were generated under an earlier output root and staged into the full14 artifact set; the expanded six pairs were generated directly under the full14 root. The seed, item subsets, condition list, reset prompt, scoring settings, and git commit match across all 14 configs.

## Exclusions

The release excludes model caches, API credentials, private logs, full exploratory run trees, hostnames, and local infrastructure references wherever possible. Exploratory open-ended work is not included as primary evidence.

The following oversized item-level diagnostic files were intentionally excluded from the GitHub release while retaining the aggregate CCTP and pressure-control tables used by the paper:

- `results/raw_aggregates/cctp/combined_alpha_diagnostics.csv`
- `results/raw_aggregates/cctp/combined_item_level.csv`
- `results/raw_aggregates/pressure_false_context/pressure_vs_false_context_item_rows.csv`

These files are useful for deep audit work but are not required for the default table reproduction scripts included in this release.
