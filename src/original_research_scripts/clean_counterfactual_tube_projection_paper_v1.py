#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EPS = 1e-12
DEFAULT_MODELS = [
    "qwen25_15b",
    "qwen25_7b",
    "qwen25_14b",
    "mistral_7b",
    "llama31_8b",
    "gemma2_2b",
    "gemma2_9b",
]
MODEL_SHORT_NAMES = {
    "qwen25_15b": "Qwen2.5-1.5B",
    "qwen25_7b": "Qwen2.5-7B",
    "qwen25_14b": "Qwen2.5-14B",
    "mistral_7b": "Mistral-7B",
    "llama31_8b": "Llama-3.1-8B",
    "gemma2_2b": "Gemma-2-2B",
    "gemma2_9b": "Gemma-2-9B",
}
DATASET_NAMES = {"truthfulqa_mc": "TruthfulQA-MC", "mmlu_pro": "MMLU-Pro"}
DEFAULT_INPUT_ROOTS = {
    "source": "runs/self_calibrated_mirror_extrapolation_v1",
}
GENERATED_LOGIT_ROOT = "runs/cctp_required_logit_artifacts_v1"
GENERATED_WIDE_REQUIRED_COLUMNS = [
    "model_key",
    "model_short_name",
    "dataset",
    "item_id",
    "correct_label",
    "advocated_wrong_label",
    "clean_option_log_scores_json",
    "clean_option_probs_json",
    "ordinary_option_log_scores_json",
    "ordinary_option_probs_json",
    "correct_pressure_option_log_scores_json",
    "correct_pressure_option_probs_json",
]
REQUIRED_SOURCE_COLUMNS = [
    "model_key",
    "model_short_name",
    "dataset",
    "item_id",
    "condition",
    "pressure_type",
    "correct_label",
    "advocated_wrong_label",
    "actual_advocated_label",
    "prediction",
    "p_correct",
    "p_wrong",
    "accuracy",
    "wrong_follow",
    "entropy",
    "max_prob",
    "hysteresis_gap",
    "flattening_flag",
    "option_log_scores_json",
    "option_probs_json",
    "clean_option_log_scores_json",
    "clean_option_probs_json",
]
ROOT_OUTPUTS = [
    "combined_item_level.csv",
    "combined_summary.csv",
    "combined_correct_pressure_control.csv",
    "combined_endpoint_comparison.csv",
    "combined_alpha_diagnostics.csv",
    "combined_tube_diagnostics.csv",
    "combined_binding_constraints.csv",
    "combined_tube_sensitivity.csv",
    "combined_path_ablation.csv",
    "combined_alpha_hysteresis_correlations.csv",
    "combined_bootstrap_cis.csv",
    "combined_alpha_bucket_summary.csv",
    "combined_pair_status.json",
    "artifact_checks.csv",
    "decision.json",
    "report.md",
]
PER_PAIR_OUTPUTS = [
    "cctp_item_level.csv",
    "cctp_summary.csv",
    "cctp_correct_pressure_control.csv",
    "cctp_alpha_diagnostics.csv",
    "cctp_tube_diagnostics.csv",
    "cctp_binding_constraints.csv",
    "cctp_endpoint_comparison.csv",
    "cctp_report.md",
]
PLOTS = [
    "alpha_cdf_by_dataset.png",
    "alpha_hist_by_dataset.png",
    "alpha_cdf_by_model_dataset.png",
    "binding_constraints_by_dataset.png",
    "binding_constraints_by_model_dataset.png",
    "alpha_vs_reset_p_wrong.png",
    "alpha_vs_reset_kl.png",
    "pair_level_mean_alpha_vs_r0_gap.png",
]
EMPTY_ITEM_COLUMNS = [
    "model_key", "model", "model_short_name", "dataset", "dataset_name", "item_id", "condition", "pressure_type",
    "tube_type", "path_type", "alpha_grid_step", "correct_label", "advocated_wrong_label", "actual_advocated_label",
    "clean_prediction", "full_history_prediction", "projected_prediction", "prediction", "alpha_star", "history_retention",
    "alpha_bucket", "p_correct", "p_wrong", "p_advocated", "accuracy", "wrong_follow", "entropy", "max_prob",
    "flattening_flag", "clean_like_pass", "clean_context_like", "clean_context_like_vs_clean", "top1_clean_match",
    "kl_clean_projected", "kl_projected_full_history", "kl_clean_full_history", "entropy_clean", "entropy_full_history",
    "entropy_projected", "entropy_delta_vs_clean", "max_prob_clean", "max_prob_full_history", "max_prob_projected",
    "max_prob_delta_vs_clean", "p_clean_correct", "p_full_history_correct", "p_projected_correct", "p_clean_wrong",
    "p_full_history_wrong", "p_projected_wrong", "binding_constraint_primary", "binding_constraints_json", "option_labels_json",
    "clean_logits_json", "full_history_logits_json", "projected_logits_json", "clean_probs_json", "full_history_probs_json",
    "projected_probs_json", "tube_pass_alpha_0", "tube_pass_alpha_1", "implementation_guarantee_passed",
    "source_artifact_path", "runtime_seconds", "R0_gap", "reset_wrong_follow", "reset_p_wrong",
    "reset_entropy_delta_vs_clean", "reset_max_prob_delta_vs_clean", "advocated_correct_label", "p_advocated_correct",
    "advocated_correct_suppressed", "accuracy_vs_clean_delta", "p_correct_vs_clean_delta",
]
EMPTY_ALPHA_COLUMNS = [
    "model_key", "model", "dataset", "item_id", "pressure_type", "condition", "alpha", "tube_pass",
    "top1_clean_match", "kl_clean_projected", "entropy_delta_vs_clean", "max_prob_delta_vs_clean",
    "p_wrong", "p_correct", "wrong_follow",
]
EMPTY_TUBE_COLUMNS = [
    "model_key", "model", "dataset", "item_id", "pressure_type", "condition", "tube_type", "alpha_star",
    "tube_pass_alpha_0", "tube_pass_alpha_1", "implementation_guarantee_passed", "binding_constraint_primary",
    "binding_constraints_json",
]
EMPTY_BINDING_COLUMNS = [
    "model", "model_key", "dataset", "item_id", "pressure_type", "condition", "alpha_star",
    "binding_constraint_primary", "binding_constraints_json", "top1_binding", "p_top_clean_binding",
    "entropy_binding", "max_prob_binding", "kl_binding",
]
EMPTY_CORRECT_PRESSURE_COLUMNS = [
    "model_key", "model", "dataset", "wrong_pressure_mean_alpha", "correct_pressure_mean_alpha",
    "wrong_pressure_median_alpha", "correct_pressure_median_alpha", "wrong_pressure_history_retention",
    "correct_pressure_history_retention", "correct_pressure_clean_like_rate", "correct_pressure_p_correct",
    "correct_pressure_accuracy", "correct_pressure_flattening_rate",
]
TUBE_DEFS = {
    "strict": {"p_slack": 0.03, "entropy_slack": 0.10, "max_slack": 0.10, "kl": 0.05},
    "main": {"p_slack": 0.05, "entropy_slack": 0.15, "max_slack": 0.15, "kl": 0.10},
    "loose": {"p_slack": 0.10, "entropy_slack": 0.20, "max_slack": 0.20, "kl": 0.20},
}
BINDING_PRIORITY = ["top1_clean_match", "p_top_clean", "KL", "entropy", "max_prob"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CCTP paper package: minimal clean-anchor recovery"
    )
    parser.add_argument("--model-keys", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=["truthfulqa_mc", "mmlu_pro"])
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--input-roots", nargs="*", default=[])
    parser.add_argument(
        "--output-root",
        default="runs/clean_counterfactual_tube_projection_paper_v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    parser.add_argument("--tubes", nargs="+", default=["main", "strict", "loose"])
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["logit_linear", "prob_linear", "geometric_prob", "kl_projection"],
    )
    parser.add_argument("--include-correct-pressure", action="store_true", default=True)
    parser.add_argument("--include-endpoints", action="store_true", default=True)
    parser.add_argument("--include-bootstrap", action="store_true", default=True)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--include-naturalistic", action="store_true", default=False)
    parser.add_argument("--include-qualitative", action="store_true", default=False)
    parser.add_argument("--source-mode", default="artifacts_only")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--skip-optional", action="store_true", default=False)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_input_roots(raw_values: Sequence[str]) -> Dict[str, Path]:
    generated_root = resolve_path(GENERATED_LOGIT_ROOT)
    if not raw_values and (generated_root / "combined_item_level.csv").is_file():
        roots = {"source": generated_root}
    else:
        roots = {name: resolve_path(path) for name, path in DEFAULT_INPUT_ROOTS.items()}
    for value in raw_values:
        if "=" not in value:
            roots["source"] = resolve_path(value)
            continue
        name, raw_path = value.split("=", 1)
        roots[name] = resolve_path(raw_path)
    return roots


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    if output_root.exists():
        existing = [name for name in ROOT_OUTPUTS if (output_root / name).exists()]
        if existing:
            raise SystemExit(
                f"Refusing to overwrite existing paper CCTP outputs in {output_root}: "
                f"{existing}. Pass --overwrite to replace."
            )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "plots").mkdir(parents=True, exist_ok=True)


def parse_json_dict(value: Any) -> Dict[str, float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out = {}
    for key, raw in parsed.items():
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            out[str(key)] = val
    return out


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def center(scores: Dict[str, float]) -> Dict[str, float]:
    vals = [v for v in scores.values() if np.isfinite(v)]
    if not vals:
        return {label: np.nan for label in scores}
    mean_val = float(np.mean(vals))
    return {label: float(value - mean_val) for label, value in scores.items()}


def softmax(scores: Dict[str, float]) -> Dict[str, float]:
    finite = {label: value for label, value in scores.items() if np.isfinite(value)}
    if not finite:
        return {label: np.nan for label in scores}
    anchor = max(finite.values())
    exp_vals = {label: math.exp(value - anchor) for label, value in finite.items()}
    denom = max(sum(exp_vals.values()), EPS)
    return {label: float(exp_vals.get(label, 0.0) / denom) for label in scores}


def normalize(probs: Dict[str, float]) -> Dict[str, float]:
    clean = {label: max(safe_float(value, 0.0), 0.0) for label, value in probs.items()}
    total = sum(clean.values())
    if total <= 0:
        return {label: np.nan for label in probs}
    return {label: float(value / total) for label, value in clean.items()}


def entropy(probs: Iterable[float]) -> float:
    vals = [max(safe_float(value, 0.0), EPS) for value in probs]
    total = sum(vals)
    if total <= 0:
        return np.nan
    vals = [value / total for value in vals]
    return float(-sum(value * math.log(max(value, EPS)) for value in vals))


def kl(p: Dict[str, float], q: Dict[str, float]) -> float:
    labels = [label for label in sorted(p) if label in q]
    if not labels:
        return np.nan
    return float(
        sum(
            max(p[label], EPS) * (math.log(max(p[label], EPS)) - math.log(max(q[label], EPS)))
            for label in labels
        )
    )


def top_label(values: Dict[str, float]) -> str:
    finite = {label: value for label, value in values.items() if np.isfinite(value)}
    return str(max(finite, key=finite.get)) if finite else ""


def alpha_values(step: float) -> List[float]:
    if step <= 0 or step > 1:
        raise ValueError("--alpha-step must be in (0, 1]")
    values = [round(float(value), 10) for value in np.arange(0.0, 1.0 + step / 2, step)]
    return sorted(set([value for value in values if value <= 1.0] + [1.0]))


def json_float(values: Dict[str, float]) -> str:
    return json.dumps(
        {str(k): (None if not np.isfinite(v) else float(v)) for k, v in sorted(values.items())},
        sort_keys=True,
    )


def alpha_bucket(alpha: float) -> str:
    if abs(alpha) <= 1e-12:
        return "alpha_000"
    if alpha <= 0.10:
        return "alpha_001_010"
    if alpha <= 0.25:
        return "alpha_011_025"
    if alpha <= 0.50:
        return "alpha_026_050"
    if alpha <= 0.75:
        return "alpha_051_075"
    if alpha < 1.0:
        return "alpha_076_099"
    return "alpha_100"


def load_source(root: Path) -> Tuple[pd.DataFrame, List[str]]:
    path = root / "combined_item_level.csv"
    if not path.is_file():
        return pd.DataFrame(), [f"missing artifact: {path}"]
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    if all(col in cols for col in GENERATED_WIDE_REQUIRED_COLUMNS) and "condition" not in cols:
        return load_generated_wide_source(path), []
    missing = [col for col in REQUIRED_SOURCE_COLUMNS if col not in cols]
    usecols = [col for col in REQUIRED_SOURCE_COLUMNS if col in cols]
    return pd.read_csv(path, usecols=usecols), missing


def generated_prediction(row: pd.Series, prefix: str) -> str:
    pred_cols = {
        "clean": "clean_prediction",
        "ordinary": "ordinary_reset_prediction",
        "correct_pressure": "correct_pressure_prediction",
        "fresh": "fresh_context_prediction",
        "truncation": "context_truncation_prediction",
        "relabel": "relabel_shuffle_prediction",
    }
    return str(row.get(pred_cols[prefix], ""))


def generated_condition_row(row: pd.Series, condition: str, prefix: str, pressure_type: str, path: str) -> Optional[Dict[str, Any]]:
    scores = parse_json_dict(row.get(f"{prefix}_option_log_scores_json"))
    probs = parse_json_dict(row.get(f"{prefix}_option_probs_json"))
    clean_scores = parse_json_dict(row.get("clean_option_log_scores_json"))
    clean_probs = parse_json_dict(row.get("clean_option_probs_json"))
    if not scores or not probs or not clean_scores or not clean_probs:
        return None
    correct = str(row.get("correct_label"))
    wrong = str(row.get("advocated_wrong_label"))
    actual = correct if pressure_type == "correct_pressure" else wrong
    pred = generated_prediction(row, prefix)
    if not pred:
        pred = top_label(probs)
    ent = entropy(probs.values())
    clean_ent = entropy(clean_probs.values())
    max_prob = max(probs.values())
    clean_max = max(clean_probs.values())
    return {
        "model_key": str(row.get("model_key")),
        "model_short_name": str(row.get("model_short_name", row.get("model_key"))),
        "dataset": str(row.get("dataset")),
        "item_id": str(row.get("item_id")),
        "condition": condition,
        "pressure_type": pressure_type,
        "correct_label": correct,
        "advocated_wrong_label": wrong,
        "actual_advocated_label": actual,
        "prediction": pred,
        "p_correct": safe_float(probs.get(correct)),
        "p_wrong": safe_float(probs.get(wrong)),
        "accuracy": float(pred == correct),
        "wrong_follow": float(pred == wrong),
        "entropy": ent,
        "max_prob": max_prob,
        "hysteresis_gap": safe_float(probs.get(wrong)) - safe_float(clean_probs.get(wrong)),
        "flattening_flag": float(ent > clean_ent + 0.15 or max_prob < clean_max - 0.15),
        "option_log_scores_json": json_float(scores),
        "option_probs_json": json_float(probs),
        "clean_option_log_scores_json": json_float(clean_scores),
        "clean_option_probs_json": json_float(clean_probs),
        "source_artifact_path": path,
    }


def load_generated_wide_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        if "item_valid_for_cctp" in df.columns and not bool(row.get("item_valid_for_cctp")):
            continue
        specs = [
            ("clean_context", "clean", "wrong_pressure"),
            ("ordinary_reset", "ordinary", "wrong_pressure"),
            ("correct_pressure_ordinary_reset", "correct_pressure", "correct_pressure"),
            ("fresh_context_deletion", "fresh", "wrong_pressure"),
            ("relabel_shuffle", "relabel", "wrong_pressure"),
            ("context_truncation", "truncation", "wrong_pressure"),
        ]
        for condition, prefix, pressure_type in specs:
            built = generated_condition_row(row, condition, prefix, pressure_type, str(path))
            if built is not None:
                rows.append(built)
    return pd.DataFrame(rows, columns=REQUIRED_SOURCE_COLUMNS + ["source_artifact_path"])


def source_lookup(df: pd.DataFrame) -> Dict[Tuple[str, str, str, str], pd.Series]:
    return {
        (str(row.model_key), str(row.dataset), str(row.item_id), str(row.condition)): row
        for row in df.itertuples(index=False)
    }


def model_dataset_status(
    source_df: pd.DataFrame,
    model_keys: Sequence[str],
    datasets: Sequence[str],
    n_items: int,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], List[str]]]:
    rows = []
    selected: Dict[Tuple[str, str], List[str]] = {}
    for model_key in model_keys:
        for dataset in datasets:
            pair = source_df[
                (source_df["model_key"].astype(str) == model_key)
                & (source_df["dataset"].astype(str) == dataset)
            ]
            ordinary = pair[pair["condition"].astype(str) == "ordinary_reset"]
            correct = pair[pair["condition"].astype(str) == "correct_pressure_ordinary_reset"]
            clean_ok = ordinary["clean_option_log_scores_json"].map(parse_json_dict).map(bool) if not ordinary.empty else pd.Series(dtype=bool)
            full_ok = ordinary["option_log_scores_json"].map(parse_json_dict).map(bool) if not ordinary.empty else pd.Series(dtype=bool)
            ordinary_items = ordinary.loc[clean_ok & full_ok, "item_id"].astype(str).drop_duplicates().tolist()
            correct_clean_ok = correct["clean_option_log_scores_json"].map(parse_json_dict).map(bool) if not correct.empty else pd.Series(dtype=bool)
            correct_full_ok = correct["option_log_scores_json"].map(parse_json_dict).map(bool) if not correct.empty else pd.Series(dtype=bool)
            correct_items = set(correct.loc[correct_clean_ok & correct_full_ok, "item_id"].astype(str).tolist())
            usable = [item for item in ordinary_items if item in correct_items]
            selected[(model_key, dataset)] = usable[:n_items]
            status = "ok" if len(usable) >= n_items else "missing_logits_or_insufficient_n"
            rows.append(
                {
                    "model_key": model_key,
                    "model": MODEL_SHORT_NAMES.get(model_key, model_key),
                    "dataset": dataset,
                    "dataset_name": DATASET_NAMES.get(dataset, dataset),
                    "n_requested": n_items,
                    "n_available_wrong_pressure": len(ordinary_items),
                    "n_available_correct_pressure": len(correct_items),
                    "n_usable_both_pressures": len(usable),
                    "status": status,
                    "missing_count": max(n_items - len(usable), 0),
                }
            )
    return pd.DataFrame(rows), selected


def distributions(row: pd.Series) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    clean_logits = parse_json_dict(row.get("clean_option_log_scores_json"))
    clean_probs = parse_json_dict(row.get("clean_option_probs_json"))
    full_logits = parse_json_dict(row.get("option_log_scores_json"))
    full_probs = parse_json_dict(row.get("option_probs_json"))
    if not clean_probs:
        clean_probs = softmax(clean_logits)
    if not full_probs:
        full_probs = softmax(full_logits)
    labels = sorted(set(clean_logits) & set(clean_probs) & set(full_logits) & set(full_probs))
    clean_logits = center({label: clean_logits[label] for label in labels})
    full_logits = center({label: full_logits[label] for label in labels})
    clean_probs = normalize({label: clean_probs[label] for label in labels})
    full_probs = normalize({label: full_probs[label] for label in labels})
    return clean_logits, clean_probs, full_logits, full_probs


def project(
    clean_logits: Dict[str, float],
    clean_probs: Dict[str, float],
    full_logits: Dict[str, float],
    full_probs: Dict[str, float],
    alpha: float,
    path: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    labels = sorted(clean_probs)
    if alpha <= 1e-12:
        return center({label: full_logits[label] for label in labels}), {label: full_probs[label] for label in labels}
    if alpha >= 1.0 - 1e-12:
        return center({label: clean_logits[label] for label in labels}), {label: clean_probs[label] for label in labels}
    if path == "prob_linear":
        probs = normalize({label: (1 - alpha) * full_probs[label] + alpha * clean_probs[label] for label in labels})
        return center({label: math.log(max(probs[label], EPS)) for label in labels}), probs
    if path == "geometric_prob":
        scores = {
            label: (1 - alpha) * math.log(max(full_probs[label], EPS))
            + alpha * math.log(max(clean_probs[label], EPS))
            for label in labels
        }
        return center(scores), softmax(scores)
    scores = {label: (1 - alpha) * full_logits[label] + alpha * clean_logits[label] for label in labels}
    scores = center(scores)
    return scores, softmax(scores)


def tube_eval(clean_probs: Dict[str, float], proj_probs: Dict[str, float], tube: str, kl_only_tau: Optional[float] = None) -> Dict[str, Any]:
    clean_top = top_label(clean_probs)
    proj_top = top_label(proj_probs)
    ent_clean = entropy(clean_probs.values())
    ent_proj = entropy(proj_probs.values())
    max_clean = max(clean_probs.values())
    max_proj = max(proj_probs.values())
    kl_val = kl(clean_probs, proj_probs)
    if kl_only_tau is not None:
        checks = {
            "top1_clean_match": True,
            "p_top_clean": True,
            "entropy": True,
            "max_prob": True,
            "KL": kl_val <= kl_only_tau + 1e-10,
        }
        thresholds = {"p_slack": np.nan, "entropy_slack": np.nan, "max_slack": np.nan, "kl": kl_only_tau}
    else:
        thresholds = TUBE_DEFS[tube]
        checks = {
            "top1_clean_match": proj_top == clean_top,
            "p_top_clean": safe_float(proj_probs.get(clean_top))
            >= safe_float(clean_probs.get(clean_top)) - thresholds["p_slack"],
            "entropy": ent_proj <= ent_clean + thresholds["entropy_slack"],
            "max_prob": max_proj >= max_clean - thresholds["max_slack"],
            "KL": kl_val <= thresholds["kl"] + 1e-10,
        }
    return {
        "tube_pass": bool(all(checks.values())),
        "checks": checks,
        "kl_clean_projected": kl_val,
        "entropy_delta_vs_clean": ent_proj - ent_clean,
        "max_prob_delta_vs_clean": max_proj - max_clean,
        "p_projected_top_clean": safe_float(proj_probs.get(clean_top)),
        "p_top_clean": safe_float(clean_probs.get(clean_top)),
        "thresholds": thresholds,
    }


def binding_from_checks(checks: Dict[str, bool]) -> Tuple[str, List[str]]:
    failed = [name for name, passed in checks.items() if not passed]
    if not failed:
        return "none", []
    for name in BINDING_PRIORITY:
        if name in failed:
            return name, failed
    return failed[0], failed


def condition_specs(paths: Sequence[str], tubes: Sequence[str]) -> List[Dict[str, Any]]:
    specs = []
    if "logit_linear" in paths and "main" in tubes:
        specs.append({"condition": "cctp_logit_linear_min_alpha", "path": "logit_linear", "tube": "main", "kl_tau": None})
        specs.append({"condition": "cctp_tube_main", "path": "logit_linear", "tube": "main", "kl_tau": None})
    if "strict" in tubes:
        specs.append({"condition": "cctp_tube_strict", "path": "logit_linear", "tube": "strict", "kl_tau": None})
    if "loose" in tubes:
        specs.append({"condition": "cctp_tube_loose", "path": "logit_linear", "tube": "loose", "kl_tau": None})
    if "prob_linear" in paths:
        specs.append({"condition": "cctp_prob_linear_min_alpha", "path": "prob_linear", "tube": "main", "kl_tau": None})
    if "geometric_prob" in paths:
        specs.append({"condition": "cctp_geometric_prob_min_alpha", "path": "geometric_prob", "tube": "main", "kl_tau": None})
    if "kl_projection" in paths:
        specs.append({"condition": "cctp_kl_projection_tau_005", "path": "logit_linear", "tube": "kl", "kl_tau": 0.05})
        specs.append({"condition": "cctp_kl_projection_tau_010", "path": "logit_linear", "tube": "kl", "kl_tau": 0.10})
    seen = set()
    out = []
    for spec in specs:
        if spec["condition"] not in seen:
            seen.add(spec["condition"])
            out.append(spec)
    return out


def run_one_condition(
    row: pd.Series,
    spec: Dict[str, Any],
    pressure_type: str,
    alphas: Sequence[float],
    source_path: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    clean_logits, clean_probs, full_logits, full_probs = distributions(row)
    correct = str(row.get("correct_label"))
    wrong = str(row.get("advocated_wrong_label"))
    actual = wrong if pressure_type == "wrong_pressure" else correct
    alpha_rows: List[Dict[str, Any]] = []
    selected: Optional[Tuple[float, Dict[str, float], Dict[str, float], Dict[str, Any], Dict[str, bool]]] = None
    prev_failed: Dict[str, bool] = {name: True for name in BINDING_PRIORITY}
    for alpha in alphas:
        scores, probs = project(clean_logits, clean_probs, full_logits, full_probs, alpha, spec["path"])
        result = tube_eval(clean_probs, probs, spec["tube"], spec.get("kl_tau"))
        pred = top_label(probs)
        alpha_rows.append(
            {
                "model_key": str(row.get("model_key")),
                "model": MODEL_SHORT_NAMES.get(str(row.get("model_key")), str(row.get("model_key"))),
                "dataset": str(row.get("dataset")),
                "item_id": str(row.get("item_id")),
                "pressure_type": pressure_type,
                "condition": spec["condition"],
                "alpha": alpha,
                "tube_pass": result["tube_pass"],
                "top1_clean_match": result["checks"].get("top1_clean_match", True),
                "kl_clean_projected": result["kl_clean_projected"],
                "entropy_delta_vs_clean": result["entropy_delta_vs_clean"],
                "max_prob_delta_vs_clean": result["max_prob_delta_vs_clean"],
                "p_wrong": safe_float(probs.get(wrong)),
                "p_correct": safe_float(probs.get(correct)),
                "wrong_follow": float(pred == wrong),
            }
        )
        if result["tube_pass"] and selected is None:
            selected = (alpha, scores, probs, result, prev_failed)
        if not result["tube_pass"]:
            prev_failed = result["checks"]
    one_scores, one_probs = project(clean_logits, clean_probs, full_logits, full_probs, 1.0, spec["path"])
    one_eval = tube_eval(clean_probs, one_probs, spec["tube"], spec.get("kl_tau"))
    guarantee = bool(
        one_eval["tube_pass"]
        and top_label(one_probs) == top_label(clean_probs)
        and abs(kl(clean_probs, one_probs)) <= 1e-8
        and abs(sum(one_probs.values()) - 1.0) <= 1e-8
        and all(np.isfinite(list(one_probs.values())))
    )
    if not guarantee:
        raise RuntimeError(f"alpha=1 guarantee failed for {row.get('model_key')} {row.get('dataset')} {row.get('item_id')} {spec['condition']}")
    if selected is None:
        raise RuntimeError("No passing alpha found despite alpha=1 guarantee")
    alpha_star, proj_scores, proj_probs, tube_result, prev_checks = selected
    binding_primary, binding_list = binding_from_checks(prev_checks)
    clean_pred = top_label(clean_probs)
    full_pred = top_label(full_probs)
    proj_pred = top_label(proj_probs)
    ent_clean = entropy(clean_probs.values())
    ent_full = entropy(full_probs.values())
    ent_proj = entropy(proj_probs.values())
    max_clean = max(clean_probs.values())
    max_full = max(full_probs.values())
    max_proj = max(proj_probs.values())
    base = {
        "model_key": str(row.get("model_key")),
        "model": MODEL_SHORT_NAMES.get(str(row.get("model_key")), str(row.get("model_key"))),
        "model_short_name": str(row.get("model_short_name", MODEL_SHORT_NAMES.get(str(row.get("model_key")), str(row.get("model_key"))))),
        "dataset": str(row.get("dataset")),
        "dataset_name": DATASET_NAMES.get(str(row.get("dataset")), str(row.get("dataset"))),
        "item_id": str(row.get("item_id")),
        "condition": spec["condition"],
        "pressure_type": pressure_type,
        "tube_type": spec["tube"],
        "path_type": spec["path"],
        "alpha_grid_step": alphas[1] - alphas[0] if len(alphas) > 1 else 1.0,
        "correct_label": correct,
        "advocated_wrong_label": wrong,
        "actual_advocated_label": actual,
        "clean_prediction": clean_pred,
        "full_history_prediction": full_pred,
        "projected_prediction": proj_pred,
        "prediction": proj_pred,
        "alpha_star": alpha_star,
        "history_retention": 1.0 - alpha_star,
        "alpha_bucket": alpha_bucket(alpha_star),
        "p_correct": safe_float(proj_probs.get(correct)),
        "p_wrong": safe_float(proj_probs.get(wrong)),
        "p_advocated": safe_float(proj_probs.get(actual)),
        "accuracy": float(proj_pred == correct),
        "wrong_follow": float(proj_pred == wrong),
        "entropy": ent_proj,
        "max_prob": max_proj,
        "flattening_flag": float(ent_proj > ent_clean + 0.15 or max_proj < max_clean - 0.15),
        "clean_like_pass": tube_result["tube_pass"],
        "clean_context_like": tube_result["tube_pass"],
        "clean_context_like_vs_clean": tube_result["tube_pass"],
        "top1_clean_match": proj_pred == clean_pred,
        "kl_clean_projected": tube_result["kl_clean_projected"],
        "kl_projected_full_history": kl(proj_probs, full_probs),
        "kl_clean_full_history": kl(clean_probs, full_probs),
        "entropy_clean": ent_clean,
        "entropy_full_history": ent_full,
        "entropy_projected": ent_proj,
        "entropy_delta_vs_clean": ent_proj - ent_clean,
        "max_prob_clean": max_clean,
        "max_prob_full_history": max_full,
        "max_prob_projected": max_proj,
        "max_prob_delta_vs_clean": max_proj - max_clean,
        "p_clean_correct": safe_float(clean_probs.get(correct)),
        "p_full_history_correct": safe_float(full_probs.get(correct)),
        "p_projected_correct": safe_float(proj_probs.get(correct)),
        "p_clean_wrong": safe_float(clean_probs.get(wrong)),
        "p_full_history_wrong": safe_float(full_probs.get(wrong)),
        "p_projected_wrong": safe_float(proj_probs.get(wrong)),
        "binding_constraint_primary": binding_primary,
        "binding_constraints_json": json.dumps(binding_list),
        "option_labels_json": json.dumps(sorted(clean_probs)),
        "clean_logits_json": json_float(clean_logits),
        "full_history_logits_json": json_float(full_logits),
        "projected_logits_json": json_float(proj_scores),
        "clean_probs_json": json_float(clean_probs),
        "full_history_probs_json": json_float(full_probs),
        "projected_probs_json": json_float(proj_probs),
        "tube_pass_alpha_0": bool(alpha_rows[0]["tube_pass"]),
        "tube_pass_alpha_1": bool(one_eval["tube_pass"]),
        "implementation_guarantee_passed": guarantee,
        "source_artifact_path": source_path,
        "runtime_seconds": np.nan,
        "R0_gap": safe_float(row.get("hysteresis_gap")),
        "reset_wrong_follow": float(full_pred == wrong),
        "reset_p_wrong": safe_float(full_probs.get(wrong)),
        "reset_entropy_delta_vs_clean": ent_full - ent_clean,
        "reset_max_prob_delta_vs_clean": max_full - max_clean,
    }
    if pressure_type == "correct_pressure":
        base.update(
            {
                "advocated_correct_label": correct,
                "p_advocated_correct": safe_float(proj_probs.get(correct)),
                "advocated_correct_suppressed": float(safe_float(proj_probs.get(correct)) < safe_float(clean_probs.get(correct)) - 0.05),
                "accuracy_vs_clean_delta": float(proj_pred == correct) - float(clean_pred == correct),
                "p_correct_vs_clean_delta": safe_float(proj_probs.get(correct)) - safe_float(clean_probs.get(correct)),
            }
        )
    else:
        base.update(
            {
                "advocated_correct_label": np.nan,
                "p_advocated_correct": np.nan,
                "advocated_correct_suppressed": np.nan,
                "accuracy_vs_clean_delta": np.nan,
                "p_correct_vs_clean_delta": np.nan,
            }
        )
    bind = {
        "model": base["model"],
        "model_key": base["model_key"],
        "dataset": base["dataset"],
        "item_id": base["item_id"],
        "pressure_type": pressure_type,
        "condition": spec["condition"],
        "alpha_star": alpha_star,
        "binding_constraint_primary": binding_primary,
        "binding_constraints_json": json.dumps(binding_list),
        "top1_binding": "top1_clean_match" in binding_list,
        "p_top_clean_binding": "p_top_clean" in binding_list,
        "entropy_binding": "entropy" in binding_list,
        "max_prob_binding": "max_prob" in binding_list,
        "kl_binding": "KL" in binding_list,
    }
    tube_row = {
        "model_key": base["model_key"],
        "model": base["model"],
        "dataset": base["dataset"],
        "item_id": base["item_id"],
        "pressure_type": pressure_type,
        "condition": spec["condition"],
        "tube_type": spec["tube"],
        "alpha_star": alpha_star,
        "tube_pass_alpha_0": base["tube_pass_alpha_0"],
        "tube_pass_alpha_1": base["tube_pass_alpha_1"],
        "implementation_guarantee_passed": guarantee,
        "binding_constraint_primary": binding_primary,
        "binding_constraints_json": json.dumps(binding_list),
    }
    return base, alpha_rows, tube_row, bind


def endpoint_row(row: pd.Series, condition: str, pressure_type: str, source_path: str) -> Optional[Dict[str, Any]]:
    clean_logits, clean_probs, _, _ = distributions(row)
    scores = parse_json_dict(row.get("option_log_scores_json"))
    probs = parse_json_dict(row.get("option_probs_json"))
    if condition == "fresh_context_deletion":
        scores = clean_logits
        probs = clean_probs
    if not probs:
        return None
    scores = center(scores) if scores else center({label: math.log(max(prob, EPS)) for label, prob in probs.items()})
    probs = normalize(probs)
    correct = str(row.get("correct_label"))
    wrong = str(row.get("advocated_wrong_label"))
    pred = top_label(probs)
    clean_pred = top_label(clean_probs)
    ent_clean = entropy(clean_probs.values())
    ent = entropy(probs.values())
    max_clean = max(clean_probs.values())
    max_prob = max(probs.values())
    tube = tube_eval(clean_probs, probs, "main")
    return {
        "model_key": str(row.get("model_key")),
        "model": MODEL_SHORT_NAMES.get(str(row.get("model_key")), str(row.get("model_key"))),
        "model_short_name": str(row.get("model_short_name", MODEL_SHORT_NAMES.get(str(row.get("model_key")), str(row.get("model_key"))))),
        "dataset": str(row.get("dataset")),
        "dataset_name": DATASET_NAMES.get(str(row.get("dataset")), str(row.get("dataset"))),
        "item_id": str(row.get("item_id")),
        "condition": condition,
        "pressure_type": pressure_type,
        "tube_type": "endpoint",
        "path_type": "endpoint",
        "alpha_grid_step": np.nan,
        "correct_label": correct,
        "advocated_wrong_label": wrong,
        "actual_advocated_label": wrong if pressure_type == "wrong_pressure" else correct,
        "clean_prediction": clean_pred,
        "full_history_prediction": pred,
        "projected_prediction": pred,
        "prediction": pred,
        "alpha_star": np.nan,
        "history_retention": np.nan,
        "alpha_bucket": "endpoint",
        "p_correct": safe_float(probs.get(correct)),
        "p_wrong": safe_float(probs.get(wrong)),
        "p_advocated": safe_float(probs.get(wrong if pressure_type == "wrong_pressure" else correct)),
        "accuracy": float(pred == correct),
        "wrong_follow": float(pred == wrong),
        "entropy": ent,
        "max_prob": max_prob,
        "flattening_flag": float(ent > ent_clean + 0.15 or max_prob < max_clean - 0.15),
        "clean_like_pass": tube["tube_pass"],
        "clean_context_like": tube["tube_pass"],
        "clean_context_like_vs_clean": tube["tube_pass"],
        "top1_clean_match": pred == clean_pred,
        "kl_clean_projected": kl(clean_probs, probs),
        "kl_projected_full_history": np.nan,
        "kl_clean_full_history": kl(clean_probs, probs),
        "entropy_clean": ent_clean,
        "entropy_full_history": ent,
        "entropy_projected": ent,
        "entropy_delta_vs_clean": ent - ent_clean,
        "max_prob_clean": max_clean,
        "max_prob_full_history": max_prob,
        "max_prob_projected": max_prob,
        "max_prob_delta_vs_clean": max_prob - max_clean,
        "p_clean_correct": safe_float(clean_probs.get(correct)),
        "p_full_history_correct": safe_float(probs.get(correct)),
        "p_projected_correct": safe_float(probs.get(correct)),
        "p_clean_wrong": safe_float(clean_probs.get(wrong)),
        "p_full_history_wrong": safe_float(probs.get(wrong)),
        "p_projected_wrong": safe_float(probs.get(wrong)),
        "binding_constraint_primary": "endpoint",
        "binding_constraints_json": "[]",
        "option_labels_json": json.dumps(sorted(clean_probs)),
        "clean_logits_json": json_float(clean_logits),
        "full_history_logits_json": json_float(scores),
        "projected_logits_json": json_float(scores),
        "clean_probs_json": json_float(clean_probs),
        "full_history_probs_json": json_float(probs),
        "projected_probs_json": json_float(probs),
        "tube_pass_alpha_0": np.nan,
        "tube_pass_alpha_1": np.nan,
        "implementation_guarantee_passed": True,
        "source_artifact_path": source_path,
        "runtime_seconds": np.nan,
        "R0_gap": safe_float(row.get("hysteresis_gap")),
        "reset_wrong_follow": float(pred == wrong),
        "reset_p_wrong": safe_float(probs.get(wrong)),
        "reset_entropy_delta_vs_clean": ent - ent_clean,
        "reset_max_prob_delta_vs_clean": max_prob - max_clean,
        "advocated_correct_label": correct if pressure_type == "correct_pressure" else np.nan,
        "p_advocated_correct": safe_float(probs.get(correct)) if pressure_type == "correct_pressure" else np.nan,
        "advocated_correct_suppressed": float(safe_float(probs.get(correct)) < safe_float(clean_probs.get(correct)) - 0.05) if pressure_type == "correct_pressure" else np.nan,
        "accuracy_vs_clean_delta": float(pred == correct) - float(clean_pred == correct) if pressure_type == "correct_pressure" else np.nan,
        "p_correct_vs_clean_delta": safe_float(probs.get(correct)) - safe_float(clean_probs.get(correct)) if pressure_type == "correct_pressure" else np.nan,
    }


def summarize(item_df: pd.DataFrame) -> pd.DataFrame:
    if item_df.empty:
        return empty_summary()
    rows = []
    for keys, group in item_df.groupby(["model_key", "model", "dataset", "condition", "pressure_type"], dropna=False):
        model_key, model, dataset, condition, pressure_type = keys
        alpha = pd.to_numeric(group["alpha_star"], errors="coerce")
        rows.append(
            {
                "model_key": model_key,
                "model": model,
                "dataset": dataset,
                "condition": condition,
                "pressure_type": pressure_type,
                "n_items": int(len(group)),
                "R0_gap": float(group["R0_gap"].mean()),
                "ordinary_reset_wrong_follow": float(group["reset_wrong_follow"].mean()),
                "ordinary_reset_p_wrong": float(group["reset_p_wrong"].mean()),
                "cctp_mean_alpha": float(alpha.mean()) if alpha.notna().any() else np.nan,
                "cctp_median_alpha": float(alpha.median()) if alpha.notna().any() else np.nan,
                "cctp_p25_alpha": float(alpha.quantile(0.25)) if alpha.notna().any() else np.nan,
                "cctp_p75_alpha": float(alpha.quantile(0.75)) if alpha.notna().any() else np.nan,
                "mean_history_retention": float(group["history_retention"].mean()),
                "fraction_alpha_0": float((alpha == 0.0).mean()) if alpha.notna().any() else np.nan,
                "fraction_alpha_le_025": float((alpha <= 0.25).mean()) if alpha.notna().any() else np.nan,
                "fraction_alpha_le_050": float((alpha <= 0.50).mean()) if alpha.notna().any() else np.nan,
                "fraction_alpha_eq_1": float((alpha == 1.0).mean()) if alpha.notna().any() else np.nan,
                "cctp_clean_like_rate": float(group["clean_context_like"].mean()),
                "cctp_wrong_follow": float(group["wrong_follow"].mean()),
                "cctp_p_wrong": float(group["p_wrong"].mean()),
                "cctp_accuracy": float(group["accuracy"].mean()),
                "cctp_p_correct": float(group["p_correct"].mean()),
                "cctp_flattening_rate": float(group["flattening_flag"].mean()),
                "mean_KL_clean_projected": float(group["kl_clean_projected"].mean()),
                "mean_entropy": float(group["entropy"].mean()),
                "mean_max_prob": float(group["max_prob"].mean()),
                "implementation_guarantee_pass_rate": float(group["implementation_guarantee_passed"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_key", "dataset", "condition", "pressure_type"])


def empty_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model_key", "model", "dataset", "condition", "pressure_type", "n_items", "R0_gap",
            "ordinary_reset_wrong_follow", "ordinary_reset_p_wrong", "cctp_mean_alpha", "cctp_median_alpha",
            "cctp_p25_alpha", "cctp_p75_alpha", "mean_history_retention", "fraction_alpha_0",
            "fraction_alpha_le_025", "fraction_alpha_le_050", "fraction_alpha_eq_1", "cctp_clean_like_rate",
            "cctp_wrong_follow", "cctp_p_wrong", "cctp_accuracy", "cctp_p_correct", "cctp_flattening_rate",
            "mean_KL_clean_projected", "mean_entropy", "mean_max_prob", "implementation_guarantee_pass_rate",
        ]
    )


def endpoint_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=["model", "dataset", "condition", "clean_like_rate", "wrong_follow_rate", "mean_p_wrong", "accuracy", "mean_p_correct", "mean_entropy", "mean_max_prob", "flattening_rate"])
    rows = []
    for row in summary_df.itertuples(index=False):
        rows.append(
            {
                "model_key": row.model_key,
                "model": row.model,
                "dataset": row.dataset,
                "condition": row.condition,
                "pressure_type": row.pressure_type,
                "clean_like_rate": row.cctp_clean_like_rate,
                "wrong_follow_rate": row.cctp_wrong_follow,
                "mean_p_wrong": row.cctp_p_wrong,
                "accuracy": row.cctp_accuracy,
                "mean_p_correct": row.cctp_p_correct,
                "mean_entropy": row.mean_entropy,
                "mean_max_prob": row.mean_max_prob,
                "flattening_rate": row.cctp_flattening_rate,
            }
        )
    return pd.DataFrame(rows)


def correct_pressure_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=EMPTY_CORRECT_PRESSURE_COLUMNS)
    main = summary_df[summary_df["condition"].eq("cctp_logit_linear_min_alpha")]
    wrong = main[main["pressure_type"].eq("wrong_pressure")]
    correct = main[main["pressure_type"].eq("correct_pressure")]
    merged = wrong.merge(correct, on=["model_key", "model", "dataset", "condition"], suffixes=("_wrong_pressure", "_correct_pressure"), how="outer")
    if merged.empty:
        return merged
    return pd.DataFrame(
        {
            "model_key": merged["model_key"],
            "model": merged["model"],
            "dataset": merged["dataset"],
            "wrong_pressure_mean_alpha": merged["cctp_mean_alpha_wrong_pressure"],
            "correct_pressure_mean_alpha": merged["cctp_mean_alpha_correct_pressure"],
            "wrong_pressure_median_alpha": merged["cctp_median_alpha_wrong_pressure"],
            "correct_pressure_median_alpha": merged["cctp_median_alpha_correct_pressure"],
            "wrong_pressure_history_retention": merged["mean_history_retention_wrong_pressure"],
            "correct_pressure_history_retention": merged["mean_history_retention_correct_pressure"],
            "correct_pressure_clean_like_rate": merged["cctp_clean_like_rate_correct_pressure"],
            "correct_pressure_p_correct": merged["cctp_p_correct_correct_pressure"],
            "correct_pressure_accuracy": merged["cctp_accuracy_correct_pressure"],
            "correct_pressure_flattening_rate": merged["cctp_flattening_rate_correct_pressure"],
        }
    )


def alpha_bucket_summary(item_df: pd.DataFrame) -> pd.DataFrame:
    primary = item_df[item_df["condition"].eq("cctp_logit_linear_min_alpha")] if not item_df.empty else item_df
    if primary.empty:
        return pd.DataFrame(columns=["model_key", "model", "dataset", "pressure_type", "condition", "alpha_bucket", "n_items", "clean_like_rate", "wrong_follow_rate", "mean_p_wrong", "accuracy", "mean_p_correct"])
    return (
        primary.groupby(["model_key", "model", "dataset", "pressure_type", "condition", "alpha_bucket"], dropna=False)
        .agg(
            n_items=("item_id", "count"),
            clean_like_rate=("clean_context_like", "mean"),
            wrong_follow_rate=("wrong_follow", "mean"),
            mean_p_wrong=("p_wrong", "mean"),
            accuracy=("accuracy", "mean"),
            mean_p_correct=("p_correct", "mean"),
        )
        .reset_index()
    )


def tube_sensitivity(summary_df: pd.DataFrame) -> pd.DataFrame:
    conditions = ["cctp_tube_strict", "cctp_tube_main", "cctp_logit_linear_min_alpha", "cctp_tube_loose"]
    df = summary_df[summary_df["condition"].isin(conditions)].copy() if not summary_df.empty else summary_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["tube", "model", "dataset", "mean_alpha", "median_alpha", "fraction_alpha_le_050", "fraction_alpha_eq_1", "clean_like_rate", "wrong_follow_rate", "mean_p_wrong"])
    tube_map = {"cctp_tube_strict": "strict", "cctp_tube_main": "main", "cctp_logit_linear_min_alpha": "main", "cctp_tube_loose": "loose"}
    df["tube"] = df["condition"].map(tube_map)
    return df.rename(
        columns={
            "cctp_mean_alpha": "mean_alpha",
            "cctp_median_alpha": "median_alpha",
            "cctp_clean_like_rate": "clean_like_rate",
            "cctp_wrong_follow": "wrong_follow_rate",
            "cctp_p_wrong": "mean_p_wrong",
        }
    )[["tube", "model_key", "model", "dataset", "pressure_type", "mean_alpha", "median_alpha", "fraction_alpha_le_050", "fraction_alpha_eq_1", "clean_like_rate", "wrong_follow_rate", "mean_p_wrong"]]


def path_ablation(summary_df: pd.DataFrame) -> pd.DataFrame:
    conditions = ["cctp_logit_linear_min_alpha", "cctp_prob_linear_min_alpha", "cctp_geometric_prob_min_alpha", "cctp_kl_projection_tau_005", "cctp_kl_projection_tau_010"]
    df = summary_df[summary_df["condition"].isin(conditions)].copy() if not summary_df.empty else summary_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["path_variant", "dataset", "mean_alpha", "median_alpha", "fraction_alpha_le_050", "fraction_alpha_eq_1", "clean_like_rate", "wrong_follow_rate", "mean_p_wrong"])
    path_map = {
        "cctp_logit_linear_min_alpha": "logit_linear",
        "cctp_prob_linear_min_alpha": "prob_linear",
        "cctp_geometric_prob_min_alpha": "geometric_prob",
        "cctp_kl_projection_tau_005": "kl_projection_tau_005",
        "cctp_kl_projection_tau_010": "kl_projection_tau_010",
    }
    df["path_variant"] = df["condition"].map(path_map)
    return df.rename(
        columns={
            "cctp_mean_alpha": "mean_alpha",
            "cctp_median_alpha": "median_alpha",
            "cctp_clean_like_rate": "clean_like_rate",
            "cctp_wrong_follow": "wrong_follow_rate",
            "cctp_p_wrong": "mean_p_wrong",
        }
    )[["path_variant", "model_key", "model", "dataset", "pressure_type", "mean_alpha", "median_alpha", "fraction_alpha_le_050", "fraction_alpha_eq_1", "clean_like_rate", "wrong_follow_rate", "mean_p_wrong"]]


def correlations(item_df: pd.DataFrame) -> pd.DataFrame:
    primary = item_df[(item_df["condition"].eq("cctp_logit_linear_min_alpha")) & (item_df["pressure_type"].eq("wrong_pressure"))] if not item_df.empty else item_df
    metrics = [
        ("p_wrong_reset", "reset_p_wrong"),
        ("wrong_follow_reset", "reset_wrong_follow"),
        ("KL_clean_reset", "kl_clean_full_history"),
        ("entropy_delta_reset", "reset_entropy_delta_vs_clean"),
        ("max_prob_delta_reset", "reset_max_prob_delta_vs_clean"),
    ]
    rows = []
    if primary.empty:
        return pd.DataFrame(columns=["level", "model_key", "model", "dataset", "x_metric", "y_metric", "pearson", "spearman", "n"])
    for (model_key, model, dataset), group in primary.groupby(["model_key", "model", "dataset"]):
        for metric_name, col in metrics:
            if group[col].notna().sum() < 3:
                continue
            rows.append(
                {
                    "level": "item",
                    "model_key": model_key,
                    "model": model,
                    "dataset": dataset,
                    "x_metric": "alpha_star",
                    "y_metric": metric_name,
                    "pearson": float(group["alpha_star"].corr(group[col], method="pearson")),
                    "spearman": float(group["alpha_star"].corr(group[col], method="spearman")),
                    "n": int(len(group)),
                }
            )
    pair = primary.groupby(["model_key", "model", "dataset"]).agg(
        mean_alpha=("alpha_star", "mean"), mean_R0_gap=("R0_gap", "mean"), ordinary_reset_wrong_follow=("reset_wrong_follow", "mean"), ordinary_reset_p_wrong=("reset_p_wrong", "mean")
    ).reset_index()
    for col in ["mean_R0_gap", "ordinary_reset_wrong_follow", "ordinary_reset_p_wrong"]:
        if len(pair) >= 3:
            rows.append(
                {
                    "level": "pair",
                    "model_key": "all",
                    "model": "all",
                    "dataset": "all",
                    "x_metric": "mean_alpha",
                    "y_metric": col,
                    "pearson": float(pair["mean_alpha"].corr(pair[col], method="pearson")),
                    "spearman": float(pair["mean_alpha"].corr(pair[col], method="spearman")),
                    "n": int(len(pair)),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_cis(item_df: pd.DataFrame, iters: int, seed: int) -> pd.DataFrame:
    primary = item_df[item_df["condition"].eq("cctp_logit_linear_min_alpha")] if not item_df.empty else item_df
    metric_specs = [
        ("mean_alpha", "alpha_star", "mean"),
        ("median_alpha", "alpha_star", "median"),
        ("mean_history_retention", "history_retention", "mean"),
        ("fraction_alpha_le_050", "alpha_le_050", "mean"),
        ("fraction_alpha_eq_1", "alpha_eq_1", "mean"),
        ("clean_like_rate", "clean_context_like", "mean"),
        ("wrong_follow_rate", "wrong_follow", "mean"),
        ("mean_p_wrong", "p_wrong", "mean"),
        ("accuracy", "accuracy", "mean"),
        ("mean_p_correct", "p_correct", "mean"),
        ("flattening_rate", "flattening_flag", "mean"),
        ("mean_KL_clean_projected", "kl_clean_projected", "mean"),
    ]
    rng = np.random.default_rng(seed)
    rows = []
    if primary.empty or iters <= 0:
        return pd.DataFrame(columns=["model", "dataset", "condition", "pressure_type", "metric", "estimate", "ci_low", "ci_high", "n_bootstrap", "seed"])

    def metric_array(group: pd.DataFrame, value_key: str) -> np.ndarray:
        if value_key == "alpha_le_050":
            return (pd.to_numeric(group["alpha_star"], errors="coerce").to_numpy(dtype=float) <= 0.50).astype(float)
        if value_key == "alpha_eq_1":
            return (pd.to_numeric(group["alpha_star"], errors="coerce").to_numpy(dtype=float) == 1.0).astype(float)
        return pd.to_numeric(group[value_key], errors="coerce").to_numpy(dtype=float)

    def estimate_and_samples(values: np.ndarray, agg: str) -> Tuple[float, np.ndarray]:
        values = np.asarray(values, dtype=float)
        if values.size == 0 or np.all(np.isnan(values)):
            return np.nan, np.full(iters, np.nan)
        sample_indices = rng.integers(0, values.size, size=(iters, values.size))
        sampled = values[sample_indices]
        if agg == "median":
            return float(np.nanmedian(values)), np.nanmedian(sampled, axis=1)
        return float(np.nanmean(values)), np.nanmean(sampled, axis=1)

    def append_metric_rows(model_key: str, model: str, dataset: str, pressure_type: str, group: pd.DataFrame) -> None:
        for metric, value_key, agg in metric_specs:
            estimate, samples = estimate_and_samples(metric_array(group, value_key), agg)
            rows.append(
                {
                    "model_key": model_key,
                    "model": model,
                    "dataset": dataset,
                    "condition": "cctp_logit_linear_min_alpha",
                    "pressure_type": pressure_type,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": float(np.nanpercentile(samples, 2.5)),
                    "ci_high": float(np.nanpercentile(samples, 97.5)),
                    "n_bootstrap": iters,
                    "seed": seed,
                }
            )

    groups = list(primary.groupby(["model_key", "model", "dataset", "pressure_type"], dropna=False))
    for (model_key, model, dataset, pressure_type), group in groups:
        append_metric_rows(str(model_key), str(model), str(dataset), str(pressure_type), group)
    for dataset, group in primary.groupby("dataset"):
        append_metric_rows("all_models", "all_models", str(dataset), "all", group)
    if set(["truthfulqa_mc", "mmlu_pro"]) <= set(primary["dataset"].unique()):
        tqa = primary[primary["dataset"].eq("truthfulqa_mc")]
        mmlu = primary[primary["dataset"].eq("mmlu_pro")]
        t_values = pd.to_numeric(tqa["alpha_star"], errors="coerce").to_numpy(dtype=float)
        m_values = pd.to_numeric(mmlu["alpha_star"], errors="coerce").to_numpy(dtype=float)
        estimate = float(np.nanmean(m_values) - np.nanmean(t_values))
        t_indices = rng.integers(0, t_values.size, size=(iters, t_values.size))
        m_indices = rng.integers(0, m_values.size, size=(iters, m_values.size))
        samples = np.nanmean(m_values[m_indices], axis=1) - np.nanmean(t_values[t_indices], axis=1)
        rows.append(
            {
                "model_key": "all_models",
                "model": "all_models",
                "dataset": "mmlu_minus_truthfulqa",
                "condition": "cctp_logit_linear_min_alpha",
                "pressure_type": "all",
                "metric": "mean_alpha_mmlu_minus_truthfulqa",
                "estimate": estimate,
                "ci_low": float(np.nanpercentile(samples, 2.5)),
                "ci_high": float(np.nanpercentile(samples, 97.5)),
                "n_bootstrap": iters,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def write_placeholder_plot(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if plt is None:
        path.write_text("matplotlib unavailable")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, "No complete artifact data", ha="center", va="center")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_plots(output_root: Path, item_df: pd.DataFrame, binding_df: pd.DataFrame, corr_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    if plt is None or item_df.empty:
        for plot in PLOTS:
            write_placeholder_plot(plot_root / plot, plot)
        return
    primary = item_df[(item_df["condition"].eq("cctp_logit_linear_min_alpha")) & (item_df["pressure_type"].eq("wrong_pressure"))]
    if primary.empty:
        for plot in PLOTS:
            write_placeholder_plot(plot_root / plot, plot)
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for dataset, group in primary.groupby("dataset"):
        values = np.sort(group["alpha_star"].astype(float).to_numpy())
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=DATASET_NAMES.get(dataset, dataset), linewidth=2)
    ax.set_xlabel("alpha*")
    ax.set_ylabel("Fraction recovered by alpha <= x")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "alpha_cdf_by_dataset.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for dataset, group in primary.groupby("dataset"):
        ax.hist(group["alpha_star"].astype(float), bins=np.linspace(0, 1, 21), alpha=0.45, label=DATASET_NAMES.get(dataset, dataset))
    ax.set_xlabel("alpha*")
    ax.set_ylabel("Items")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "alpha_hist_by_dataset.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, dataset), group in primary.groupby(["model", "dataset"]):
        values = np.sort(group["alpha_star"].astype(float).to_numpy())
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), alpha=0.45, label=f"{model} {dataset}")
    ax.set_xlabel("alpha*")
    ax.set_ylabel("Fraction recovered")
    if primary[["model", "dataset"]].drop_duplicates().shape[0] <= 8:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_root / "alpha_cdf_by_model_dataset.png", dpi=180)
    plt.close(fig)

    for filename, by_cols in [
        ("binding_constraints_by_dataset.png", ["dataset", "binding_constraint_primary"]),
        ("binding_constraints_by_model_dataset.png", ["model", "dataset", "binding_constraint_primary"]),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        if binding_df.empty:
            ax.text(0.5, 0.5, "No binding data", ha="center", va="center")
        else:
            bind = binding_df[binding_df["condition"].eq("cctp_logit_linear_min_alpha")]
            pivot = bind.groupby(by_cols).size().reset_index(name="n")
            x_col = by_cols[0] if filename == "binding_constraints_by_dataset.png" else "model"
            table = pivot.pivot_table(index=x_col, columns="binding_constraint_primary", values="n", aggfunc="sum", fill_value=0)
            table.div(table.sum(axis=1), axis=0).plot(kind="bar", stacked=True, ax=ax)
            ax.set_ylabel("Fraction")
        fig.tight_layout()
        fig.savefig(plot_root / filename, dpi=180)
        plt.close(fig)

    scatter_specs = [
        ("alpha_vs_reset_p_wrong.png", "reset_p_wrong", "Reset p_wrong"),
        ("alpha_vs_reset_kl.png", "kl_clean_full_history", "KL(clean || reset)"),
    ]
    for filename, col, ylabel in scatter_specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(primary[col], primary["alpha_star"], alpha=0.55)
        ax.set_xlabel(ylabel)
        ax.set_ylabel("alpha*")
        fig.tight_layout()
        fig.savefig(plot_root / filename, dpi=180)
        plt.close(fig)

    pair = summary_df[(summary_df["condition"].eq("cctp_logit_linear_min_alpha")) & (summary_df["pressure_type"].eq("wrong_pressure"))]
    fig, ax = plt.subplots(figsize=(6, 4))
    if pair.empty:
        ax.text(0.5, 0.5, "No pair data", ha="center", va="center")
    else:
        ax.scatter(pair["R0_gap"], pair["cctp_mean_alpha"])
        ax.set_xlabel("Mean R0 gap")
        ax.set_ylabel("Mean alpha*")
    fig.tight_layout()
    fig.savefig(plot_root / "pair_level_mean_alpha_vs_r0_gap.png", dpi=180)
    plt.close(fig)


def qualitative_examples(output_root: Path, item_df: pd.DataFrame) -> None:
    primary = item_df[(item_df["condition"].eq("cctp_logit_linear_min_alpha")) & (item_df["pressure_type"].eq("wrong_pressure"))] if not item_df.empty else item_df
    examples = []
    specs = [
        ("low_alpha_recovery", primary[primary["alpha_star"] <= 0.10].head(1)),
        ("medium_alpha_recovery", primary[(primary["alpha_star"] >= 0.40) & (primary["alpha_star"] <= 0.60)].head(1)),
        ("high_alpha_recovery", primary[primary["alpha_star"] >= 0.90].head(1)),
        ("mmlu_distribution_shape_binding", primary[(primary["dataset"].eq("mmlu_pro")) & (primary["binding_constraint_primary"].isin(["entropy", "max_prob", "KL"]))].head(1)),
        ("truthfulqa_small_correction", primary[(primary["dataset"].eq("truthfulqa_mc")) & (primary["alpha_star"] <= 0.25)].head(1)),
    ]
    for name, frame in specs:
        if frame.empty:
            continue
        row = frame.iloc[0].to_dict()
        examples.append(
            {
                "example_type": name,
                "model": row.get("model"),
                "dataset": row.get("dataset"),
                "item_id": row.get("item_id"),
                "question": "omitted_dataset_text",
                "answer_options": "omitted_dataset_text",
                "correct_label": row.get("correct_label"),
                "advocated_wrong_label": row.get("advocated_wrong_label"),
                "alpha_star": row.get("alpha_star"),
                "binding_constraint": row.get("binding_constraint_primary"),
                "clean_prediction": row.get("clean_prediction"),
                "ordinary_reset_prediction": row.get("full_history_prediction"),
                "cctp_prediction": row.get("prediction"),
                "p_wrong_alpha_0": row.get("p_full_history_wrong"),
                "p_wrong_alpha_star": row.get("p_projected_wrong"),
                "p_wrong_alpha_1": row.get("p_clean_wrong"),
                "p_correct_alpha_0": row.get("p_full_history_correct"),
                "p_correct_alpha_star": row.get("p_projected_correct"),
                "p_correct_alpha_1": row.get("p_clean_correct"),
                "entropy_alpha_0": row.get("entropy_full_history"),
                "entropy_alpha_star": row.get("entropy_projected"),
                "entropy_alpha_1": row.get("entropy_clean"),
                "max_prob_alpha_0": row.get("max_prob_full_history"),
                "max_prob_alpha_star": row.get("max_prob_projected"),
                "max_prob_alpha_1": row.get("max_prob_clean"),
            }
        )
    (output_root / "qualitative_examples.json").write_text(json.dumps(examples, indent=2, sort_keys=True))
    lines = ["# Qualitative Examples", "", "Dataset text omitted; labels and probabilities retained.", ""]
    for example in examples:
        lines.append(f"## {example['example_type']}")
        for key, value in example.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    (output_root / "qualitative_examples.md").write_text("\n".join(lines))


def make_empty_outputs(output_root: Path) -> None:
    pd.DataFrame(columns=EMPTY_ITEM_COLUMNS).to_csv(output_root / "combined_item_level.csv", index=False)
    empty_summary().to_csv(output_root / "combined_summary.csv", index=False)
    pd.DataFrame(columns=EMPTY_CORRECT_PRESSURE_COLUMNS).to_csv(output_root / "combined_correct_pressure_control.csv", index=False)
    pd.DataFrame().to_csv(output_root / "combined_endpoint_comparison.csv", index=False)
    pd.DataFrame(columns=EMPTY_ALPHA_COLUMNS).to_csv(output_root / "combined_alpha_diagnostics.csv", index=False)
    pd.DataFrame(columns=EMPTY_TUBE_COLUMNS).to_csv(output_root / "combined_tube_diagnostics.csv", index=False)
    pd.DataFrame(columns=EMPTY_BINDING_COLUMNS).to_csv(output_root / "combined_binding_constraints.csv", index=False)
    pd.DataFrame().to_csv(output_root / "combined_tube_sensitivity.csv", index=False)
    pd.DataFrame().to_csv(output_root / "combined_path_ablation.csv", index=False)
    pd.DataFrame().to_csv(output_root / "combined_alpha_hysteresis_correlations.csv", index=False)
    pd.DataFrame().to_csv(output_root / "combined_bootstrap_cis.csv", index=False)
    pd.DataFrame().to_csv(output_root / "combined_alpha_bucket_summary.csv", index=False)


def write_per_pair_outputs(output_root: Path, item_df: pd.DataFrame, summary_df: pd.DataFrame, correct_df: pd.DataFrame, alpha_df: pd.DataFrame, tube_df: pd.DataFrame, binding_df: pd.DataFrame, endpoint_df: pd.DataFrame) -> None:
    if item_df.empty or not {"model_key", "dataset", "item_id"}.issubset(item_df.columns):
        return
    for (model_key, dataset), group in item_df.groupby(["model_key", "dataset"], dropna=False):
        pair_root = output_root / str(model_key) / f"{dataset}_n{group['item_id'].nunique()}"
        pair_root.mkdir(parents=True, exist_ok=True)
        group.to_csv(pair_root / "cctp_item_level.csv", index=False)
        summary_df[(summary_df["model_key"].eq(model_key)) & (summary_df["dataset"].eq(dataset))].to_csv(pair_root / "cctp_summary.csv", index=False)
        correct_df[(correct_df["model_key"].eq(model_key)) & (correct_df["dataset"].eq(dataset))].to_csv(pair_root / "cctp_correct_pressure_control.csv", index=False)
        alpha_df[(alpha_df["model_key"].eq(model_key)) & (alpha_df["dataset"].eq(dataset))].to_csv(pair_root / "cctp_alpha_diagnostics.csv", index=False)
        tube_df[(tube_df["model_key"].eq(model_key)) & (tube_df["dataset"].eq(dataset))].to_csv(pair_root / "cctp_tube_diagnostics.csv", index=False)
        binding_df[(binding_df["model_key"].eq(model_key)) & (binding_df["dataset"].eq(dataset))].to_csv(pair_root / "cctp_binding_constraints.csv", index=False)
        endpoint_df[(endpoint_df["model_key"].eq(model_key)) & (endpoint_df["dataset"].eq(dataset))].to_csv(pair_root / "cctp_endpoint_comparison.csv", index=False)
        (pair_root / "cctp_report.md").write_text("# CCTP Pair Report\n\nSee root report for interpretation.\n")


def artifact_checks(output_root: Path) -> pd.DataFrame:
    rows = []
    for name in ROOT_OUTPUTS:
        path = output_root / name
        rows.append({"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0})
    for plot in PLOTS:
        path = output_root / "plots" / plot
        rows.append({"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    display = df.head(max_rows).copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    cols = list(display.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def write_report(output_root: Path, decision: Dict[str, Any], status_df: pd.DataFrame, summary_df: pd.DataFrame, correct_df: pd.DataFrame, endpoint_df: pd.DataFrame, sensitivity_df: pd.DataFrame, path_df: pd.DataFrame, corr_df: pd.DataFrame, ci_df: pd.DataFrame) -> None:
    main = summary_df[(summary_df["condition"].eq("cctp_logit_linear_min_alpha")) & (summary_df["pressure_type"].eq("wrong_pressure"))] if not summary_df.empty else summary_df
    compact_cols = ["model", "dataset", "R0_gap", "cctp_mean_alpha", "cctp_median_alpha", "mean_history_retention", "fraction_alpha_le_050", "cctp_clean_like_rate", "cctp_wrong_follow"]
    lines = [
        "# CCTP Paper Package Report",
        "",
        "## Executive Summary",
        f"- Recommendation: `{decision['recommendation']}`.",
        f"- Reason: `{decision['reason']}`.",
        f"- Complete pairs: `{decision['complete_pairs']}/{decision['requested_pairs']}`.",
        f"- Artifact coverage: `{decision['artifact_coverage']:.4f}`.",
        "",
        "CCTP is not a history-preserving intervention. It is a clean-counterfactual anchored recovery operator.",
        "",
        "Fresh deletion answers entirely from the clean branch. CCTP returns the closest point along a full-history-to-clean path that satisfies the clean tube.",
        "",
        "## Method Definition",
        "`z_alpha = (1 - alpha) z_H + alpha z_0`, `p_alpha = softmax(z_alpha)`. Alpha* is the smallest alpha entering the clean tube.",
        "",
        "## Artifact Coverage and Model-Dataset Pairs",
        md_table(status_df),
        "",
        "## Main 14-Pair n=500 Results",
        md_table(main[[col for col in compact_cols if col in main.columns]] if not main.empty else main),
        "",
        "## Correct-Pressure CCTP Control",
        md_table(correct_df),
        "",
        "## Endpoint Comparison",
        md_table(endpoint_df),
        "",
        "## Alpha Distribution Analysis",
        "See plots/alpha_cdf_by_dataset.png and plots/alpha_hist_by_dataset.png.",
        "",
        "## Binding Constraint Analysis",
        "See combined_binding_constraints.csv and binding constraint plots.",
        "",
        "## Tube Sensitivity",
        md_table(sensitivity_df),
        "",
        "## Path Variant Ablation",
        md_table(path_df),
        "",
        "## Alpha-Hysteresis Correlation",
        md_table(corr_df),
        "",
        "## Bootstrap Confidence Intervals",
        md_table(ci_df),
        "",
        "## Naturalistic Pressure CCTP",
        "Not run unless `--include-naturalistic` artifacts are available.",
        "",
        "## Qualitative Examples",
        "Dataset text is omitted unless `--include-qualitative` is used; examples are label/probability based.",
        "",
        "## Main Paper Takeaways",
        "CCTP shows that a clean counterfactual anchor is sufficient for guaranteed recovery where required logits exist. The required alpha* quantifies how far the post-reset state remains from clean-context behavior.",
        "",
        "## Limitations",
        "CCTP is not evidence that contaminated state is internally repaired. Missing artifact coverage requires rerunning source logits, not inference inside this package.",
        "",
    ]
    (output_root / "report.md").write_text("\n".join(lines))


def decide(status_df: pd.DataFrame, summary_df: pd.DataFrame, ci_df: pd.DataFrame, binding_df: pd.DataFrame) -> Dict[str, Any]:
    requested = int(len(status_df))
    complete = int((status_df["status"] == "ok").sum()) if not status_df.empty else 0
    coverage = complete / max(requested, 1)
    if complete < requested:
        recommendation = "missing_logits_rerun_required"
        reason = "missing_clean_or_full_history_logits"
        scale_ready = False
    else:
        primary = summary_df[summary_df["condition"].eq("cctp_logit_linear_min_alpha")]
        guarantee = float(primary["implementation_guarantee_pass_rate"].min()) if not primary.empty else 0.0
        cp = primary[primary["pressure_type"].eq("correct_pressure")]
        cp_pass = float(cp["cctp_clean_like_rate"].min()) if not cp.empty else 0.0
        if guarantee < 1.0:
            recommendation = "fix_implementation"
            reason = "alpha_one_failed_clean_tube"
            scale_ready = False
        elif cp_pass >= 0.99 and not ci_df.empty and not binding_df.empty:
            mean_retention = float(primary["mean_history_retention"].mean())
            if mean_retention >= 0.25:
                recommendation = "strong_positive_scale_result"
                reason = "minimal_clean_anchor_recovers_clean_like_with_partial_history_retention"
            else:
                recommendation = "report_as_clean_anchor_operator"
                reason = "recovery_requires_near_fresh_deletion"
            scale_ready = recommendation == "strong_positive_scale_result"
        else:
            recommendation = "report_as_clean_anchor_operator"
            reason = "clean_anchor_operator_outputs_incomplete_for_scale_ready_claim"
            scale_ready = False
    return {
        "recommendation": recommendation,
        "reason": reason,
        "scale_to_n500": scale_ready,
        "requested_pairs": requested,
        "complete_pairs": complete,
        "artifact_coverage": coverage,
        "bootstrap_cis_generated": not ci_df.empty,
        "binding_constraints_generated": not binding_df.empty,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.source_mode != "artifacts_only":
        raise RuntimeError("paper CCTP package currently supports artifacts_only only")
    input_roots = resolve_input_roots(args.input_roots)
    output_root = resolve_path(args.output_root)
    prepare_output_root(output_root, args.overwrite)
    source_path = input_roots["source"] / "combined_item_level.csv"
    source_df, missing_cols = load_source(input_roots["source"])
    if missing_cols:
        status_df = pd.DataFrame(
            [
                {
                    "model_key": model,
                    "model": MODEL_SHORT_NAMES.get(model, model),
                    "dataset": dataset,
                    "n_requested": args.n,
                    "n_available_wrong_pressure": 0,
                    "n_available_correct_pressure": 0,
                    "n_usable_both_pressures": 0,
                    "status": "missing_source_columns",
                    "missing_count": args.n,
                    "missing_columns": json.dumps(missing_cols),
                }
                for model in args.model_keys
                for dataset in args.datasets
            ]
        )
        make_empty_outputs(output_root)
        status = {f"{row.model_key}:{row.dataset}": row._asdict() for row in status_df.itertuples(index=False)}
        (output_root / "combined_pair_status.json").write_text(json.dumps(status, indent=2, sort_keys=True))
        decision = decide(status_df, empty_summary(), pd.DataFrame(), pd.DataFrame())
        decision.update({"missing_source_columns": missing_cols, "source_mode": args.source_mode, "input_artifacts_used": {k: str(v) for k, v in input_roots.items()}})
        (output_root / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True))
        write_plots(output_root, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        write_report(output_root, decision, status_df, empty_summary(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        artifact_checks(output_root).to_csv(output_root / "artifact_checks.csv", index=False)
        artifact_checks(output_root).to_csv(output_root / "artifact_checks.csv", index=False)
        return decision

    status_df, selected = model_dataset_status(source_df, args.model_keys, args.datasets, args.n)
    status = {f"{row.model_key}:{row.dataset}": row._asdict() for row in status_df.itertuples(index=False)}
    lookup = source_lookup(source_df)
    specs = condition_specs(args.paths, args.tubes)
    alphas = alpha_values(args.alpha_step)
    item_rows: List[Dict[str, Any]] = []
    alpha_rows: List[Dict[str, Any]] = []
    tube_rows: List[Dict[str, Any]] = []
    binding_rows: List[Dict[str, Any]] = []
    for pair_row in status_df.itertuples(index=False):
        if pair_row.status != "ok":
            continue
        for item_id in selected[(pair_row.model_key, pair_row.dataset)][: args.n]:
            for pressure_type, source_condition in [("wrong_pressure", "ordinary_reset"), ("correct_pressure", "correct_pressure_ordinary_reset")]:
                source_row = lookup.get((pair_row.model_key, pair_row.dataset, item_id, source_condition))
                if source_row is None:
                    continue
                source_series = pd.Series(source_row._asdict() if hasattr(source_row, "_asdict") else source_row)
                for spec in specs:
                    item, alpha_diag, tube, bind = run_one_condition(source_series, spec, pressure_type, alphas, str(source_path))
                    item_rows.append(item)
                    alpha_rows.extend(alpha_diag)
                    tube_rows.append(tube)
                    binding_rows.append(bind)
                if args.include_endpoints:
                    for endpoint_condition in ["ordinary_reset", "relabel_shuffle", "fresh_context_deletion"]:
                        cond = endpoint_condition if pressure_type == "wrong_pressure" else f"correct_pressure_{endpoint_condition}"
                        if endpoint_condition == "fresh_context_deletion":
                            cond = "fresh_context_deletion"
                        endpoint_source = lookup.get((pair_row.model_key, pair_row.dataset, item_id, cond))
                        if endpoint_source is None:
                            continue
                        endpoint_series = pd.Series(endpoint_source._asdict() if hasattr(endpoint_source, "_asdict") else endpoint_source)
                        endpoint = endpoint_row(endpoint_series, endpoint_condition, pressure_type, str(source_path))
                        if endpoint is not None:
                            item_rows.append(endpoint)
    item_df = pd.DataFrame(item_rows)
    alpha_df = pd.DataFrame(alpha_rows)
    tube_df = pd.DataFrame(tube_rows)
    binding_df = pd.DataFrame(binding_rows)
    if item_df.empty:
        item_df = pd.DataFrame(columns=EMPTY_ITEM_COLUMNS)
    if alpha_df.empty:
        alpha_df = pd.DataFrame(columns=EMPTY_ALPHA_COLUMNS)
    if tube_df.empty:
        tube_df = pd.DataFrame(columns=EMPTY_TUBE_COLUMNS)
    if binding_df.empty:
        binding_df = pd.DataFrame(columns=EMPTY_BINDING_COLUMNS)
    summary_df = summarize(item_df)
    correct_df = correct_pressure_summary(summary_df)
    endpoint_df = endpoint_summary(summary_df)
    sensitivity_df = tube_sensitivity(summary_df)
    path_df = path_ablation(summary_df)
    corr_df = correlations(item_df)
    ci_df = bootstrap_cis(item_df, args.bootstrap_iters, args.seed) if args.include_bootstrap else pd.DataFrame()
    bucket_df = alpha_bucket_summary(item_df)

    item_df.to_csv(output_root / "combined_item_level.csv", index=False)
    summary_df.to_csv(output_root / "combined_summary.csv", index=False)
    correct_df.to_csv(output_root / "combined_correct_pressure_control.csv", index=False)
    endpoint_df.to_csv(output_root / "combined_endpoint_comparison.csv", index=False)
    alpha_df.to_csv(output_root / "combined_alpha_diagnostics.csv", index=False)
    tube_df.to_csv(output_root / "combined_tube_diagnostics.csv", index=False)
    binding_df.to_csv(output_root / "combined_binding_constraints.csv", index=False)
    sensitivity_df.to_csv(output_root / "combined_tube_sensitivity.csv", index=False)
    path_df.to_csv(output_root / "combined_path_ablation.csv", index=False)
    corr_df.to_csv(output_root / "combined_alpha_hysteresis_correlations.csv", index=False)
    ci_df.to_csv(output_root / "combined_bootstrap_cis.csv", index=False)
    bucket_df.to_csv(output_root / "combined_alpha_bucket_summary.csv", index=False)
    (output_root / "combined_pair_status.json").write_text(json.dumps(status, indent=2, sort_keys=True))
    write_per_pair_outputs(output_root, item_df, summary_df, correct_df, alpha_df, tube_df, binding_df, endpoint_df)
    write_plots(output_root, item_df, binding_df, corr_df, summary_df)
    if args.include_qualitative:
        qualitative_examples(output_root, item_df)
    decision = decide(status_df, summary_df, ci_df, binding_df)
    decision.update(
        {
            "source_mode": args.source_mode,
            "output_root": str(output_root),
            "input_artifacts_used": {k: str(v) for k, v in input_roots.items()},
            "model_keys": list(args.model_keys),
            "datasets": list(args.datasets),
            "n": args.n,
            "alpha_step": args.alpha_step,
            "missing_pairs": status_df[status_df["status"] != "ok"].to_dict("records"),
        }
    )
    (output_root / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True))
    write_report(output_root, decision, status_df, summary_df, correct_df, endpoint_df, sensitivity_df, path_df, corr_df, ci_df)
    artifact_checks(output_root).to_csv(output_root / "artifact_checks.csv", index=False)
    artifact_checks(output_root).to_csv(output_root / "artifact_checks.csv", index=False)
    return decision


def main() -> None:
    args = parse_args()
    decision = run(args)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()