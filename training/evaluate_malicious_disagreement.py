#!/usr/bin/env python3
"""
RolloutSecBench — Phase 4:
Deterministic malicious revision disagreement analysis.

This script derives malicious disagreement evidence ONLY from the already
frozen PRIMARY strict-test paired-prediction artifact.

It does NOT:
- read strict_test_nonoverlap.csv;
- read official_test_full.csv;
- read final_train or validation;
- load the preprocessor;
- load V1 or V2 models;
- perform inference;
- fit/refit anything;
- tune thresholds;
- perform rollout analysis;
- invent a practical MRD threshold.

All operational artifact paths and hashes are resolved from the supplied
static-test evaluation manifest and its referenced artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Required JSON artifact does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object in {path}")

    return obj


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Required artifact does not exist: {path}")

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)

    if actual.lower() != str(expected).lower():
        raise RuntimeError(
            f"{label} SHA-256 mismatch.\n"
            f"Path:     {path}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    print(f"[PASS] {label} SHA-256: {actual}")
    return actual


def resolve_repo_root(static_manifest_path: Path) -> Path:
    """
    The supplied manifest is expected to live somewhere beneath the repository.

    Resolve the repository root from the manifest itself by walking upward until
    the manifest-relative paths it contains become valid. We do not assume a
    fixed absolute repository location.
    """
    static_manifest_path = static_manifest_path.resolve()
    manifest = load_json(static_manifest_path)

    refs = []

    authorization = manifest.get("authorization", {})
    if authorization.get("gate1_path"):
        refs.append(authorization["gate1_path"])

    outputs = manifest.get("outputs", {})
    for key in ("paired_predictions", "static_metrics"):
        entry = outputs.get(key, {})
        if entry.get("path"):
            refs.append(entry["path"])

    if not refs:
        raise RuntimeError(
            "Static evaluation manifest contains no usable repository-relative "
            "artifact references."
        )

    for candidate in [static_manifest_path.parent, *static_manifest_path.parents]:
        if all((candidate / Path(ref)).exists() for ref in refs):
            return candidate.resolve()

    raise RuntimeError(
        "Could not derive repository root from the static evaluation manifest "
        "and its referenced artifacts."
    )


def repo_path(repo_root: Path, relative_path: str) -> Path:
    p = Path(relative_path)

    if p.is_absolute():
        raise RuntimeError(
            f"Expected repository-relative path, got absolute path: {p}"
        )

    resolved = (repo_root / p).resolve()

    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"Manifest path escapes repository root: {relative_path}"
        ) from exc

    return resolved


def write_json_exclusive(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------

def validate_static_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("artifact") != "RolloutSecBench":
        raise RuntimeError("Unexpected artifact identity.")

    if manifest.get("phase") != "Primary Static Test Evaluation":
        raise RuntimeError(
            "Input manifest is not the frozen Primary Static Test Evaluation."
        )

    scope = manifest.get("analysis_scope", {})

    expected_scope = {
        "full_population_static_metrics_computed": True,
        "malicious_disagreement_computed": False,
        "mrd_computed": False,
        "rollout_analysis_performed": False,
    }

    for key, expected in expected_scope.items():
        if scope.get(key) is not expected:
            raise RuntimeError(
                f"Unexpected static-manifest analysis scope for {key}: "
                f"{scope.get(key)!r}"
            )

    access = manifest.get("data_access", {})

    if access.get("strict_test_nonoverlap_read") is not True:
        raise RuntimeError(
            "Static manifest does not establish primary strict-test access."
        )

    for field in (
        "official_test_full_read",
        "validation_read",
        "final_train_read",
    ):
        if access.get(field) is not False:
            raise RuntimeError(
                f"Static manifest unexpectedly reports {field}=True."
            )

    auth = manifest.get("authorization", {})

    if auth.get("gate1_status") != "PASS":
        raise RuntimeError("Static evaluation was not authorized by Gate 1 PASS.")

    print("[PASS] Frozen static-evaluation manifest scope verified.")


def validate_static_metrics(
    metrics: dict[str, Any],
    static_manifest: dict[str, Any],
) -> None:
    if metrics.get("artifact") != "RolloutSecBench":
        raise RuntimeError("Unexpected static metrics artifact identity.")

    if metrics.get("phase") != "Primary Static Test Evaluation":
        raise RuntimeError(
            "Static metrics artifact has unexpected phase."
        )

    evaluation_set = metrics.get("evaluation_set", {})

    if evaluation_set.get("name") != "STRICT_NONOVERLAP_TEST":
        raise RuntimeError(
            "Static metrics were not produced on STRICT_NONOVERLAP_TEST."
        )

    if evaluation_set.get("role") != "PRIMARY_EVALUATION":
        raise RuntimeError(
            "Static metrics do not identify the set as PRIMARY_EVALUATION."
        )

    constraints = metrics.get("scientific_constraints", {})

    required_false = (
        "hyperparameter_tuning_performed",
        "malicious_disagreement_computed",
        "model_fitting_performed",
        "official_test_full_read",
        "preprocessor_refit",
        "rollout_analysis_performed",
        "threshold_tuning_performed",
    )

    for field in required_false:
        if constraints.get(field) is not False:
            raise RuntimeError(
                f"Static metrics scientific constraint violated: {field}"
            )

    frozen_test = static_manifest["frozen_inputs"]["strict_test"]

    if evaluation_set.get("sha256") != frozen_test.get("sha256"):
        raise RuntimeError(
            "Static metrics strict-test SHA-256 disagrees with "
            "the static evaluation manifest."
        )

    if int(evaluation_set.get("rows")) != int(frozen_test.get("rows")):
        raise RuntimeError(
            "Static metrics strict-test row count disagrees with "
            "the static evaluation manifest."
        )

    print("[PASS] Static metrics provenance verified.")


# ---------------------------------------------------------------------
# Paired prediction validation
# ---------------------------------------------------------------------

def validate_prediction_table(
    df: pd.DataFrame,
    expected_rows: int,
) -> None:

    required_columns = {
        "label",
        "v1_probability",
        "v1_prediction",
        "v2_probability",
        "v2_prediction",
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise RuntimeError(
            "Paired predictions missing required columns: "
            + ", ".join(sorted(missing))
        )

    if len(df) != expected_rows:
        raise RuntimeError(
            f"Paired prediction row-count mismatch: expected "
            f"{expected_rows}, observed {len(df)}."
        )

    if df["label"].isna().any():
        raise RuntimeError("Ground-truth label contains missing values.")

    observed_labels = set(df["label"].astype(int).unique())

    if observed_labels != {0, 1}:
        raise RuntimeError(
            f"Expected binary ground truth {{0,1}}, observed {observed_labels}."
        )

    for column in ("v1_prediction", "v2_prediction"):
        if df[column].isna().any():
            raise RuntimeError(f"{column} contains missing values.")

        observed = set(df[column].astype(int).unique())

        if not observed.issubset({0, 1}):
            raise RuntimeError(
                f"{column} contains values outside {{0,1}}: {observed}"
            )

    for column in ("v1_probability", "v2_probability"):
        values = df[column].to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise RuntimeError(
                f"{column} contains NaN or infinite values."
            )

        if ((values < 0.0) | (values > 1.0)).any():
            raise RuntimeError(
                f"{column} contains probabilities outside [0,1]."
            )

    print(
        f"[PASS] Paired prediction table validated: "
        f"{len(df)} rows."
    )


# ---------------------------------------------------------------------
# Disagreement analysis
# ---------------------------------------------------------------------

def compute_malicious_disagreement(
    df: pd.DataFrame,
    static_metrics: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:

    y = df["label"].to_numpy(dtype=np.int8)
    p1 = df["v1_prediction"].to_numpy(dtype=np.int8)
    p2 = df["v2_prediction"].to_numpy(dtype=np.int8)

    malicious_mask = y == 1

    malicious = df.loc[malicious_mask].copy()

    malicious_count = int(malicious_mask.sum())

    if malicious_count == 0:
        raise RuntimeError(
            "Primary paired predictions contain no malicious observations."
        )

    m1 = p1[malicious_mask]
    m2 = p2[malicious_mask]

    both_detect_mask = (m1 == 1) & (m2 == 1)
    v1_detect_v2_miss_mask = (m1 == 1) & (m2 == 0)
    v2_detect_v1_miss_mask = (m1 == 0) & (m2 == 1)
    both_miss_mask = (m1 == 0) & (m2 == 0)

    both_detect = int(both_detect_mask.sum())
    v1_detect_v2_miss = int(v1_detect_v2_miss_mask.sum())
    v2_detect_v1_miss = int(v2_detect_v1_miss_mask.sum())
    both_miss = int(both_miss_mask.sum())

    partition_sum = (
        both_detect
        + v1_detect_v2_miss
        + v2_detect_v1_miss
        + both_miss
    )

    if partition_sum != malicious_count:
        raise RuntimeError(
            "Malicious paired-outcome partition does not sum to "
            "the malicious population."
        )

    disagreement_count = (
        v1_detect_v2_miss + v2_detect_v1_miss
    )

    mrd = disagreement_count / malicious_count

    v1_misses = v2_detect_v1_miss + both_miss
    v2_misses = v1_detect_v2_miss + both_miss

    v1_detects = both_detect + v1_detect_v2_miss
    v2_detects = both_detect + v2_detect_v1_miss

    static_v1_fn = int(static_metrics["metrics"]["v1"]["fn"])
    static_v1_tp = int(static_metrics["metrics"]["v1"]["tp"])
    static_v2_fn = int(static_metrics["metrics"]["v2"]["fn"])
    static_v2_tp = int(static_metrics["metrics"]["v2"]["tp"])

    if v1_misses != static_v1_fn:
        raise RuntimeError(
            f"V1 malicious miss count disagrees with frozen static metrics: "
            f"paired={v1_misses}, static FN={static_v1_fn}."
        )

    if v2_misses != static_v2_fn:
        raise RuntimeError(
            f"V2 malicious miss count disagrees with frozen static metrics: "
            f"paired={v2_misses}, static FN={static_v2_fn}."
        )

    if v1_detects != static_v1_tp:
        raise RuntimeError(
            f"V1 malicious detection count disagrees with frozen static "
            f"metrics: paired={v1_detects}, static TP={static_v1_tp}."
        )

    if v2_detects != static_v2_tp:
        raise RuntimeError(
            f"V2 malicious detection count disagrees with frozen static "
            f"metrics: paired={v2_detects}, static TP={static_v2_tp}."
        )

    # Full malicious table with an explicit mutually-exclusive outcome.
    conditions = [
        (malicious["v1_prediction"] == 1)
        & (malicious["v2_prediction"] == 1),

        (malicious["v1_prediction"] == 1)
        & (malicious["v2_prediction"] == 0),

        (malicious["v1_prediction"] == 0)
        & (malicious["v2_prediction"] == 1),

        (malicious["v1_prediction"] == 0)
        & (malicious["v2_prediction"] == 0),
    ]

    outcome_names = [
        "both_detect",
        "v1_detects_v2_misses",
        "v2_detects_v1_misses",
        "both_miss",
    ]

    malicious["paired_outcome"] = np.select(
        conditions,
        outcome_names,
        default="INVALID",
    )

    if (malicious["paired_outcome"] == "INVALID").any():
        raise RuntimeError(
            "At least one malicious row could not be assigned a paired outcome."
        )

    result = {
        "malicious_population": {
            "definition": "label == 1",
            "count": malicious_count,
        },
        "paired_outcomes": {
            "both_detect": both_detect,
            "v1_detects_v2_misses": v1_detect_v2_miss,
            "v2_detects_v1_misses": v2_detect_v1_miss,
            "both_miss": both_miss,
            "partition_sum": partition_sum,
        },
        "model_level_consistency": {
            "v1": {
                "detects": v1_detects,
                "misses": v1_misses,
                "miss_rate": v1_misses / malicious_count,
                "static_tp_crosscheck": static_v1_tp,
                "static_fn_crosscheck": static_v1_fn,
            },
            "v2": {
                "detects": v2_detects,
                "misses": v2_misses,
                "miss_rate": v2_misses / malicious_count,
                "static_tp_crosscheck": static_v2_tp,
                "static_fn_crosscheck": static_v2_fn,
            },
        },
        "malicious_revision_disagreement": {
            "definition": (
                "label == 1 and v1_prediction != v2_prediction"
            ),
            "disagreement_count": disagreement_count,
            "v1_detects_v2_misses": v1_detect_v2_miss,
            "v2_detects_v1_misses": v2_detect_v1_miss,
            "mrd_definition": "|X_D| / |X_m|",
            "mrd": mrd,
            "formal_nonzero_condition": bool(disagreement_count > 0),
        },
    }

    return result, malicious


def compute_attack_family_breakdown(
    malicious: pd.DataFrame,
) -> tuple[list[dict[str, Any]], bool]:

    if "attack_cat" not in malicious.columns:
        return [], False

    family_series = (
        malicious["attack_cat"]
        .fillna("__MISSING__")
        .astype(str)
    )

    work = malicious.copy()
    work["attack_cat"] = family_series

    rows: list[dict[str, Any]] = []

    for attack_cat, group in work.groupby(
        "attack_cat",
        dropna=False,
        sort=True,
    ):
        n = len(group)

        both_detect = int(
            (group["paired_outcome"] == "both_detect").sum()
        )
        v1_detect_v2_miss = int(
            (
                group["paired_outcome"]
                == "v1_detects_v2_misses"
            ).sum()
        )
        v2_detect_v1_miss = int(
            (
                group["paired_outcome"]
                == "v2_detects_v1_misses"
            ).sum()
        )
        both_miss = int(
            (group["paired_outcome"] == "both_miss").sum()
        )

        disagreement = (
            v1_detect_v2_miss + v2_detect_v1_miss
        )

        if (
            both_detect
            + v1_detect_v2_miss
            + v2_detect_v1_miss
            + both_miss
            != n
        ):
            raise RuntimeError(
                f"Attack-family partition failed for {attack_cat!r}."
            )

        rows.append(
            {
                "attack_cat": str(attack_cat),
                "malicious_count": int(n),
                "both_detect": both_detect,
                "v1_detects_v2_misses": v1_detect_v2_miss,
                "v2_detects_v1_misses": v2_detect_v1_miss,
                "both_miss": both_miss,
                "disagreement_count": disagreement,
                "mrd_within_attack_family": (
                    disagreement / n if n else None
                ),
            }
        )

    if sum(row["malicious_count"] for row in rows) != len(work):
        raise RuntimeError(
            "Attack-family counts do not sum to the malicious population."
        )

    return rows, True


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute deterministic malicious revision disagreement from "
            "a frozen primary static-evaluation manifest."
        )
    )

    parser.add_argument(
        "--static-manifest",
        required=True,
        type=Path,
        help=(
            "Path to the frozen Primary Static Test Evaluation manifest."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    static_manifest_path = args.static_manifest.resolve()

    static_manifest = load_json(static_manifest_path)
    validate_static_manifest(static_manifest)

    repo_root = resolve_repo_root(static_manifest_path)

    print("============================================================")
    print("RolloutSecBench — MALICIOUS REVISION DISAGREEMENT")
    print("============================================================")
    print(f"[INFO] Repository root: {repo_root}")
    print(
        "[INFO] Static manifest: "
        f"{static_manifest_path.relative_to(repo_root)}"
    )

    static_manifest_hash = sha256_file(static_manifest_path)

    print(
        "[INFO] Static evaluation manifest SHA-256: "
        f"{static_manifest_hash}"
    )

    # -------------------------------------------------------------
    # Resolve frozen upstream evidence ONLY through static manifest.
    # -------------------------------------------------------------

    gate1_entry = static_manifest["authorization"]

    gate1_path = repo_path(
        repo_root,
        gate1_entry["gate1_path"],
    )

    gate1_hash = require_hash(
        gate1_path,
        gate1_entry["gate1_sha256"],
        "Gate 1 acceptance artifact",
    )

    gate1 = load_json(gate1_path)

    if gate1.get("gate_result", {}).get("status") != "PASS":
        raise RuntimeError(
            "Referenced Gate 1 artifact no longer reports PASS."
        )

    outputs = static_manifest["outputs"]

    prediction_entry = outputs["paired_predictions"]
    metrics_entry = outputs["static_metrics"]

    predictions_path = repo_path(
        repo_root,
        prediction_entry["path"],
    )

    static_metrics_path = repo_path(
        repo_root,
        metrics_entry["path"],
    )

    predictions_hash = require_hash(
        predictions_path,
        prediction_entry["sha256"],
        "Frozen paired predictions",
    )

    static_metrics_hash = require_hash(
        static_metrics_path,
        metrics_entry["sha256"],
        "Frozen static metrics",
    )

    static_metrics = load_json(static_metrics_path)

    validate_static_metrics(
        static_metrics,
        static_manifest,
    )

    expected_rows = int(prediction_entry["rows"])

    # -------------------------------------------------------------
    # This is the ONLY CSV read performed by Phase 4.
    # It reads the already-frozen paired prediction artifact.
    # -------------------------------------------------------------

    print(
        "[INFO] Reading frozen paired predictions only: "
        f"{predictions_path.relative_to(repo_root)}"
    )

    predictions = pd.read_csv(predictions_path)

    validate_prediction_table(
        predictions,
        expected_rows,
    )

    disagreement, malicious = compute_malicious_disagreement(
        predictions,
        static_metrics,
    )

    family_rows, family_available = compute_attack_family_breakdown(
        malicious
    )

    # -------------------------------------------------------------
    # Output locations are derived relative to the directory already
    # containing the frozen static outputs.
    # -------------------------------------------------------------

    output_dir = predictions_path.parent

    result_path = (
        output_dir / "malicious_disagreement_strict.json"
    )

    malicious_rows_path = (
        output_dir / "malicious_predictions_strict.csv"
    )

    family_path = (
        output_dir / "malicious_disagreement_by_attack_family_strict.csv"
    )

    manifest_path = (
        output_dir / "malicious_disagreement_manifest.json"
    )

    candidate_outputs = [
        result_path,
        malicious_rows_path,
        manifest_path,
    ]

    if family_available:
        candidate_outputs.append(family_path)

    existing = [p for p in candidate_outputs if p.exists()]

    if existing:
        joined = "\n".join(
            str(p.relative_to(repo_root))
            for p in existing
        )
        raise RuntimeError(
            "Refusing to overwrite existing Phase 4 scientific artifact(s):\n"
            + joined
        )

    # -------------------------------------------------------------
    # Write full malicious population, NOT only disagreement rows.
    # This preserves complete evidence and prevents favorable-subset
    # reporting.
    # -------------------------------------------------------------

    with malicious_rows_path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as f:
        malicious.to_csv(f, index=False)

    malicious_rows_hash = sha256_file(malicious_rows_path)

    if family_available:
        family_df = pd.DataFrame(family_rows)

        with family_path.open(
            "x",
            encoding="utf-8",
            newline="",
        ) as f:
            family_df.to_csv(f, index=False)

        family_hash = sha256_file(family_path)
    else:
        family_hash = None

    result_artifact = {
        "artifact": "RolloutSecBench",
        "schema_version": "1.0",
        "phase": "Malicious Revision Disagreement Analysis",
        "evaluation_population": {
            "source": "frozen primary paired predictions",
            "primary_test_name": "STRICT_NONOVERLAP_TEST",
            "full_prediction_rows": expected_rows,
        },
        **disagreement,
        "attack_family_analysis": {
            "available": family_available,
            "family_count": (
                len(family_rows) if family_available else 0
            ),
            "interpretation_scope": (
                "Descriptive within-family disagreement only."
                if family_available
                else "attack_cat was unavailable."
            ),
        },
        "gate2_evidence": {
            "formal_criterion_evaluated": "MRD > 0",
            "formal_nonzero_condition": disagreement[
                "malicious_revision_disagreement"
            ]["formal_nonzero_condition"],
            "practical_meaningfulness_threshold_applied": False,
            "note": (
                "No post-hoc numerical threshold for practical "
                "meaningfulness is introduced by this analysis."
            ),
        },
        "scientific_constraints": {
            "strict_test_raw_read": False,
            "official_test_full_read": False,
            "validation_read": False,
            "final_train_read": False,
            "model_loading_performed": False,
            "model_inference_performed": False,
            "model_fitting_performed": False,
            "preprocessor_loaded": False,
            "preprocessor_refit": False,
            "threshold_tuning_performed": False,
            "hyperparameter_tuning_performed": False,
            "rollout_analysis_performed": False,
            "only_disagreement_rows_reported": False,
        },
    }

    write_json_exclusive(
        result_path,
        result_artifact,
    )

    result_hash = sha256_file(result_path)

    disagreement_manifest = {
        "artifact": "RolloutSecBench",
        "schema_version": "1.0",
        "phase": "Malicious Revision Disagreement Analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_evidence": {
            "static_evaluation_manifest": {
                "path": str(
                    static_manifest_path.relative_to(repo_root)
                ),
                "sha256": static_manifest_hash,
            },
            "gate1_acceptance": {
                "path": str(gate1_path.relative_to(repo_root)),
                "sha256": gate1_hash,
                "status": gate1["gate_result"]["status"],
            },
            "paired_predictions": {
                "path": str(predictions_path.relative_to(repo_root)),
                "sha256": predictions_hash,
                "rows": expected_rows,
            },
            "static_metrics": {
                "path": str(static_metrics_path.relative_to(repo_root)),
                "sha256": static_metrics_hash,
            },
        },
        "derivation": {
            "malicious_population_definition": "label == 1",
            "disagreement_definition": (
                "label == 1 and v1_prediction != v2_prediction"
            ),
            "mrd_definition": "|X_D| / |X_m|",
            "paired_partition": [
                "both_detect",
                "v1_detects_v2_misses",
                "v2_detects_v1_misses",
                "both_miss",
            ],
            "attack_family_column": (
                "attack_cat" if family_available else None
            ),
            "model_reinference_performed": False,
        },
        "outputs": {
            "malicious_disagreement": {
                "path": str(result_path.relative_to(repo_root)),
                "sha256": result_hash,
            },
            "full_malicious_predictions": {
                "path": str(
                    malicious_rows_path.relative_to(repo_root)
                ),
                "sha256": malicious_rows_hash,
                "rows": int(len(malicious)),
            },
            "attack_family_breakdown": (
                {
                    "path": str(family_path.relative_to(repo_root)),
                    "sha256": family_hash,
                    "rows": len(family_rows),
                }
                if family_available
                else None
            ),
        },
        "analysis_scope": {
            "malicious_disagreement_computed": True,
            "mrd_computed": True,
            "attack_family_breakdown_computed": family_available,
            "inferential_significance_testing_performed": False,
            "practical_mrd_threshold_applied": False,
            "rollout_analysis_performed": False,
        },
    }

    write_json_exclusive(
        manifest_path,
        disagreement_manifest,
    )

    manifest_hash = sha256_file(manifest_path)

    d = disagreement["malicious_revision_disagreement"]
    paired = disagreement["paired_outcomes"]

    print()
    print("============================================================")
    print("MALICIOUS REVISION DISAGREEMENT COMPLETE")
    print("============================================================")
    print(
        "Malicious observations:",
        disagreement["malicious_population"]["count"],
    )
    print("Both detect:", paired["both_detect"])
    print(
        "V1 detects / V2 misses:",
        paired["v1_detects_v2_misses"],
    )
    print(
        "V2 detects / V1 misses:",
        paired["v2_detects_v1_misses"],
    )
    print("Both miss:", paired["both_miss"])
    print("Malicious disagreement:", d["disagreement_count"])
    print("MRD:", d["mrd"])
    print(
        "Formal MRD > 0 condition:",
        d["formal_nonzero_condition"],
    )

    print()
    print("[OUTPUT] disagreement result:")
    print(f"  {result_path.relative_to(repo_root)}")
    print(f"  SHA256: {result_hash}")

    print("[OUTPUT] full malicious paired predictions:")
    print(f"  {malicious_rows_path.relative_to(repo_root)}")
    print(f"  SHA256: {malicious_rows_hash}")

    if family_available:
        print("[OUTPUT] attack-family breakdown:")
        print(f"  {family_path.relative_to(repo_root)}")
        print(f"  SHA256: {family_hash}")

    print("[OUTPUT] disagreement manifest:")
    print(f"  {manifest_path.relative_to(repo_root)}")
    print(f"  SHA256: {manifest_hash}")

    print()
    print(
        "No model inference was repeated. "
        "No strict-test raw data or official_test_full data was read. "
        "No practical MRD threshold was introduced. "
        "No rollout analysis was performed."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
