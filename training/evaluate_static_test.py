#!/usr/bin/env python3
"""
RolloutSecBench — Phase 3/4 boundary:
Frozen-model static evaluation on the PRIMARY strict non-overlap test set.

Scientific constraints
----------------------
- Requires the previously completed validation-only Gate 1 to PASS.
- Uses only the prospectively accepted frozen V1/V2 models.
- Uses only the frozen preprocessor and frozen source-feature order.
- Reads ONLY the primary STRICT_NONOVERLAP_TEST.
- Does NOT read OFFICIAL_TEST_FULL.
- Does NOT fit/refit models or preprocessing.
- Does NOT tune thresholds or hyperparameters.
- Does NOT compute malicious disagreement/MRD.
- Does NOT perform rollout analysis.
- Refuses to overwrite existing scientific result artifacts.

All operational paths and hashes are resolved from repository manifests/artifacts
where those manifests define them. Scientific definitions are not inferred from
model output.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# Repository anchor.
# This is structural, not experimental evidence.
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_MANIFEST_PATH = REPO_ROOT / "data/manifests/dataset_manifest.json"
SPLIT_MANIFEST_PATH = REPO_ROOT / "data/manifests/split_manifest.json"
PREPROCESSING_MANIFEST_PATH = (
    REPO_ROOT / "data/manifests/preprocessing_manifest.json"
)
ACCEPTANCE_PATH = (
    REPO_ROOT / "results/derived/model_revision_acceptance.json"
)

OUTPUT_METRICS = (
    REPO_ROOT / "results/derived/static_model_metrics_strict.json"
)
OUTPUT_PREDICTIONS = (
    REPO_ROOT / "results/derived/pairwise_predictions_strict.csv"
)
OUTPUT_MANIFEST = (
    REPO_ROOT / "results/derived/static_test_evaluation_manifest.json"
)


# ---------------------------------------------------------------------
# Generic integrity helpers
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


def repo_path(relative_path: str) -> Path:
    p = Path(relative_path)

    if p.is_absolute():
        raise RuntimeError(
            f"Manifest path must be repository-relative, got: {relative_path}"
        )

    resolved = (REPO_ROOT / p).resolve()

    try:
        resolved.relative_to(REPO_ROOT.resolve())
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


def refuse_existing_outputs() -> None:
    existing = [
        p for p in
        (OUTPUT_METRICS, OUTPUT_PREDICTIONS, OUTPUT_MANIFEST)
        if p.exists()
    ]

    if existing:
        joined = "\n".join(str(p.relative_to(REPO_ROOT)) for p in existing)
        raise RuntimeError(
            "Refusing to overwrite existing scientific result artifact(s):\n"
            f"{joined}"
        )


# ---------------------------------------------------------------------
# Gate 1 authorization
# ---------------------------------------------------------------------

def validate_gate1(acceptance: dict[str, Any]) -> None:
    if acceptance.get("gate") != "model_revision_validation_acceptance":
        raise RuntimeError("Unexpected Gate 1 artifact type.")

    gate_result = acceptance.get("gate_result", {})

    if gate_result.get("status") != "PASS":
        raise RuntimeError("Gate 1 status is not PASS.")

    if gate_result.get("all_required_conditions_passed") is not True:
        raise RuntimeError(
            "Gate 1 does not report all required conditions as passed."
        )

    evidence = acceptance.get("evidence_scope", {})

    if evidence.get("validation_only") is not True:
        raise RuntimeError(
            "Gate 1 artifact does not identify itself as validation-only."
        )

    if evidence.get("test_data_read_by_this_script") is not False:
        raise RuntimeError(
            "Gate 1 artifact does not establish test-data non-access."
        )

    if evidence.get("malicious_disagreement_computed") is not False:
        raise RuntimeError(
            "Gate 1 artifact unexpectedly reports prior disagreement analysis."
        )

    if evidence.get("rollout_analysis_performed") is not False:
        raise RuntimeError(
            "Gate 1 artifact unexpectedly reports rollout analysis."
        )

    print("[PASS] Gate 1 authorization verified.")


# ---------------------------------------------------------------------
# Manifest resolution
# ---------------------------------------------------------------------

def resolve_model_manifest(
    acceptance: dict[str, Any],
    version: str,
) -> tuple[Path, dict[str, Any], str]:

    entry = acceptance["input_manifests"][version]

    path = repo_path(entry["path"])
    digest = require_hash(
        path,
        entry["sha256"],
        f"{version.upper()} model manifest",
    )

    return path, load_json(path), digest


def resolve_model_artifact(
    manifest: dict[str, Any],
    version: str,
) -> tuple[Path, str]:

    entry = manifest["artifacts"]["model"]
    path = repo_path(entry["path"])

    digest = require_hash(
        path,
        entry["sha256"],
        f"{version.upper()} model artifact",
    )

    return path, digest


def threshold_from_manifest(
    manifest: dict[str, Any],
    version: str,
) -> float:

    if version == "v1":
        threshold = manifest["training"]["threshold"]
        policy = manifest["training"]["threshold_policy"]

    elif version == "v2":
        threshold = manifest["decision_policy"]["decision_threshold"]
        policy = manifest["decision_policy"]["threshold_policy"]

    else:
        raise RuntimeError(f"Unsupported version: {version}")

    if policy.get("type") != "FIXED":
        raise RuntimeError(
            f"{version.upper()} threshold policy is not FIXED."
        )

    if policy.get("threshold_selection_data") != "NONE":
        raise RuntimeError(
            f"{version.upper()} threshold unexpectedly used selection data."
        )

    threshold = float(threshold)

    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError(
            f"{version.upper()} threshold outside [0,1]: {threshold}"
        )

    if float(policy["threshold"]) != threshold:
        raise RuntimeError(
            f"{version.upper()} threshold and threshold-policy disagree."
        )

    return threshold


# ---------------------------------------------------------------------
# Dataset/preprocessing resolution
# ---------------------------------------------------------------------

def resolve_primary_test(
    dataset_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
) -> tuple[Path, str, int, int]:

    evaluation_policy = dataset_manifest["evaluation_policy"]

    if (
        evaluation_policy.get("primary_static_evaluation_set")
        != "STRICT_NONOVERLAP_TEST"
    ):
        raise RuntimeError(
            "Dataset manifest no longer declares STRICT_NONOVERLAP_TEST "
            "as the primary static evaluation set."
        )

    strict_policy = dataset_manifest["split_policy"]["strict_test_nonoverlap"]

    if strict_policy.get("role") != "PRIMARY_EVALUATION":
        raise RuntimeError(
            "strict_test_nonoverlap is not marked PRIMARY_EVALUATION."
        )

    if strict_policy.get("membership_uses_labels") is not False:
        raise RuntimeError(
            "Strict-test membership unexpectedly uses labels."
        )

    if strict_policy.get("membership_uses_model_outputs") is not False:
        raise RuntimeError(
            "Strict-test membership unexpectedly uses model outputs."
        )

    if strict_policy.get("membership_uses_model_performance") is not False:
        raise RuntimeError(
            "Strict-test membership unexpectedly uses model performance."
        )

    artifact = split_manifest["artifacts"]["strict_test_nonoverlap"]

    path = repo_path(artifact["relative_path"])
    digest = require_hash(
        path,
        artifact["sha256"],
        "Primary strict test",
    )

    rows = int(artifact["rows"])
    columns = int(artifact["columns"])

    return path, digest, rows, columns


def resolve_preprocessing(
    dataset_manifest: dict[str, Any],
    preprocessing_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
) -> tuple[
    Path,
    str,
    Path,
    str,
    list[str],
    int,
]:

    dataset_policy = dataset_manifest["preprocessing_policy"]

    if dataset_policy.get("fit_scope") != "final_train_only":
        raise RuntimeError(
            "Frozen preprocessing fit scope is not final_train_only."
        )

    if dataset_policy.get("test_data_may_influence_fit") is not False:
        raise RuntimeError(
            "Dataset manifest unexpectedly permits test influence on "
            "preprocessing."
        )

    prep_policy = preprocessing_manifest["policy"]

    if prep_policy.get("fit_scope") != "final_train_only":
        raise RuntimeError(
            "Preprocessing manifest fit scope is not final_train_only."
        )

    if prep_policy.get("test_may_influence_fit") is not False:
        raise RuntimeError(
            "Preprocessing manifest unexpectedly permits test influence."
        )

    preprocessor_entry = preprocessing_manifest["artifacts"]["preprocessor"]
    feature_entry = preprocessing_manifest["artifacts"]["feature_names"]

    preprocessor_path = repo_path(preprocessor_entry["path"])
    feature_names_path = repo_path(feature_entry["path"])

    preprocessor_hash = require_hash(
        preprocessor_path,
        preprocessor_entry["sha256"],
        "Frozen preprocessor",
    )

    feature_names_hash = require_hash(
        feature_names_path,
        feature_entry["sha256"],
        "Frozen feature-names artifact",
    )

    feature_names = load_json(feature_names_path)

    source_order = feature_names.get("source_feature_order")

    if not isinstance(source_order, list):
        raise RuntimeError(
            "feature_names.json has no valid source_feature_order."
        )

    split_features = split_manifest["feature_representation"]["features"]

    if source_order != split_features:
        raise RuntimeError(
            "Frozen feature_names source order does not exactly match "
            "split_manifest feature order."
        )

    raw_count = int(
        preprocessing_manifest["feature_space"]["source_feature_count"]
    )

    transformed_count = int(
        preprocessing_manifest["feature_space"]["transformed_feature_count"]
    )

    if len(source_order) != raw_count:
        raise RuntimeError(
            "Frozen source feature count does not match source order."
        )

    if int(feature_names["raw_feature_count"]) != raw_count:
        raise RuntimeError(
            "feature_names raw feature count disagrees with preprocessing "
            "manifest."
        )

    if (
        int(feature_names["transformed_feature_count"])
        != transformed_count
    ):
        raise RuntimeError(
            "feature_names transformed count disagrees with preprocessing "
            "manifest."
        )

    return (
        preprocessor_path,
        preprocessor_hash,
        feature_names_path,
        feature_names_hash,
        source_order,
        transformed_count,
    )


# ---------------------------------------------------------------------
# Static metrics
# ---------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Predicted probabilities contain NaN or infinity.")

    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise RuntimeError("Predicted probabilities fall outside [0,1].")

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])

    if cm.shape != (2, 2):
        raise RuntimeError(
            f"Unexpected confusion-matrix shape: {cm.shape}"
        )

    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    denominator = fp + tn

    if denominator == 0:
        raise RuntimeError(
            "Cannot calculate FPR: test set contains no negative examples."
        )

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(
            average_precision_score(y_true, probabilities)
        ),
        "fpr": float(fp / denominator),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    refuse_existing_outputs()

    print("============================================================")
    print("RolloutSecBench — PRIMARY STATIC TEST EVALUATION")
    print("============================================================")

    acceptance = load_json(ACCEPTANCE_PATH)
    acceptance_hash = sha256_file(ACCEPTANCE_PATH)

    print(
        "[INFO] Gate 1 artifact SHA-256: "
        f"{acceptance_hash}"
    )

    validate_gate1(acceptance)

    dataset_manifest = load_json(DATASET_MANIFEST_PATH)
    split_manifest = load_json(SPLIT_MANIFEST_PATH)
    preprocessing_manifest = load_json(PREPROCESSING_MANIFEST_PATH)

    dataset_manifest_hash = sha256_file(DATASET_MANIFEST_PATH)
    split_manifest_hash = sha256_file(SPLIT_MANIFEST_PATH)
    preprocessing_manifest_hash = sha256_file(
        PREPROCESSING_MANIFEST_PATH
    )

    v1_manifest_path, v1_manifest, v1_manifest_hash = (
        resolve_model_manifest(acceptance, "v1")
    )
    v2_manifest_path, v2_manifest, v2_manifest_hash = (
        resolve_model_manifest(acceptance, "v2")
    )

    v1_model_path, v1_model_hash = resolve_model_artifact(
        v1_manifest, "v1"
    )
    v2_model_path, v2_model_hash = resolve_model_artifact(
        v2_manifest, "v2"
    )

    threshold_v1 = threshold_from_manifest(v1_manifest, "v1")
    threshold_v2 = threshold_from_manifest(v2_manifest, "v2")

    gate_threshold = float(
        acceptance["shared_frozen_inputs"]["decision_threshold"]
    )

    if threshold_v1 != gate_threshold or threshold_v2 != gate_threshold:
        raise RuntimeError(
            "Current frozen model thresholds disagree with the "
            "Gate 1 accepted threshold."
        )

    print(f"[PASS] V1 threshold: {threshold_v1}")
    print(f"[PASS] V2 threshold: {threshold_v2}")

    strict_path, strict_hash, expected_rows, expected_columns = (
        resolve_primary_test(dataset_manifest, split_manifest)
    )

    (
        preprocessor_path,
        preprocessor_hash,
        feature_names_path,
        feature_names_hash,
        source_features,
        transformed_count,
    ) = resolve_preprocessing(
        dataset_manifest,
        preprocessing_manifest,
        split_manifest,
    )

    gate_inputs = acceptance["shared_frozen_inputs"]

    if gate_inputs["preprocessor_sha256"] != preprocessor_hash:
        raise RuntimeError(
            "Preprocessor hash differs from Gate 1 frozen input."
        )

    if gate_inputs["feature_names_sha256"] != feature_names_hash:
        raise RuntimeError(
            "Feature-names hash differs from Gate 1 frozen input."
        )

    if int(gate_inputs["raw_feature_count"]) != len(source_features):
        raise RuntimeError(
            "Gate 1 raw feature count disagrees with current frozen "
            "representation."
        )

    if (
        int(gate_inputs["transformed_feature_count"])
        != transformed_count
    ):
        raise RuntimeError(
            "Gate 1 transformed feature count disagrees with current "
            "frozen representation."
        )

    # -------------------------------------------------------------
    # FIRST AND ONLY TEST DATA READ IN THIS SCRIPT
    # -------------------------------------------------------------

    print(
        "[INFO] Opening PRIMARY strict non-overlap test only: "
        f"{strict_path.relative_to(REPO_ROOT)}"
    )

    test_df = pd.read_csv(strict_path)

    if len(test_df) != expected_rows:
        raise RuntimeError(
            f"Strict-test row-count mismatch: "
            f"expected {expected_rows}, observed {len(test_df)}."
        )

    if len(test_df.columns) != expected_columns:
        raise RuntimeError(
            f"Strict-test column-count mismatch: "
            f"expected {expected_columns}, observed "
            f"{len(test_df.columns)}."
        )

    dataset_schema = dataset_manifest["selected_file_schema"]

    expected_schema = dataset_schema["columns"]

    if list(test_df.columns) != expected_schema:
        raise RuntimeError(
            "Strict-test column names/order do not exactly match "
            "the frozen selected-file schema."
        )

    label_column = dataset_schema["binary_label_column"]
    attack_column = dataset_schema["multiclass_label_column"]

    expected_labels = set(
        int(x) for x in dataset_schema["expected_binary_label_values"]
    )

    observed_labels = set(
        int(x) for x in pd.unique(test_df[label_column])
    )

    if observed_labels != expected_labels:
        raise RuntimeError(
            f"Unexpected strict-test binary labels. "
            f"Expected {expected_labels}, observed {observed_labels}."
        )

    missing_features = [
        c for c in source_features if c not in test_df.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Strict test is missing frozen source features: "
            + ", ".join(missing_features)
        )

    y_true = test_df[label_column].to_numpy(dtype=np.int8)

    preprocessor = joblib.load(preprocessor_path)

    if hasattr(preprocessor, "feature_names_in_"):
        learned_order = list(preprocessor.feature_names_in_)

        if learned_order != source_features:
            raise RuntimeError(
                "Serialized preprocessor feature_names_in_ does not "
                "match frozen source-feature order."
            )

    # Transform only. No fit/refit occurs here.
    X_test = preprocessor.transform(test_df[source_features])

    if X_test.shape[0] != expected_rows:
        raise RuntimeError(
            "Transformed strict-test row count changed unexpectedly."
        )

    if X_test.shape[1] != transformed_count:
        raise RuntimeError(
            f"Transformed strict-test width mismatch: expected "
            f"{transformed_count}, observed {X_test.shape[1]}."
        )

    print(
        f"[PASS] Strict test transformed: "
        f"{X_test.shape[0]} x {X_test.shape[1]}"
    )

    v1 = XGBClassifier()
    v1.load_model(str(v1_model_path))

    v2 = XGBClassifier()
    v2.load_model(str(v2_model_path))

    for name, model in (("V1", v1), ("V2", v2)):
        model_width = getattr(model, "n_features_in_", None)

        if model_width is None:
            raise RuntimeError(
                f"{name} serialized model does not report n_features_in_."
            )

        if int(model_width) != transformed_count:
            raise RuntimeError(
                f"{name} model width mismatch: expected "
                f"{transformed_count}, observed {model_width}."
            )

    p1 = v1.predict_proba(X_test)[:, 1].astype(np.float64)
    p2 = v2.predict_proba(X_test)[:, 1].astype(np.float64)

    pred1 = (p1 >= threshold_v1).astype(np.int8)
    pred2 = (p2 >= threshold_v2).astype(np.int8)

    metrics_v1 = compute_metrics(y_true, p1, pred1)
    metrics_v2 = compute_metrics(y_true, p2, pred2)

    # Preserve row-level paired outputs for later Phase 4 disagreement
    # analysis. No disagreement statistic is computed here.
    prediction_df = pd.DataFrame()

    if "id" in test_df.columns:
        prediction_df["id"] = test_df["id"].to_numpy()

    if attack_column in test_df.columns:
        prediction_df[attack_column] = (
            test_df[attack_column]
            .fillna("__MISSING__")
            .astype(str)
            .to_numpy()
        )

    prediction_df[label_column] = y_true
    prediction_df["v1_probability"] = p1
    prediction_df["v1_prediction"] = pred1
    prediction_df["v2_probability"] = p2
    prediction_df["v2_prediction"] = pred2

    OUTPUT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents silent overwrite.
    with OUTPUT_PREDICTIONS.open(
        "x", encoding="utf-8", newline=""
    ) as f:
        prediction_df.to_csv(f, index=False)

    predictions_hash = sha256_file(OUTPUT_PREDICTIONS)

    metrics_artifact = {
        "artifact": "RolloutSecBench",
        "schema_version": "1.0",
        "phase": "Primary Static Test Evaluation",
        "evaluation_set": {
            "name": "STRICT_NONOVERLAP_TEST",
            "role": "PRIMARY_EVALUATION",
            "path": str(strict_path.relative_to(REPO_ROOT)),
            "sha256": strict_hash,
            "rows": expected_rows,
            "columns": expected_columns,
        },
        "decision_thresholds": {
            "v1": threshold_v1,
            "v2": threshold_v2,
        },
        "metrics": {
            "v1": metrics_v1,
            "v2": metrics_v2,
        },
        "scientific_constraints": {
            "model_fitting_performed": False,
            "preprocessor_refit": False,
            "threshold_tuning_performed": False,
            "hyperparameter_tuning_performed": False,
            "official_test_full_read": False,
            "malicious_disagreement_computed": False,
            "rollout_analysis_performed": False,
        },
    }

    write_json_exclusive(OUTPUT_METRICS, metrics_artifact)
    metrics_hash = sha256_file(OUTPUT_METRICS)

    evaluation_manifest = {
        "artifact": "RolloutSecBench",
        "schema_version": "1.0",
        "phase": "Primary Static Test Evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": {
            "gate1_path": str(ACCEPTANCE_PATH.relative_to(REPO_ROOT)),
            "gate1_sha256": acceptance_hash,
            "gate1_status": acceptance["gate_result"]["status"],
        },
        "input_manifests": {
            "dataset_manifest": {
                "path": str(
                    DATASET_MANIFEST_PATH.relative_to(REPO_ROOT)
                ),
                "sha256_at_evaluation": dataset_manifest_hash,
            },
            "split_manifest": {
                "path": str(
                    SPLIT_MANIFEST_PATH.relative_to(REPO_ROOT)
                ),
                "sha256_at_evaluation": split_manifest_hash,
            },
            "preprocessing_manifest": {
                "path": str(
                    PREPROCESSING_MANIFEST_PATH.relative_to(REPO_ROOT)
                ),
                "sha256_at_evaluation": preprocessing_manifest_hash,
            },
            "v1_model_manifest": {
                "path": str(v1_manifest_path.relative_to(REPO_ROOT)),
                "sha256": v1_manifest_hash,
            },
            "v2_model_manifest": {
                "path": str(v2_manifest_path.relative_to(REPO_ROOT)),
                "sha256": v2_manifest_hash,
            },
        },
        "frozen_inputs": {
            "strict_test": {
                "path": str(strict_path.relative_to(REPO_ROOT)),
                "sha256": strict_hash,
                "rows": expected_rows,
            },
            "preprocessor": {
                "path": str(
                    preprocessor_path.relative_to(REPO_ROOT)
                ),
                "sha256": preprocessor_hash,
            },
            "feature_names": {
                "path": str(
                    feature_names_path.relative_to(REPO_ROOT)
                ),
                "sha256": feature_names_hash,
            },
            "v1_model": {
                "path": str(v1_model_path.relative_to(REPO_ROOT)),
                "sha256": v1_model_hash,
            },
            "v2_model": {
                "path": str(v2_model_path.relative_to(REPO_ROOT)),
                "sha256": v2_model_hash,
            },
            "source_feature_count": len(source_features),
            "transformed_feature_count": transformed_count,
            "v1_threshold": threshold_v1,
            "v2_threshold": threshold_v2,
        },
        "outputs": {
            "static_metrics": {
                "path": str(OUTPUT_METRICS.relative_to(REPO_ROOT)),
                "sha256": metrics_hash,
            },
            "paired_predictions": {
                "path": str(
                    OUTPUT_PREDICTIONS.relative_to(REPO_ROOT)
                ),
                "sha256": predictions_hash,
                "rows": int(len(prediction_df)),
            },
        },
        "data_access": {
            "strict_test_nonoverlap_read": True,
            "official_test_full_read": False,
            "validation_read": False,
            "final_train_read": False,
        },
        "analysis_scope": {
            "full_population_static_metrics_computed": True,
            "malicious_disagreement_computed": False,
            "mrd_computed": False,
            "rollout_analysis_performed": False,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }

    write_json_exclusive(OUTPUT_MANIFEST, evaluation_manifest)
    manifest_hash = sha256_file(OUTPUT_MANIFEST)

    print()
    print("============================================================")
    print("PRIMARY STATIC TEST EVALUATION COMPLETE")
    print("============================================================")
    print(f"Test rows: {expected_rows}")
    print()
    print("V1:")
    for k, v in metrics_v1.items():
        print(f"  {k}: {v}")

    print()
    print("V2:")
    for k, v in metrics_v2.items():
        print(f"  {k}: {v}")

    print()
    print("[OUTPUT] static metrics:")
    print(
        f"  {OUTPUT_METRICS.relative_to(REPO_ROOT)}"
    )
    print(f"  SHA256: {metrics_hash}")

    print("[OUTPUT] paired predictions:")
    print(
        f"  {OUTPUT_PREDICTIONS.relative_to(REPO_ROOT)}"
    )
    print(f"  SHA256: {predictions_hash}")

    print("[OUTPUT] evaluation manifest:")
    print(
        f"  {OUTPUT_MANIFEST.relative_to(REPO_ROOT)}"
    )
    print(f"  SHA256: {manifest_hash}")

    print()
    print(
        "No malicious disagreement/MRD has been computed. "
        "No official_test_full data was read. "
        "Next step after reviewing/freezing these outputs: "
        "Phase 4 malicious disagreement analysis."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
