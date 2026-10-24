#!/usr/bin/env python3
"""
RolloutSecBench - Train frozen V1 baseline model.

This script:
- verifies the frozen Phase-2 model-revision protocol and its SHA-256;
- verifies the exact training/validation/preprocessing artifacts referenced by the protocol;
- refuses to read either test set;
- fits V1 only on final_train using the frozen preprocessor;
- evaluates V1 only on validation;
- serializes the model;
- verifies reload-equivalent predictions;
- writes validation predictions and a model manifest with hashes.

It does NOT:
- tune hyperparameters;
- change the frozen threshold;
- inspect strict_test_nonoverlap or official_test_full;
- alter preprocessing;
- decide anything about V2 or rollout disagreement.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
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


REPO_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_PATH = REPO_ROOT / "data" / "manifests" / "model_revision_protocol_v1.0.json"
PROTOCOL_SHA_PATH = REPO_ROOT / "data" / "manifests" / "model_revision_protocol_v1.0.sha256"
DATASET_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "dataset_manifest.json"

FINAL_TRAIN_PATH = REPO_ROOT / "data" / "splits" / "final_train.csv"
VALIDATION_PATH = REPO_ROOT / "data" / "splits" / "validation.csv"
PREPROCESSOR_PATH = REPO_ROOT / "models" / "preprocessing" / "preprocessor.joblib"
FEATURE_NAMES_PATH = REPO_ROOT / "data" / "processed" / "feature_names.json"
PREPROCESSING_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "preprocessing_manifest.json"

MODEL_DIR = REPO_ROOT / "models" / "v1"
MODEL_PATH = MODEL_DIR / "model.json"
MODEL_MANIFEST_PATH = MODEL_DIR / "model_manifest.json"
VALIDATION_PREDICTIONS_PATH = (
    REPO_ROOT / "results" / "derived" / "v1_validation_predictions.csv"
)

OUTPUT_PATHS = [
    MODEL_PATH,
    MODEL_MANIFEST_PATH,
    VALIDATION_PREDICTIONS_PATH,
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_json_write(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps(obj, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_csv_write(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def parse_sha256_sidecar(path: Path) -> tuple[str, str]:
    line = path.read_text(encoding="utf-8").strip()
    parts = line.split()
    if len(parts) < 2:
        raise RuntimeError(f"Invalid SHA-256 sidecar: {path}")
    return parts[0].lower(), parts[-1]


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA-256 mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )


def ensure_outputs_absent() -> None:
    existing = [p for p in OUTPUT_PATHS if p.exists()]
    if existing:
        lines = "\n".join(f"  - {p.relative_to(REPO_ROOT)}" for p in existing)
        raise FileExistsError(
            "Refusing to overwrite frozen V1 artifacts:\n"
            f"{lines}\n"
            "If this was a deliberate failed dry run, archive it first; do not silently overwrite."
        )


def verify_protocol_integrity(protocol: dict[str, Any]) -> str:
    expected_protocol_hash, recorded_name = parse_sha256_sidecar(PROTOCOL_SHA_PATH)
    actual_protocol_hash = sha256_file(PROTOCOL_PATH)

    if actual_protocol_hash.lower() != expected_protocol_hash.lower():
        raise RuntimeError(
            "Frozen model-revision protocol hash check FAILED.\n"
            f"Sidecar expected: {expected_protocol_hash}\n"
            f"Actual:           {actual_protocol_hash}"
        )

    if Path(recorded_name).name != PROTOCOL_PATH.name:
        raise RuntimeError(
            "Protocol SHA-256 sidecar refers to an unexpected filename: "
            f"{recorded_name}"
        )

    if protocol.get("protocol_status") != "FROZEN_PRETRAINING":
        raise RuntimeError("Protocol status must be FROZEN_PRETRAINING.")

    return actual_protocol_hash


def verify_training_boundary() -> str:
    dataset_manifest = load_json(DATASET_MANIFEST_PATH)
    tb = dataset_manifest.get("training_boundary", {})
    if tb.get("model_training_allowed_now") is not True:
        raise RuntimeError(
            "dataset_manifest training boundary is not open. "
            "model_training_allowed_now must be true."
        )
    return sha256_file(DATASET_MANIFEST_PATH)


def protocol_artifact_hash(
    protocol: dict[str, Any],
    section: str,
    key: str = "sha256",
) -> str:
    try:
        return str(protocol["frozen_inputs"][section][key])
    except KeyError as exc:
        raise KeyError(
            f"Frozen protocol missing frozen_inputs.{section}.{key}"
        ) from exc


def verify_frozen_inputs(protocol: dict[str, Any]) -> None:
    require_hash(
        FINAL_TRAIN_PATH,
        protocol_artifact_hash(protocol, "training_data"),
        "final_train",
    )
    require_hash(
        VALIDATION_PATH,
        protocol_artifact_hash(protocol, "validation_data"),
        "validation",
    )
    require_hash(
        PREPROCESSOR_PATH,
        protocol_artifact_hash(protocol, "preprocessor"),
        "preprocessor",
    )
    require_hash(
        FEATURE_NAMES_PATH,
        protocol_artifact_hash(protocol, "feature_names"),
        "feature_names",
    )
    require_hash(
        PREPROCESSING_MANIFEST_PATH,
        protocol_artifact_hash(protocol, "preprocessing_manifest"),
        "preprocessing_manifest",
    )

    # Deliberately do NOT open either test file in this training script.


def get_source_feature_order() -> list[str]:
    doc = load_json(FEATURE_NAMES_PATH)
    features = doc.get("source_feature_order")
    if not isinstance(features, list) or len(features) != 42:
        raise RuntimeError(
            "feature_names.json must contain source_feature_order with exactly 42 features."
        )
    if any(c in features for c in ["id", "attack_cat", "label"]):
        raise RuntimeError("Non-model metadata appears in source_feature_order.")
    return [str(x) for x in features]


def clean_xgb_params(params: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "objective",
        "eval_metric",
        "tree_method",
        "random_state",
        "n_jobs",
    }
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise RuntimeError(
            f"Unexpected V1 hyperparameters in frozen protocol: {unknown}"
        )
    return {k: params[k] for k in params}


def validate_shared_policy(protocol: dict[str, Any]) -> None:
    shared = protocol.get("shared_training_policy", {})
    required = {
        "training_source": "final_train_only",
        "validation_source": "validation_only",
        "test_data_used_for_training": False,
        "test_data_used_for_hyperparameter_selection": False,
        "test_data_used_for_threshold_selection": False,
        "test_data_used_for_model_acceptance": False,
        "preprocessor_refitting_allowed": False,
        "feature_order_modification_allowed": False,
    }
    for key, expected in required.items():
        if shared.get(key) != expected:
            raise RuntimeError(
                f"Frozen shared_training_policy.{key} must be {expected!r}; "
                f"found {shared.get(key)!r}"
            )

    selection = protocol.get("model_selection_policy", {})
    for key in [
        "hyperparameter_search_permitted",
        "posthoc_pair_selection_permitted",
        "selection_based_on_malicious_disagreement_permitted",
        "selection_based_on_mrd_permitted",
        "selection_based_on_rollout_asr_permitted",
    ]:
        if selection.get(key) is not False:
            raise RuntimeError(f"Frozen model_selection_policy.{key} must be false.")


def threshold_from_protocol(protocol: dict[str, Any]) -> float:
    threshold_policy = protocol["revision_v1"]["threshold_policy"]
    if threshold_policy.get("type") != "FIXED":
        raise RuntimeError("V1 threshold policy must be FIXED.")
    threshold = float(threshold_policy["threshold"])
    if not (0.0 < threshold < 1.0):
        raise RuntimeError(f"Invalid V1 threshold: {threshold}")
    return threshold


def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "fpr": fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    print("=== RolloutSecBench V1 Frozen Training ===")

    for required in [
        PROTOCOL_PATH,
        PROTOCOL_SHA_PATH,
        DATASET_MANIFEST_PATH,
        FINAL_TRAIN_PATH,
        VALIDATION_PATH,
        PREPROCESSOR_PATH,
        FEATURE_NAMES_PATH,
        PREPROCESSING_MANIFEST_PATH,
    ]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")

    ensure_outputs_absent()

    protocol = load_json(PROTOCOL_PATH)
    protocol_hash = verify_protocol_integrity(protocol)
    dataset_manifest_hash = verify_training_boundary()
    validate_shared_policy(protocol)
    verify_frozen_inputs(protocol)

    source_features = get_source_feature_order()
    threshold = threshold_from_protocol(protocol)

    v1 = protocol.get("revision_v1", {})
    if v1.get("revision_role") != "baseline_production_revision":
        raise RuntimeError(
            "revision_v1.revision_role must be baseline_production_revision."
        )

    params = clean_xgb_params(dict(v1["hyperparameters"]))

    print("\n=== FROZEN V1 CONTRACT ===")
    print(f"[+] Protocol SHA-256:       {protocol_hash}")
    print(f"[+] Training source:        final_train ONLY")
    print(f"[+] Validation source:      validation ONLY")
    print(f"[+] Raw source features:    {len(source_features)}")
    print(f"[+] Decision threshold:     {threshold}")
    print(f"[+] Hyperparameter search:  DISABLED")
    print(f"[+] Test access:            DISABLED")
    print(f"[+] V1 hyperparameters:     {json.dumps(params, sort_keys=True)}")

    train_df = pd.read_csv(FINAL_TRAIN_PATH)
    val_df = pd.read_csv(VALIDATION_PATH)

    for split_name, df in [("final_train", train_df), ("validation", val_df)]:
        missing = [c for c in source_features + ["label"] if c not in df.columns]
        if missing:
            raise RuntimeError(f"{split_name} missing columns: {missing}")

    y_train = train_df["label"].to_numpy(dtype=np.int8)
    y_val = val_df["label"].to_numpy(dtype=np.int8)

    if set(np.unique(y_train)) - {0, 1}:
        raise RuntimeError("final_train label contains values other than 0/1.")
    if set(np.unique(y_val)) - {0, 1}:
        raise RuntimeError("validation label contains values other than 0/1.")

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Transform only. The frozen preprocessor is NEVER refit here.
    X_train = preprocessor.transform(train_df[source_features])
    X_val = preprocessor.transform(val_df[source_features])

    expected_transformed_features = int(
        protocol["frozen_inputs"]["feature_names"]["transformed_feature_count"]
    )
    if X_train.shape[1] != expected_transformed_features:
        raise RuntimeError(
            f"Training transformed width mismatch: "
            f"{X_train.shape[1]} != {expected_transformed_features}"
        )
    if X_val.shape[1] != expected_transformed_features:
        raise RuntimeError(
            f"Validation transformed width mismatch: "
            f"{X_val.shape[1]} != {expected_transformed_features}"
        )

    print("\n=== TRAINING MATRIX ===")
    print(f"[+] X_train: {X_train.shape}")
    print(f"[+] y_train: {y_train.shape}")
    print(f"[+] X_val:   {X_val.shape}")
    print(f"[+] y_val:   {y_val.shape}")

    model = XGBClassifier(**params)

    # No early stopping and no validation-driven training adaptation:
    # V1 architecture/round count is already frozen prospectively.
    model.fit(X_train, y_train, verbose=False)

    probabilities = model.predict_proba(X_val)[:, 1].astype(np.float64)

    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("Validation probabilities contain NaN or Inf.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise RuntimeError("Validation probabilities fall outside [0, 1].")

    predictions = (probabilities >= threshold).astype(np.int8)
    metrics = compute_metrics(y_val, probabilities, predictions)

    print("\n=== V1 VALIDATION METRICS ===")
    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "fpr",
    ]:
        print(f"[+] {key:10s}: {metrics[key]:.10f}")
    print(
        f"[+] confusion: TN={metrics['tn']} FP={metrics['fp']} "
        f"FN={metrics['fn']} TP={metrics['tp']}"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save model to a temporary JSON and atomically move it into place.
    tmp_model = MODEL_PATH.with_name(
    MODEL_PATH.stem + ".tmp.json"
)
    try:
        model.save_model(str(tmp_model))
        os.replace(tmp_model, MODEL_PATH)
    finally:
        if tmp_model.exists():
            tmp_model.unlink()

    # Reload-equivalence check.
    reloaded = XGBClassifier()
    reloaded.load_model(str(MODEL_PATH))
    reload_probabilities = (
        reloaded.predict_proba(X_val)[:, 1].astype(np.float64)
    )

    max_abs_diff = float(
        np.max(np.abs(probabilities - reload_probabilities))
    )
    reload_equivalent = bool(
        np.allclose(
            probabilities,
            reload_probabilities,
            rtol=0.0,
            atol=1e-12,
        )
    )
    if not reload_equivalent:
        raise RuntimeError(
            "Reload-equivalence check FAILED. "
            f"Maximum absolute probability difference: {max_abs_diff}"
        )

    output_columns: dict[str, Any] = {}
    if "id" in val_df.columns:
        output_columns["id"] = val_df["id"].to_numpy()
    if "attack_cat" in val_df.columns:
        output_columns["attack_cat"] = val_df["attack_cat"].fillna("__MISSING__").astype(str).to_numpy()

    output_columns["label"] = y_val
    output_columns["probability"] = probabilities
    output_columns["prediction"] = predictions

    pred_df = pd.DataFrame(output_columns)
    atomic_csv_write(VALIDATION_PREDICTIONS_PATH, pred_df)

    model_hash = sha256_file(MODEL_PATH)
    predictions_hash = sha256_file(VALIDATION_PREDICTIONS_PATH)

    manifest = {
        "model_manifest_schema_version": "1.0",
        "artifact": "RolloutSecBench",
        "revision": "V1",
        "revision_role": "baseline_production_revision",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "sha256": protocol_hash,
            "sidecar_path": str(PROTOCOL_SHA_PATH.relative_to(REPO_ROOT)),
        },
        "dataset_manifest": {
            "path": str(DATASET_MANIFEST_PATH.relative_to(REPO_ROOT)),
            "sha256_at_training": dataset_manifest_hash,
            "training_boundary_open": True,
        },
        "training_data": {
            "path": str(FINAL_TRAIN_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(FINAL_TRAIN_PATH),
            "rows": int(len(train_df)),
            "used_for_fit": True,
        },
        "validation_data": {
            "path": str(VALIDATION_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(VALIDATION_PATH),
            "rows": int(len(val_df)),
            "used_for_fit": False,
            "used_for_evaluation": True,
        },
        "test_data_access": {
            "strict_test_nonoverlap_read_by_training_script": False,
            "official_test_full_read_by_training_script": False,
        },
        "preprocessing": {
            "path": str(PREPROCESSOR_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(PREPROCESSOR_PATH),
            "refit": False,
            "raw_feature_count": len(source_features),
            "transformed_feature_count": int(X_train.shape[1]),
            "feature_names_path": str(FEATURE_NAMES_PATH.relative_to(REPO_ROOT)),
            "feature_names_sha256": sha256_file(FEATURE_NAMES_PATH),
        },
        "training": {
            "model_class": "xgboost.XGBClassifier",
            "hyperparameters": params,
            "threshold": threshold,
            "threshold_policy": v1["threshold_policy"],
            "train_rows": int(X_train.shape[0]),
            "train_columns": int(X_train.shape[1]),
        },
        "validation": {
            "rows": int(X_val.shape[0]),
            "columns": int(X_val.shape[1]),
            "metrics": metrics,
            "probabilities_finite": True,
            "probabilities_within_unit_interval": True,
        },
        "serialization_validation": {
            "reload_equivalent": reload_equivalent,
            "comparison_rtol": 0.0,
            "comparison_atol": 1e-12,
            "max_absolute_probability_difference": max_abs_diff,
        },
        "artifacts": {
            "model": {
                "path": str(MODEL_PATH.relative_to(REPO_ROOT)),
                "sha256": model_hash,
            },
            "validation_predictions": {
                "path": str(
                    VALIDATION_PREDICTIONS_PATH.relative_to(REPO_ROOT)
                ),
                "sha256": predictions_hash,
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "joblib": joblib.__version__,
        },
        "acceptance_status": {
            "v1_core_checks_passed": True,
            "note": (
                "V1 training, validation evaluation, serialization, and reload-equivalence "
                "checks passed. This does not evaluate V2 relative noninferiority and does "
                "not authorize test-set analysis."
            ),
        },
    }

    atomic_json_write(MODEL_MANIFEST_PATH, manifest)
    model_manifest_hash = sha256_file(MODEL_MANIFEST_PATH)

    print("\n=== V1 ARTIFACTS ===")
    print(f"[+] model.json:                  {model_hash}")
    print(f"[+] validation predictions:      {predictions_hash}")
    print(f"[+] model_manifest.json:          {model_manifest_hash}")
    print(f"[+] reload-equivalent:            {reload_equivalent}")
    print(f"[+] reload max abs prob diff:     {max_abs_diff:.17g}")

    print(
        "\n=== V1 TRAINING PASSED ===\n"
        "V1 has been trained only on final_train and evaluated only on validation.\n"
        "Do NOT inspect either test set yet. Next, review/freeze the V1 artifacts, "
        "then train the prospectively frozen V2 and apply the validation-only "
        "V1/V2 acceptance gate."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
