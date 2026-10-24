#!/usr/bin/env python3
"""
RolloutSecBench — Train prospectively frozen V2 model revision.

Scientific constraints enforced:
- Reads the frozen model_revision_protocol_v1.0.json.
- Uses revision_v2 hyperparameters exactly as frozen.
- Fits ONLY on final_train.
- Evaluates ONLY on validation.
- Reuses the already-frozen preprocessor; never refits it.
- Does NOT read either test split.
- Does NOT perform hyperparameter search or threshold optimization.
- Uses the frozen V2 threshold from the protocol.
- Checks serialized-model reload equivalence.
- Performs an independent same-process deterministic retraining check.
- Writes V2 model, validation predictions, and model manifest.
- Refuses to overwrite existing V2 outputs.

The V1/V2 relative noninferiority gate is intentionally evaluated separately
after V2 training so that training and acceptance remain distinct operations.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import tempfile
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
# Repository
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_PATH = (
    REPO_ROOT / "data" / "manifests" / "model_revision_protocol_v1.0.json"
)

PROTOCOL_SHA_PATH = (
    REPO_ROOT / "data" / "manifests" / "model_revision_protocol_v1.0.sha256"
)

MODEL_DIR = REPO_ROOT / "models" / "v2"
MODEL_PATH = MODEL_DIR / "model.json"
MODEL_MANIFEST_PATH = MODEL_DIR / "model_manifest.json"

RESULTS_DERIVED = REPO_ROOT / "results" / "derived"
VAL_PREDICTIONS_PATH = RESULTS_DERIVED / "v2_validation_predictions.csv"

EXPECTED_PROTOCOL_STATUS = "FROZEN_PRETRAINING"

NON_MODEL_COLUMNS = {"id", "attack_cat", "label"}

HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


# ---------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.stem + ".",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)

    os.replace(tmp_path, path)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=path.stem + ".",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def find_unique(filename: str) -> Path:
    """
    Locate an existing repository artifact without guessing its directory.

    Fails if zero or multiple matches are found.
    """
    matches = [
        p for p in REPO_ROOT.rglob(filename)
        if p.is_file()
        and ".git" not in p.parts
        and ".venv" not in p.parts
        and ".venv-1" not in p.parts
    ]

    if not matches:
        raise FileNotFoundError(
            f"Required frozen artifact not found anywhere in repository: {filename}"
        )

    if len(matches) > 1:
        rendered = "\n".join(f"  - {p}" for p in matches)
        raise RuntimeError(
            f"Ambiguous artifact name {filename!r}; multiple matches found:\n"
            f"{rendered}\n"
            "Resolve the ambiguity rather than allowing the trainer to guess."
        )

    return matches[0]


def collect_sha256_strings(obj: Any) -> set[str]:
    """
    Recursively collect all SHA-256-looking strings from frozen protocol.

    This lets us confirm that current frozen inputs correspond to hashes
    cryptographically recorded somewhere in the prospectively frozen protocol
    without assuming a particular JSON nesting layout.
    """
    found: set[str] = set()

    if isinstance(obj, dict):
        for value in obj.values():
            found.update(collect_sha256_strings(value))

    elif isinstance(obj, list):
        for value in obj:
            found.update(collect_sha256_strings(value))

    elif isinstance(obj, str):
        candidate = obj.strip()
        if HEX64.fullmatch(candidate):
            found.add(candidate.lower())

    return found


def verify_hash_is_frozen(
    path: Path,
    frozen_hashes: set[str],
    label: str,
) -> str:
    digest = sha256_file(path)

    if digest.lower() not in frozen_hashes:
        raise RuntimeError(
            f"{label} SHA-256 is not present in the frozen protocol.\n"
            f"Path: {path}\n"
            f"Observed SHA-256: {digest}\n"
            "Training aborted."
        )

    print(f"[+] {label:<25} {digest}")
    return digest


def max_abs_difference(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise RuntimeError(
            f"Prediction shape mismatch: {a.shape} vs {b.shape}"
        )

    if a.size == 0:
        return 0.0

    return float(np.max(np.abs(a - b)))


# ---------------------------------------------------------------------
# Frozen protocol
# ---------------------------------------------------------------------

def load_and_verify_protocol() -> tuple[dict[str, Any], str]:

    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(PROTOCOL_PATH)

    protocol_sha = sha256_file(PROTOCOL_PATH)

    if PROTOCOL_SHA_PATH.exists():
        sidecar_text = PROTOCOL_SHA_PATH.read_text(
            encoding="utf-8"
        ).strip()

        expected_sha = sidecar_text.split()[0]

        if protocol_sha != expected_sha:
            raise RuntimeError(
                "Frozen protocol SHA-256 does not match its sidecar.\n"
                f"Expected: {expected_sha}\n"
                f"Observed: {protocol_sha}"
            )

    protocol = json.loads(
        PROTOCOL_PATH.read_text(encoding="utf-8")
    )

    status = protocol.get("protocol_status")

    if status != EXPECTED_PROTOCOL_STATUS:
        raise RuntimeError(
            f"Unexpected protocol_status={status!r}; "
            f"expected {EXPECTED_PROTOCOL_STATUS!r}."
        )

    if "revision_v2" not in protocol:
        raise RuntimeError(
            "Frozen protocol contains no revision_v2 section."
        )

    return protocol, protocol_sha


def get_v2_configuration(
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], float]:

    revision = protocol["revision_v2"]

    role = revision.get("revision_role")

    if role != "prospective_legitimate_model_revision":
        raise RuntimeError(
            "Unexpected V2 revision_role.\n"
            f"Observed: {role!r}"
        )

    params = revision.get("hyperparameters")

    if not isinstance(params, dict) or not params:
        raise RuntimeError(
            "Frozen revision_v2.hyperparameters is missing or invalid."
        )

    threshold_policy = revision.get("threshold_policy", {})

    if threshold_policy.get("type") != "FIXED":
        raise RuntimeError(
            "This trainer expects the prospectively frozen V2 "
            "threshold policy to be FIXED."
        )

    threshold = threshold_policy.get("threshold")

    if threshold is None:
        raise RuntimeError(
            "Frozen V2 decision threshold is missing."
        )

    threshold = float(threshold)

    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError(
            f"Invalid frozen decision threshold: {threshold}"
        )

    if threshold_policy.get("threshold_selection_data") != "NONE":
        raise RuntimeError(
            "Expected threshold_selection_data='NONE' for frozen "
            "V2 threshold."
        )

    required_exact = {
        "n_estimators": 450,
        "max_depth": 5,
        "learning_rate": 0.04,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 2,
        "gamma": 0.1,
        "reg_alpha": 0.05,
        "reg_lambda": 1.5,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 1,
    }

    if params != required_exact:
        raise RuntimeError(
            "revision_v2 hyperparameters no longer match the "
            "prospectively frozen configuration printed and verified "
            "before V2 execution.\n\n"
            f"Expected:\n{json.dumps(required_exact, indent=2, sort_keys=True)}\n\n"
            f"Observed:\n{json.dumps(params, indent=2, sort_keys=True)}"
        )

    if threshold != 0.5:
        raise RuntimeError(
            "Frozen V2 threshold no longer equals the prospectively "
            "verified value 0.5."
        )

    return params, threshold


# ---------------------------------------------------------------------
# Data / preprocessing
# ---------------------------------------------------------------------

def load_frozen_inputs(
    protocol: dict[str, Any],
) -> dict[str, Any]:

    frozen_hashes = collect_sha256_strings(protocol)

    if not frozen_hashes:
        raise RuntimeError(
            "No SHA-256 values were found in the frozen model protocol."
        )

    final_train_path = find_unique("final_train.csv")
    validation_path = find_unique("validation.csv")
    preprocessor_path = find_unique("preprocessor.joblib")
    feature_names_path = find_unique("feature_names.json")
    preprocessing_manifest_path = find_unique(
        "preprocessing_manifest.json"
    )

    print("\n=== FROZEN INPUT HASH VERIFICATION ===")

    hashes = {
        "final_train_sha256": verify_hash_is_frozen(
            final_train_path,
            frozen_hashes,
            "final_train",
        ),
        "validation_sha256": verify_hash_is_frozen(
            validation_path,
            frozen_hashes,
            "validation",
        ),
        "preprocessor_sha256": verify_hash_is_frozen(
            preprocessor_path,
            frozen_hashes,
            "preprocessor",
        ),
        "feature_names_sha256": verify_hash_is_frozen(
            feature_names_path,
            frozen_hashes,
            "feature_names",
        ),
        "preprocessing_manifest_sha256": verify_hash_is_frozen(
            preprocessing_manifest_path,
            frozen_hashes,
            "preprocessing_manifest",
        ),
    }

    train_df = pd.read_csv(final_train_path)
    val_df = pd.read_csv(validation_path)

    if "label" not in train_df.columns:
        raise RuntimeError("final_train.csv has no label column.")

    if "label" not in val_df.columns:
        raise RuntimeError("validation.csv has no label column.")

    preprocessor = joblib.load(preprocessor_path)

    # Prefer the exact feature order learned by the frozen preprocessor.
    if hasattr(preprocessor, "feature_names_in_"):
        source_features = list(preprocessor.feature_names_in_)
    else:
        source_features = [
            c
            for c in train_df.columns
            if c not in NON_MODEL_COLUMNS
        ]

    if len(source_features) != 42:
        raise RuntimeError(
            f"Expected 42 raw source features; observed "
            f"{len(source_features)}."
        )

    forbidden = NON_MODEL_COLUMNS.intersection(source_features)

    if forbidden:
        raise RuntimeError(
            "Non-model columns unexpectedly included in source features: "
            + ", ".join(sorted(forbidden))
        )

    train_missing = [
        c for c in source_features if c not in train_df.columns
    ]
    val_missing = [
        c for c in source_features if c not in val_df.columns
    ]

    if train_missing:
        raise RuntimeError(
            f"final_train missing source features: {train_missing}"
        )

    if val_missing:
        raise RuntimeError(
            f"validation missing source features: {val_missing}"
        )

    X_train_raw = train_df[source_features]
    X_val_raw = val_df[source_features]

    y_train = train_df["label"].to_numpy(dtype=np.int64)
    y_val = val_df["label"].to_numpy(dtype=np.int64)

    if not set(np.unique(y_train)).issubset({0, 1}):
        raise RuntimeError("Training labels are not binary {0,1}.")

    if not set(np.unique(y_val)).issubset({0, 1}):
        raise RuntimeError("Validation labels are not binary {0,1}.")

    # IMPORTANT: transform only. Never fit or fit_transform.
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    if X_train.shape[1] != X_val.shape[1]:
        raise RuntimeError(
            "Train/validation transformed feature widths differ."
        )

    if X_train.shape[1] != 192:
        raise RuntimeError(
            f"Expected frozen transformed width 192; "
            f"observed {X_train.shape[1]}."
        )

    return {
        "paths": {
            "final_train": final_train_path,
            "validation": validation_path,
            "preprocessor": preprocessor_path,
            "feature_names": feature_names_path,
            "preprocessing_manifest": preprocessing_manifest_path,
        },
        "hashes": hashes,
        "train_df": train_df,
        "val_df": val_df,
        "source_features": source_features,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
    }


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:

    if probabilities.ndim != 1:
        probabilities = probabilities.reshape(-1)

    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError(
            "Validation probabilities contain NaN or infinity."
        )

    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise RuntimeError(
            "Validation probabilities fall outside [0,1]."
        )

    predictions = (probabilities >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr_denominator = fp + tn

    if fpr_denominator == 0:
        raise RuntimeError(
            "Cannot calculate FPR because validation contains "
            "no negative samples."
        )

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "auroc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "auprc": float(
            average_precision_score(y_true, probabilities)
        ),
        "fpr": float(fp / fpr_denominator),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics, predictions


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print("=== RolloutSecBench V2 Frozen Training ===")

    # Refuse accidental overwrite.
    existing_outputs = [
        p
        for p in (
            MODEL_PATH,
            MODEL_MANIFEST_PATH,
            VAL_PREDICTIONS_PATH,
        )
        if p.exists()
    ]

    if existing_outputs:
        rendered = "\n".join(
            f"  - {p}" for p in existing_outputs
        )

        raise RuntimeError(
            "V2 output artifact(s) already exist. "
            "Refusing to overwrite:\n"
            f"{rendered}\n"
            "Inspect/preserve existing artifacts before rerunning."
        )

    protocol, protocol_sha = load_and_verify_protocol()
    params, threshold = get_v2_configuration(protocol)

    print("\n=== FROZEN V2 CONTRACT ===")
    print(f"[+] Protocol SHA-256:       {protocol_sha}")
    print("[+] Training source:        final_train ONLY")
    print("[+] Validation source:      validation ONLY")
    print("[+] Raw source features:    42")
    print(f"[+] Decision threshold:     {threshold}")
    print("[+] Hyperparameter search:  DISABLED")
    print("[+] Threshold tuning:       DISABLED")
    print("[+] Test access:            DISABLED")
    print(
        "[+] V2 hyperparameters:     "
        + json.dumps(params, sort_keys=True)
    )

    frozen = load_frozen_inputs(protocol)

    X_train = frozen["X_train"]
    X_val = frozen["X_val"]
    y_train = frozen["y_train"]
    y_val = frozen["y_val"]

    print("\n=== TRAINING MATRIX ===")
    print(f"[+] X_train: {X_train.shape}")
    print(f"[+] y_train: {y_train.shape}")
    print(f"[+] X_val:   {X_val.shape}")
    print(f"[+] y_val:   {y_val.shape}")

    # --------------------------------------------------------------
    # Primary prospective V2 training
    # --------------------------------------------------------------

    model = XGBClassifier(**params)

    # No validation-driven early stopping:
    # n_estimators is already prospectively frozen.
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]

    metrics, val_pred = calculate_metrics(
        y_val,
        val_prob,
        threshold,
    )

    print("\n=== V2 VALIDATION METRICS ===")
    for name in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "fpr",
    ):
        print(f"[+] {name:<10}: {metrics[name]:.10f}")

    print(
        "[+] confusion: "
        f"TN={metrics['tn']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"TP={metrics['tp']}"
    )

    # --------------------------------------------------------------
    # Deterministic retraining check
    # --------------------------------------------------------------

    print("\n=== DETERMINISTIC RETRAINING CHECK ===")

    reproducibility_model = XGBClassifier(**params)
    reproducibility_model.fit(X_train, y_train)

    reproducibility_prob = (
        reproducibility_model.predict_proba(X_val)[:, 1]
    )

    deterministic_max_diff = max_abs_difference(
        val_prob,
        reproducibility_prob,
    )

    deterministic_equivalent = np.allclose(
        val_prob,
        reproducibility_prob,
        rtol=0.0,
        atol=1e-12,
    )

    print(
        "[+] same-seed validation prediction equivalence: "
        f"{deterministic_equivalent}"
    )
    print(
        "[+] max abs probability difference:              "
        f"{deterministic_max_diff:.18g}"
    )

    if not deterministic_equivalent:
        raise RuntimeError(
            "Deterministic retraining check FAILED. "
            "No V2 artifacts will be frozen."
        )

    # --------------------------------------------------------------
    # Model JSON serialization
    # --------------------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DERIVED.mkdir(parents=True, exist_ok=True)

    # Keep .json as the FINAL extension so XGBoost writes JSON rather
    # than falling back to UBJSON.
    tmp_model = MODEL_PATH.with_name(
        MODEL_PATH.stem + ".tmp.json"
    )

    if tmp_model.exists():
        tmp_model.unlink()

    try:
        model.save_model(str(tmp_model))
        os.replace(tmp_model, MODEL_PATH)
    finally:
        if tmp_model.exists():
            tmp_model.unlink()

    # --------------------------------------------------------------
    # Reload equivalence
    # --------------------------------------------------------------

    reloaded = XGBClassifier()
    reloaded.load_model(str(MODEL_PATH))

    reloaded_prob = reloaded.predict_proba(X_val)[:, 1]

    reload_max_diff = max_abs_difference(
        val_prob,
        reloaded_prob,
    )

    reload_equivalent = np.allclose(
        val_prob,
        reloaded_prob,
        rtol=0.0,
        atol=1e-12,
    )

    if not reload_equivalent:
        raise RuntimeError(
            "Serialized V2 model reload does not reproduce "
            "the original validation probabilities."
        )

    # --------------------------------------------------------------
    # Validation prediction artifact
    # --------------------------------------------------------------

    val_df = frozen["val_df"]

    prediction_df = pd.DataFrame()

    if "id" in val_df.columns:
        prediction_df["id"] = val_df["id"]

    if "attack_cat" in val_df.columns:
        prediction_df["attack_cat"] = (
            val_df["attack_cat"]
            .fillna("__MISSING__")
            .astype(str)
        )

    prediction_df["label"] = y_val
    prediction_df["probability"] = val_prob
    prediction_df["prediction"] = val_pred

    atomic_write_csv(
        prediction_df,
        VAL_PREDICTIONS_PATH,
    )

    # --------------------------------------------------------------
    # Artifact hashes
    # --------------------------------------------------------------

    model_sha = sha256_file(MODEL_PATH)
    val_predictions_sha = sha256_file(
        VAL_PREDICTIONS_PATH
    )

    # --------------------------------------------------------------
    # Manifest
    # --------------------------------------------------------------

    v2_acceptance_policy = (
        protocol
        .get("model_acceptance_policy", {})
        .get("v2")
    )

    manifest = {
        "schema_version": "1.0",
        "model_version": "v2",
        "revision_role": protocol["revision_v2"][
            "revision_role"
        ],
        "revision_rationale": protocol["revision_v2"][
            "revision_rationale"
        ],

        "protocol": {
            "path": str(
                PROTOCOL_PATH.relative_to(REPO_ROOT)
            ),
            "sha256": protocol_sha,
            "status": protocol.get("protocol_status"),
        },

        "data_access": {
            "training_split": "final_train",
            "validation_split": "validation",
            "test_data_accessed": False,
            "strict_test_nonoverlap_accessed": False,
            "official_test_full_accessed": False,
        },

        "training": {
            "algorithm": "XGBoost",
            "training_seed": params["random_state"],
            "hyperparameters": params,
            "training_rows": int(X_train.shape[0]),
            "validation_rows": int(X_val.shape[0]),
            "raw_feature_count": int(
                len(frozen["source_features"])
            ),
            "transformed_feature_count": int(
                X_train.shape[1]
            ),
        },

        "decision_policy": {
            "decision_threshold": threshold,
            "threshold_policy": protocol[
                "revision_v2"
            ]["threshold_policy"],
        },

        "frozen_inputs": {
            "final_train_sha256": frozen[
                "hashes"
            ]["final_train_sha256"],
            "validation_sha256": frozen[
                "hashes"
            ]["validation_sha256"],
            "preprocessor_sha256": frozen[
                "hashes"
            ]["preprocessor_sha256"],
            "feature_names_sha256": frozen[
                "hashes"
            ]["feature_names_sha256"],
            "preprocessing_manifest_sha256": frozen[
                "hashes"
            ]["preprocessing_manifest_sha256"],
        },

        "validation_metrics": metrics,

        "serialization_check": {
            "reload_equivalent": bool(
                reload_equivalent
            ),
            "max_abs_probability_difference": (
                reload_max_diff
            ),
            "absolute_tolerance": 1e-12,
            "relative_tolerance": 0.0,
        },

        "deterministic_retraining_check": {
            "passed": bool(
                deterministic_equivalent
            ),
            "same_hyperparameters": True,
            "same_seed": True,
            "same_training_data": True,
            "same_preprocessor": True,
            "max_abs_probability_difference": (
                deterministic_max_diff
            ),
            "absolute_tolerance": 1e-12,
            "relative_tolerance": 0.0,
        },

        "acceptance_policy_reference": (
            v2_acceptance_policy
        ),

        "acceptance_status": {
            "training_completed": True,
            "validation_probabilities_finite": True,
            "validation_probabilities_in_unit_interval": True,
            "required_metrics_recorded": True,
            "serialization_reload_equivalent": bool(
                reload_equivalent
            ),
            "deterministic_reproduction_passed": bool(
                deterministic_equivalent
            ),
            "relative_v1_v2_noninferiority_evaluated": False,
            "gate_1_final_status": "PENDING_V1_V2_ACCEPTANCE_COMPARISON",
        },

        "artifacts": {
            "model": {
                "path": str(
                    MODEL_PATH.relative_to(REPO_ROOT)
                ),
                "sha256": model_sha,
                "format": "XGBoost JSON",
            },
            "validation_predictions": {
                "path": str(
                    VAL_PREDICTIONS_PATH.relative_to(
                        REPO_ROOT
                    )
                ),
                "sha256": val_predictions_sha,
            },
        },

        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": getattr(
                joblib,
                "__version__",
                "UNKNOWN",
            ),
        },

        "scientific_constraints": {
            "hyperparameter_search_performed": False,
            "validation_driven_early_stopping": False,
            "threshold_tuning_performed": False,
            "preprocessor_refit": False,
            "test_set_used_for_training": False,
            "test_set_used_for_validation": False,
            "test_set_used_for_threshold_selection": False,
            "test_set_used_for_model_selection": False,
            "malicious_disagreement_used_for_model_selection": False,
            "rollout_asr_used_for_model_selection": False,
        },

        "note": (
            "V2 has been trained only on final_train and "
            "evaluated only on validation. This manifest does "
            "not itself declare the V1/V2 pair acceptable. "
            "The prospectively frozen relative noninferiority "
            "requirements must be evaluated separately before "
            "either test split or malicious-disagreement "
            "analysis is opened."
        ),
    }

    atomic_write_text(
        MODEL_MANIFEST_PATH,
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    model_manifest_sha = sha256_file(
        MODEL_MANIFEST_PATH
    )

    print("\n=== V2 ARTIFACTS ===")
    print(
        f"[+] model.json:                  "
        f"{model_sha}"
    )
    print(
        f"[+] validation predictions:      "
        f"{val_predictions_sha}"
    )
    print(
        f"[+] model_manifest.json:          "
        f"{model_manifest_sha}"
    )
    print(
        f"[+] reload-equivalent:            "
        f"{reload_equivalent}"
    )
    print(
        f"[+] reload max abs prob diff:     "
        f"{reload_max_diff:.18g}"
    )
    print(
        f"[+] deterministic retrain:        "
        f"{deterministic_equivalent}"
    )
    print(
        f"[+] deterministic max abs diff:   "
        f"{deterministic_max_diff:.18g}"
    )

    print("\n=== V2 TRAINING PASSED ===")
    print(
        "V2 has been trained only on final_train and "
        "evaluated only on validation."
    )
    print(
        "Do NOT inspect either test set yet. "
        "Next apply the prospectively frozen validation-only "
        "V1/V2 acceptance gate."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\n[ERROR] {exc}",
            file=sys.stderr,
        )
        raise
