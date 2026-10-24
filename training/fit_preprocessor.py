#!/usr/bin/env python3
"""
RolloutSecBench - Phase 1 Frozen Preprocessor Fit

Fits the already-frozen preprocessing policy using FINAL_TRAIN ONLY.

Inputs
------
data/manifests/dataset_manifest.json
data/manifests/split_manifest.json
data/splits/final_train.csv
data/splits/validation.csv
data/splits/strict_test_nonoverlap.csv
data/splits/official_test_full.csv

Outputs
-------
models/preprocessing/preprocessor.joblib
data/processed/feature_names.json
data/manifests/preprocessing_manifest.json
data/processed/preprocessing_summary.json

Scientific invariants
---------------------
1. Fit occurs ONLY on final_train.
2. id, attack_cat, and label never enter the model matrix.
3. Frozen raw model feature count is exactly 42.
4. Frozen categorical features are exactly proto, service, state.
5. Numeric features are passed through without scaling.
6. Categorical features use OneHotEncoder(handle_unknown="ignore").
7. No validation/test information influences fitting.
8. All frozen datasets must transform into the identical output feature space.
9. Preprocessing aborts on NaN/Inf rather than silently imputing.
10. Existing frozen preprocessing artifacts are never silently overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "dataset_manifest.json"
SPLIT_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "split_manifest.json"

FINAL_TRAIN_PATH = REPO_ROOT / "data" / "splits" / "final_train.csv"
VALIDATION_PATH = REPO_ROOT / "data" / "splits" / "validation.csv"
STRICT_TEST_PATH = REPO_ROOT / "data" / "splits" / "strict_test_nonoverlap.csv"
OFFICIAL_TEST_PATH = REPO_ROOT / "data" / "splits" / "official_test_full.csv"

PREPROCESSOR_DIR = REPO_ROOT / "models" / "preprocessing"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

PREPROCESSOR_PATH = PREPROCESSOR_DIR / "preprocessor.joblib"
FEATURE_NAMES_PATH = PROCESSED_DIR / "feature_names.json"
PREPROCESSING_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "preprocessing_manifest.json"
PREPROCESSING_SUMMARY_PATH = PROCESSED_DIR / "preprocessing_summary.json"

OUTPUT_PATHS = [
    PREPROCESSOR_PATH,
    FEATURE_NAMES_PATH,
    PREPROCESSING_MANIFEST_PATH,
    PREPROCESSING_SUMMARY_PATH,
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: list[str]) -> str:
    payload = "\n".join(values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def ensure_no_existing_outputs() -> None:
    existing = [p for p in OUTPUT_PATHS if p.exists()]
    if existing:
        rel = "\n".join(f"  - {p.relative_to(REPO_ROOT)}" for p in existing)
        raise FileExistsError(
            "Refusing to overwrite existing preprocessing artifacts:\n"
            f"{rel}\n"
            "Archive/remove only deliberate pre-freeze test artifacts before rerunning."
        )


def verify_file_hash(path: Path, expected_hash: str, description: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != expected_hash.lower():
        raise RuntimeError(
            f"{description} SHA-256 mismatch.\nExpected: {expected_hash}\nActual:   {actual}"
        )


def artifact_entry(split_manifest: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return split_manifest["artifacts"][name]
    except KeyError as exc:
        raise KeyError(f"split_manifest missing artifact {name!r}") from exc


def verify_contract(
    dataset_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    pp = dataset_manifest.get("preprocessing_policy", {})
    if pp.get("status") != "FROZEN_PHASE1":
        raise RuntimeError("preprocessing_policy.status must be FROZEN_PHASE1.")
    if pp.get("fit_scope") != "final_train_only":
        raise RuntimeError("preprocessing fit_scope must be final_train_only.")
    if pp.get("raw_feature_count") != 42:
        raise RuntimeError("Frozen raw feature count must be 42.")

    categorical = list(pp.get("categorical_features", []))
    if categorical != ["proto", "service", "state"]:
        raise RuntimeError(
            "Frozen categorical feature order must be ['proto', 'service', 'state']."
        )

    non_model = list(pp.get("non_model_columns", []))
    if set(non_model) != {"id", "attack_cat", "label"}:
        raise RuntimeError("Frozen non-model columns must be id, attack_cat, label.")

    feature_repr = split_manifest.get("feature_representation", {})
    source_features = list(feature_repr.get("features", []))

    if len(source_features) != 42:
        raise RuntimeError(
            f"split_manifest must contain exactly 42 source features; found {len(source_features)}."
        )

    if any(col in source_features for col in non_model):
        raise RuntimeError("Non-model columns appear in split_manifest feature representation.")

    numeric = [c for c in source_features if c not in categorical]

    if len(numeric) != 39:
        raise RuntimeError(
            f"Expected 39 numeric + 3 categorical source features; found "
            f"{len(numeric)} numeric + {len(categorical)} categorical."
        )

    return source_features, categorical, numeric


def check_raw_feature_frame(
    df: pd.DataFrame,
    source_features: list[str],
    categorical: list[str],
    numeric: list[str],
    split_name: str,
    require_no_missing: bool = True,
) -> None:
    missing_columns = [c for c in source_features if c not in df.columns]
    if missing_columns:
        raise RuntimeError(f"{split_name} missing source features: {missing_columns}")

    extra_non_model_in_features = [c for c in ["id", "attack_cat", "label"] if c in source_features]
    if extra_non_model_in_features:
        raise RuntimeError(
            f"{split_name}: non-model columns entered feature list: {extra_non_model_in_features}"
        )

    if require_no_missing:
        missing_total = int(df[source_features].isna().sum().sum())
        if missing_total:
            raise RuntimeError(
                f"{split_name} contains {missing_total} missing source-feature values; "
                "frozen preprocessing policy specifies no imputation."
            )

    numeric_frame = df[numeric]
    non_numeric_numeric_fields = [
        c for c in numeric if not pd.api.types.is_numeric_dtype(numeric_frame[c])
    ]
    if non_numeric_numeric_fields:
        raise RuntimeError(
            f"{split_name} expected numeric fields have nonnumeric dtype: "
            f"{non_numeric_numeric_fields}"
        )

    arr = numeric_frame.to_numpy()
    inf_count = int(np.isinf(arr).sum())
    if inf_count:
        raise RuntimeError(
            f"{split_name} contains {inf_count} infinite numeric feature values."
        )

    bad_categorical = [
        c for c in categorical
        if not (
            pd.api.types.is_object_dtype(df[c])
            or pd.api.types.is_string_dtype(df[c])
        )
    ]
    if bad_categorical:
        raise RuntimeError(
            f"{split_name} categorical fields have unexpected dtype: {bad_categorical}"
        )


def build_preprocessor(
    categorical: list[str],
    numeric: list[str],
) -> ColumnTransformer:
    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop=None,
        sparse_output=True,
        dtype=np.float32,
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_encoder, categorical),
            ("numeric", "passthrough", numeric),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def matrix_quality_summary(matrix: Any) -> dict[str, Any]:
    if sparse.issparse(matrix):
        data = matrix.data
        missing = int(np.isnan(data).sum())
        infinite = int(np.isinf(data).sum())
        return {
            "matrix_type": type(matrix).__name__,
            "rows": int(matrix.shape[0]),
            "columns": int(matrix.shape[1]),
            "nnz": int(matrix.nnz),
            "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
            "nan_values_in_stored_entries": missing,
            "infinite_values_in_stored_entries": infinite,
        }

    array = np.asarray(matrix)
    return {
        "matrix_type": type(matrix).__name__,
        "rows": int(array.shape[0]),
        "columns": int(array.shape[1]),
        "nan_values": int(np.isnan(array).sum()),
        "infinite_values": int(np.isinf(array).sum()),
    }


def transformed_has_bad_values(matrix: Any) -> tuple[int, int]:
    if sparse.issparse(matrix):
        return int(np.isnan(matrix.data).sum()), int(np.isinf(matrix.data).sum())

    array = np.asarray(matrix)
    return int(np.isnan(array).sum()), int(np.isinf(array).sum())


def main() -> None:
    print("=== RolloutSecBench Frozen Preprocessor Fit ===")

    for required in [
        DATASET_MANIFEST_PATH,
        SPLIT_MANIFEST_PATH,
        FINAL_TRAIN_PATH,
        VALIDATION_PATH,
        STRICT_TEST_PATH,
        OFFICIAL_TEST_PATH,
    ]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required artifact: {required}")

    dataset_manifest = load_json(DATASET_MANIFEST_PATH)
    split_manifest = load_json(SPLIT_MANIFEST_PATH)

    source_features, categorical, numeric = verify_contract(
        dataset_manifest, split_manifest
    )

    # Verify frozen split bytes before fitting.
    split_paths = {
        "final_train": FINAL_TRAIN_PATH,
        "validation": VALIDATION_PATH,
        "strict_test_nonoverlap": STRICT_TEST_PATH,
        "official_test_full": OFFICIAL_TEST_PATH,
    }

    for name, path in split_paths.items():
        entry = artifact_entry(split_manifest, name)
        verify_file_hash(path, entry["sha256"], name)

    ensure_no_existing_outputs()
    PREPROCESSOR_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSING_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    frames = {
        "final_train": pd.read_csv(FINAL_TRAIN_PATH),
        "validation": pd.read_csv(VALIDATION_PATH),
        "strict_test_nonoverlap": pd.read_csv(STRICT_TEST_PATH),
        "official_test_full": pd.read_csv(OFFICIAL_TEST_PATH),
    }

    for name, df in frames.items():
        check_raw_feature_frame(
            df=df,
            source_features=source_features,
            categorical=categorical,
            numeric=numeric,
            split_name=name,
            require_no_missing=True,
        )

    # Fit ONLY here, and ONLY on final_train.
    preprocessor = build_preprocessor(categorical, numeric)
    preprocessor.fit(frames["final_train"][source_features])

    transformed = {}
    summaries = {}

    for name, df in frames.items():
        matrix = preprocessor.transform(df[source_features])
        transformed[name] = matrix
        summaries[name] = matrix_quality_summary(matrix)

        nan_count, inf_count = transformed_has_bad_values(matrix)
        if nan_count or inf_count:
            raise RuntimeError(
                f"{name} transformed matrix contains invalid values: "
                f"NaN={nan_count}, Inf={inf_count}"
            )

    output_dims = {matrix.shape[1] for matrix in transformed.values()}
    if len(output_dims) != 1:
        raise RuntimeError(
            f"Transformed feature dimensionality differs across datasets: {output_dims}"
        )

    transformed_feature_count = next(iter(output_dims))

    feature_names = [str(x) for x in preprocessor.get_feature_names_out().tolist()]
    if len(feature_names) != transformed_feature_count:
        raise RuntimeError(
            "get_feature_names_out count does not equal transformed matrix width."
        )
    if len(set(feature_names)) != len(feature_names):
        raise RuntimeError("Transformed feature names are not unique.")

    # Record categorical vocabularies learned ONLY from final_train.
    fitted_ohe = preprocessor.named_transformers_["categorical"]
    category_vocabularies = {
        feature: [str(v) for v in values.tolist()]
        for feature, values in zip(categorical, fitted_ohe.categories_)
    }

    # Explicitly characterize unseen categories in evaluation splits without refitting.
    unseen_categories: dict[str, dict[str, list[str]]] = {}
    training_categories = {
        feature: set(str(v) for v in values.tolist())
        for feature, values in zip(categorical, fitted_ohe.categories_)
    }

    for split_name, df in frames.items():
        if split_name == "final_train":
            continue
        split_result: dict[str, list[str]] = {}
        for feature in categorical:
            observed = set(df[feature].astype(str).unique().tolist())
            unseen = sorted(observed - training_categories[feature])
            split_result[feature] = unseen
        unseen_categories[split_name] = split_result

    # Serialize only after all transformations pass.
    tmp_preprocessor = PREPROCESSOR_PATH.with_suffix(".joblib.tmp")
    try:
        joblib.dump(preprocessor, tmp_preprocessor)
        os.replace(tmp_preprocessor, PREPROCESSOR_PATH)
    finally:
        if tmp_preprocessor.exists():
            tmp_preprocessor.unlink()

    feature_names_document = {
        "schema_version": "1.0",
        "raw_feature_count": len(source_features),
        "transformed_feature_count": len(feature_names),
        "source_feature_order": source_features,
        "transformed_feature_order": feature_names,
        "source_feature_list_sha256": sha256_lines(source_features),
        "transformed_feature_list_sha256": sha256_lines(feature_names),
    }
    atomic_json_write(FEATURE_NAMES_PATH, feature_names_document)

    preprocessing_summary = {
        "schema_version": "1.0",
        "fit_scope": "final_train_only",
        "input": {
            "raw_feature_count": len(source_features),
            "categorical_feature_count": len(categorical),
            "numeric_feature_count": len(numeric),
            "categorical_features": categorical,
            "numeric_features": numeric,
        },
        "categorical_cardinality_final_train": {
            feature: len(values)
            for feature, values in category_vocabularies.items()
        },
        "category_vocabularies_final_train": category_vocabularies,
        "unseen_categories_after_frozen_fit": unseen_categories,
        "transformed_feature_count": transformed_feature_count,
        "transformed_split_summaries": summaries,
    }
    atomic_json_write(PREPROCESSING_SUMMARY_PATH, preprocessing_summary)

    preprocessing_manifest = {
        "preprocessing_manifest_schema_version": "1.0",
        "artifact": "RolloutSecBench",
        "phase": "Phase 1 - Frozen Preprocessing",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "fit_scope": "final_train_only",
            "feature_representation": "SOURCE_CANDIDATE_FEATURES",
            "raw_feature_count": 42,
            "categorical_features": categorical,
            "numeric_features": numeric,
            "categorical_encoder": {
                "class": "sklearn.preprocessing.OneHotEncoder",
                "handle_unknown": "ignore",
                "drop": None,
                "sparse_output": True,
                "dtype": "float32",
            },
            "numeric_transform": "passthrough",
            "numeric_scaling": "none",
            "imputation": "none",
            "validation_may_influence_fit": False,
            "test_may_influence_fit": False,
        },
        "provenance": {
            "dataset_manifest_path": str(DATASET_MANIFEST_PATH.relative_to(REPO_ROOT)),
            "dataset_manifest_sha256": sha256_file(DATASET_MANIFEST_PATH),
            "split_manifest_path": str(SPLIT_MANIFEST_PATH.relative_to(REPO_ROOT)),
            "split_manifest_sha256": sha256_file(SPLIT_MANIFEST_PATH),
            "fit_source_path": str(FINAL_TRAIN_PATH.relative_to(REPO_ROOT)),
            "fit_source_sha256": sha256_file(FINAL_TRAIN_PATH),
        },
        "libraries": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "feature_space": {
            "source_feature_count": len(source_features),
            "source_feature_list_sha256": sha256_lines(source_features),
            "transformed_feature_count": len(feature_names),
            "transformed_feature_list_sha256": sha256_lines(feature_names),
        },
        "artifacts": {
            "preprocessor": {
                "path": str(PREPROCESSOR_PATH.relative_to(REPO_ROOT)),
                "sha256": sha256_file(PREPROCESSOR_PATH),
            },
            "feature_names": {
                "path": str(FEATURE_NAMES_PATH.relative_to(REPO_ROOT)),
                "sha256": sha256_file(FEATURE_NAMES_PATH),
            },
            "preprocessing_summary": {
                "path": str(PREPROCESSING_SUMMARY_PATH.relative_to(REPO_ROOT)),
                "sha256": sha256_file(PREPROCESSING_SUMMARY_PATH),
            },
        },
        "validation": {
            "all_splits_same_transformed_feature_count": True,
            "all_transformed_splits_nan_free": True,
            "all_transformed_splits_infinite_free": True,
            "feature_names_unique": True,
            "unseen_categories_recorded_and_handled_by_ignore": True,
        },
        "training_boundary": {
            "preprocessing_gate_passed": True,
            "model_training_may_be_unblocked_by_explicit_dataset_manifest_update": True,
            "note": (
                "This script does not silently modify dataset_manifest.json. "
                "Open the model-training boundary only after reviewing these artifacts/hashes."
            ),
        },
    }
    atomic_json_write(PREPROCESSING_MANIFEST_PATH, preprocessing_manifest)

    print("\n=== FROZEN INPUT POLICY ===")
    print(f"[+] Raw model features:        {len(source_features)}")
    print(f"[+] Categorical features:      {categorical}")
    print(f"[+] Numeric features:          {len(numeric)}")
    print("[+] Numeric scaling:           NONE")
    print("[+] Imputation:                NONE")
    print("[+] Encoder:                   OneHotEncoder(handle_unknown='ignore')")
    print("[+] Fit source:                final_train ONLY")

    print("\n=== TRANSFORMED FEATURE SPACE ===")
    print(f"[+] Transformed feature count: {len(feature_names)}")
    for name, summary in summaries.items():
        print(
            f"[+] {name}: rows={summary['rows']} "
            f"columns={summary['columns']} "
            f"type={summary['matrix_type']}"
        )

    print("\n=== CATEGORICAL CARDINALITY LEARNED FROM FINAL_TRAIN ===")
    for feature, values in category_vocabularies.items():
        print(f"[+] {feature}: {len(values)}")

    print("\n=== UNSEEN EVALUATION CATEGORIES ===")
    for split_name, per_feature in unseen_categories.items():
        print(f"[+] {split_name}: {per_feature}")

    print("\n=== ARTIFACT HASHES ===")
    print(
        f"[+] preprocessor.joblib:       {sha256_file(PREPROCESSOR_PATH)}"
    )
    print(
        f"[+] feature_names.json:        {sha256_file(FEATURE_NAMES_PATH)}"
    )
    print(
        f"[+] preprocessing_summary.json:{sha256_file(PREPROCESSING_SUMMARY_PATH)}"
    )
    print(
        f"[+] preprocessing_manifest.json:{sha256_file(PREPROCESSING_MANIFEST_PATH)}"
    )

    print(
        "\n=== PREPROCESSING FREEZE PASSED ===\n"
        "Preprocessing has been fitted ONLY on final_train and validated across all "
        "frozen splits. Review and record the artifact hashes, then explicitly open "
        "the model-training boundary in dataset_manifest.json before training V1/V2."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
