#!/usr/bin/env python3
"""
RolloutSecBench - Deterministic Phase 1 Split Generation

Creates immutable derived split artifacts from the already-audited
UNSW-NB15 source partitions.

Outputs
-------
data/splits/final_train.csv
data/splits/validation.csv
data/splits/strict_test_nonoverlap.csv
data/splits/official_test_full.csv
data/manifests/split_manifest.json

Scientific invariants
---------------------
1. Validation is derived ONLY from official train_raw.
2. official_test_full is a byte-for-byte copy of official test_raw.
3. strict_test_nonoverlap membership is features-only:
      keep test row iff its complete SOURCE_CANDIDATE_FEATURES vector
      does not exactly occur anywhere in official train_raw.
4. Labels, attack categories, predictions, probabilities, and model
   performance never influence strict-test membership.
5. Exact equality is verified on actual feature values; pandas 64-bit
   hashes are screening only.
6. Existing split artifacts are never silently overwritten.
7. Expected strict-test counts are cross-checked against the completed
   dataset audit before files are committed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "dataset_manifest.json"
AUDIT_PATH = REPO_ROOT / "results" / "derived" / "dataset_audit.json"

SPLITS_DIR = REPO_ROOT / "data" / "splits"
SPLIT_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "split_manifest.json"

FINAL_TRAIN_PATH = SPLITS_DIR / "final_train.csv"
VALIDATION_PATH = SPLITS_DIR / "validation.csv"
STRICT_TEST_PATH = SPLITS_DIR / "strict_test_nonoverlap.csv"
OFFICIAL_TEST_FULL_PATH = SPLITS_DIR / "official_test_full.csv"

ARTIFACT_PATHS = [
    FINAL_TRAIN_PATH,
    VALIDATION_PATH,
    STRICT_TEST_PATH,
    OFFICIAL_TEST_FULL_PATH,
    SPLIT_MANIFEST_PATH,
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


def sha256_text_lines(values: list[str]) -> str:
    payload = "\n".join(values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_scalar(value: Any) -> Any:
    if pd.isna(value):
        return ("__MISSING__",)
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_row(values: list[Any]) -> tuple[Any, ...]:
    return tuple(canonical_scalar(value) for value in values)


def get_file_entry(manifest: dict[str, Any], role: str) -> dict[str, Any]:
    matches = [item for item in manifest.get("files", []) if item.get("role") == role]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one manifest file entry for role={role!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def resolve_manifest_file(entry: dict[str, Any]) -> Path:
    rel = entry.get("relative_path")
    if not rel:
        raise ValueError("Manifest file entry is missing relative_path.")
    return REPO_ROOT / rel


def ensure_no_existing_artifacts() -> None:
    existing = [path for path in ARTIFACT_PATHS if path.exists()]
    if existing:
        message = "\n".join(f"  - {p.relative_to(REPO_ROOT)}" for p in existing)
        raise FileExistsError(
            "Refusing to overwrite existing frozen/derived artifacts:\n"
            f"{message}\n"
            "If these are pre-freeze test artifacts, archive/remove them deliberately "
            "and rerun. Do not overwrite frozen artifacts silently."
        )


def verify_manifest_contract(manifest: dict[str, Any]) -> None:
    problems: list[str] = []

    if manifest.get("manifest_schema_version") != "1.4":
        problems.append("manifest_schema_version must be '1.4'.")

    feature_cfg = manifest.get("feature_policy", {})
    if feature_cfg.get("policy_status") != "FROZEN_PHASE1":
        problems.append("feature_policy.policy_status must be FROZEN_PHASE1.")
    if feature_cfg.get("policy_excluded_features") != []:
        problems.append(
            "policy_excluded_features must be [] for the frozen 42-feature policy."
        )

    split_cfg = manifest.get("split_policy", {})
    if split_cfg.get("status") != "FROZEN_PHASE1":
        problems.append("split_policy.status must be FROZEN_PHASE1.")
    if split_cfg.get("validation_fraction") != 0.20:
        problems.append("split_policy.validation_fraction must be 0.20.")
    if split_cfg.get("random_seed") != 42:
        problems.append("split_policy.random_seed must be 42.")
    if (
        split_cfg.get("stratification_policy")
        != "BINARY_LABEL_STRATIFIED_RANDOM_SPLIT"
    ):
        problems.append(
            "split_policy.stratification_policy must be "
            "BINARY_LABEL_STRATIFIED_RANDOM_SPLIT."
        )

    strict_cfg = split_cfg.get("strict_test_nonoverlap", {})
    if strict_cfg.get("feature_representation") != "SOURCE_CANDIDATE_FEATURES":
        problems.append(
            "strict_test_nonoverlap.feature_representation must be "
            "SOURCE_CANDIDATE_FEATURES."
        )
    if strict_cfg.get("reference_partition") != "UNSW_NB15_training-set.csv":
        problems.append(
            "strict_test_nonoverlap.reference_partition must be the official "
            "training partition."
        )

    overlap_cfg = manifest.get("cross_split_overlap_policy", {})
    if (
        overlap_cfg.get("resolution_status")
        != "RESOLVED_BY_PRIMARY_NONOVERLAP_EVALUATION_POLICY"
    ):
        problems.append("Cross-split overlap resolution policy is not frozen/resolved.")

    if problems:
        raise RuntimeError(
            "Dataset manifest is not ready for deterministic split generation:\n- "
            + "\n- ".join(problems)
        )


def verify_source_integrity(
    train_path: Path,
    test_path: Path,
    train_entry: dict[str, Any],
    test_entry: dict[str, Any],
) -> None:
    for path, entry, role in [
        (train_path, train_entry, "train_raw"),
        (test_path, test_entry, "test_raw"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{role} missing: {path}")

        actual_hash = sha256_file(path)
        expected_hash = str(entry.get("sha256", "")).lower()

        if actual_hash.lower() != expected_hash:
            raise RuntimeError(
                f"{role} SHA-256 mismatch.\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {actual_hash}"
            )


def exact_overlap_test_positions(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[set[int], dict[str, int]]:
    """
    Return exact-overlap test row positions.

    pandas 64-bit row hashes are screening only. Every candidate hash
    match is verified by canonical feature-value tuple equality.
    """
    train_hashes = pd.util.hash_pandas_object(
        df_train[feature_columns], index=False
    ).astype("uint64")

    test_hashes = pd.util.hash_pandas_object(
        df_test[feature_columns], index=False
    ).astype("uint64")

    train_hash_values = [int(x) for x in train_hashes.tolist()]
    test_hash_values = [int(x) for x in test_hashes.tolist()]

    shared_hashes = set(train_hash_values) & set(test_hash_values)

    train_groups: dict[int, list[int]] = defaultdict(list)
    test_groups: dict[int, list[int]] = defaultdict(list)

    for pos, fingerprint in enumerate(train_hash_values):
        if fingerprint in shared_hashes:
            train_groups[fingerprint].append(pos)

    for pos, fingerprint in enumerate(test_hash_values):
        if fingerprint in shared_hashes:
            test_groups[fingerprint].append(pos)

    exact_test_positions: set[int] = set()
    confirmed_unique_vectors: set[tuple[Any, ...]] = set()
    shared_hashes_with_exact_match = 0

    for fingerprint in shared_hashes:
        train_vectors: set[tuple[Any, ...]] = set()

        for pos in train_groups[fingerprint]:
            key = canonical_row(
                df_train.iloc[pos][feature_columns].tolist()
            )
            train_vectors.add(key)

        exact_found_for_hash = False

        for pos in test_groups[fingerprint]:
            key = canonical_row(
                df_test.iloc[pos][feature_columns].tolist()
            )
            if key in train_vectors:
                exact_test_positions.add(pos)
                confirmed_unique_vectors.add(key)
                exact_found_for_hash = True

        if exact_found_for_hash:
            shared_hashes_with_exact_match += 1

    diagnostics = {
        "screening_shared_unique_hashes": len(shared_hashes),
        "shared_hashes_with_exact_match": shared_hashes_with_exact_match,
        "shared_hashes_without_exact_match": (
            len(shared_hashes) - shared_hashes_with_exact_match
        ),
        "confirmed_exact_unique_feature_vectors": len(confirmed_unique_vectors),
        "test_rows_involved_in_exact_overlap": len(exact_test_positions),
    }

    return exact_test_positions, diagnostics


def write_csv_atomically(df: pd.DataFrame, destination: Path) -> None:
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        df.to_csv(tmp, index=False, lineterminator="\n")
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            tmp.unlink()


def copy_file_atomically(source: Path, destination: Path) -> None:
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            tmp.unlink()


def class_counts(df: pd.DataFrame, label_column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in df[label_column].value_counts(dropna=False).sort_index().items()
    }


def main() -> None:
    print("=== RolloutSecBench Deterministic Split Generation ===")

    if not DATASET_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {DATASET_MANIFEST_PATH}")

    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            "Completed Phase 1 dataset audit is required before split generation: "
            f"{AUDIT_PATH}"
        )

    manifest = load_json(DATASET_MANIFEST_PATH)
    audit = load_json(AUDIT_PATH)

    verify_manifest_contract(manifest)

    if audit.get("status") != "PASSED":
        raise RuntimeError(
            f"Dataset audit status must be PASSED; got {audit.get('status')!r}."
        )

    # Ensure the audit was produced against the current dataset manifest.
    current_manifest_hash = sha256_file(DATASET_MANIFEST_PATH)
    audited_manifest_hash = audit.get("manifest", {}).get("sha256")

    # The split-policy freeze changes the manifest after the raw audit, so a hash
    # mismatch is expected. We therefore validate the scientific quantities from
    # the audit rather than requiring the entire manifest byte hash to remain equal.
    print(f"[+] Current dataset manifest SHA-256: {current_manifest_hash}")
    print(f"[+] Dataset audit manifest SHA-256:   {audited_manifest_hash}")

    train_entry = get_file_entry(manifest, "train_raw")
    test_entry = get_file_entry(manifest, "test_raw")

    train_path = resolve_manifest_file(train_entry)
    test_path = resolve_manifest_file(test_entry)

    verify_source_integrity(
        train_path=train_path,
        test_path=test_path,
        train_entry=train_entry,
        test_entry=test_entry,
    )

    ensure_no_existing_artifacts()
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    schema_cfg = manifest["selected_file_schema"]
    feature_cfg = manifest["feature_policy"]
    split_cfg = manifest["split_policy"]

    expected_columns = schema_cfg["columns"]
    if list(df_train.columns) != expected_columns:
        raise RuntimeError("train_raw schema/order differs from dataset manifest.")
    if list(df_test.columns) != expected_columns:
        raise RuntimeError("test_raw schema/order differs from dataset manifest.")

    label_column = schema_cfg["binary_label_column"]
    non_model_columns = set(feature_cfg["non_model_columns"])
    source_candidate_features = [
        column for column in expected_columns if column not in non_model_columns
    ]

    if len(source_candidate_features) != 42:
        raise RuntimeError(
            "Frozen source-candidate representation must contain 42 features; "
            f"found {len(source_candidate_features)}."
        )

    # -----------------------------------------------------------------
    # A. Deterministic development split
    # -----------------------------------------------------------------
    validation_fraction = float(split_cfg["validation_fraction"])
    random_seed = int(split_cfg["random_seed"])

    all_positions = np.arange(len(df_train))

    train_positions, validation_positions = train_test_split(
        all_positions,
        test_size=validation_fraction,
        random_state=random_seed,
        shuffle=True,
        stratify=df_train[label_column],
    )

    # Preserve source-row order inside each artifact after deterministic
    # membership selection. This improves traceability without changing
    # membership.
    train_positions = np.sort(train_positions)
    validation_positions = np.sort(validation_positions)

    if set(train_positions).intersection(set(validation_positions)):
        raise RuntimeError("final_train and validation memberships overlap.")

    if len(train_positions) + len(validation_positions) != len(df_train):
        raise RuntimeError("final_train + validation do not partition train_raw exactly.")

    df_final_train = df_train.iloc[train_positions].copy()
    df_validation = df_train.iloc[validation_positions].copy()

    # -----------------------------------------------------------------
    # B. Strict non-overlap test
    # -----------------------------------------------------------------
    overlap_test_positions, overlap_diag = exact_overlap_test_positions(
        df_train=df_train,
        df_test=df_test,
        feature_columns=source_candidate_features,
    )

    strict_mask = np.ones(len(df_test), dtype=bool)
    if overlap_test_positions:
        strict_mask[list(overlap_test_positions)] = False

    df_strict_test = df_test.iloc[np.flatnonzero(strict_mask)].copy()

    audit_projection = (
        audit.get("cross_split_overlap", {})
        .get("primary_evaluation_projection", {})
    )

    expected_excluded = int(
        audit_projection.get("strict_test_excluded_rows", -1)
    )
    expected_eligible = int(
        audit_projection.get("strict_test_eligible_rows", -1)
    )

    actual_excluded = len(overlap_test_positions)
    actual_eligible = len(df_strict_test)

    if actual_excluded != expected_excluded:
        raise RuntimeError(
            "Strict-test excluded-row count does not reproduce the completed audit.\n"
            f"Audit expected: {expected_excluded}\n"
            f"Generated:      {actual_excluded}"
        )

    if actual_eligible != expected_eligible:
        raise RuntimeError(
            "Strict-test eligible-row count does not reproduce the completed audit.\n"
            f"Audit expected: {expected_eligible}\n"
            f"Generated:      {actual_eligible}"
        )

    # Strong expected-count guard from the completed Phase 1 audit.
    if actual_excluded != 8541 or actual_eligible != 73791:
        raise RuntimeError(
            "Strict-test counts differ from the Phase 1 frozen evidence "
            "(excluded=8541, eligible=73791). "
            f"Observed excluded={actual_excluded}, eligible={actual_eligible}."
        )

    # -----------------------------------------------------------------
    # C. Materialize artifacts
    # -----------------------------------------------------------------
    write_csv_atomically(df_final_train, FINAL_TRAIN_PATH)
    write_csv_atomically(df_validation, VALIDATION_PATH)
    write_csv_atomically(df_strict_test, STRICT_TEST_PATH)

    # Preserve official test exactly: byte-for-byte copy of source.
    copy_file_atomically(test_path, OFFICIAL_TEST_FULL_PATH)

    if sha256_file(OFFICIAL_TEST_FULL_PATH) != sha256_file(test_path):
        raise RuntimeError("official_test_full.csv is not byte-identical to test_raw.")

    # -----------------------------------------------------------------
    # D. Verify written artifacts
    # -----------------------------------------------------------------
    reread_final_train = pd.read_csv(FINAL_TRAIN_PATH)
    reread_validation = pd.read_csv(VALIDATION_PATH)
    reread_strict = pd.read_csv(STRICT_TEST_PATH)
    reread_official = pd.read_csv(OFFICIAL_TEST_FULL_PATH)

    if len(reread_final_train) != len(df_final_train):
        raise RuntimeError("final_train row-count changed during serialization.")
    if len(reread_validation) != len(df_validation):
        raise RuntimeError("validation row-count changed during serialization.")
    if len(reread_strict) != actual_eligible:
        raise RuntimeError("strict_test_nonoverlap row-count changed during serialization.")
    if len(reread_official) != len(df_test):
        raise RuntimeError("official_test_full row-count differs from official test_raw.")

    # -----------------------------------------------------------------
    # E. Provenance manifest
    # -----------------------------------------------------------------
    artifact_info = {}

    for name, path, source_role in [
        ("final_train", FINAL_TRAIN_PATH, "derived_from_train_raw"),
        ("validation", VALIDATION_PATH, "derived_from_train_raw"),
        ("strict_test_nonoverlap", STRICT_TEST_PATH, "derived_from_test_raw"),
        ("official_test_full", OFFICIAL_TEST_FULL_PATH, "byte_copy_of_test_raw"),
    ]:
        frame = pd.read_csv(path)
        artifact_info[name] = {
            "relative_path": str(path.relative_to(REPO_ROOT)),
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "sha256": sha256_file(path),
            "source_role": source_role,
            "binary_label_counts": class_counts(frame, label_column),
        }

    split_manifest = {
        "split_manifest_schema_version": "1.0",
        "artifact": "RolloutSecBench",
        "phase": "Phase 1 - Derived Split Freeze",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "method": "sklearn.model_selection.train_test_split + exact features-only non-overlap filtering",
            "validation_fraction": validation_fraction,
            "random_seed": random_seed,
            "stratification_policy": split_cfg["stratification_policy"],
            "stratification_column": label_column,
            "source_row_order_preserved_within_each_output": True,
        },
        "source_integrity": {
            "dataset_manifest_path": str(DATASET_MANIFEST_PATH.relative_to(REPO_ROOT)),
            "dataset_manifest_sha256": current_manifest_hash,
            "dataset_audit_path": str(AUDIT_PATH.relative_to(REPO_ROOT)),
            "dataset_audit_sha256": sha256_file(AUDIT_PATH),
            "train_raw_path": str(train_path.relative_to(REPO_ROOT)),
            "train_raw_sha256": sha256_file(train_path),
            "test_raw_path": str(test_path.relative_to(REPO_ROOT)),
            "test_raw_sha256": sha256_file(test_path),
        },
        "feature_representation": {
            "name": "SOURCE_CANDIDATE_FEATURES",
            "feature_count": len(source_candidate_features),
            "features": source_candidate_features,
            "feature_list_sha256": sha256_text_lines(source_candidate_features),
            "non_model_columns": sorted(non_model_columns),
        },
        "development_split": {
            "source_rows": int(len(df_train)),
            "final_train_rows": int(len(df_final_train)),
            "validation_rows": int(len(df_validation)),
            "partition_complete": (
                len(df_final_train) + len(df_validation) == len(df_train)
            ),
            "membership_overlap_rows": 0,
        },
        "strict_test_policy": {
            "source": "official_test_raw",
            "reference_partition": "official_train_raw",
            "membership_representation": "SOURCE_CANDIDATE_FEATURES",
            "membership_uses_labels": False,
            "membership_uses_attack_categories": False,
            "membership_uses_model_outputs": False,
            "membership_uses_model_performance": False,
            "screening_method": "pandas_64bit_row_fingerprint",
            "exact_verification_method": "canonical_feature_value_tuple_equality",
            "official_test_rows": int(len(df_test)),
            "excluded_overlap_rows": int(actual_excluded),
            "strict_test_rows": int(actual_eligible),
            "strict_test_percentage": float(100.0 * actual_eligible / len(df_test)),
            "screening_shared_unique_hashes": int(
                overlap_diag["screening_shared_unique_hashes"]
            ),
            "confirmed_exact_unique_feature_vectors": int(
                overlap_diag["confirmed_exact_unique_feature_vectors"]
            ),
            "shared_hashes_without_exact_match": int(
                overlap_diag["shared_hashes_without_exact_match"]
            ),
        },
        "artifacts": artifact_info,
        "training_boundary": {
            "model_training_allowed_after_split_generation": False,
            "reason": (
                "Split artifacts are frozen, but preprocessing must still be "
                "frozen, fitted on final_train only, serialized, and hashed."
            ),
        },
    }

    tmp_manifest = SPLIT_MANIFEST_PATH.with_suffix(".json.tmp")
    try:
        tmp_manifest.write_text(
            json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_manifest, SPLIT_MANIFEST_PATH)
    finally:
        if tmp_manifest.exists():
            tmp_manifest.unlink()

    # -----------------------------------------------------------------
    # F. Final human-readable summary
    # -----------------------------------------------------------------
    print("\n=== DEVELOPMENT SPLIT ===")
    print(f"[+] train_raw rows:      {len(df_train)}")
    print(f"[+] final_train rows:    {len(df_final_train)}")
    print(f"[+] validation rows:     {len(df_validation)}")
    print(f"[+] validation fraction: {validation_fraction}")
    print(f"[+] random seed:         {random_seed}")
    print(f"[+] stratification:      {split_cfg['stratification_policy']}")

    print("\n=== STRICT TEST ===")
    print(f"[+] official test rows:            {len(df_test)}")
    print(f"[+] strict-test excluded rows:     {actual_excluded}")
    print(f"[+] strict-test eligible rows:     {actual_eligible}")
    print(
        "[+] strict-test eligible percent:  "
        f"{100.0 * actual_eligible / len(df_test):.6f}"
    )
    print(
        "[+] exact shared feature vectors:  "
        f"{overlap_diag['confirmed_exact_unique_feature_vectors']}"
    )
    print(
        "[+] fingerprint false matches:     "
        f"{overlap_diag['shared_hashes_without_exact_match']}"
    )

    print("\n=== ARTIFACTS ===")
    for name, info in artifact_info.items():
        print(
            f"[+] {name}: rows={info['rows']} sha256={info['sha256']}"
        )

    print(
        f"[+] split_manifest: {SPLIT_MANIFEST_PATH.relative_to(REPO_ROOT)} "
        f"sha256={sha256_file(SPLIT_MANIFEST_PATH)}"
    )

    print(
        "\n=== SPLIT GENERATION PASSED ===\n"
        "Derived split artifacts are now frozen. "
        "Do NOT begin model training yet: next freeze the preprocessing policy, "
        "fit preprocessing on final_train only, serialize it, and hash it."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
