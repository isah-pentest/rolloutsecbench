#!/usr/bin/env python3
"""
RolloutSecBench - Phase 1 Raw Dataset Audit

Purpose
-------
Validate the acquired raw dataset BEFORE split generation,
preprocessing, model training, or test-set use.

The audit is manifest-driven. Dataset policy and expectations are
declared in:

    data/manifests/dataset_manifest.json

Measured observations are written to:

    results/derived/dataset_audit.json
    results/derived/dataset_class_distribution.csv

This script does NOT:
- create train/validation/test splits,
- fit preprocessing,
- remove duplicate rows,
- delete cross-split overlaps,
- create the strict non-overlap test artifact,
- impute missing values,
- modify labels,
- train models.

Cross-split overlap methodology
-------------------------------
A pandas 64-bit row fingerprint is used only as a screening mechanism.

Any candidate fingerprint match is then verified using actual feature
values. A fingerprint match alone is never treated as proof of exact
feature-vector equality.

The audit distinguishes:
- unique shared feature vectors,
- actual training rows participating in overlap,
- actual test rows participating in overlap,
- label-set agreement/ambiguity,
- strict-test eligible versus excluded test rows.

Status semantics
----------------
PASSED:
    No blocking or review-required findings remain.

REVIEW_REQUIRED:
    No blocking defect was found, but one or more findings require
    a documented protocol decision.

FAILED:
    At least one blocking integrity, schema, label, value-validation,
    policy-consistency, or unresolved contamination finding remains.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import load_config, resolve_path


# ---------------------------------------------------------------------
# Repository / configured paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

_paths_cfg = load_config("paths")

RAW_TRAIN_PATH = Path(
    resolve_path(
        _paths_cfg["data"]["raw_train_file"]
    )
)

RAW_TEST_PATH = Path(
    resolve_path(
        _paths_cfg["data"]["raw_test_file"]
    )
)

RESULTS_DIR = Path(
    resolve_path(
        _paths_cfg["results"]["derived_dir"]
    )
)

AUDIT_JSON_PATH = Path(
    resolve_path(
        _paths_cfg["results"]["dataset_audit"]
    )
)

CLASS_DIST_PATH = Path(
    resolve_path(
        _paths_cfg["results"]["dataset_class_distribution"]
    )
)

MANIFEST_PATH = (
    REPO_ROOT
    / "data"
    / "manifests"
    / "dataset_manifest.json"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Generic utility functions
# ---------------------------------------------------------------------

def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Return SHA-256 digest without loading the whole file into memory.
    """

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:
    """
    Load a JSON document.
    """

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def json_safe_counts(
    series: pd.Series,
) -> dict[str, int]:
    """
    Convert pandas value counts to JSON-safe values.
    """

    counts = series.value_counts(
        dropna=False
    )

    output: dict[str, int] = {}

    for key, value in counts.items():
        if pd.isna(key):
            normalized_key = "__MISSING__"
        else:
            normalized_key = str(key)

        output[normalized_key] = int(
            value
        )

    return output


def numeric_infinite_count(
    df: pd.DataFrame,
) -> tuple[int, dict[str, int]]:
    """
    Return total and per-column counts of +/- infinity.
    """

    numeric = df.select_dtypes(
        include=[np.number]
    )

    per_column: dict[str, int] = {}
    total = 0

    for column in numeric.columns:
        values = numeric[
            column
        ].to_numpy()

        count = int(
            np.isinf(
                values
            ).sum()
        )

        if count > 0:
            per_column[
                column
            ] = count

            total += count

    return total, per_column


def missing_count_by_column(
    df: pd.DataFrame,
) -> dict[str, int]:
    """
    Return non-zero missing-value counts by column.
    """

    counts = df.isna().sum()

    return {
        str(column): int(count)
        for column, count in counts.items()
        if int(count) > 0
    }


def duplicate_row_count(
    df: pd.DataFrame,
) -> int:
    """
    Count exact duplicate raw rows.
    """

    return int(
        df.duplicated().sum()
    )


def get_manifest_file_entry(
    manifest: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    """
    Return exactly one manifest file entry for the requested role.
    """

    files = manifest.get(
        "files",
        [],
    )

    matching = [
        item
        for item in files
        if item.get("role") == role
    ]

    if len(matching) != 1:
        raise ValueError(
            "Dataset manifest must contain exactly one "
            f"file entry with role={role!r}; "
            f"found {len(matching)}."
        )

    return matching[0]


def canonical_scalar(
    value: Any,
) -> Any:
    """
    Normalize a scalar for exact tuple comparison.

    Missing values are represented explicitly so NaN != NaN semantics
    cannot create false inequality.

    NumPy scalar types are converted to native Python scalars.
    """

    if pd.isna(value):
        return (
            "__MISSING__",
        )

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    return value


def canonical_row(
    values: list[Any],
) -> tuple[Any, ...]:
    """
    Convert one feature row to a stable, hashable tuple.
    """

    return tuple(
        canonical_scalar(value)
        for value in values
    )


# ---------------------------------------------------------------------
# Class-distribution output
# ---------------------------------------------------------------------

def build_class_distribution(
    df: pd.DataFrame,
    split_name: str,
    binary_label_column: str,
    attack_category_column: str,
) -> list[dict[str, Any]]:
    """
    Return binary and attack-category distribution records.

    Missing attack-category values remain explicitly missing and
    are never silently converted to "Normal".
    """

    records: list[
        dict[str, Any]
    ] = []

    binary_counts = df[
        binary_label_column
    ].value_counts(
        dropna=False
    )

    for value, count in binary_counts.items():
        category = (
            "__MISSING__"
            if pd.isna(value)
            else str(value)
        )

        records.append(
            {
                "split": split_name,
                "label_type": "binary",
                "category": category,
                "count": int(count),
                "percentage": float(
                    (
                        count
                        / len(df)
                    )
                    * 100.0
                ),
            }
        )

    attack_series = (
        df[
            attack_category_column
        ]
        .astype("string")
        .str.strip()
        .fillna("__MISSING__")
    )

    attack_counts = (
        attack_series
        .value_counts(
            dropna=False
        )
    )

    for value, count in attack_counts.items():
        records.append(
            {
                "split": split_name,
                "label_type": "attack_category",
                "category": str(value),
                "count": int(count),
                "percentage": float(
                    (
                        count
                        / len(df)
                    )
                    * 100.0
                ),
            }
        )

    return records


# ---------------------------------------------------------------------
# Exact cross-split overlap characterization
# ---------------------------------------------------------------------

def exact_cross_split_overlap(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: list[str],
    binary_label_column: str,
) -> dict[str, Any]:
    """
    Characterize cross-split overlap.

    Stage 1:
        pandas 64-bit row fingerprints identify candidate matches.

    Stage 2:
        Candidate matches are verified using actual feature values.

    The final contamination-relevant counts are based on confirmed
    exact feature-vector equality, not fingerprint equality alone.

    The function reports:
    - unique exact shared feature vectors,
    - actual training rows participating in those vectors,
    - actual test rows participating in those vectors,
    - strict-test eligible/excluded row counts,
    - train/test label-set relationships.

    No observations are modified or deleted.
    """

    if not feature_columns:
        return {
            "feature_count": 0,

            "screening_method": (
                "pandas_64bit_row_fingerprint"
            ),

            "screening_shared_unique_hashes": 0,

            "exact_verification_method": (
                "field_value_tuple_equality"
            ),

            "shared_hashes_with_exact_match": 0,

            "shared_hashes_without_exact_match": 0,

            "confirmed_exact_unique_feature_vectors": 0,

            "train_rows_involved_in_exact_overlap": 0,

            "test_rows_involved_in_exact_overlap": 0,

            "strict_test_excluded_rows": 0,

            "strict_test_eligible_rows": int(
                len(df_test)
            ),

            "train_overlap_row_percentage": 0.0,

            "test_overlap_row_percentage": 0.0,

            "strict_test_eligible_percentage": (
                100.0
                if len(df_test) > 0
                else 0.0
            ),

            "exact_vectors_with_identical_label_sets": 0,

            "exact_vectors_with_different_label_sets": 0,

            "exact_vectors_with_within_train_label_ambiguity": 0,

            "exact_vectors_with_within_test_label_ambiguity": 0,
        }

    train_hashes = (
        pd.util.hash_pandas_object(
            df_train[
                feature_columns
            ],
            index=False,
        )
    )

    test_hashes = (
        pd.util.hash_pandas_object(
            df_test[
                feature_columns
            ],
            index=False,
        )
    )

    train_hash_values = [
        int(value)
        for value in train_hashes.tolist()
    ]

    test_hash_values = [
        int(value)
        for value in test_hashes.tolist()
    ]

    shared_hashes = (
        set(
            train_hash_values
        )
        & set(
            test_hash_values
        )
    )

    train_groups: dict[
        int,
        list[int],
    ] = defaultdict(list)

    test_groups: dict[
        int,
        list[int],
    ] = defaultdict(list)

    for row_position, fingerprint in enumerate(
        train_hash_values
    ):
        if fingerprint in shared_hashes:
            train_groups[
                fingerprint
            ].append(
                row_position
            )

    for row_position, fingerprint in enumerate(
        test_hash_values
    ):
        if fingerprint in shared_hashes:
            test_groups[
                fingerprint
            ].append(
                row_position
            )

    exact_vectors: set[
        tuple[Any, ...]
    ] = set()

    exact_train_row_positions: set[int] = set()
    exact_test_row_positions: set[int] = set()

    train_label_sets: dict[
        tuple[Any, ...],
        set[Any],
    ] = defaultdict(set)

    test_label_sets: dict[
        tuple[Any, ...],
        set[Any],
    ] = defaultdict(set)

    hashes_with_exact_match = 0

    for fingerprint in shared_hashes:
        train_vectors: dict[
            tuple[Any, ...],
            list[int],
        ] = defaultdict(list)

        test_vectors: dict[
            tuple[Any, ...],
            list[int],
        ] = defaultdict(list)

        for row_position in train_groups[
            fingerprint
        ]:
            values = (
                df_train.iloc[
                    row_position
                ][
                    feature_columns
                ]
                .tolist()
            )

            key = canonical_row(
                values
            )

            train_vectors[
                key
            ].append(
                row_position
            )

        for row_position in test_groups[
            fingerprint
        ]:
            values = (
                df_test.iloc[
                    row_position
                ][
                    feature_columns
                ]
                .tolist()
            )

            key = canonical_row(
                values
            )

            test_vectors[
                key
            ].append(
                row_position
            )

        exact_for_hash = (
            set(
                train_vectors
            )
            & set(
                test_vectors
            )
        )

        if exact_for_hash:
            hashes_with_exact_match += 1

        for key in exact_for_hash:
            exact_vectors.add(
                key
            )

            exact_train_row_positions.update(
                train_vectors[
                    key
                ]
            )

            exact_test_row_positions.update(
                test_vectors[
                    key
                ]
            )

            for row_position in train_vectors[
                key
            ]:
                label = canonical_scalar(
                    df_train.iloc[
                        row_position
                    ][
                        binary_label_column
                    ]
                )

                train_label_sets[
                    key
                ].add(
                    label
                )

            for row_position in test_vectors[
                key
            ]:
                label = canonical_scalar(
                    df_test.iloc[
                        row_position
                    ][
                        binary_label_column
                    ]
                )

                test_label_sets[
                    key
                ].add(
                    label
                )

    identical_label_sets = 0
    different_label_sets = 0
    within_train_ambiguity = 0
    within_test_ambiguity = 0

    for key in exact_vectors:
        train_labels = (
            train_label_sets[
                key
            ]
        )

        test_labels = (
            test_label_sets[
                key
            ]
        )

        if train_labels == test_labels:
            identical_label_sets += 1
        else:
            different_label_sets += 1

        if len(
            train_labels
        ) > 1:
            within_train_ambiguity += 1

        if len(
            test_labels
        ) > 1:
            within_test_ambiguity += 1

    train_rows_in_overlap = len(
        exact_train_row_positions
    )

    test_rows_in_overlap = len(
        exact_test_row_positions
    )

    strict_test_excluded_rows = (
        test_rows_in_overlap
    )

    strict_test_eligible_rows = (
        len(df_test)
        - strict_test_excluded_rows
    )

    train_overlap_percentage = (
        (
            train_rows_in_overlap
            / len(df_train)
        )
        * 100.0
        if len(df_train) > 0
        else 0.0
    )

    test_overlap_percentage = (
        (
            test_rows_in_overlap
            / len(df_test)
        )
        * 100.0
        if len(df_test) > 0
        else 0.0
    )

    strict_test_eligible_percentage = (
        (
            strict_test_eligible_rows
            / len(df_test)
        )
        * 100.0
        if len(df_test) > 0
        else 0.0
    )

    return {
        "feature_count": int(
            len(
                feature_columns
            )
        ),

        "screening_method": (
            "pandas_64bit_row_fingerprint"
        ),

        "screening_shared_unique_hashes": int(
            len(
                shared_hashes
            )
        ),

        "exact_verification_method": (
            "field_value_tuple_equality"
        ),

        "shared_hashes_with_exact_match": int(
            hashes_with_exact_match
        ),

        "shared_hashes_without_exact_match": int(
            len(
                shared_hashes
            )
            - hashes_with_exact_match
        ),

        "confirmed_exact_unique_feature_vectors": int(
            len(
                exact_vectors
            )
        ),

        "train_rows_involved_in_exact_overlap": int(
            train_rows_in_overlap
        ),

        "test_rows_involved_in_exact_overlap": int(
            test_rows_in_overlap
        ),

        "strict_test_excluded_rows": int(
            strict_test_excluded_rows
        ),

        "strict_test_eligible_rows": int(
            strict_test_eligible_rows
        ),

        "train_overlap_row_percentage": float(
            train_overlap_percentage
        ),

        "test_overlap_row_percentage": float(
            test_overlap_percentage
        ),

        "strict_test_eligible_percentage": float(
            strict_test_eligible_percentage
        ),

        "exact_vectors_with_identical_label_sets": int(
            identical_label_sets
        ),

        "exact_vectors_with_different_label_sets": int(
            different_label_sets
        ),

        "exact_vectors_with_within_train_label_ambiguity": int(
            within_train_ambiguity
        ),

        "exact_vectors_with_within_test_label_ambiguity": int(
            within_test_ambiguity
        ),
    }


# ---------------------------------------------------------------------
# Primary non-overlap policy validation
# ---------------------------------------------------------------------

def validate_primary_nonoverlap_policy(
    overlap_cfg: dict[str, Any],
) -> tuple[
    bool,
    list[str],
]:
    """
    Validate a manifest-declared resolution based on a strict
    non-overlapping primary evaluation subset.

    This validates policy consistency only. It does not create the
    strict-test artifact.

    The resolution is considered internally valid only if membership
    does not depend on labels, attack categories, model predictions,
    probabilities, or model performance and does not modify raw data.
    """

    problems: list[str] = []

    resolution_status = overlap_cfg.get(
        "resolution_status"
    )

    if (
        resolution_status
        != "RESOLVED_BY_PRIMARY_NONOVERLAP_EVALUATION_POLICY"
    ):
        return False, problems

    primary_cfg = overlap_cfg.get(
        "primary_evaluation_policy",
        {},
    )

    if not primary_cfg:
        problems.append(
            "Overlap resolution declares a primary non-overlap "
            "evaluation policy but primary_evaluation_policy is absent."
        )

        return False, problems

    if (
        primary_cfg.get(
            "name"
        )
        != "STRICT_NONOVERLAP_TEST"
    ):
        problems.append(
            "primary_evaluation_policy.name must identify the "
            "strict non-overlap evaluation policy."
        )

    prohibited_true_fields = [
        "membership_uses_binary_label",
        "membership_uses_attack_category",
        "membership_uses_model_prediction",
        "membership_uses_model_probability",
        "membership_uses_model_performance",
        "raw_test_rows_deleted",
        "raw_training_rows_deleted",
    ]

    for field in prohibited_true_fields:
        value = primary_cfg.get(
            field
        )

        if value is not False:
            problems.append(
                f"primary_evaluation_policy.{field} "
                "must be explicitly false."
            )

    membership_rule = primary_cfg.get(
        "membership_rule"
    )

    if not membership_rule:
        problems.append(
            "primary_evaluation_policy.membership_rule "
            "must be explicitly documented."
        )

    return (
        len(problems) == 0,
        problems,
    )


# ---------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------

def run_audit() -> None:
    """
    Execute Phase 1 raw dataset audit.
    """

    print(
        "=== RolloutSecBench Phase 1 Raw Dataset Audit ==="
    )

    blocking_findings: list[str] = []
    review_findings: list[str] = []
    informational_findings: list[str] = []

    # -----------------------------------------------------------------
    # 1. Manifest
    # -----------------------------------------------------------------

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {MANIFEST_PATH}"
        )

    manifest = load_json(
        MANIFEST_PATH
    )

    manifest_sha256 = sha256_file(
        MANIFEST_PATH
    )

    dataset_cfg = manifest.get(
        "dataset",
        {},
    )

    schema_cfg = manifest.get(
        "selected_file_schema",
        {},
    )

    feature_cfg = manifest.get(
        "feature_policy",
        {},
    )

    integrity_cfg = manifest.get(
        "integrity_policy",
        {},
    )

    duplicate_cfg = manifest.get(
        "duplicate_policy",
        {},
    )

    missing_cfg = manifest.get(
        "missing_value_policy",
        {},
    )

    value_validation_cfg = manifest.get(
        "value_validation_policy",
        {},
    )

    overlap_cfg = manifest.get(
        "cross_split_overlap_policy",
        {},
    )

    temporal_cfg = manifest.get(
        "temporal_ordering_assessment",
        {},
    )

    train_manifest = get_manifest_file_entry(
        manifest,
        "train_raw",
    )

    test_manifest = get_manifest_file_entry(
        manifest,
        "test_raw",
    )

    # -----------------------------------------------------------------
    # 2. Raw file existence
    # -----------------------------------------------------------------

    required_files = [
        (
            RAW_TRAIN_PATH,
            "train_raw",
        ),
        (
            RAW_TEST_PATH,
            "test_raw",
        ),
    ]

    missing_files = []

    for path, role in required_files:
        if not path.exists():
            missing_files.append(
                f"{role}: {path}"
            )

    if missing_files:
        raise FileNotFoundError(
            "Missing required raw dataset files: "
            + "; ".join(
                missing_files
            )
        )

    # -----------------------------------------------------------------
    # 3. Cryptographic verification before CSV parsing
    # -----------------------------------------------------------------

    actual_train_sha256 = sha256_file(
        RAW_TRAIN_PATH
    )

    actual_test_sha256 = sha256_file(
        RAW_TEST_PATH
    )

    expected_train_sha256 = str(
        train_manifest.get(
            "sha256",
            "",
        )
    ).lower()

    expected_test_sha256 = str(
        test_manifest.get(
            "sha256",
            "",
        )
    ).lower()

    train_hash_matches = (
        actual_train_sha256.lower()
        == expected_train_sha256
    )

    test_hash_matches = (
        actual_test_sha256.lower()
        == expected_test_sha256
    )

    if integrity_cfg.get(
        "verify_sha256_before_processing",
        True,
    ):
        if not train_hash_matches:
            blocking_findings.append(
                "train_raw SHA-256 does not match "
                "dataset_manifest.json."
            )

        if not test_hash_matches:
            blocking_findings.append(
                "test_raw SHA-256 does not match "
                "dataset_manifest.json."
            )

    # -----------------------------------------------------------------
    # 4. Load raw files
    # -----------------------------------------------------------------

    df_train = pd.read_csv(
        RAW_TRAIN_PATH
    )

    df_test = pd.read_csv(
        RAW_TEST_PATH
    )

    df_full = pd.concat(
        [
            df_train,
            df_test,
        ],
        ignore_index=True,
    )

    # -----------------------------------------------------------------
    # 5. Manifest row-count checks
    # -----------------------------------------------------------------

    expected_train_rows = int(
        train_manifest[
            "expected_row_count"
        ]
    )

    expected_test_rows = int(
        test_manifest[
            "expected_row_count"
        ]
    )

    expected_total_rows = int(
        schema_cfg[
            "expected_total_records"
        ]
    )

    train_row_count_matches = (
        len(
            df_train
        )
        == expected_train_rows
    )

    test_row_count_matches = (
        len(
            df_test
        )
        == expected_test_rows
    )

    total_row_count_matches = (
        len(
            df_full
        )
        == expected_total_rows
    )

    if integrity_cfg.get(
        "verify_manifest_row_counts",
        True,
    ):
        if not train_row_count_matches:
            blocking_findings.append(
                "train_raw row count does not match "
                "the manifest expected row count."
            )

        if not test_row_count_matches:
            blocking_findings.append(
                "test_raw row count does not match "
                "the manifest expected row count."
            )

        if not total_row_count_matches:
            blocking_findings.append(
                "Combined raw row count does not match "
                "selected_file_schema.expected_total_records."
            )

    # -----------------------------------------------------------------
    # 6. Raw schema validation
    # -----------------------------------------------------------------

    train_columns = list(
        df_train.columns
    )

    test_columns = list(
        df_test.columns
    )

    same_column_set = (
        set(
            train_columns
        )
        == set(
            test_columns
        )
    )

    same_column_order = (
        train_columns
        == test_columns
    )

    if integrity_cfg.get(
        "require_matching_train_test_schema",
        True,
    ):
        if not same_column_set:
            blocking_findings.append(
                "Raw train/test column sets differ."
            )

        elif not same_column_order:
            review_findings.append(
                "Raw train/test contain the same columns "
                "but in different order."
            )

    expected_raw_column_count = int(
        schema_cfg[
            "raw_column_count"
        ]
    )

    if (
        len(
            train_columns
        )
        != expected_raw_column_count
    ):
        blocking_findings.append(
            f"train_raw has {len(train_columns)} columns; "
            f"manifest expects {expected_raw_column_count}."
        )

    if (
        len(
            test_columns
        )
        != expected_raw_column_count
    ):
        blocking_findings.append(
            f"test_raw has {len(test_columns)} columns; "
            f"manifest expects {expected_raw_column_count}."
        )

    expected_columns = schema_cfg.get(
        "columns",
        [],
    )

    expected_columns_present = bool(
        expected_columns
    )

    train_matches_expected_columns = (
        train_columns
        == expected_columns
        if expected_columns_present
        else None
    )

    test_matches_expected_columns = (
        test_columns
        == expected_columns
        if expected_columns_present
        else None
    )

    if expected_columns_present:
        if not train_matches_expected_columns:
            blocking_findings.append(
                "train_raw column names/order do not match "
                "selected_file_schema.columns."
            )

        if not test_matches_expected_columns:
            blocking_findings.append(
                "test_raw column names/order do not match "
                "selected_file_schema.columns."
            )

    binary_label_column = schema_cfg[
        "binary_label_column"
    ]

    attack_category_column = schema_cfg[
        "multiclass_label_column"
    ]

    identifier_columns = list(
        schema_cfg.get(
            "identifier_columns",
            [],
        )
    )

    required_columns = {
        binary_label_column,
        attack_category_column,
    }

    required_columns.update(
        identifier_columns
    )

    missing_required_train = sorted(
        required_columns.difference(
            df_train.columns
        )
    )

    missing_required_test = sorted(
        required_columns.difference(
            df_test.columns
        )
    )

    if missing_required_train:
        blocking_findings.append(
            "train_raw is missing required columns: "
            + ", ".join(
                missing_required_train
            )
        )

    if missing_required_test:
        blocking_findings.append(
            "test_raw is missing required columns: "
            + ", ".join(
                missing_required_test
            )
        )

    if (
        missing_required_train
        or missing_required_test
    ):
        raise ValueError(
            "Required identifier/label columns are missing; "
            "remaining audit checks cannot be executed safely."
        )

    # -----------------------------------------------------------------
    # 7. Label integrity
    # -----------------------------------------------------------------

    expected_binary_labels = set(
        schema_cfg[
            "expected_binary_label_values"
        ]
    )

    train_missing_binary_labels = int(
        df_train[
            binary_label_column
        ]
        .isna()
        .sum()
    )

    test_missing_binary_labels = int(
        df_test[
            binary_label_column
        ]
        .isna()
        .sum()
    )

    missing_binary_allowed = bool(
        integrity_cfg.get(
            "missing_binary_labels_allowed",
            False,
        )
    )

    if not missing_binary_allowed:
        if train_missing_binary_labels > 0:
            blocking_findings.append(
                f"train_raw contains "
                f"{train_missing_binary_labels} missing "
                "binary labels."
            )

        if test_missing_binary_labels > 0:
            blocking_findings.append(
                f"test_raw contains "
                f"{test_missing_binary_labels} missing "
                "binary labels."
            )

    train_binary_values = set(
        df_train[
            binary_label_column
        ]
        .dropna()
        .unique()
        .tolist()
    )

    test_binary_values = set(
        df_test[
            binary_label_column
        ]
        .dropna()
        .unique()
        .tolist()
    )

    unexpected_train_labels = sorted(
        train_binary_values.difference(
            expected_binary_labels
        )
    )

    unexpected_test_labels = sorted(
        test_binary_values.difference(
            expected_binary_labels
        )
    )

    unexpected_labels_allowed = bool(
        integrity_cfg.get(
            "unexpected_binary_label_values_allowed",
            False,
        )
    )

    if not unexpected_labels_allowed:
        if unexpected_train_labels:
            blocking_findings.append(
                "Unexpected binary labels in train_raw: "
                + ", ".join(
                    str(value)
                    for value in unexpected_train_labels
                )
            )

        if unexpected_test_labels:
            blocking_findings.append(
                "Unexpected binary labels in test_raw: "
                + ", ".join(
                    str(value)
                    for value in unexpected_test_labels
                )
            )

    # -----------------------------------------------------------------
    # 8. Feature policy derived from actual schema
    # -----------------------------------------------------------------

    non_model_columns = set(
        feature_cfg.get(
            "non_model_columns",
            [],
        )
    )

    policy_excluded_features = set(
        feature_cfg.get(
            "policy_excluded_features",
            [],
        )
    )

    missing_non_model_columns = sorted(
        non_model_columns.difference(
            train_columns
        )
    )

    if missing_non_model_columns:
        blocking_findings.append(
            "Manifest non-model columns absent from train_raw: "
            + ", ".join(
                missing_non_model_columns
            )
        )

    present_policy_excluded_features = sorted(
        policy_excluded_features.intersection(
            train_columns
        )
    )

    absent_policy_excluded_features = sorted(
        policy_excluded_features.difference(
            train_columns
        )
    )

    source_candidate_features = [
        column
        for column in train_columns
        if column not in non_model_columns
    ]

    effective_candidate_features = [
        column
        for column in source_candidate_features
        if column not in policy_excluded_features
    ]

    if not source_candidate_features:
        blocking_findings.append(
            "No source candidate model features remain after "
            "removing manifest-defined non-model columns."
        )

    if not effective_candidate_features:
        blocking_findings.append(
            "No effective candidate model features remain "
            "after applying the feature-exclusion policy."
        )

    feature_policy_status = feature_cfg.get(
        "policy_status"
    )

    feature_exclusion_rationale_status = feature_cfg.get(
        "policy_exclusion_rationale_status"
    )

    if (
        feature_policy_status
        == "PROVISIONAL_PENDING_PHASE1_RESOLUTION"
    ):
        review_findings.append(
            "Feature policy remains provisional and must be "
            "formally resolved before Phase 1 is frozen."
        )

    if (
        feature_exclusion_rationale_status
        == "REQUIRES_FINAL_DOCUMENTED_JUSTIFICATION_BEFORE_SPLIT_FREEZE"
        and present_policy_excluded_features
    ):
        review_findings.append(
            "Present policy-excluded features require a final "
            "documented justification before split freeze: "
            + ", ".join(
                present_policy_excluded_features
            )
        )

    # -----------------------------------------------------------------
    # 9. Missing and infinite values
    # -----------------------------------------------------------------

    train_missing_by_column = missing_count_by_column(
        df_train
    )

    test_missing_by_column = missing_count_by_column(
        df_test
    )

    train_missing_total = int(
        df_train
        .isna()
        .sum()
        .sum()
    )

    test_missing_total = int(
        df_test
        .isna()
        .sum()
        .sum()
    )

    train_inf_total, train_inf_by_column = (
        numeric_infinite_count(
            df_train
        )
    )

    test_inf_total, test_inf_by_column = (
        numeric_infinite_count(
            df_test
        )
    )

    infinite_allowed = bool(
        integrity_cfg.get(
            "infinite_numeric_values_allowed",
            False,
        )
    )

    if not infinite_allowed:
        if train_inf_total > 0:
            blocking_findings.append(
                f"train_raw contains {train_inf_total} "
                "infinite numeric values."
            )

        if test_inf_total > 0:
            blocking_findings.append(
                f"test_raw contains {test_inf_total} "
                "infinite numeric values."
            )

    train_feature_missing = {
        column: int(
            df_train[
                column
            ]
            .isna()
            .sum()
        )
        for column in source_candidate_features
        if int(
            df_train[
                column
            ]
            .isna()
            .sum()
        ) > 0
    }

    test_feature_missing = {
        column: int(
            df_test[
                column
            ]
            .isna()
            .sum()
        )
        for column in source_candidate_features
        if int(
            df_test[
                column
            ]
            .isna()
            .sum()
        ) > 0
    }

    feature_missing_policy = missing_cfg.get(
        "feature_missing_values",
        "REPORT_AND_RESOLVE_BEFORE_PREPROCESSING_FREEZE",
    )

    if train_feature_missing:
        if feature_missing_policy == "BLOCK":
            blocking_findings.append(
                "Missing candidate-feature values occur in "
                "train_raw."
            )
        else:
            review_findings.append(
                "Missing candidate-feature values occur in "
                "train_raw and require a documented "
                "preprocessing decision."
            )

    if test_feature_missing:
        if feature_missing_policy == "BLOCK":
            blocking_findings.append(
                "Missing candidate-feature values occur in "
                "test_raw."
            )
        else:
            review_findings.append(
                "Missing candidate-feature values occur in "
                "test_raw; any treatment must use the frozen "
                "train-fitted preprocessing policy."
            )

    train_attack_category_missing = int(
        df_train[
            attack_category_column
        ]
        .isna()
        .sum()
    )

    test_attack_category_missing = int(
        df_test[
            attack_category_column
        ]
        .isna()
        .sum()
    )

    # -----------------------------------------------------------------
    # 10. Duplicate and identifier checks
    # -----------------------------------------------------------------

    train_duplicate_rows = duplicate_row_count(
        df_train
    )

    test_duplicate_rows = duplicate_row_count(
        df_test
    )

    duplicate_policy = duplicate_cfg.get(
        "within_split_exact_duplicates",
        "REVIEW_AND_DOCUMENT",
    )

    if train_duplicate_rows > 0:
        if duplicate_policy == "BLOCK":
            blocking_findings.append(
                f"train_raw contains {train_duplicate_rows} "
                "exact duplicate raw rows."
            )
        else:
            review_findings.append(
                f"train_raw contains {train_duplicate_rows} "
                "exact duplicate raw rows; no rows were removed."
            )

    if test_duplicate_rows > 0:
        if duplicate_policy == "BLOCK":
            blocking_findings.append(
                f"test_raw contains {test_duplicate_rows} "
                "exact duplicate raw rows."
            )
        else:
            review_findings.append(
                f"test_raw contains {test_duplicate_rows} "
                "exact duplicate raw rows; no rows were removed."
            )

    identifier_duplicate_summary: dict[
        str,
        Any,
    ] = {}

    for column in identifier_columns:
        if (
            column in df_train.columns
            and column in df_test.columns
        ):
            train_duplicate_identifiers = int(
                df_train[
                    column
                ]
                .duplicated()
                .sum()
            )

            test_duplicate_identifiers = int(
                df_test[
                    column
                ]
                .duplicated()
                .sum()
            )

            train_identifiers = set(
                df_train[
                    column
                ]
                .dropna()
                .tolist()
            )

            test_identifiers = set(
                df_test[
                    column
                ]
                .dropna()
                .tolist()
            )

            cross_split_identifier_overlap = len(
                train_identifiers.intersection(
                    test_identifiers
                )
            )

            identifier_duplicate_summary[
                column
            ] = {
                "train_duplicate_count": int(
                    train_duplicate_identifiers
                ),

                "test_duplicate_count": int(
                    test_duplicate_identifiers
                ),

                "cross_split_unique_value_overlap": int(
                    cross_split_identifier_overlap
                ),
            }

    # -----------------------------------------------------------------
    # 11. Cross-split feature-vector overlap
    # -----------------------------------------------------------------

    representation_map = {
        "SOURCE_CANDIDATE_FEATURES": (
            source_candidate_features
        ),

        "EFFECTIVE_CANDIDATE_FEATURES": (
            effective_candidate_features
        ),
    }

    requested_representations = overlap_cfg.get(
        "evaluate_representations"
    )

    if requested_representations is None:
        comparison_representation = overlap_cfg.get(
            "comparison_representation"
        )

        if comparison_representation:
            requested_representations = [
                comparison_representation
            ]
        else:
            requested_representations = list(
                representation_map.keys()
            )

    unknown_representations = [
        representation
        for representation in requested_representations
        if representation not in representation_map
    ]

    if unknown_representations:
        blocking_findings.append(
            "Unknown cross-split representation(s) "
            "declared in manifest: "
            + ", ".join(
                unknown_representations
            )
        )

    overlap_results: dict[
        str,
        Any,
    ] = {}

    for representation in requested_representations:
        if representation not in representation_map:
            continue

        overlap_results[
            representation
        ] = exact_cross_split_overlap(
            df_train=df_train,
            df_test=df_test,
            feature_columns=representation_map[
                representation
            ],
            binary_label_column=binary_label_column,
        )

    source_overlap = overlap_results.get(
        "SOURCE_CANDIDATE_FEATURES"
    )

    if source_overlap is None:
        source_overlap = exact_cross_split_overlap(
            df_train=df_train,
            df_test=df_test,
            feature_columns=source_candidate_features,
            binary_label_column=binary_label_column,
        )

        overlap_results[
            "SOURCE_CANDIDATE_FEATURES"
        ] = source_overlap

    effective_overlap = overlap_results.get(
        "EFFECTIVE_CANDIDATE_FEATURES"
    )

    if effective_overlap is None:
        effective_overlap = exact_cross_split_overlap(
            df_train=df_train,
            df_test=df_test,
            feature_columns=effective_candidate_features,
            binary_label_column=binary_label_column,
        )

        overlap_results[
            "EFFECTIVE_CANDIDATE_FEATURES"
        ] = effective_overlap

    overlap_resolution_status = overlap_cfg.get(
        "resolution_status"
    )

    (
        strict_policy_valid,
        strict_policy_problems,
    ) = validate_primary_nonoverlap_policy(
        overlap_cfg
    )

    if strict_policy_problems:
        for problem in strict_policy_problems:
            blocking_findings.append(
                problem
            )

    effective_overlap_action = overlap_cfg.get(
        "effective_overlap_action"
    )

    confirmed_effective_overlap = int(
        effective_overlap.get(
            "confirmed_exact_unique_feature_vectors",
            0,
        )
    )

    if confirmed_effective_overlap > 0:
        if (
            overlap_resolution_status
            == "RESOLVED_BY_PRIMARY_NONOVERLAP_EVALUATION_POLICY"
            and strict_policy_valid
        ):
            informational_findings.append(
                "Exact cross-split model-input overlap exists, "
                "but the manifest resolves this source-partition "
                "characteristic through a predeclared strict "
                "non-overlap primary evaluation policy. "
                "Raw source partitions remain unchanged."
            )

        elif (
            effective_overlap_action
            == "BLOCK_UNTIL_RESOLVED"
        ):
            blocking_findings.append(
                "Confirmed cross-split effective-feature overlap: "
                f"{confirmed_effective_overlap} unique exact "
                "model-input feature vectors occur in both "
                "supplied source partitions."
            )

        else:
            review_findings.append(
                "Confirmed cross-split model-input overlap is "
                "present but no fully resolved strict primary "
                "evaluation policy is currently validated."
            )

    conflicting_overlap_labels = int(
        source_overlap.get(
            "exact_vectors_with_different_label_sets",
            0,
        )
    )

    if conflicting_overlap_labels > 0:
        if (
            overlap_resolution_status
            == "RESOLVED_BY_PRIMARY_NONOVERLAP_EVALUATION_POLICY"
            and strict_policy_valid
        ):
            informational_findings.append(
                "Exact shared source-candidate feature vectors "
                "with different binary label sets are present: "
                f"{conflicting_overlap_labels} unique vectors. "
                "They are reported as feature-space label "
                "ambiguity and are excluded from the strict "
                "non-overlap primary evaluation by construction."
            )
        else:
            review_findings.append(
                "Some exact cross-split feature vectors have "
                "different binary label sets across the supplied "
                "partitions: "
                f"{conflicting_overlap_labels} unique vectors."
            )

    # -----------------------------------------------------------------
    # 12. Constant-feature check
    # -----------------------------------------------------------------

    train_constant_features = [
        column
        for column in effective_candidate_features
        if (
            df_train[
                column
            ]
            .nunique(
                dropna=False
            )
            <= 1
        )
    ]

    if train_constant_features:
        review_findings.append(
            "Constant effective candidate features occur in "
            "train_raw: "
            + ", ".join(
                train_constant_features
            )
            + ". Their treatment must be documented before "
            "preprocessing freeze."
        )

    # -----------------------------------------------------------------
    # 13. Manifest-driven numeric value validation
    # -----------------------------------------------------------------

    declared_non_negative_fields = list(
        value_validation_cfg.get(
            "non_negative_numeric_fields",
            [],
        )
    )

    value_violation_action = (
        value_validation_cfg.get(
            "action_on_violation",
            "BLOCK",
        )
    )

    missing_declared_field_action = (
        value_validation_cfg.get(
            "action_on_missing_declared_field",
            "REVIEW_REQUIRED",
        )
    )

    missing_declared_non_negative_fields = [
        column
        for column in declared_non_negative_fields
        if (
            column not in df_train.columns
            or column not in df_test.columns
        )
    ]

    if missing_declared_non_negative_fields:
        finding = (
            "Manifest-declared non-negative validation fields "
            "are absent from one or both source partitions: "
            + ", ".join(
                missing_declared_non_negative_fields
            )
        )

        if missing_declared_field_action == "BLOCK":
            blocking_findings.append(
                finding
            )
        else:
            review_findings.append(
                finding
            )

    impossible_negative_values: dict[
        str,
        dict[str, int],
    ] = {
        "train_raw": {},
        "test_raw": {},
    }

    for column in declared_non_negative_fields:
        if column in df_train.columns:
            count = int(
                (
                    df_train[
                        column
                    ]
                    < 0
                ).sum()
            )

            if count > 0:
                impossible_negative_values[
                    "train_raw"
                ][
                    column
                ] = count

        if column in df_test.columns:
            count = int(
                (
                    df_test[
                        column
                    ]
                    < 0
                ).sum()
            )

            if count > 0:
                impossible_negative_values[
                    "test_raw"
                ][
                    column
                ] = count

    for split_name in [
        "train_raw",
        "test_raw",
    ]:
        if impossible_negative_values[
            split_name
        ]:
            finding = (
                f"Negative values detected in {split_name} "
                "for fields declared non-negative by the "
                "manifest."
            )

            if value_violation_action == "BLOCK":
                blocking_findings.append(
                    finding
                )
            else:
                review_findings.append(
                    finding
                )

    # -----------------------------------------------------------------
    # 14. Temporal-ordering assessment
    # -----------------------------------------------------------------

    configured_time_columns = list(
        temporal_cfg.get(
            "configured_time_columns",
            [],
        )
    )

    present_time_columns = [
        column
        for column in configured_time_columns
        if (
            column in df_train.columns
            and column in df_test.columns
        )
    ]

    temporal_status = temporal_cfg.get(
        "status",
        "UNSPECIFIED",
    )

    explicit_timestamp_available = temporal_cfg.get(
        "explicit_timestamp_available_in_selected_split_files"
    )

    chronological_split_claimed = bool(
        temporal_cfg.get(
            "chronological_split_claimed",
            False,
        )
    )

    if (
        temporal_status
        == "PENDING_DATA_DICTIONARY_REVIEW"
    ):
        review_findings.append(
            "Temporal-ordering assessment remains pending."
        )

    elif (
        temporal_status
        == "ASSESSED_NOT_VERIFIABLE_FROM_SELECTED_DERIVED_FILES"
    ):
        if chronological_split_claimed:
            blocking_findings.append(
                "Temporal configuration is internally "
                "inconsistent: chronological_split_claimed=true "
                "while temporal ordering is marked not verifiable."
            )

        if explicit_timestamp_available is True:
            blocking_findings.append(
                "Temporal configuration is internally "
                "inconsistent: explicit timestamp availability "
                "is true while temporal ordering is marked "
                "not verifiable from the selected files."
            )

    elif configured_time_columns:
        missing_configured_time_columns = [
            column
            for column in configured_time_columns
            if column not in present_time_columns
        ]

        if missing_configured_time_columns:
            review_findings.append(
                "Configured temporal columns are missing from "
                "one or both source partitions: "
                + ", ".join(
                    missing_configured_time_columns
                )
            )

    temporal_assessment = {
        "configured_time_columns": (
            configured_time_columns
        ),

        "present_in_both_source_splits": (
            present_time_columns
        ),

        "explicit_timestamp_available_in_selected_split_files": (
            explicit_timestamp_available
        ),

        "chronological_split_claimed": (
            chronological_split_claimed
        ),

        "status": temporal_status,

        "interpretation": temporal_cfg.get(
            "interpretation"
        ),
    }

    # -----------------------------------------------------------------
    # 15. Class distributions
    # -----------------------------------------------------------------

    class_distribution_records: list[
        dict[str, Any]
    ] = []

    class_distribution_records.extend(
        build_class_distribution(
            df_train,
            "train_raw",
            binary_label_column,
            attack_category_column,
        )
    )

    class_distribution_records.extend(
        build_class_distribution(
            df_test,
            "test_raw",
            binary_label_column,
            attack_category_column,
        )
    )

    class_distribution_records.extend(
        build_class_distribution(
            df_full,
            "combined_raw",
            binary_label_column,
            attack_category_column,
        )
    )

    class_distribution_df = pd.DataFrame(
        class_distribution_records
    )

    CLASS_DIST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_distribution_df.to_csv(
        CLASS_DIST_PATH,
        index=False,
    )

    # -----------------------------------------------------------------
    # 16. Determine status
    # -----------------------------------------------------------------

    if blocking_findings:
        status = "FAILED"

    elif review_findings:
        status = "REVIEW_REQUIRED"

    else:
        status = "PASSED"

    # -----------------------------------------------------------------
    # 17. Build audit result
    # -----------------------------------------------------------------

    audit_results = {
        "audit_schema_version": "1.3",

        "audit_timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": status,

        "manifest": {
            "path": str(
                MANIFEST_PATH.relative_to(
                    REPO_ROOT
                )
            ),

            "sha256": manifest_sha256,

            "manifest_schema_version": (
                manifest.get(
                    "manifest_schema_version"
                )
            ),

            "dataset_name": (
                dataset_cfg.get(
                    "dataset_name"
                )
            ),

            "dataset_release": (
                dataset_cfg.get(
                    "dataset_release"
                )
            ),
        },

        "integrity_checks": {
            "train_raw": {
                "path": str(
                    RAW_TRAIN_PATH
                ),

                "manifest_sha256": (
                    expected_train_sha256
                ),

                "actual_sha256": (
                    actual_train_sha256
                ),

                "sha256_match": bool(
                    train_hash_matches
                ),

                "manifest_expected_row_count": (
                    expected_train_rows
                ),

                "actual_row_count": int(
                    len(
                        df_train
                    )
                ),

                "row_count_match": bool(
                    train_row_count_matches
                ),
            },

            "test_raw": {
                "path": str(
                    RAW_TEST_PATH
                ),

                "manifest_sha256": (
                    expected_test_sha256
                ),

                "actual_sha256": (
                    actual_test_sha256
                ),

                "sha256_match": bool(
                    test_hash_matches
                ),

                "manifest_expected_row_count": (
                    expected_test_rows
                ),

                "actual_row_count": int(
                    len(
                        df_test
                    )
                ),

                "row_count_match": bool(
                    test_row_count_matches
                ),
            },

            "combined_expected_rows": (
                expected_total_rows
            ),

            "combined_actual_rows": int(
                len(
                    df_full
                )
            ),

            "combined_row_count_match": bool(
                total_row_count_matches
            ),
        },

        "schema_checks": {
            "train_column_count": int(
                len(
                    train_columns
                )
            ),

            "test_column_count": int(
                len(
                    test_columns
                )
            ),

            "expected_raw_column_count": (
                expected_raw_column_count
            ),

            "same_column_set": bool(
                same_column_set
            ),

            "same_column_order": bool(
                same_column_order
            ),

            "expected_columns_declared": bool(
                expected_columns_present
            ),

            "train_matches_expected_columns": (
                train_matches_expected_columns
            ),

            "test_matches_expected_columns": (
                test_matches_expected_columns
            ),

            "observed_columns": (
                train_columns
            ),
        },

        "split_summary": {
            "train_raw_rows": int(
                len(
                    df_train
                )
            ),

            "test_raw_rows": int(
                len(
                    df_test
                )
            ),

            "combined_rows": int(
                len(
                    df_full
                )
            ),
        },

        "label_checks": {
            "binary_label_column": (
                binary_label_column
            ),

            "attack_category_column": (
                attack_category_column
            ),

            "expected_binary_values": sorted(
                expected_binary_labels
            ),

            "train_binary_counts": (
                json_safe_counts(
                    df_train[
                        binary_label_column
                    ]
                )
            ),

            "test_binary_counts": (
                json_safe_counts(
                    df_test[
                        binary_label_column
                    ]
                )
            ),

            "train_missing_binary_labels": (
                train_missing_binary_labels
            ),

            "test_missing_binary_labels": (
                test_missing_binary_labels
            ),

            "unexpected_train_binary_labels": (
                unexpected_train_labels
            ),

            "unexpected_test_binary_labels": (
                unexpected_test_labels
            ),

            "train_missing_attack_category": (
                train_attack_category_missing
            ),

            "test_missing_attack_category": (
                test_attack_category_missing
            ),
        },

        "missing_and_infinite_values": {
            "train_raw": {
                "missing_total": (
                    train_missing_total
                ),

                "missing_by_column": (
                    train_missing_by_column
                ),

                "candidate_feature_missing_by_column": (
                    train_feature_missing
                ),

                "infinite_total": (
                    train_inf_total
                ),

                "infinite_by_column": (
                    train_inf_by_column
                ),
            },

            "test_raw": {
                "missing_total": (
                    test_missing_total
                ),

                "missing_by_column": (
                    test_missing_by_column
                ),

                "candidate_feature_missing_by_column": (
                    test_feature_missing
                ),

                "infinite_total": (
                    test_inf_total
                ),

                "infinite_by_column": (
                    test_inf_by_column
                ),
            },
        },

        "duplicate_checks": {
            "train_exact_duplicate_rows": int(
                train_duplicate_rows
            ),

            "test_exact_duplicate_rows": int(
                test_duplicate_rows
            ),

            "identifier_summary": (
                identifier_duplicate_summary
            ),
        },

        "feature_policy": {
            "policy_status": (
                feature_policy_status
            ),

            "non_model_columns": sorted(
                non_model_columns
            ),

            "policy_excluded_features": sorted(
                policy_excluded_features
            ),

            "present_policy_excluded_features": (
                present_policy_excluded_features
            ),

            "absent_policy_excluded_features": (
                absent_policy_excluded_features
            ),

            "source_candidate_features": (
                source_candidate_features
            ),

            "source_candidate_feature_count": int(
                len(
                    source_candidate_features
                )
            ),

            "effective_candidate_features": (
                effective_candidate_features
            ),

            "effective_candidate_feature_count": int(
                len(
                    effective_candidate_features
                )
            ),

            "train_constant_effective_features": (
                train_constant_features
            ),
        },

        "cross_split_overlap": {
            "policy": {
                "screening_method": (
                    overlap_cfg.get(
                        "screening_method"
                    )
                ),

                "exact_verification_required": (
                    overlap_cfg.get(
                        "exact_verification_required"
                    )
                ),

                "resolution_status": (
                    overlap_resolution_status
                ),

                "strict_primary_policy_valid": bool(
                    strict_policy_valid
                ),

                "effective_overlap_action": (
                    effective_overlap_action
                ),
            },

            "representations": (
                overlap_results
            ),

            "primary_evaluation_projection": {
                "reference_representation": (
                    "SOURCE_CANDIDATE_FEATURES"
                ),

                "official_test_rows": int(
                    len(
                        df_test
                    )
                ),

                "strict_test_excluded_rows": int(
                    source_overlap.get(
                        "strict_test_excluded_rows",
                        0,
                    )
                ),

                "strict_test_eligible_rows": int(
                    source_overlap.get(
                        "strict_test_eligible_rows",
                        len(df_test),
                    )
                ),

                "strict_test_eligible_percentage": float(
                    source_overlap.get(
                        "strict_test_eligible_percentage",
                        100.0,
                    )
                ),

                "artifact_created_by_audit": False,
            },

            "interpretation": (
                "Fingerprint equality is used only as a "
                "screening mechanism. Confirmed exact overlap "
                "is based on actual feature-value equality. "
                "The audit reports the projected strict-test "
                "membership counts but does not modify either "
                "source partition or create the strict-test artifact."
            ),
        },

        "value_validation": {
            "non_negative_numeric_fields": (
                declared_non_negative_fields
            ),

            "missing_declared_fields": (
                missing_declared_non_negative_fields
            ),

            "negative_values": (
                impossible_negative_values
            ),
        },

        "temporal_ordering_assessment": (
            temporal_assessment
        ),

        "feature_datatypes": {
            str(column): str(dtype)
            for column, dtype
            in df_train.dtypes.items()
        },

        "blocking_findings": (
            blocking_findings
        ),

        "review_findings": (
            review_findings
        ),

        "informational_findings": (
            informational_findings
        ),
    }

    # -----------------------------------------------------------------
    # 18. Write outputs
    # -----------------------------------------------------------------

    AUDIT_JSON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with AUDIT_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            audit_results,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    print(
        f"[+] Audit JSON: {AUDIT_JSON_PATH}"
    )

    print(
        f"[+] Class distribution: {CLASS_DIST_PATH}"
    )

    print(
        f"[+] Final audit status: {status}"
    )

    # -----------------------------------------------------------------
    # 19. Human-readable diagnostic summary
    # -----------------------------------------------------------------

    print(
        "\n=== FEATURE REPRESENTATION SUMMARY ==="
    )

    print(
        "[+] Source candidate feature count:",
        len(
            source_candidate_features
        ),
    )

    print(
        "[+] Effective candidate feature count:",
        len(
            effective_candidate_features
        ),
    )

    print(
        "[+] Present policy exclusions:",
        present_policy_excluded_features,
    )

    print(
        "[+] Absent policy exclusions:",
        absent_policy_excluded_features,
    )

    print(
        "\n=== CROSS-SPLIT OVERLAP SUMMARY ==="
    )

    print(
        "[+] Source-candidate shared fingerprints:",
        source_overlap.get(
            "screening_shared_unique_hashes",
            "NOT_EVALUATED",
        ),
    )

    print(
        "[+] Source-candidate confirmed exact vectors:",
        source_overlap.get(
            "confirmed_exact_unique_feature_vectors",
            "NOT_EVALUATED",
        ),
    )

    print(
        "[+] Source exact-overlap train rows:",
        source_overlap.get(
            "train_rows_involved_in_exact_overlap",
            "NOT_EVALUATED",
        ),
    )

    print(
        "[+] Source exact-overlap test rows:",
        source_overlap.get(
            "test_rows_involved_in_exact_overlap",
            "NOT_EVALUATED",
        ),
    )

    print(
        "[+] Source overlap test-row percentage:",
        source_overlap.get(
            "test_overlap_row_percentage",
            "NOT_EVALUATED",
        ),
    )

    print(
        "[+] Strict-test excluded rows:",
        source_overlap.get(
            "strict_test_excluded_rows",
            "NOT_EVALUATED",
        ),
    )

    print(
        "[+] Strict-test eligible rows:",
        source_overlap.get(
            "strict_test_eligible_rows",
            "NOT_EVALUATED",
        ),
    )

    print(
        "[+] Strict-test eligible percentage:",
        source_overlap.get(
            "strict_test_eligible_percentage",
            "NOT_EVALUATED",
        ),
    )

    print(
        "[+] Source exact vectors with different label sets:",
        source_overlap.get(
            "exact_vectors_with_different_label_sets",
            "NOT_EVALUATED",
        ),
    )

    print(
        "[+] Effective-candidate shared fingerprints:",
        effective_overlap.get(
            "screening_shared_unique_hashes",
            "NOT_EVALUATED",
        ),
    )

    print(
        "[+] Effective-candidate confirmed exact vectors:",
        effective_overlap.get(
            "confirmed_exact_unique_feature_vectors",
            "NOT_EVALUATED",
        ),
    )

    print(
        "[+] Effective fingerprint matches without exact match:",
        effective_overlap.get(
            "shared_hashes_without_exact_match",
            "NOT_EVALUATED",
        ),
    )

    print(
        "\n=== OVERLAP POLICY SUMMARY ==="
    )

    print(
        "[+] Resolution status:",
        overlap_resolution_status,
    )

    print(
        "[+] Strict primary policy valid:",
        strict_policy_valid,
    )

    if informational_findings:
        print(
            "\n=== INFORMATIONAL FINDINGS ==="
        )

        for index, finding in enumerate(
            informational_findings,
            start=1,
        ):
            print(
                f"[INFO {index}] {finding}"
            )

    if blocking_findings:
        print(
            "\n=== BLOCKING FINDINGS ==="
        )

        for index, finding in enumerate(
            blocking_findings,
            start=1,
        ):
            print(
                f"[BLOCK {index}] {finding}"
            )

    if review_findings:
        print(
            "\n=== REVIEW-REQUIRED FINDINGS ==="
        )

        for index, finding in enumerate(
            review_findings,
            start=1,
        ):
            print(
                f"[REVIEW {index}] {finding}"
            )

    # -----------------------------------------------------------------
    # 20. Enforce phase gate
    # -----------------------------------------------------------------

    if status == "FAILED":
        raise RuntimeError(
            "Dataset audit FAILED. "
            "Resolve and document blocking findings before "
            "continuing Phase 1. "
            f"See {AUDIT_JSON_PATH}."
        )

    if status == "REVIEW_REQUIRED":
        raise RuntimeError(
            "Dataset audit completed with REVIEW_REQUIRED "
            "findings. Resolve and document them before "
            "freezing Phase 1. "
            f"See {AUDIT_JSON_PATH}."
        )

    print(
        "\n=== Dataset audit PASSED. "
        "Next step: freeze remaining split-generation parameters "
        "and generate immutable derived split artifacts. "
        "Do NOT begin model training before the split and "
        "preprocessing policies are fully frozen. ==="
    )


if __name__ == "__main__":
    run_audit()