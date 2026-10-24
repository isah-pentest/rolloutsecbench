#!/usr/bin/env python3
"""
RolloutSecBench — Validation-only V1/V2 model-revision acceptance gate.

Purpose
-------
Mechanically evaluate the prospectively frozen V2 noninferiority criteria
using ONLY:
    - frozen model revision protocol
    - V1 model manifest
    - V2 model manifest
    - frozen model/prediction artifacts referenced by those manifests

This script:
    * verifies protocol integrity
    * verifies both manifests bind to the same frozen protocol
    * verifies V1/V2 model and validation-prediction artifact hashes
    * verifies the same frozen train/validation/preprocessing chain
    * verifies validation-only core checks
    * reads acceptance tolerances from the frozen protocol
    * computes V2 >= V1 - tolerance
    * DOES NOT read either test dataset
    * DOES NOT compute disagreement/MRD
    * refuses to overwrite an existing acceptance artifact
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_PATH = (
    REPO_ROOT / "data/manifests/model_revision_protocol_v1.0.json"
)
PROTOCOL_SHA_PATH = (
    REPO_ROOT / "data/manifests/model_revision_protocol_v1.0.sha256"
)

V1_MANIFEST_PATH = REPO_ROOT / "models/v1/model_manifest.json"
V2_MANIFEST_PATH = REPO_ROOT / "models/v2/model_manifest.json"

OUTPUT_PATH = (
    REPO_ROOT / "results/derived/model_revision_acceptance.json"
)

EXPECTED_PROTOCOL_STATUS = "FROZEN_PRETRAINING"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def repo_path(relative_path: str) -> Path:
    p = (REPO_ROOT / relative_path).resolve()

    try:
        p.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"Artifact path escapes repository: {relative_path}"
        ) from exc

    return p


def verify_file_hash(
    path: Path,
    expected_sha256: str,
    label: str,
) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")

    observed = sha256_file(path)

    if observed != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch.\n"
            f"Expected: {expected_sha256}\n"
            f"Observed: {observed}"
        )

    print(f"[+] {label:<32} PASS")
    return observed


def require_true(value: Any, label: str) -> None:
    if value is not True:
        raise RuntimeError(
            f"Required condition is not true: {label} = {value!r}"
        )
    print(f"[+] {label:<32} PASS")


def require_false(value: Any, label: str) -> None:
    if value is not False:
        raise RuntimeError(
            f"Required condition is not false: {label} = {value!r}"
        )
    print(f"[+] {label:<32} PASS")


def finite_metric(value: Any, label: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{label} is not numeric: {value!r}"
        ) from exc

    if not math.isfinite(x):
        raise RuntimeError(f"{label} is not finite: {x}")

    return x


def main() -> None:
    print("=== RolloutSecBench Model Revision Acceptance Gate ===")

    if OUTPUT_PATH.exists():
        raise RuntimeError(
            "Acceptance artifact already exists; refusing to overwrite:\n"
            f"{OUTPUT_PATH}"
        )

    # ------------------------------------------------------------
    # 1. Frozen protocol integrity
    # ------------------------------------------------------------

    protocol = load_json(PROTOCOL_PATH)
    protocol_sha = sha256_file(PROTOCOL_PATH)

    if not PROTOCOL_SHA_PATH.is_file():
        raise FileNotFoundError(
            f"Protocol SHA sidecar missing: {PROTOCOL_SHA_PATH}"
        )

    expected_protocol_sha = (
        PROTOCOL_SHA_PATH.read_text(encoding="utf-8")
        .strip()
        .split()[0]
    )

    if protocol_sha != expected_protocol_sha:
        raise RuntimeError(
            "Frozen protocol integrity check failed.\n"
            f"Expected: {expected_protocol_sha}\n"
            f"Observed: {protocol_sha}"
        )

    if protocol.get("protocol_status") != EXPECTED_PROTOCOL_STATUS:
        raise RuntimeError(
            "Unexpected frozen protocol status: "
            f"{protocol.get('protocol_status')!r}"
        )

    print(f"[+] Frozen protocol integrity       PASS")
    print(f"[+] Protocol SHA-256: {protocol_sha}")

    # ------------------------------------------------------------
    # 2. Load existing V1 and V2 manifests
    # ------------------------------------------------------------

    v1 = load_json(V1_MANIFEST_PATH)
    v2 = load_json(V2_MANIFEST_PATH)

    v1_manifest_sha = sha256_file(V1_MANIFEST_PATH)
    v2_manifest_sha = sha256_file(V2_MANIFEST_PATH)

    print(f"[+] V1 manifest loaded              PASS")
    print(f"[+] V2 manifest loaded              PASS")

    # ------------------------------------------------------------
    # 3. Bind both revisions to same frozen protocol
    # ------------------------------------------------------------

    v1_protocol_sha = v1["protocol"]["sha256"]
    v2_protocol_sha = v2["protocol"]["sha256"]

    if not (
        v1_protocol_sha
        == v2_protocol_sha
        == protocol_sha
    ):
        raise RuntimeError(
            "V1/V2 protocol binding mismatch.\n"
            f"Frozen protocol: {protocol_sha}\n"
            f"V1 manifest:     {v1_protocol_sha}\n"
            f"V2 manifest:     {v2_protocol_sha}"
        )

    print(f"[+] V1/V2 protocol binding          PASS")

    # ------------------------------------------------------------
    # 4. Verify model + validation prediction artifacts
    # ------------------------------------------------------------

    v1_model = v1["artifacts"]["model"]
    v1_predictions = v1["artifacts"]["validation_predictions"]

    v2_model = v2["artifacts"]["model"]
    v2_predictions = v2["artifacts"]["validation_predictions"]

    verify_file_hash(
        repo_path(v1_model["path"]),
        v1_model["sha256"],
        "V1 model artifact",
    )

    verify_file_hash(
        repo_path(v1_predictions["path"]),
        v1_predictions["sha256"],
        "V1 validation predictions",
    )

    verify_file_hash(
        repo_path(v2_model["path"]),
        v2_model["sha256"],
        "V2 model artifact",
    )

    verify_file_hash(
        repo_path(v2_predictions["path"]),
        v2_predictions["sha256"],
        "V2 validation predictions",
    )

    # ------------------------------------------------------------
    # 5. Verify shared frozen experimental inputs
    # ------------------------------------------------------------

    v1_train_sha = v1["training_data"]["sha256"]
    v2_train_sha = v2["frozen_inputs"]["final_train_sha256"]

    if v1_train_sha != v2_train_sha:
        raise RuntimeError(
            "V1/V2 final_train SHA-256 mismatch."
        )

    print(f"[+] Shared final_train               PASS")

    v1_validation_sha = v1["validation_data"]["sha256"]
    v2_validation_sha = v2["frozen_inputs"]["validation_sha256"]

    if v1_validation_sha != v2_validation_sha:
        raise RuntimeError(
            "V1/V2 validation SHA-256 mismatch."
        )

    print(f"[+] Shared validation                PASS")

    v1_preprocessor_sha = v1["preprocessing"]["sha256"]
    v2_preprocessor_sha = v2["frozen_inputs"]["preprocessor_sha256"]

    if v1_preprocessor_sha != v2_preprocessor_sha:
        raise RuntimeError(
            "V1/V2 preprocessor SHA-256 mismatch."
        )

    print(f"[+] Shared frozen preprocessor       PASS")

    v1_features_sha = v1["preprocessing"]["feature_names_sha256"]
    v2_features_sha = v2["frozen_inputs"]["feature_names_sha256"]

    if v1_features_sha != v2_features_sha:
        raise RuntimeError(
            "V1/V2 feature_names SHA-256 mismatch."
        )

    print(f"[+] Shared feature representation    PASS")

    if v1["preprocessing"]["raw_feature_count"] != 42:
        raise RuntimeError("V1 raw feature count is not 42.")

    if v2["training"]["raw_feature_count"] != 42:
        raise RuntimeError("V2 raw feature count is not 42.")

    if v1["preprocessing"]["transformed_feature_count"] != 192:
        raise RuntimeError(
            "V1 transformed feature count is not 192."
        )

    if v2["training"]["transformed_feature_count"] != 192:
        raise RuntimeError(
            "V2 transformed feature count is not 192."
        )

    print(f"[+] Shared 42 -> 192 representation  PASS")

    # ------------------------------------------------------------
    # 6. Verify thresholds remain fixed and equal
    # ------------------------------------------------------------

    v1_threshold = finite_metric(
        v1["training"]["threshold"],
        "V1 threshold",
    )

    v2_threshold = finite_metric(
        v2["decision_policy"]["decision_threshold"],
        "V2 threshold",
    )

    if v1_threshold != 0.5 or v2_threshold != 0.5:
        raise RuntimeError(
            "Expected frozen V1/V2 threshold = 0.5."
        )

    if v1_threshold != v2_threshold:
        raise RuntimeError(
            "V1/V2 decision thresholds differ."
        )

    print(f"[+] Shared frozen threshold 0.5      PASS")

    # ------------------------------------------------------------
    # 7. Verify validation-only/core checks
    # ------------------------------------------------------------

    require_true(
        v1["acceptance_status"]["v1_core_checks_passed"],
        "V1 core checks",
    )

    require_true(
        v1["validation"]["probabilities_finite"],
        "V1 probabilities finite",
    )

    require_true(
        v1["validation"]["probabilities_within_unit_interval"],
        "V1 probabilities in [0,1]",
    )

    require_true(
        v1["serialization_validation"]["reload_equivalent"],
        "V1 reload equivalence",
    )

    require_false(
        v1["test_data_access"][
            "official_test_full_read_by_training_script"
        ],
        "V1 official test access",
    )

    require_false(
        v1["test_data_access"][
            "strict_test_nonoverlap_read_by_training_script"
        ],
        "V1 strict test access",
    )

    require_true(
        v2["acceptance_status"]["training_completed"],
        "V2 training completed",
    )

    require_true(
        v2["acceptance_status"][
            "validation_probabilities_finite"
        ],
        "V2 probabilities finite",
    )

    require_true(
        v2["acceptance_status"][
            "validation_probabilities_in_unit_interval"
        ],
        "V2 probabilities in [0,1]",
    )

    require_true(
        v2["acceptance_status"]["required_metrics_recorded"],
        "V2 metrics recorded",
    )

    require_true(
        v2["serialization_check"]["reload_equivalent"],
        "V2 reload equivalence",
    )

    require_true(
        v2["deterministic_retraining_check"]["passed"],
        "V2 deterministic reproduction",
    )

    require_false(
        v2["data_access"]["test_data_accessed"],
        "V2 test access",
    )

    require_false(
        v2["data_access"]["official_test_full_accessed"],
        "V2 official test access",
    )

    require_false(
        v2["data_access"]["strict_test_nonoverlap_accessed"],
        "V2 strict test access",
    )

    # ------------------------------------------------------------
    # 8. Read acceptance policy FROM FROZEN PROTOCOL
    # ------------------------------------------------------------

    policy = protocol["model_acceptance_policy"]["v2"]

    if policy.get("required") is not True:
        raise RuntimeError(
            "Frozen protocol does not mark V2 acceptance as required."
        )

    requirements = policy[
        "relative_noninferiority_requirements"
    ]

    f1_tolerance = finite_metric(
        requirements["f1"]["absolute_tolerance"],
        "F1 tolerance",
    )

    auroc_tolerance = finite_metric(
        requirements["auroc"]["absolute_tolerance"],
        "AUROC tolerance",
    )

    if requirements["f1"]["rule"] != "V2 >= V1 - tolerance":
        raise RuntimeError(
            "Unexpected frozen F1 acceptance rule."
        )

    if requirements["auroc"]["rule"] != "V2 >= V1 - tolerance":
        raise RuntimeError(
            "Unexpected frozen AUROC acceptance rule."
        )

    # ------------------------------------------------------------
    # 9. Extract metrics from EXISTING manifest schemas
    # ------------------------------------------------------------

    v1_metrics = v1["validation"]["metrics"]
    v2_metrics = v2["validation_metrics"]

    required_metrics = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "fpr",
        "tn",
        "fp",
        "fn",
        "tp",
    )

    for metric in required_metrics:
        if metric not in v1_metrics:
            raise RuntimeError(
                f"V1 required metric missing: {metric}"
            )
        if metric not in v2_metrics:
            raise RuntimeError(
                f"V2 required metric missing: {metric}"
            )

    v1_f1 = finite_metric(v1_metrics["f1"], "V1 F1")
    v2_f1 = finite_metric(v2_metrics["f1"], "V2 F1")

    v1_auroc = finite_metric(
        v1_metrics["auroc"],
        "V1 AUROC",
    )
    v2_auroc = finite_metric(
        v2_metrics["auroc"],
        "V2 AUROC",
    )

    f1_minimum = v1_f1 - f1_tolerance
    auroc_minimum = v1_auroc - auroc_tolerance

    f1_pass = v2_f1 >= f1_minimum
    auroc_pass = v2_auroc >= auroc_minimum

    core_checks_pass = True
    gate_pass = (
        core_checks_pass
        and f1_pass
        and auroc_pass
    )

    print("\n=== PROSPECTIVELY FROZEN NONINFERIORITY GATE ===")

    print(
        f"V1 F1:               {v1_f1:.10f}"
    )
    print(
        f"F1 tolerance:         {f1_tolerance:.10f}"
    )
    print(
        f"Required V2 F1 >=     {f1_minimum:.10f}"
    )
    print(
        f"Observed V2 F1:       {v2_f1:.10f}"
    )
    print(
        f"F1 criterion:         {'PASS' if f1_pass else 'FAIL'}"
    )

    print()

    print(
        f"V1 AUROC:             {v1_auroc:.10f}"
    )
    print(
        f"AUROC tolerance:      {auroc_tolerance:.10f}"
    )
    print(
        f"Required V2 AUROC >=  {auroc_minimum:.10f}"
    )
    print(
        f"Observed V2 AUROC:    {v2_auroc:.10f}"
    )
    print(
        f"AUROC criterion:      {'PASS' if auroc_pass else 'FAIL'}"
    )

    # ------------------------------------------------------------
    # 10. Write immutable acceptance artifact
    # ------------------------------------------------------------

    output = {
        "schema_version": "1.0",
        "artifact": "RolloutSecBench",
        "gate": "model_revision_validation_acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),

        "evidence_scope": {
            "validation_only": True,
            "test_data_read_by_this_script": False,
            "malicious_disagreement_computed": False,
            "rollout_analysis_performed": False,
        },

        "frozen_protocol": {
            "path": str(
                PROTOCOL_PATH.relative_to(REPO_ROOT)
            ),
            "sha256": protocol_sha,
            "status": protocol["protocol_status"],
        },

        "input_manifests": {
            "v1": {
                "path": str(
                    V1_MANIFEST_PATH.relative_to(REPO_ROOT)
                ),
                "sha256": v1_manifest_sha,
            },
            "v2": {
                "path": str(
                    V2_MANIFEST_PATH.relative_to(REPO_ROOT)
                ),
                "sha256": v2_manifest_sha,
            },
        },

        "shared_frozen_inputs": {
            "final_train_sha256": v1_train_sha,
            "validation_sha256": v1_validation_sha,
            "preprocessor_sha256": v1_preprocessor_sha,
            "feature_names_sha256": v1_features_sha,
            "raw_feature_count": 42,
            "transformed_feature_count": 192,
            "decision_threshold": 0.5,
        },

        "validation_metrics": {
            "v1": v1_metrics,
            "v2": v2_metrics,
        },

        "prospective_acceptance_policy": {
            "source": (
                "model_revision_protocol_v1.0.json:"
                "model_acceptance_policy.v2"
            ),
            "f1": {
                "rule": requirements["f1"]["rule"],
                "absolute_tolerance": f1_tolerance,
                "v1": v1_f1,
                "minimum_required_v2": f1_minimum,
                "observed_v2": v2_f1,
                "delta_v2_minus_v1": v2_f1 - v1_f1,
                "passed": f1_pass,
            },
            "auroc": {
                "rule": requirements["auroc"]["rule"],
                "absolute_tolerance": auroc_tolerance,
                "v1": v1_auroc,
                "minimum_required_v2": auroc_minimum,
                "observed_v2": v2_auroc,
                "delta_v2_minus_v1": v2_auroc - v1_auroc,
                "passed": auroc_pass,
            },
        },

        "gate_result": {
            "v1_core_checks_passed": True,
            "v2_core_checks_passed": True,
            "f1_noninferiority_passed": f1_pass,
            "auroc_noninferiority_passed": auroc_pass,
            "all_required_conditions_passed": gate_pass,
            "status": "PASS" if gate_pass else "FAIL",
        },

        "authorization": {
            "static_test_evaluation_allowed": gate_pass,
            "malicious_disagreement_analysis_allowed": gate_pass,
            "rollout_analysis_allowed": False,
            "note": (
                "Passing this gate establishes independent "
                "validation acceptability of V1 and V2. "
                "It does not itself constitute test-set, "
                "malicious-disagreement, or rollout evidence."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation: refuse overwrite even under race/re-run.
    with OUTPUT_PATH.open("x", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    output_sha = sha256_file(OUTPUT_PATH)

    print("\n=== ACCEPTANCE ARTIFACT ===")
    print(
        f"[+] path:   {OUTPUT_PATH.relative_to(REPO_ROOT)}"
    )
    print(
        f"[+] SHA256: {output_sha}"
    )

    print("\n=== GATE 1 RESULT ===")
    print(f"STATUS: {'PASS' if gate_pass else 'FAIL'}")

    if not gate_pass:
        print(
            "STOP: model pair did not satisfy the "
            "prospectively frozen validation gate."
        )
        sys.exit(2)

    print(
        "V1 and V2 satisfy the prospectively frozen "
        "validation-only acceptance gate."
    )
    print(
        "Static test evaluation may now proceed under the "
        "predefined primary/secondary test policy."
    )


if __name__ == "__main__":
    main()
