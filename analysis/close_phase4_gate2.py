#!/usr/bin/env python3
"""
RolloutSecBench — Phase 4 statistical closure and Gate 2 decision.

Purpose
-------
Close Phase 4 using ONLY already-frozen Phase-4 evidence.

Computes:
- MRD point estimate cross-check
- exact 95% Clopper-Pearson CI for MRD
- exact two-sided McNemar/binomial test using discordant cells
- directional disagreement proportion and exact CI
- paired malicious detection-rate difference
- immutable Gate-2 decision under the prospectively defined criterion MRD > 0

Does NOT:
- read raw train/validation/test data
- load V1/V2 models
- run inference
- recompute predictions
- tune anything
- introduce a practical MRD threshold
- apply Holm correction prematurely
- perform rollout analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from scipy.stats import binomtest


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Required JSON artifact missing: {path}")

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")

    return obj


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Required artifact missing: {path}")

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


def serialize_json(obj: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_bytes_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("xb") as f:
        f.write(data)


def repo_path(repo_root: Path, relative_path: str) -> Path:
    p = Path(relative_path)

    if p.is_absolute():
        raise RuntimeError(
            f"Manifest path must be repository-relative: {relative_path}"
        )

    resolved = (repo_root / p).resolve()

    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"Manifest path escapes repository root: {relative_path}"
        ) from exc

    return resolved


def resolve_repo_root(manifest_path: Path) -> Path:
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)

    refs: list[str] = []

    upstream = manifest.get("upstream_evidence", {})

    for key in (
        "static_evaluation_manifest",
        "gate1_acceptance",
        "paired_predictions",
        "static_metrics",
    ):
        entry = upstream.get(key)
        if isinstance(entry, dict) and entry.get("path"):
            refs.append(entry["path"])

    outputs = manifest.get("outputs", {})

    for entry in outputs.values():
        if isinstance(entry, dict) and entry.get("path"):
            refs.append(entry["path"])

    if not refs:
        raise RuntimeError(
            "Phase-4 manifest contains no usable repository-relative references."
        )

    candidates = [
        manifest_path.parent,
        *manifest_path.parents,
    ]

    for candidate in candidates:
        if all((candidate / ref).exists() for ref in refs):
            return candidate.resolve()

    raise RuntimeError(
        "Could not infer repository root from Phase-4 manifest."
    )


# ---------------------------------------------------------------------
# Phase-4 integrity validation
# ---------------------------------------------------------------------

def validate_phase4_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("artifact") != "RolloutSecBench":
        raise RuntimeError("Unexpected artifact identity.")

    if manifest.get("phase") != "Malicious Revision Disagreement Analysis":
        raise RuntimeError(
            "Input is not a Phase-4 malicious-disagreement manifest."
        )

    scope = manifest.get("analysis_scope", {})

    if scope.get("malicious_disagreement_computed") is not True:
        raise RuntimeError(
            "Phase-4 manifest does not report disagreement computation."
        )

    if scope.get("mrd_computed") is not True:
        raise RuntimeError(
            "Phase-4 manifest does not report MRD computation."
        )

    if scope.get("practical_mrd_threshold_applied") is not False:
        raise RuntimeError(
            "A practical MRD threshold was unexpectedly applied."
        )

    if scope.get("rollout_analysis_performed") is not False:
        raise RuntimeError(
            "Phase-4 manifest unexpectedly reports rollout analysis."
        )

    print("[PASS] Phase-4 analysis scope verified.")


def validate_result(result: dict[str, Any]) -> None:
    if result.get("artifact") != "RolloutSecBench":
        raise RuntimeError("Unexpected disagreement-result artifact identity.")

    if result.get("phase") != "Malicious Revision Disagreement Analysis":
        raise RuntimeError(
            "Unexpected phase in disagreement-result artifact."
        )

    gate = result.get("gate2_evidence", {})

    if gate.get("practical_meaningfulness_threshold_applied") is not False:
        raise RuntimeError(
            "Disagreement result indicates a practical threshold was applied."
        )

    constraints = result.get("scientific_constraints", {})

    expected_false = (
        "strict_test_raw_read",
        "official_test_full_read",
        "validation_read",
        "final_train_read",
        "model_loading_performed",
        "model_inference_performed",
        "model_fitting_performed",
        "preprocessor_loaded",
        "preprocessor_refit",
        "threshold_tuning_performed",
        "hyperparameter_tuning_performed",
        "rollout_analysis_performed",
        "only_disagreement_rows_reported",
    )

    for field in expected_false:
        if constraints.get(field) is not False:
            raise RuntimeError(
                f"Unexpected Phase-4 scientific constraint: {field}"
            )

    print("[PASS] Phase-4 disagreement-result constraints verified.")


def extract_and_crosscheck_counts(
    result: dict[str, Any],
) -> dict[str, int | float]:

    n = int(result["malicious_population"]["count"])

    paired = result["paired_outcomes"]

    both_detect = int(paired["both_detect"])
    b = int(paired["v1_detects_v2_misses"])
    c = int(paired["v2_detects_v1_misses"])
    both_miss = int(paired["both_miss"])

    partition_sum = (
        both_detect
        + b
        + c
        + both_miss
    )

    if partition_sum != n:
        raise RuntimeError(
            f"Malicious paired partition mismatch: "
            f"{partition_sum} != {n}"
        )

    disagreement = b + c

    d_obj = result["malicious_revision_disagreement"]

    recorded_disagreement = int(d_obj["disagreement_count"])
    recorded_mrd = float(d_obj["mrd"])

    if recorded_disagreement != disagreement:
        raise RuntimeError(
            "Recorded disagreement count does not equal "
            "the two discordant cells."
        )

    recomputed_mrd = disagreement / n

    if not math.isclose(
        recomputed_mrd,
        recorded_mrd,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError(
            f"Recorded MRD mismatch: "
            f"recorded={recorded_mrd}, "
            f"recomputed={recomputed_mrd}"
        )

    expected_nonzero = disagreement > 0

    if bool(d_obj["formal_nonzero_condition"]) != expected_nonzero:
        raise RuntimeError(
            "Recorded formal_nonzero_condition is inconsistent "
            "with disagreement count."
        )

    model_consistency = result["model_level_consistency"]

    v1_misses = c + both_miss
    v2_misses = b + both_miss

    if int(model_consistency["v1"]["misses"]) != v1_misses:
        raise RuntimeError("V1 miss-count consistency check failed.")

    if int(model_consistency["v2"]["misses"]) != v2_misses:
        raise RuntimeError("V2 miss-count consistency check failed.")

    print("[PASS] Phase-4 paired-count arithmetic verified.")

    return {
        "n": n,
        "both_detect": both_detect,
        "b_v1_detects_v2_misses": b,
        "c_v2_detects_v1_misses": c,
        "both_miss": both_miss,
        "disagreement": disagreement,
        "mrd": recomputed_mrd,
        "v1_misses": v1_misses,
        "v2_misses": v2_misses,
    }


# ---------------------------------------------------------------------
# Statistical closure
# ---------------------------------------------------------------------

def exact_binomial_ci(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:

    if trials <= 0:
        raise RuntimeError("Binomial CI requires trials > 0.")

    result = binomtest(successes, trials)

    ci = result.proportion_ci(
        confidence_level=confidence_level,
        method="exact",
    )

    return float(ci.low), float(ci.high)


def compute_statistics(
    counts: dict[str, int | float],
) -> dict[str, Any]:

    n = int(counts["n"])
    d = int(counts["disagreement"])
    b = int(counts["b_v1_detects_v2_misses"])
    c = int(counts["c_v2_detects_v1_misses"])

    mrd = float(counts["mrd"])

    mrd_ci_low, mrd_ci_high = exact_binomial_ci(
        d,
        n,
        confidence_level=0.95,
    )

    # Exact conditional McNemar test:
    # Under equality of paired marginal detection rates, conditional on
    # a discordance, either direction has probability 0.5.
    discordant = b + c

    if discordant == 0:
        mcnemar_p = 1.0
        directional_share = None
        directional_ci_low = None
        directional_ci_high = None
    else:
        mc = binomtest(
            b,
            discordant,
            p=0.5,
            alternative="two-sided",
        )

        mcnemar_p = float(mc.pvalue)
        directional_share = b / discordant

        directional_ci = mc.proportion_ci(
            confidence_level=0.95,
            method="exact",
        )

        directional_ci_low = float(directional_ci.low)
        directional_ci_high = float(directional_ci.high)

    # Difference in malicious detection rates:
    #
    # detection_rate_v1 - detection_rate_v2
    # = (b - c) / N
    #
    # because concordant outcomes cancel.
    paired_detection_rate_difference = (b - c) / n

    return {
        "mrd": {
            "estimate": mrd,
            "numerator_disagreement": d,
            "denominator_malicious": n,
            "confidence_level": 0.95,
            "confidence_interval_method": "clopper_pearson_exact",
            "ci_low": mrd_ci_low,
            "ci_high": mrd_ci_high,
        },
        "directional_disagreement": {
            "discordant_total": discordant,
            "v1_detects_v2_misses": b,
            "v2_detects_v1_misses": c,
            "v1_detects_v2_misses_share_of_discordance": directional_share,
            "share_confidence_level": 0.95,
            "share_confidence_interval_method": "clopper_pearson_exact",
            "share_ci_low": directional_ci_low,
            "share_ci_high": directional_ci_high,
        },
        "paired_effect_characterization": {
            "malicious_detection_rate_difference_v1_minus_v2": (
                paired_detection_rate_difference
            ),
            "malicious_detection_rate_difference_percentage_points": (
                100.0 * paired_detection_rate_difference
            ),
            "interpretation": (
                "Positive values indicate higher malicious detection "
                "rate for V1; negative values indicate higher malicious "
                "detection rate for V2."
            ),
        },
        "mcnemar": {
            "implementation": "exact_conditional_binomial_two_sided",
            "null_hypothesis": (
                "Conditional on a discordant malicious prediction, "
                "the two disagreement directions are equally likely "
                "(p = 0.5), equivalent to equal paired marginal "
                "malicious-detection rates."
            ),
            "discordant_total": discordant,
            "b_v1_detects_v2_misses": b,
            "c_v2_detects_v1_misses": c,
            "raw_two_sided_p_value": mcnemar_p,
            "holm_bonferroni_adjusted_p_value": None,
            "multiple_testing_status": (
                "DEFERRED_UNTIL_H2_H5_RESULTS_EXIST"
            ),
            "important_scope_note": (
                "McNemar evaluates directional asymmetry / paired "
                "marginal detection-rate difference. It is not the "
                "Gate-2 test of whether MRD is nonzero."
            ),
        },
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase4-manifest",
        required=True,
        type=Path,
        help=(
            "Frozen malicious_disagreement_manifest.json from Phase 4."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    phase4_manifest_path = args.phase4_manifest.resolve()

    phase4_manifest = load_json(phase4_manifest_path)

    validate_phase4_manifest(phase4_manifest)

    repo_root = resolve_repo_root(phase4_manifest_path)

    print("============================================================")
    print("RolloutSecBench — PHASE 4 STATISTICAL CLOSURE")
    print("============================================================")
    print(f"[INFO] Repository root: {repo_root}")

    phase4_manifest_hash = sha256_file(phase4_manifest_path)

    print(
        "[INFO] Phase-4 manifest SHA-256: "
        f"{phase4_manifest_hash}"
    )

    # -------------------------------------------------------------
    # Verify upstream evidence chain carried into Phase 4.
    # -------------------------------------------------------------

    upstream = phase4_manifest["upstream_evidence"]

    static_entry = upstream["static_evaluation_manifest"]
    gate1_entry = upstream["gate1_acceptance"]
    paired_entry = upstream["paired_predictions"]
    metrics_entry = upstream["static_metrics"]

    static_path = repo_path(repo_root, static_entry["path"])
    gate1_path = repo_path(repo_root, gate1_entry["path"])
    paired_path = repo_path(repo_root, paired_entry["path"])
    metrics_path = repo_path(repo_root, metrics_entry["path"])

    static_hash = require_hash(
        static_path,
        static_entry["sha256"],
        "Static evaluation manifest",
    )

    gate1_hash = require_hash(
        gate1_path,
        gate1_entry["sha256"],
        "Gate 1 acceptance",
    )

    require_hash(
        paired_path,
        paired_entry["sha256"],
        "Frozen paired predictions",
    )

    require_hash(
        metrics_path,
        metrics_entry["sha256"],
        "Frozen static metrics",
    )

    if gate1_entry.get("status") != "PASS":
        raise RuntimeError(
            "Phase-4 manifest does not bind to Gate 1 PASS."
        )

    # -------------------------------------------------------------
    # Verify every Phase-4 output listed in its manifest.
    # -------------------------------------------------------------

    outputs = phase4_manifest["outputs"]

    disagreement_entry = outputs["malicious_disagreement"]

    disagreement_path = repo_path(
        repo_root,
        disagreement_entry["path"],
    )

    disagreement_hash = require_hash(
        disagreement_path,
        disagreement_entry["sha256"],
        "Malicious disagreement result",
    )

    malicious_entry = outputs["full_malicious_predictions"]

    malicious_path = repo_path(
        repo_root,
        malicious_entry["path"],
    )

    malicious_hash = require_hash(
        malicious_path,
        malicious_entry["sha256"],
        "Full malicious prediction artifact",
    )

    family_entry = outputs.get("attack_family_breakdown")

    family_hash = None
    family_path = None

    if isinstance(family_entry, dict):
        family_path = repo_path(
            repo_root,
            family_entry["path"],
        )

        family_hash = require_hash(
            family_path,
            family_entry["sha256"],
            "Attack-family disagreement artifact",
        )

    disagreement_result = load_json(disagreement_path)

    validate_result(disagreement_result)

    counts = extract_and_crosscheck_counts(
        disagreement_result
    )

    statistics = compute_statistics(counts)

    # -------------------------------------------------------------
    # Gate 2 follows ONLY the prospectively frozen operational rule.
    # -------------------------------------------------------------

    gate2_pass = float(counts["mrd"]) > 0.0

    gate2_status = "PASS" if gate2_pass else "FAIL"

    next_phase = (
        "PHASE_5_SERVING_VALIDATION"
        if gate2_pass
        else "STOP_AND_REASSESS_SCIENTIFIC_PREMISE"
    )

    statistical_summary = {
        "artifact": "RolloutSecBench",
        "schema_version": "1.0",
        "phase": "Phase 4 Statistical Closure",
        "analysis_population": {
            "definition": "malicious samples: label == 1",
            "count": int(counts["n"]),
        },
        "paired_outcome_counts": {
            "both_detect": int(counts["both_detect"]),
            "v1_detects_v2_misses": int(
                counts["b_v1_detects_v2_misses"]
            ),
            "v2_detects_v1_misses": int(
                counts["c_v2_detects_v1_misses"]
            ),
            "both_miss": int(counts["both_miss"]),
            "malicious_disagreement": int(
                counts["disagreement"]
            ),
        },
        "statistics": statistics,
        "multiple_testing": {
            "predeclared_family": "H1_H2_H3_H4_H5",
            "holm_bonferroni_required": True,
            "holm_bonferroni_applied_now": False,
            "reason": (
                "H2-H5 have not yet been evaluated. "
                "No primary-hypothesis significance claim is made "
                "from the unadjusted McNemar p-value."
            ),
        },
        "method_freeze_note": {
            "predeclared_requirements": [
                "disagreement counts",
                "confidence intervals",
                "McNemar test",
            ],
            "implementation_fixed_for_phase4_closure": {
                "mrd_ci": "95% Clopper-Pearson exact binomial interval",
                "mcnemar": "exact conditional two-sided binomial test",
            },
            "posthoc_method_switching_allowed": False,
        },
        "scientific_scope": {
            "raw_dataset_read": False,
            "paired_predictions_recomputed": False,
            "model_inference_performed": False,
            "rollout_analysis_performed": False,
            "practical_mrd_threshold_applied": False,
        },
    }

    gate2_artifact = {
        "artifact": "RolloutSecBench",
        "schema_version": "1.0",
        "gate": "Gate 2",
        "name": "Malicious Revision Disagreement",
        "prospective_operational_criterion": "MRD > 0",
        "observed": {
            "malicious_count": int(counts["n"]),
            "disagreement_count": int(counts["disagreement"]),
            "mrd": float(counts["mrd"]),
            "v1_detects_v2_misses": int(
                counts["b_v1_detects_v2_misses"]
            ),
            "v2_detects_v1_misses": int(
                counts["c_v2_detects_v1_misses"]
            ),
            "mrd_ci_95_exact": {
                "low": statistics["mrd"]["ci_low"],
                "high": statistics["mrd"]["ci_high"],
            },
        },
        "decision": {
            "status": gate2_status,
            "formal_criterion_passed": gate2_pass,
            "next_phase": next_phase,
            "main_experiment_authorized": False,
        },
        "interpretation_constraints": {
            "practical_magnitude_threshold_predeclared": False,
            "practical_magnitude_threshold_applied": False,
            "posthoc_practical_threshold_introduced": False,
            "mcnemar_used_to_decide_gate2": False,
            "note": (
                "Gate 2 is decided solely by the prospectively "
                "specified MRD > 0 criterion. The magnitude of MRD "
                "must still be reported transparently and must not "
                "be described as operationally large without "
                "deployment evidence."
            ),
        },
        "hypothesis_status": {
            "H1_formal_condition": "MRD(V1,V2) > 0",
            "H1_condition_observed": gate2_pass,
            "familywise_significance_claim_available": False,
            "holm_bonferroni_status": (
                "DEFERRED_UNTIL_H2_H5_RESULTS_EXIST"
            ),
        },
    }

    # -------------------------------------------------------------
    # Deterministically serialize before writing so all hashes are
    # known and no partial scientific artifact is required to build
    # the closure manifest.
    # -------------------------------------------------------------

    summary_bytes = serialize_json(statistical_summary)
    summary_hash = sha256_bytes(summary_bytes)

    gate2_bytes = serialize_json(gate2_artifact)
    gate2_hash = sha256_bytes(gate2_bytes)

    output_dir = disagreement_path.parent

    summary_path = output_dir / "phase4_statistical_summary.json"
    gate2_path = output_dir / "gate2_malicious_disagreement.json"
    closure_manifest_path = output_dir / "phase4_closure_manifest.json"

    targets = [
        summary_path,
        gate2_path,
        closure_manifest_path,
    ]

    existing = [p for p in targets if p.exists()]

    if existing:
        joined = "\n".join(
            str(p.relative_to(repo_root))
            for p in existing
        )

        raise RuntimeError(
            "Refusing to overwrite existing Phase-4 closure artifact(s):\n"
            + joined
        )

    closure_manifest = {
        "artifact": "RolloutSecBench",
        "schema_version": "1.0",
        "phase": "Phase 4 Closure",
        "upstream_evidence": {
            "gate1_acceptance": {
                "path": str(gate1_path.relative_to(repo_root)),
                "sha256": gate1_hash,
                "status": "PASS",
            },
            "static_evaluation_manifest": {
                "path": str(static_path.relative_to(repo_root)),
                "sha256": static_hash,
            },
            "phase4_disagreement_manifest": {
                "path": str(
                    phase4_manifest_path.relative_to(repo_root)
                ),
                "sha256": phase4_manifest_hash,
            },
            "malicious_disagreement_result": {
                "path": str(
                    disagreement_path.relative_to(repo_root)
                ),
                "sha256": disagreement_hash,
            },
            "full_malicious_predictions": {
                "path": str(
                    malicious_path.relative_to(repo_root)
                ),
                "sha256": malicious_hash,
            },
            "attack_family_breakdown": (
                {
                    "path": str(
                        family_path.relative_to(repo_root)
                    ),
                    "sha256": family_hash,
                }
                if family_path is not None
                else None
            ),
        },
        "statistical_specification": {
            "mrd_confidence_level": 0.95,
            "mrd_ci_method": "clopper_pearson_exact",
            "mcnemar_method": (
                "exact_conditional_binomial_two_sided"
            ),
            "mcnemar_null_direction_probability": 0.5,
            "holm_bonferroni_applied": False,
            "holm_bonferroni_deferred_until": (
                "H2_H3_H4_H5_RESULTS_AVAILABLE"
            ),
        },
        "outputs": {
            "phase4_statistical_summary": {
                "path": str(summary_path.relative_to(repo_root)),
                "sha256": summary_hash,
            },
            "gate2_decision": {
                "path": str(gate2_path.relative_to(repo_root)),
                "sha256": gate2_hash,
                "status": gate2_status,
            },
        },
        "continuation": {
            "phase4_closed": True,
            "gate2_status": gate2_status,
            "phase5_serving_validation_authorized": gate2_pass,
            "phase6_router_validation_authorized": False,
            "phase7_pilot_authorized": False,
            "phase9_main_experiment_authorized": False,
        },
    }

    closure_bytes = serialize_json(closure_manifest)
    closure_hash = sha256_bytes(closure_bytes)

    # All validation and calculations have completed before any output
    # is created.
    write_bytes_exclusive(summary_path, summary_bytes)
    write_bytes_exclusive(gate2_path, gate2_bytes)
    write_bytes_exclusive(
        closure_manifest_path,
        closure_bytes,
    )

    print()
    print("============================================================")
    print("PHASE 4 STATISTICAL CLOSURE COMPLETE")
    print("============================================================")

    print(f"Malicious N: {counts['n']}")
    print(f"Disagreement N: {counts['disagreement']}")
    print(f"MRD: {statistics['mrd']['estimate']}")
    print(
        "MRD 95% exact CI: "
        f"[{statistics['mrd']['ci_low']}, "
        f"{statistics['mrd']['ci_high']}]"
    )
    print(
        "V1 detects / V2 misses: "
        f"{counts['b_v1_detects_v2_misses']}"
    )
    print(
        "V2 detects / V1 misses: "
        f"{counts['c_v2_detects_v1_misses']}"
    )
    print(
        "Exact McNemar two-sided p: "
        f"{statistics['mcnemar']['raw_two_sided_p_value']}"
    )
    print(
        "Paired malicious detection-rate difference "
        "(V1 - V2): "
        f"{statistics['paired_effect_characterization']['malicious_detection_rate_difference_v1_minus_v2']}"
    )
    print()
    print(f"GATE 2: {gate2_status}")
    print(f"Next authorized phase: {next_phase}")
    print(
        "Holm-Bonferroni: DEFERRED until H2-H5 results exist."
    )
    print(
        "No practical MRD threshold was introduced."
    )

    print()
    print("[OUTPUT] Phase-4 statistical summary:")
    print(f"  {summary_path.relative_to(repo_root)}")
    print(f"  SHA256: {summary_hash}")

    print("[OUTPUT] Gate-2 decision:")
    print(f"  {gate2_path.relative_to(repo_root)}")
    print(f"  SHA256: {gate2_hash}")

    print("[OUTPUT] Phase-4 closure manifest:")
    print(f"  {closure_manifest_path.relative_to(repo_root)}")
    print(f"  SHA256: {closure_hash}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
