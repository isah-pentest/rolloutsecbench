#!/usr/bin/env python3
"""
RolloutSecBench Phase-9 v3 run validator. Standard library only. Read-only on the run directory;
writes <run_dir>/validation.json.

Validity is decided ONLY by objective, pre-specified apparatus criteria. ASR and every other
scientific outcome are computed in a separate "outcomes" block and NEVER consulted for validity.

Validity criteria (all must hold)
  V1 evidence completeness  requests.jsonl has exactly expected_requests records; request_sequence
                            is exactly 1..N; request_id unique; checksums.sha256 verifies
  V2 transport              every record HTTP 200, no transport error
  V3 route integrity        for every record, u = SHA256(basis|run_id|seq)[:8]/2^64 reconstructs the
                            logged routing_uniform_value exactly, and selected_revision == (v2 iff u < w2)
  V4 serving identity       model_sha256 is the frozen hash of the selected revision; preprocessor
                            sha256 is the frozen preprocessor hash
  V5 served == frozen       prediction equals the frozen offline prediction of the selected revision
                            and |probability - frozen| < 1e-4 (Section K), for every record
  V6 offered load           achieved QPS within +/-10% of target (Section AH)
  V7 router evidence        (optional, when --router-evidence is given) exactly one router record per
                            request_id of this run, with identical selected revision and prediction

Usage
  python3 workloads/validate_v3_run.py --run-dir results/raw/phase9_v3/V3-R1-A0-P01 \
      --route-basis fa1c7a14b56398343fa0ea09ba200ab77bb67c55c4fc1c5aefd1da229fba9c1f \
      [--router-evidence /path/to/router_records_for_this_run.jsonl]
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xm_loader  # noqa: E402

V1_SHA = "6c0aa79af5568231202f2aa150e2a388855a058601333cbdb067b54b097ac14c"
V2_SHA = "ea66463097938adf80145a872882dbb45cb8fdb3ffabdde5b23b633b586ca874"
PRE_SHA = "4ea29c9353505b423ce37b077f3ef821f76da426230518d771ebe217e3c2408f"
PAIRWISE = "results/derived/pairwise_predictions_strict.csv"
PAIRWISE_SHA = "8994cc82239894c51dc4a8c9131872321a04d1814bd69306fbdf1cdc60f1b11c"
STATES = {"R0": 0.0, "R1": 0.1, "R2": 0.5, "R3": 1.0}
EPS = 1e-4


def route(basis, run_id, seq, w2):
    d = hashlib.sha256(f"{basis}|{run_id}|{seq}".encode("utf-8")).digest()
    u = int.from_bytes(d[:8], "big", signed=False) / (2 ** 64)
    return u, ("v2" if u < w2 else "v1")


def frozen_predictions(root, mode):
    """sample_id -> {'v1': (pred, prob), 'v2': (pred, prob)} from the frozen offline evaluation."""
    if mode == "SCIENCE":
        path, sha = root / xm_loader.FROZEN_XM, xm_loader.FROZEN_XM_SHA256
    else:
        path, sha = root / PAIRWISE, PAIRWISE_SHA
    got = xm_loader._sha256(path)
    if got != sha:
        raise SystemExit(f"frozen reference hash mismatch: {path} {got}")
    ref = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            ref[r["id"]] = {"v1": (int(r["v1_prediction"]), float(r["v1_probability"])),
                            "v2": (int(r["v2_prediction"]), float(r["v2_probability"]))}
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--route-basis", required=True)
    ap.add_argument("--router-evidence")
    ap.add_argument("--root", default=str(HERE.parent))
    a = ap.parse_args()
    root = Path(a.root).resolve()
    xm_loader.ROOT = root
    rd = Path(a.run_dir)
    man = json.loads((rd / "run_manifest.json").read_text())
    summ = json.loads((rd / "run_summary.json").read_text())
    run_id, state, mode = man["run_id"], man["rollout_state"], man["mode"]
    w2 = STATES[state]
    expected = man["expected_requests"]
    attempts = man["attempts_per_sample"]

    fails = Counter()
    examples = {}

    def fail(code, detail):
        fails[code] += 1
        examples.setdefault(code, detail)

    # checksums
    for line in (rd / "checksums.sha256").read_text().splitlines():
        h, name = line.split(None, 1)
        if xm_loader._sha256(rd / name.strip()) != h:
            fail("V1_checksum_mismatch", name)

    ref = frozen_predictions(root, mode)
    recs = []
    with open(rd / "requests.jsonl", encoding="utf-8") as fh:
        for line in fh:
            recs.append(json.loads(line))
    recs.sort(key=lambda r: r["request_sequence"])

    # V1 completeness
    if len(recs) != expected:
        fail("V1_record_count", f"{len(recs)} != {expected}")
    seqs = [r["request_sequence"] for r in recs]
    if seqs != list(range(1, len(recs) + 1)):
        fail("V1_sequence_not_contiguous", f"first gap near {next((i+1 for i,s in enumerate(seqs) if s != i+1), None)}")
    if len({r["request_id"] for r in recs}) != len(recs):
        fail("V1_duplicate_request_id", "")

    per_sample = Counter(r["sample_id"] for r in recs)
    if any(c != attempts for c in per_sample.values()):
        fail("V1_attempts_per_sample", f"{Counter(per_sample.values())}")

    rev_count = Counter()
    for r in recs:
        if r["http_status"] != 200 or r["transport_error"]:
            fail("V2_transport", f"seq {r['request_sequence']}: {r['http_status']} {r['transport_error']} {r.get('error_detail')}")
            continue
        u, sel = route(a.route_basis, run_id, r["request_sequence"], w2)
        if r["routing_uniform_value"] != u:
            fail("V3_uniform_mismatch", f"seq {r['request_sequence']}: logged {r['routing_uniform_value']} vs {u}")
        if r["selected_revision"] != sel:
            fail("V3_revision_mismatch", f"seq {r['request_sequence']}: logged {r['selected_revision']} vs {sel}")
        if r["configured_v2_weight"] != w2:
            fail("V3_weight_mismatch", f"seq {r['request_sequence']}: {r['configured_v2_weight']}")
        rev = r["selected_revision"]
        rev_count[rev] += 1
        if r["model_sha256"] != (V1_SHA if rev == "v1" else V2_SHA):
            fail("V4_model_sha", f"seq {r['request_sequence']}: {r['model_sha256']}")
        if r["preprocessor_sha256"] != PRE_SHA:
            fail("V4_preprocessor_sha", f"seq {r['request_sequence']}")
        fr = ref.get(str(r["sample_id"]))
        if fr is None or rev not in fr:
            fail("V5_sample_not_in_frozen_reference", str(r["sample_id"]))
            continue
        fp, fb = fr[rev]
        if int(r["prediction"]) != fp:
            fail("V5_prediction_mismatch", f"seq {r['request_sequence']} sample {r['sample_id']} {rev}: {r['prediction']} vs {fp}")
        if abs(float(r["probability"]) - fb) >= EPS:
            fail("V5_probability_mismatch", f"seq {r['request_sequence']}: |dp|={abs(float(r['probability'])-fb):.2e}")

    target = man["target_qps"]
    achieved = summ.get("achieved_qps") or 0
    if not (0.9 * target <= achieved <= 1.1 * target):
        fail("V6_offered_load", f"achieved {achieved} vs target {target}")

    if a.router_evidence:
        ev = [json.loads(l) for l in open(a.router_evidence, encoding="utf-8") if l.strip()]
        ev = [e for e in ev if e.get("run_id") == run_id]
        by_id = Counter(e["request_id"] for e in ev)
        if set(by_id) != {r["request_id"] for r in recs} or any(c != 1 for c in by_id.values()):
            fail("V7_router_evidence_join", f"router records {len(ev)}, unique {len(by_id)}, client {len(recs)}")
        evm = {e["request_id"]: e for e in ev}
        for r in recs:
            e = evm.get(r["request_id"])
            if e and (e["selected_revision"] != r["selected_revision"] or e["prediction"] != r["prediction"]):
                fail("V7_router_client_disagreement", r["request_id"])

    valid = not fails

    # ---- outcomes: computed AFTER validity, never used for it ----
    first_evasion = {}
    for r in recs:
        if r.get("prediction") == 0 and r["sample_id"] not in first_evasion:
            first_evasion[r["sample_id"]] = r["attempt_number"]
    n = len(per_sample)
    outcomes = {
        "population": n,
        "requests_by_revision": dict(rev_count),
        "realized_v2_share": rev_count["v2"] / max(1, sum(rev_count.values())),
        "evasions_attempt1": sum(1 for v in first_evasion.values() if v == 1),
        "cumulative_evasions_by_n": {str(k): sum(1 for v in first_evasion.values() if v <= k)
                                     for k in range(1, attempts + 1)},
    } if mode == "SCIENCE" else {
        "population": n, "requests_by_revision": dict(rev_count),
        "note": "qualification run: benign traffic, no scientific outcome",
    }

    report = {
        "validator": "phase9-v3-validator-1.0",
        "validator_sha256": xm_loader._sha256(Path(__file__)),
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "mode": mode, "rollout_state": state,
        "disposition": "VALID" if valid else "INVALID",
        "failed_criteria": dict(fails), "first_example_per_criterion": examples,
        "checks": ["V1_completeness", "V2_transport", "V3_route_integrity", "V4_serving_identity",
                   "V5_served_equals_frozen", "V6_offered_load"] + (["V7_router_evidence"] if a.router_evidence else []),
        "achieved_qps": achieved, "target_qps": target,
        "outcomes_not_used_for_validity": outcomes,
    }
    out = rd / "validation.json"
    with open(out, "x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"{run_id}: {report['disposition']}  failures={dict(fails)}")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
