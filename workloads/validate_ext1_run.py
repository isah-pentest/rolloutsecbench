#!/usr/bin/env python3
"""
RolloutSecBench EXT1 run validator (revision r1, E3). Derived from workloads/validate_v3_run.py (pinned dc968cab...).
Read-only on the run directory; writes <run_dir>/validation.json. Validity uses only objective apparatus criteria;
outcomes are computed afterwards and never consulted.

V1-V7 as in validate_v3_run.py, with V3 generalised to the declared routing policy (pre-registration Section 5,
"every routing decision reconstructs exactly under the declared policy key"):
  V3 route integrity  for every record: session_id equals the session rule of the run; routing_policy equals the
                      manifest policy; u = SHA256(basis|run_id|key)[:8]/2^64 reconstructs the logged
                      routing_uniform_value exactly, with key = request_sequence (P1), session_id (P2) or
                      SHA256_hex(canonical feature vector) (P3, computed here from the frozen features);
                      selected_revision == (v2 iff u < w2)
  V7 router evidence  as in v3, plus identical routing_policy, session_id and routing_uniform_value per request
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xm_loader  # noqa: E402
import v3_client  # noqa: E402
import validate_v3_run as v3v  # noqa: E402  (frozen hashes, frozen_predictions, EPS reused unchanged)

STATES = {"R0": 0.0, "R1": 0.1, "R2": 0.5, "R3": 1.0}


def canon_key(features, order):
    wire = json.loads(json.dumps(features))
    canon = "|".join(wire[n] if isinstance(wire[n], str) else repr(wire[n]) for n in order)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def session_rule(run_id, sample_id, attempt, attacker):
    return f"{run_id}-S{sample_id}-A{attempt}" if attacker == "A1-multi" else f"{run_id}-S{sample_id}"


def uniform(basis, run_id, key):
    d = hashlib.sha256(f"{basis}|{run_id}|{key}".encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big", signed=False) / (2 ** 64)


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
    policy, attacker = man["routing_policy"], man["attacker_type"]
    w2 = STATES[state]
    expected, attempts = man["expected_requests"], man["attempts_per_sample"]

    fails = Counter()
    examples = {}

    def fail(code, detail):
        fails[code] += 1
        examples.setdefault(code, detail)

    for line in (rd / "checksums.sha256").read_text().splitlines():
        h, name = line.split(None, 1)
        if xm_loader._sha256(rd / name.strip()) != h:
            fail("V1_checksum_mismatch", name)

    ref = v3v.frozen_predictions(root, mode)
    order = json.loads((root / xm_loader.FEATURE_NAMES).read_text())["source_feature_order"]
    pop = xm_loader.load_xm(root)[0] if mode == "SCIENCE" else v3_client.qualification_population(root, man["population_size"])
    p3 = {str(s["sample_id"]): canon_key(s["features"], order) for s in pop} if policy == "P3" else {}

    recs = [json.loads(l) for l in open(rd / "requests.jsonl", encoding="utf-8")]
    recs.sort(key=lambda r: r["request_sequence"])

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
        sq = r["request_sequence"]
        exp_sess = session_rule(run_id, r["sample_id"], r["attempt_number"], attacker)
        if r.get("session_id") != exp_sess:
            fail("V3_session_id_mismatch", f"seq {sq}: {r.get('session_id')} vs {exp_sess}")
        if r.get("routing_policy") != policy:
            fail("V3_policy_mismatch", f"seq {sq}: {r.get('routing_policy')} vs {policy}")
        key = str(sq) if policy == "P1" else exp_sess if policy == "P2" else p3.get(str(r["sample_id"]))
        u = uniform(a.route_basis, run_id, key)
        sel = "v2" if u < w2 else "v1"
        if r["routing_uniform_value"] != u:
            fail("V3_uniform_mismatch", f"seq {sq}: logged {r['routing_uniform_value']} vs {u}")
        if r["selected_revision"] != sel:
            fail("V3_revision_mismatch", f"seq {sq}: logged {r['selected_revision']} vs {sel}")
        if r["configured_v2_weight"] != w2:
            fail("V3_weight_mismatch", f"seq {sq}: {r['configured_v2_weight']}")
        rev = r["selected_revision"]
        rev_count[rev] += 1
        if r["model_sha256"] != (v3v.V1_SHA if rev == "v1" else v3v.V2_SHA):
            fail("V4_model_sha", f"seq {sq}: {r['model_sha256']}")
        if r["preprocessor_sha256"] != v3v.PRE_SHA:
            fail("V4_preprocessor_sha", f"seq {sq}")
        fr = ref.get(str(r["sample_id"]))
        if fr is None or rev not in fr:
            fail("V5_sample_not_in_frozen_reference", str(r["sample_id"]))
            continue
        fp, fb = fr[rev]
        if int(r["prediction"]) != fp:
            fail("V5_prediction_mismatch", f"seq {sq} sample {r['sample_id']} {rev}: {r['prediction']} vs {fp}")
        if abs(float(r["probability"]) - fb) >= v3v.EPS:
            fail("V5_probability_mismatch", f"seq {sq}: |dp|={abs(float(r['probability'])-fb):.2e}")

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
            if e and (e["selected_revision"] != r["selected_revision"] or e["prediction"] != r["prediction"]
                      or e.get("routing_policy") != policy or e.get("session_id") != r.get("session_id")
                      or e.get("routing_uniform_value") != r["routing_uniform_value"]):
                fail("V7_router_client_disagreement", r["request_id"])

    valid = not fails

    # ---- outcomes: computed AFTER validity, never used for it ----
    first_evasion = {}
    for r in recs:
        if r.get("prediction") == 0 and r["sample_id"] not in first_evasion:
            first_evasion[r["sample_id"]] = r["attempt_number"]
    n = len(per_sample)
    rev_seq = "".join("2" if r["selected_revision"] == "v2" else "1" for r in recs)
    dec_seq = "".join(str(r["prediction"]) for r in recs)
    outcomes = {
        "population": n, "requests_by_revision": dict(rev_count),
        "realized_v2_share": rev_count["v2"] / max(1, sum(rev_count.values())),
        "cumulative_evasions_by_n": {str(k): sum(1 for v in first_evasion.values() if v <= k) for k in range(1, attempts + 1)},
        "revision_sequence_sha256": hashlib.sha256(rev_seq.encode()).hexdigest(),
        "decision_sequence_sha256": hashlib.sha256(dec_seq.encode()).hexdigest(),
    } if mode == "SCIENCE" else {
        "population": n, "requests_by_revision": dict(rev_count),
        "note": "qualification run: benign traffic, no scientific outcome",
    }

    report = {
        "validator": "ext1-validator-1.0",
        "validator_sha256": xm_loader._sha256(Path(__file__)),
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "mode": mode, "rollout_state": state, "routing_policy": policy, "attacker": attacker,
        "disposition": "VALID" if valid else "INVALID",
        "failed_criteria": dict(fails), "first_example_per_criterion": examples,
        "checks": ["V1_completeness", "V2_transport", "V3_route_integrity_under_policy", "V4_serving_identity",
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
