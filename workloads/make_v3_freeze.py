#!/usr/bin/env python3
"""
RolloutSecBench - generate the Phase-9 v3 confirmatory freeze manifest (run ONCE, before any science traffic).

Writes data/manifests/phase9_v3/phase9_v3_freeze.json (+ .sha256). Refuses to overwrite.

Contents
  * design: 4x2 factorial, 5 primary + 5 ordered reserve identities per cell, A0=1 / A1=5 attempts, 16 QPS
  * amendment A1-B5-L16 with rationale; Gate-4 re-specification (analytic routing null) with the
    superseded MDE retained as a reported, pre-registered secondary comparison
  * randomized execution order of the 40 primaries (fixed, recorded seed)
  * pinned hashes: client, validator, loader, campaign runner, frozen inputs, router source/protocol,
    live image IDs and router deployment environment (captured from the cluster at freeze time)
  * PRE-REGISTERED PREDICTED OUTCOMES for every identity, computed from the frozen offline predictions
    and the deterministic routing rule. A VALID run must reproduce them exactly (validity requires
    served == frozen and route == reconstructed), so any discrepancy is an apparatus defect.

Usage (repository root, port-forward to the router on 18000 active):
  python3 workloads/make_v3_freeze.py --endpoint http://127.0.0.1:18000
  python3 workloads/make_v3_freeze.py --offline      # tests only: skips cluster capture
"""
import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xm_loader  # noqa: E402

NS = "rolloutsecbench"
BASIS = "fa1c7a14b56398343fa0ea09ba200ab77bb67c55c4fc1c5aefd1da229fba9c1f"
STATES = {"R0": 0.0, "R1": 0.1, "R2": 0.5, "R3": 1.0}
ATTEMPTS = {"A0": 1, "A1": 5}
PRIMARY, RESERVE = 5, 5
QPS = 16.0
SEED = 2026101901
OUT_DIR = "data/manifests/phase9_v3"
PINNED_CODE = ["workloads/xm_loader.py", "workloads/v3_client.py", "workloads/validate_v3_run.py",
               "workloads/v3_campaign.py", "workloads/make_v3_freeze.py"]
FROZEN_INPUTS = {
    "data/splits/strict_test_nonoverlap.csv": xm_loader.STRICT_TEST_SHA256,
    "data/processed/feature_names.json": xm_loader.FEATURE_NAMES_SHA256,
    "results/derived/malicious_predictions_strict.csv": xm_loader.FROZEN_XM_SHA256,
    "results/derived/pairwise_predictions_strict.csv": "8994cc82239894c51dc4a8c9131872321a04d1814bd69306fbdf1cdc60f1b11c",
}


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{r.stderr[:400]}")
    return r.stdout


def capture_runtime(endpoint):
    health = json.loads(urllib.request.urlopen(endpoint.rstrip("/") + "/health", timeout=10).read())
    pods = json.loads(sh(["kubectl", "-n", NS, "get", "pods", "-o", "json"]))["items"]
    dep = json.loads(sh(["kubectl", "-n", NS, "get", "deploy", "router", "-o", "json"]))
    cms = json.loads(sh(["kubectl", "-n", NS, "get", "configmap", "-o", "json"]))["items"]
    c = dep["spec"]["template"]["spec"]["containers"][0]
    cm = {m["metadata"]["name"]: {k: hashlib.sha256(v.encode()).hexdigest() for k, v in m.get("data", {}).items()}
          for m in cms if m["metadata"]["name"].startswith("phase7-router")}
    nodes = sh(["kubectl", "get", "nodes", "-o", "name"]).split()
    return {
        "router_health": health,
        "router_protocol_sha256": health.get("protocol_sha256"),
        "router_image": c.get("image"),
        "router_command": c.get("command", []) + c.get("args", []),
        "router_env": {e["name"]: e.get("value") for e in c.get("env", [])},
        "router_source_configmaps_sha256": cm,
        "pods": [{"name": p["metadata"]["name"],
                  "image": [s.get("image") for s in p["status"].get("containerStatuses", [])],
                  "imageID": [s.get("imageID") for s in p["status"].get("containerStatuses", [])]}
                 for p in pods],
        "node": nodes[0].split("/", 1)[-1] if nodes else None,
    }


def route(run_id, seq, w2):
    d = hashlib.sha256(f"{BASIS}|{run_id}|{seq}".encode()).digest()
    return "v2" if int.from_bytes(d[:8], "big") / 2 ** 64 < w2 else "v1"


def predict(run_id, state, attacker, order, frozen):
    w2, k = STATES[state], ATTEMPTS[attacker]
    seq, v2 = 0, 0
    first = {}
    for sid in order:
        for att in range(1, k + 1):
            seq += 1
            rev = route(run_id, seq, w2)
            v2 += rev == "v2"
            if frozen[sid][rev] == 0 and sid not in first:
                first[sid] = att
    out = {"requests": seq, "realized_v2_share": v2 / seq,
           "evasions_attempt1": sum(1 for a in first.values() if a == 1)}
    if attacker == "A1":
        out["cumulative_evasions_by_n"] = {str(n): sum(1 for a in first.values() if a <= n) for n in range(1, k + 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:18000")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--root", default=str(HERE.parent))
    a = ap.parse_args()
    root = Path(a.root).resolve()
    xm_loader.ROOT = root
    out_dir = root / OUT_DIR
    target = out_dir / "phase9_v3_freeze.json"
    if target.exists():
        raise SystemExit(f"FAIL_CLOSED: {target} already exists (a freeze is never regenerated)")

    for rel, sha in FROZEN_INPUTS.items():
        if xm_loader._sha256(root / rel) != sha:
            raise SystemExit(f"frozen input hash mismatch: {rel}")
    code = {}
    for rel in PINNED_CODE:
        p = root / rel
        if not p.is_file():
            raise SystemExit(f"missing pinned file: {rel}")
        code[rel] = xm_loader._sha256(p)

    samples, prov = xm_loader.load_xm(root)
    order = [s["sample_id"] for s in samples]
    frozen = {}
    with open(root / xm_loader.FROZEN_XM, newline="") as fh:
        for r in csv.DictReader(fh):
            frozen[r["id"]] = {"v1": int(r["v1_prediction"]), "v2": int(r["v2_prediction"])}

    identities = []
    for state in STATES:
        for attacker in ATTEMPTS:
            cell = f"{state}-{attacker}"
            for kind, n in (("PRIMARY", PRIMARY), ("RESERVE", RESERVE)):
                for i in range(1, n + 1):
                    rid = f"V3-{cell}-{'P' if kind == 'PRIMARY' else 'R'}{i:02d}"
                    identities.append({
                        "run_id": rid, "cell": cell, "rollout_state": state, "attacker": attacker,
                        "attempts_per_sample": ATTEMPTS[attacker], "configured_v2_weight": STATES[state],
                        "identity_type": kind, "reserve_order": i if kind == "RESERVE" else None,
                        "expected_requests": len(order) * ATTEMPTS[attacker],
                        "predicted_outcome": predict(rid, state, attacker, order, frozen),
                    })
    primaries = [x["run_id"] for x in identities if x["identity_type"] == "PRIMARY"]
    rng = random.Random(SEED)
    exec_order = primaries[:]
    rng.shuffle(exec_order)

    runtime = None if a.offline else capture_runtime(a.endpoint)
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None

    freeze = {
        "artifact": "RolloutSecBench",
        "freeze": "phase9-v3-confirmatory-1.0",
        "status": "FROZEN_PREEXECUTION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_at_freeze": git,
        "design": {"factorial": "4x2 Rollout x Attacker", "cells": 8, "primary_per_cell": PRIMARY,
                   "reserve_per_cell": RESERVE, "required_valid_per_cell": 5, "population": len(order),
                   "attempts": ATTEMPTS, "a1_attempt_policy": "all attempts always sent; no early stop",
                   "target_qps": QPS, "qps_tolerance": 0.10, "client_workers": 16,
                   "randomization_seed": SEED, "execution_order_primaries": exec_order},
        "amendment_A1_B5_L16": {
            "changes": {"a1_budget": "20 -> 5", "fixed_load_qps": "64 -> 16 (50% -> 12.5% of QPS_max 128)"},
            "rationale": ("Phase-7 runs at 64 QPS were invalidated by HTTP-502 transport failures; load is not on the "
                          "hypothesized causal chain (Blueprint Section P) and H6 is deferred. Budget 5 keeps total "
                          "runtime unchanged at the lower load; analytic power for H4/H5 at budget 5 exceeds 0.999 "
                          "(retry gain 34.2 and 69.8 samples at R1/R2 vs SE 2.2/2.7)."),
            "consequential_changes": ["H4 omnibus over n in {1..5}; effect sizes Delta(2..5)",
                                      "primary A1 outcome C(5)",
                                      "Figure 4 beyond n=5 shown only as labelled model extrapolation"],
            "disclosure": "Parameters equal those of the archived invalid v2 execution, whose outcomes carried no valid information.",
        },
        "gate4": {
            "superseded_rule": {"mde_absolute_asr": 3.64705466658e-05, "source": "Phase-8 freeze",
                                "status": "reported as pre-registered secondary comparison",
                                "reason": "under correct routing >=1 of 5 runs exceeds it with p>0.999 (3 of 4 pilot runs already do)"},
            "primary_rule": ("Per mixed A0 cell (R1, R2): statistic = mean d_r over valid runs, d_r = observed ASR - "
                             "(w1_obs*ASR_R0 + w2_obs*ASR_R3). Null: K12~Bin(92,w_obs), K21~Bin(57,w_obs) independent; "
                             "two-sided Monte-Carlo p (1e6 draws, seed 2026101902); Holm across R1,R2; alpha 0.05."),
            "mechanism_check": "report realized V2 share within X_1->2 and X_2->1 vs global share per run",
        },
        "validity_criteria": ["V1 completeness", "V2 transport (HTTP 200, no retries)", "V3 route reconstruction",
                              "V4 model/preprocessor identity", "V5 served == frozen (exact decision, |dp|<1e-4)",
                              "V6 offered load within +/-10%", "V7 router-evidence join",
                              "P1 pod UIDs and restart counts unchanged across the run",
                              "P2 router /health protocol hash, source configmap and deployment env equal the pinned values"],
        "validity_never_uses": ["ASR", "C(n)", "effect direction or size", "hypothesis agreement"],
        "replacement_policy": {"reserve_activation": "only after an objective INVALID in the same cell",
                               "reserve_order": "ascending within cell", "valid_run_replacement": False,
                               "consumed_identity_rerun": False, "posthoc_identity_creation": False,
                               "reserve_exhaustion": "stop cell, report incomplete coverage",
                               "consumption_boundary": "first request sent"},
        "routing": {"basis_sha256": BASIS, "material": "basis|run_id|request_sequence",
                    "rule": "v2 iff SHA256(material)[:8]/2^64 < w2"},
        "pinned_code_sha256": code,
        "frozen_inputs_sha256": FROZEN_INPUTS,
        "loader_provenance": prov,
        "runtime_at_freeze": runtime,
        "identities": identities,
        "registry": "results/raw/phase9_v3/registry.csv",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(target, "x", encoding="utf-8") as f:
        json.dump(freeze, f, indent=2, sort_keys=True)
        f.write("\n")
    h = xm_loader._sha256(target)
    (out_dir / "phase9_v3_freeze.json.sha256").write_text(f"{h}  phase9_v3_freeze.json\n")
    print(f"FREEZE WRITTEN {target}\nsha256 {h}\nidentities {len(identities)}  primaries in order: {exec_order[:4]} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
