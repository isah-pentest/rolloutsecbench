#!/usr/bin/env python3
"""
Revision r1, Phase 6 live: write the EXT1 live freeze (pre-registration Sections 2 and 5; DEV-R1-005).
Run after the EXT1 router is deployed (ROUTING_POLICY=P1) and the benign qualification has PASSED, and before any
EXT1- science request. Writes data/manifests/ext1/ext1_live_freeze.json and its .sha256 sidecar (never overwrites).

Contents: the 24 live identities from E3_FREEZE.json (12 primary, 12 reserve), the pre-registered execution order,
the predicted outcome of every identity from live_predictions.json (computed and committed before any router change),
the routing basis, design values (target QPS and client workers as in the published campaign; 2 valid runs per
cell), the pinned code and inputs, and the runtime captured from the cluster (node, pods, image IDs, router module,
env, ConfigMap hashes, protocol hash).
"""
import argparse, hashlib, json, pathlib, subprocess, sys, urllib.request
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workloads"))
import xm_loader  # noqa: E402

PINS = {"revision/prereg/E3_FREEZE.json": "4623d2e75ea6d2615f50601d237ecb8c1ec54ae031ee9c61e4fa7400178c4706",
        "revision/results/e3_sim/live_predictions.json": "24f754648e5dc197df91f869d0547e14eb955c6857875107eb05f3d4a55208e9",
        "data/manifests/ext1/ext1_router_protocol_v1.0.json": "af8c8243662e13cf0c29d5a97bc19878287d91da52655764c757c9954cd2bfde",
        "data/manifests/phase9_v3/phase9_v3_freeze.json": "67d0ac984710b5a3748c44de577eece02bf7652deb9969fc792decf6b75dfcf1"}
CODE = ["workloads/ext1_client.py", "workloads/validate_ext1_run.py", "workloads/ext1_campaign.py",
        "workloads/ext1_serving_gate.py", "workloads/v3_client.py", "workloads/validate_v3_run.py",
        "workloads/xm_loader.py", "serving/router/ext1/router_ext1_v1.py", "workloads/make_ext1_live_freeze.py"]
INPUTS = ["data/splits/strict_test_nonoverlap.csv", "data/processed/feature_names.json",
          "results/derived/malicious_predictions_strict.csv", "results/derived/pairwise_predictions_strict.csv",
          "data/manifests/phase5_serving_protocol_v1.0.json", *PINS,
          "kubernetes/ext1/configmap-ext1-router-source.json", "kubernetes/ext1/configmap-ext1-router-protocol.json",
          "kubernetes/ext1/deployment-router-ext1.json", "revision/results/ext1_rebuild/serving_gate.json"]
NS = "rolloutsecbench"
sha = lambda rel: xm_loader._sha256(ROOT / rel)


def kubectl_json(*a):
    r = subprocess.run(["kubectl", "-n", NS, *a, "-o", "json"], capture_output=True, text=True, timeout=60)
    if r.returncode: sys.exit(f"kubectl {' '.join(a)} failed: {r.stderr[:300]}")
    return json.loads(r.stdout)


def runtime(endpoint):
    pods = []
    for p in kubectl_json("get", "pods")["items"]:
        if p["metadata"].get("deletionTimestamp"): continue
        cs = p["status"]["containerStatuses"][0]
        pods.append({"name": p["metadata"]["name"], "image": [cs["image"]], "imageID": [cs["imageID"]]})
    dep = kubectl_json("get", "deploy", "router")["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e.get("value") for e in dep.get("env", [])}
    if env.get("ROUTING_POLICY") != "P1": sys.exit("freeze must be captured with ROUTING_POLICY=P1 (the default)")
    cms = {m["metadata"]["name"]: {k: hashlib.sha256(v.encode()).hexdigest() for k, v in m.get("data", {}).items()}
           for m in kubectl_json("get", "configmap")["items"] if m["metadata"]["name"].startswith("ext1-router")}
    health = json.loads(urllib.request.urlopen(endpoint + "/health", timeout=5).read())
    nodes = subprocess.run(["kubectl", "get", "nodes", "-o", "jsonpath={.items[0].metadata.name}"], capture_output=True, text=True).stdout
    return {"node": nodes, "pods": pods, "router_image": dep["image"], "router_command": dep.get("command"),
            "router_args": dep.get("args"), "router_env": env, "router_health": health,
            "router_protocol_sha256": health["protocol_sha256"], "router_source_configmaps_sha256": cms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qualification", required=True, help="results/raw/ext1/qualification_<tag>.json (must be PASS)")
    ap.add_argument("--endpoint", default="http://127.0.0.1:18000")
    ap.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    for rel, h in PINS.items():
        if sha(rel) != h and not a.test_mode: sys.exit(f"hash mismatch: {rel}")
    out = ROOT / "data/manifests/ext1/ext1_live_freeze.json"
    if out.exists(): sys.exit("FAIL_CLOSED: live freeze already exists")
    q = json.loads((ROOT / a.qualification).read_text())
    if q["decision"] != "PASS": sys.exit("qualification did not PASS")
    e3 = json.loads((ROOT / "revision/prereg/E3_FREEZE.json").read_text())
    pred = json.loads((ROOT / "revision/results/e3_sim/live_predictions.json").read_text())
    p9 = json.loads((ROOT / "data/manifests/phase9_v3/phase9_v3_freeze.json").read_text())
    proto = json.loads((ROOT / "data/manifests/ext1/ext1_router_protocol_v1.0.json").read_text())
    idents = []
    for cell, grp in e3["live_identities"].items():
        pol, st, att = cell.split("-", 2)
        for kind, lst in (("PRIMARY", grp["primary"]), ("RESERVE", grp["reserve"])):
            for k, rid in enumerate(lst, 1):
                if rid not in pred: sys.exit(f"no prediction for {rid}")
                attempts = 1 if att == "A0" else 5
                idents.append({"run_id": rid, "cell": cell, "policy": pol, "rollout_state": st, "attacker": att,
                               "identity_type": kind, "reserve_order": k if kind == "RESERVE" else None,
                               "attempts_per_sample": attempts, "expected_requests": xm_loader.N_M * attempts,
                               "predicted_outcome": pred[rid]})
    if len(idents) != 24 or set(pred) != {i["run_id"] for i in idents}: sys.exit("identity/prediction set mismatch")
    freeze = {
        "artifact": "RolloutSecBench EXT1 live freeze (revision r1, E3)", "status": "FROZEN_PREEXECUTION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "routing": {"basis_sha256": proto["routing"]["route_seed_basis_sha256"], "policies": proto["policies"]},
        "design": {"target_qps": p9["design"]["target_qps"], "client_workers": p9["design"]["client_workers"],
                   "execution_order_primaries": e3["live_execution_order"], "valid_runs_per_cell": 2},
        "identities": idents,
        "qualification": {"path": a.qualification, "sha256": sha(a.qualification), "decision": q["decision"]},
        "deviations_in_force": ["DEV-R1-005 (apparatus rebuilt; acceptance met, revision/results/ext1_rebuild)"],
        "pinned_code_sha256": {rel: sha(rel) for rel in CODE},
        "frozen_inputs_sha256": {rel: sha(rel) for rel in INPUTS},
        "runtime_at_freeze": None if a.test_mode else runtime(a.endpoint),
        "validity_never_uses": ["C(n)", "Delta(n)", "predicted outcome", "agreement with prediction"],
    }
    out.write_text(json.dumps(freeze, indent=1, sort_keys=True) + "\n")
    h = sha("data/manifests/ext1/ext1_live_freeze.json")
    (out.parent / (out.name + ".sha256")).write_text(f"{h}  {out.name}\n")
    print("identities:", len(idents), "| order:", e3["live_execution_order"])
    print("wrote", out, h)


if __name__ == "__main__":
    main()
