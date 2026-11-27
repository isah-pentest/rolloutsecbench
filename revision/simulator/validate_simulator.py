"""E1 validation: the simulator must reproduce every valid phase9_v3 run exactly.

Per request: sample, attempt, routing value u, selected revision (client log and router evidence), decision.
Per run: realized V2 share, evasions on attempt 1 and the request count against the pre-registered prediction
in the freeze; C(1..n), K12, K21 and d from the logs against the simulator.
Pass = every check holds for all valid runs. Writes revision/results/e1_validation/validation_report.json.
"""
import csv, hashlib, json, pathlib, sys, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from simulate import Simulator, ROOT

RAW = ROOT / "results/raw/phase9_v3"
OUT = ROOT / "revision/results/e1_validation"; OUT.mkdir(parents=True, exist_ok=True)
sim = Simulator()
pre = {x["run_id"]: x for x in sim.freeze["identities"]}
report = {}
all_ok = True
for row in csv.DictReader(open(RAW / "registry.csv")):
    if row["disposition"] != "VALID":
        continue
    rid = row["run_id"]; ident = pre[rid]
    state, attacker = ident["rollout_state"], ident["attacker"]
    s = sim.run(rid, state, attacker, "P1", keep_requests=True)
    recs = sorted((json.loads(l) for l in open(RAW / rid / "requests.jsonl")), key=lambda q: q["request_sequence"])
    router = {}
    with open(RAW / rid / "router_evidence.jsonl") as f:
        for l in f:
            e = json.loads(l); router[e["request_sequence"]] = e["backend_model_version"]
    mism = collections.Counter(); max_du = 0.0
    first_evade = {}
    for q, (seq, sid, a, u, rev, dec) in zip(recs, s["requests_detail"]):
        mism["sequence"] += q["request_sequence"] != seq
        mism["sample"] += str(q["sample_id"]) != sid
        mism["attempt"] += q["attempt_number"] != a
        du = abs(q["routing_uniform_value"] - u); max_du = max(max_du, du); mism["u"] += du != 0.0
        mism["revision_client"] += q["selected_revision"] != rev
        mism["revision_router"] += router.get(seq) != rev
        mism["decision"] += int(q["prediction"]) != dec
        if int(q["prediction"]) == 0 and q["sample_id"] not in first_evade:
            first_evade[q["sample_id"]] = q["attempt_number"]
    mism["count"] += len(recs) != s["requests"]
    A = max(s["C"])
    C_obs = {m: sum(1 for x in first_evade.values() if x <= m) for m in range(1, A + 1)}
    po = ident["predicted_outcome"]
    run_checks = {
        "C_logs_eq_sim": C_obs == s["C"],
        "requests_eq_prereg": po["requests"] == s["requests"],
        "evasions_attempt1_eq_prereg": po["evasions_attempt1"] == s["C"][1],
        "realized_share_eq_prereg": abs(po["realized_v2_share"] - s["realized_v2_share"]) < 1e-12,
    }
    ok = sum(mism.values()) == 0 and all(run_checks.values())
    all_ok &= ok
    report[rid] = {"ok": ok, "mismatches": dict(mism), "max_abs_u_diff": max_du, **run_checks,
                   "realized_v2_share": s["realized_v2_share"], "C": s["C"], "K12": s["K12"], "K21": s["K21"],
                   "d": s.get("d")}
    print(f"{rid}: {'OK' if ok else 'FAIL'}  C1={s['C'][1]} K12={s['K12']} K21={s['K21']}"
          + (f" d={s['d']:.2f}" if 'd' in s else "") + ("" if ok else f"  {dict(mism)} {run_checks}"))
out = OUT / "validation_report.json"
json.dump({"all_valid_runs_reproduced": all_ok, "runs": report}, open(out, "w"), indent=2, sort_keys=True)
print(f"\nRUNS: {len(report)}  ALL REPRODUCED EXACTLY: {all_ok}")
print("wrote", out, hashlib.sha256(out.read_bytes()).hexdigest())
sys.exit(0 if all_ok else 1)
