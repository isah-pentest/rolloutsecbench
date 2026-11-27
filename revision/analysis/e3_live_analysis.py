"""Revision r1, Phase 6, E3 live analysis (pre-registration Section 5, verification criteria). Computes no new outcome:
compares each executed EXT1 run with the prediction frozen in ext1_live_freeze.json (computed before any router
change). Run after the campaign reports COMPLETE.

A run "equals its predicted outcome" when ALL of these are equal: C(1..5); the realised V2 share; the SHA-256 of the
per-request revision sequence; the SHA-256 of the per-request decision sequence. Equal revision and decision
sequences imply equal K12, K21 and every request-level outcome.
  V-P2-one : every valid P2 A1-one run equals its prediction, and Delta(5) = 0
  V-P2-multi: every valid P2 A1-multi run equals its prediction
  V-P3     : every valid P3 A1-one run equals its prediction, and Delta(5) = 0
Runs that are INVALID are listed with their reasons. A valid run that differs is reported as a finding, not removed.

Outputs (revision/results/e3_live/): e3_live_results.json, the registry, campaign log, policy switches, activations,
and each run's validation.json, disposition.json, run_manifest.json and run_summary.json; plus
raw_evidence.sha256 (every file of every EXT1 run directory, hashed in place; raw files stay out of git).
"""
import csv, hashlib, json, pathlib, shutil, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/ext1"
FREEZE = ROOT / "data/manifests/ext1/ext1_live_freeze.json"
FREEZE_SHA = "feefb9a8adf83044f787486ef293cf511ff67c102e9d6adc90ae85f6b3415747"
OUT = ROOT / "revision/results/e3_live"
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

if sha(FREEZE) != FREEZE_SHA: sys.exit("live freeze hash mismatch")
fz = json.load(open(FREEZE)); ids = {i["run_id"]: i for i in fz["identities"]}
status = json.load(open(RAW / "campaign_status.json"))
if status["state"] != "COMPLETE": sys.exit(f"campaign not COMPLETE: {status['state']}")
reg = list(csv.DictReader(open(RAW / "registry.csv")))
if OUT.exists(): sys.exit("e3_live output folder already exists")
OUT.mkdir(parents=True)

runs, invalid = {}, []
for row in reg:
    r = row["run_id"]; d = RAW / r; ident = ids[r]
    if row["disposition"] != "VALID":
        invalid.append({"run_id": r, "cell": row["cell"], "reasons": row["reasons"]}); continue
    v = json.load(open(d / "validation.json")); o = v["outcomes_not_used_for_validity"]; p = ident["predicted_outcome"]
    live_C = [o["cumulative_evasions_by_n"][str(n)] for n in range(1, 6)]; pred_C = [p["C"][str(n)] for n in range(1, 6)]
    checks = {"C": live_C == pred_C, "realized_v2_share": o["realized_v2_share"] == p["realized_v2_share"],
              "revision_sequence": o["revision_sequence_sha256"] == p["revision_sequence_sha256"],
              "decision_sequence": o["decision_sequence_sha256"] == p["decision_sequence_sha256"]}
    runs[r] = {"cell": ident["cell"], "policy": ident["policy"], "attacker": ident["attacker"], "identity_type": ident["identity_type"],
               "C_live": live_C, "C_predicted": pred_C, "Delta5_live": live_C[4] - live_C[0], "Delta5_predicted": p["Delta5"],
               "realized_v2_share": o["realized_v2_share"], "checks": checks, "equals_prediction": all(checks.values())}

valid_per_cell = {}
for r, x in runs.items(): valid_per_cell.setdefault(x["cell"], []).append(r)


def crit(policy, attacker, need_zero):
    rs = [x for x in runs.values() if x["policy"] == policy and x["attacker"] == attacker]
    return {"valid_runs": len(rs), "all_equal_prediction": all(x["equals_prediction"] for x in rs),
            **({"all_Delta5_zero": all(x["Delta5_live"] == 0 for x in rs)} if need_zero else {}),
            "met": bool(rs) and all(x["equals_prediction"] for x in rs) and (not need_zero or all(x["Delta5_live"] == 0 for x in rs))}


res = {"freeze_sha256": FREEZE_SHA, "campaign_status": status, "invalid_runs": invalid,
       "valid_per_cell": {c: len(v) for c, v in sorted(valid_per_cell.items())},
       "coverage_complete": all(len(valid_per_cell.get(c, [])) >= fz["design"]["valid_runs_per_cell"] for c in {i["cell"] for i in fz["identities"]}),
       "runs": runs,
       "verification": {"V-P2-one": crit("P2", "A1-one", True), "V-P2-multi": crit("P2", "A1-multi", False),
                        "V-P3": crit("P3", "A1-one", True)},
       "findings_valid_runs_differing_from_prediction": [r for r, x in runs.items() if not x["equals_prediction"]]}
json.dump(res, open(OUT / "e3_live_results.json", "w"), indent=1, sort_keys=True)

for f in ("registry.csv", "campaign_status.json", "policy_switches.jsonl", "activations.jsonl", "_campaign/campaign.log"):
    if (RAW / f).exists(): shutil.copy2(RAW / f, OUT / pathlib.Path(f).name)
(OUT / "runs").mkdir()
for row in reg:
    for f in ("validation.json", "disposition.json", "run_manifest.json", "run_summary.json"):
        if (RAW / row["run_id"] / f).exists(): shutil.copy2(RAW / row["run_id"] / f, OUT / "runs" / f"{row['run_id']}_{f}")
with open(OUT / "raw_evidence.sha256", "w") as fh:
    for d in sorted(RAW.glob("EXT1-*")):
        for p in sorted(d.iterdir()):
            if p.is_file(): fh.write(f"{sha(p)}  {p.relative_to(ROOT)}\n")

for r, x in runs.items():
    print(f"{r}: equals_prediction={x['equals_prediction']} C live {x['C_live']} Delta5 {x['Delta5_live']} (pred {x['Delta5_predicted']})")
print("invalid runs:", invalid)
print("verification:", json.dumps(res["verification"]))
print("coverage complete:", res["coverage_complete"], "| findings:", res["findings_valid_runs_differing_from_prediction"])
print("wrote", OUT / "e3_live_results.json", sha(OUT / "e3_live_results.json"))
