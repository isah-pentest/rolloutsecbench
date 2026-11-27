"""Revision r1, Phase 7, E4 decomposition (DEV-R1-007; descriptive, deterministic). Reads committed E4 outputs only.

The fresh-variant attack (attempt a sends variant a) is evaluated at R0 (V1 only) and R3 (V2 only). With no routing,
Delta(5) there is evasion by feature variation alone. The rollout-attributable increment at R1 and R2 is the
simulated mean Delta(5) (e4_summary.json) minus the single-model value:
  increment vs R0 = E[Delta5(R_w)] - Delta5(R0)   (against the pre-rollout status quo, V1 only)
  increment vs R3 = E[Delta5(R_w)] - Delta5(R3)   (against the completed rollout, V2 only)
Also checks whether the XD samples whose disagreement persists in all 5 variants are exactly the unperturbed ones.
Output: revision/results/e4/e4_decomposition.json
"""
import csv, hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
E4 = ROOT / "revision/results/e4"
MAN = ROOT / "revision/manifests/phase7_e4.sha256"
FROZEN = ROOT / "results/derived/malicious_predictions_strict.csv"
FROZEN_SHA = "1a2b0e104eb497d7ea50e5dadd9834e5795e66ae24fb74d32d6bc19f67b0c04f"
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
man = {l.split(None, 1)[1].strip(): l.split()[0] for l in MAN.read_text().splitlines() if l.strip()}
for rel in ("revision/results/e4/variants.csv", "revision/results/e4/e4_summary.json"):
    if sha(ROOT / rel) != man[rel]: sys.exit(f"hash mismatch vs phase7_e4 manifest: {rel}")
if sha(FROZEN) != FROZEN_SHA: sys.exit("frozen predictions hash mismatch")

orig = {r["id"]: (int(r["v1_prediction"]), int(r["v2_prediction"])) for r in csv.DictReader(open(FROZEN))}
var, unpert = {}, set()
for r in csv.DictReader(open(E4 / "variants.csv")):
    var.setdefault(r["sample_id"], {})[int(r["variant"])] = (int(r["v1"]), int(r["v2"]))
    if r["donor_id"] == "": unpert.add(r["sample_id"])
if set(var) != set(orig) or any(sorted(v) != [1, 2, 3, 4, 5] for v in var.values()): sys.exit("variants.csv incomplete")


def C(rev):
    first = {}
    for s, v in var.items():
        f = next((a for a in range(1, 6) if v[a][rev] == 0), 0)
        if f: first[s] = f
    return [sum(1 for f in first.values() if f <= n) for n in range(1, 6)]


single = {}
for st, rev in (("R0", 0), ("R3", 1)):
    c = C(rev); single[st] = {"C": c, "Delta5": c[4] - c[0]}
summ = json.load(open(E4 / "e4_summary.json"))
mix = {st: summ["simulated"][f"{st}-variant"]["D5"] for st in ("R1", "R2")}
incr = {st: {"mean_Delta5": mix[st]["mean"], "p2.5": mix[st]["p2.5"], "p97.5": mix[st]["p97.5"],
             "increment_vs_R0": mix[st]["mean"] - single["R0"]["Delta5"],
             "increment_vs_R3": mix[st]["mean"] - single["R3"]["Delta5"]} for st in mix}
xd = [s for s, (a, b) in orig.items() if a != b]
persist_all = {s for s in xd if all(var[s][a] == orig[s] for a in range(1, 6))}
unpert_xd = {s for s in xd if s in unpert}
out = {"definitions": "DEV-R1-007", "single_model_fresh_variant": single, "mixed_states": incr,
       "XD_persisting_all5": len(persist_all), "unperturbed_XD": len(unpert_xd),
       "persisting_all5_equals_unperturbed_XD": persist_all == unpert_xd,
       "perturbed_XD_persisting_all5": len(persist_all - unpert_xd),
       "inputs_sha256": {k: man[k] for k in ("revision/results/e4/variants.csv", "revision/results/e4/e4_summary.json")}}
p = E4 / "e4_decomposition.json"
if p.exists(): sys.exit("e4_decomposition.json already exists")
json.dump(out, open(p, "w"), indent=1, sort_keys=True)
print("single model (pure variation):", {k: v for k, v in single.items()})
print("mixed:", {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in incr.items()})
print("persisting all 5 == unperturbed XD:", out["persisting_all5_equals_unperturbed_XD"], "| perturbed XD persisting all 5:", out["perturbed_XD_persisting_all5"])
print("wrote", p, sha(p))
