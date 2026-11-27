"""Revision r1, Phase 8, E5 mitigations (pre-registration Section 7; operational definitions DEV-R1-008).
Offline; reads committed results only (each checked against its committed manifest). Output: revision/results/e5/

E5.1 directional-disagreement gate (computed): a revision is blocked when |X12| / n_m > t, t in {0.05%, 0.1%, 0.2%,
     0.5%}. Applied to every Gate-1-passing E2 pair, per dataset (UNSW-NB15: n_m = 38,263; CIC-IDS2017: n_m from the
     pair records), and to the original V1->V2 pair. For each t: pairs blocked, and the detections forgone =
     |X21| of the blocked pairs (samples the blocked V2 would have detected and V1 misses).
E5.2 sticky routing: E3 results for P2 (A1-one and A1-multi), copied from e3_summary.json.
E5.3 shortened mixed window: the rollout-schedule exposure from e1_summary.json (Section 3), copied.
E5.4 rate limiting of identical submissions: analytic statement only, not evaluated, labelled as such.
"""
import hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "revision/results/e5"
T = [0.0005, 0.001, 0.002, 0.005]
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def manifest(rel):
    return {l.split(None, 1)[1].strip(): l.split()[0] for l in (ROOT / rel).read_text().splitlines() if l.strip()}


def check(man, rel):
    # phase4_s2.sha256 was written from inside revision/ (paths without the "revision/" prefix)
    want = man.get(rel) or man.get(rel.removeprefix("revision/"))
    if want is None: sys.exit(f"not listed in manifest: {rel}")
    if sha(ROOT / rel) != want: sys.exit(f"hash mismatch vs manifest: {rel}")
    return json.load(open(ROOT / rel))


m2, mc = manifest("revision/manifests/phase5_e2.sha256"), manifest("revision/manifests/cic/phase5b_e2_cic.sha256")
m3, m1 = manifest("revision/manifests/phase6_e3_sim.sha256"), manifest("revision/manifests/phase4_s2.sha256")
unsw = [check(m2, k) for k in sorted(m2) if k.startswith("revision/results/e2/pairs/")]
cic = [check(mc, k) for k in sorted(mc) if k.startswith("revision/results/e2_cic/pairs/")]
step0 = check(m2, "revision/results/e2/step0_reproduction.json")
e3 = check(m3, "revision/results/e3_sim/e3_summary.json")
e1 = check(m1, "revision/results/e1_distributions/e1_summary.json")
if len(unsw) != 48 or len(cic) != 24: sys.exit(f"pair counts {len(unsw)} {len(cic)}")
N_UNSW = 38263; N_CIC = check(mc, "revision/results/e2_cic/e2_cic_summary.json")["malicious_strict_test_rows"]


def gate(pairs, n):
    res = {}
    for t in T:
        blocked = [p for p in pairs if p["X12"] / n > t]
        res[f"{t:.2%}"] = {"threshold_X12": t * n, "pairs_evaluated": len(pairs), "pairs_blocked": len(blocked),
                           "blocked": sorted(p["pair"] for p in blocked),
                           "detections_forgone_total_X21": sum(p["X21"] for p in blocked),
                           "detections_forgone_per_blocked_pair": {p["pair"]: p["X21"] for p in sorted(blocked, key=lambda p: p["pair"])}}
    return res


orig = step0["original_pair"]
e5 = {
    "definitions": "DEV-R1-008",
    "E5.1_directional_gate": {
        "UNSW-NB15": gate([p for p in unsw if p["gate1_pass"]], N_UNSW),
        "CIC-IDS2017": gate([p for p in cic if p["gate1_pass"]], N_CIC),
        "original_pair": {"X12": orig["X12"], "X21": orig["X21"], "X12_share": orig["X12"] / N_UNSW,
                          **{f"blocked_at_{t:.2%}": orig["X12"] / N_UNSW > t for t in T}},
        "n_m": {"UNSW-NB15": N_UNSW, "CIC-IDS2017": N_CIC}},
    "E5.2_sticky_routing_from_E3": {"P2": {c: e3["cells"][c] for c in sorted(e3["cells"]) if c.startswith("P2-")},
                                    "P1_reference_from_S2": {c: e1["cells"][c] for c in sorted(e1["cells"]) if c.endswith("-A1")}},
    "E5.3_shortened_mixed_window_from_schedule": e1["schedule"],
    "E5.4_rate_limiting": {"evaluated": False,
        "statement": "Returning the first decision for any identical feature vector submitted within a window W removes "
                     "the retry gain for exact replays within W, like content-keyed routing (P3, E3). It does not "
                     "address near-duplicate variants (E4.3). Its cost is that legitimate repeated flows with identical "
                     "features also receive the cached decision, and it requires the router to keep per-vector state "
                     "for W. Not evaluated empirically in this study."},
}
OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "e5_summary.json"
if p.exists(): sys.exit("e5_summary.json already exists")
json.dump(e5, open(p, "w"), indent=1, sort_keys=True)
g = e5["E5.1_directional_gate"]
for ds in ("UNSW-NB15", "CIC-IDS2017"):
    print(ds, {t: (v["pairs_blocked"], v["pairs_evaluated"], v["detections_forgone_total_X21"]) for t, v in g[ds].items()})
print("original pair:", g["original_pair"])
print("wrote", p, sha(p))
