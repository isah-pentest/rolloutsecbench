"""Revision r1, Phase 7, E4 attacker realism (pre-registration Section 6; operational definitions DEV-R1-006).
Offline. Uses the recorded A1 logs, the validated simulator, and the frozen preprocessor and models.
Run with ~/.venvs/rsb-revision/bin/python.

Step 0 (all must hold, otherwise STOP before any estimand):
  0a  scoring X_m with the frozen preprocessor + models reproduces every frozen prediction exactly
  0b  the per-sample attempt machinery reproduces C(1..5) of all 20 valid A1 runs (validation report)
  0c  the simulation shortcut reproduces C(1..5) of the same 20 runs from their run ids
Estimands
  E4.1 detection cost k in {1,2,3,5}: from the 20 valid A1 logs, and simulated (P1, R1 and R2, the S2 identities
       SIM-P1-{R1,R2}-A1-00001..10000)
  E4.2 early stopping: mean attempts per sample and C(5), from the same logs and simulation
  E4.3 near-duplicate replay: variants, persistence of XD, new disagreements, and Delta(5) with a fresh variant
       per attempt (P1, R1 and R2, the same S2 identities)
  E4.4 campaign view (derived): P(at least one of m flows of a family is missed), m in {1,5,10,50}, A0 and A1, R0-R3
Outputs: revision/results/e4/
"""
import csv, hashlib, json, math, os, pathlib, sys
from multiprocessing import Pool
import numpy as np, pandas as pd, joblib
from xgboost import XGBClassifier

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "revision/results/e4"
VAL = ROOT / "revision/results/e1_validation/validation_report.json"; VAL_SHA = "57427065fc8c63447392a3c2f14e29fa832ff04ecac55d0ca2f03f00725cb496"
E2S = ROOT / "revision/results/e2/e2_summary.json"; E2S_SHA = "02c94f55787f84885460399813f6277147e9217e96d175c61a6afe7b9e60b38d"
IDS = ROOT / "revision/results/e1_distributions/identities.txt"; IDS_MANIFEST = ROOT / "revision/manifests/phase4_s2.sha256"
PRE = {"models/v1/model.json": "6c0aa79af5568231202f2aa150e2a388855a058601333cbdb067b54b097ac14c",
       "models/v2/model.json": "ea66463097938adf80145a872882dbb45cb8fdb3ffabdde5b23b633b586ca874",
       "models/preprocessing/preprocessor.joblib": "4ea29c9353505b423ce37b077f3ef821f76da426230518d771ebe217e3c2408f"}
PERTURB = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload", "sinpkt", "dinpkt", "sjit", "djit",
           "tcprtt", "synack", "ackdat", "smean", "dmean"]
SEED, NVAR, KS, STATES, MS = 11242026, 5, (1, 2, 3, 5), {"R0": 0.0, "R1": 0.1, "R2": 0.5, "R3": 1.0}, (1, 5, 10, 50)
G = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def u(prefix, seq): return int.from_bytes(hashlib.sha256(prefix + b"%d" % seq).digest()[:8], "big") / 2**64


def outcome(decs, k=5):
    """decs: decisions (1 detected, 0 missed) of a sample's attempts in order. Returns (first evasion attempt or 0,
    attempts used under early stopping, evaded under detection cost k)."""
    first = next((a for a, d in enumerate(decs, 1) if d == 0), 0)
    det_before = 0; ev_k = False
    for d in decs:
        if d == 0: ev_k = True; break
        det_before += 1
        if det_before >= k: break
    return first, (first if first else len(decs)), ev_k


def init(g): G.update(g)


def sim_task(args):
    """One identity (P1, A1 schedule). For each routing-relevant sample, route its 5 attempts and take the decision
    of the routed revision for the payload of that attempt. Returns per-k evasions, early-stop attempts, C(1..5)."""
    run_id, w2, payload = args            # payload: "orig" or "variant"
    pre = f"{G['basis']}|{run_id}|".encode()
    C = [0] * 5; Ck = {k: 0 for k in KS}; att = 0
    for i, d1, d2 in G["rel"][payload]:   # i = position in order; d1/d2 = 5 decisions of V1/V2 per attempt
        decs = [(d2[a] if u(pre, 5 * i + a + 1) < w2 else d1[a]) if d1[a] != d2[a] else d1[a] for a in range(5)]
        first, used, _ = outcome(decs)
        att += used
        for n in range(5):
            C[n] += 0 < first <= n + 1
        for k in KS:
            Ck[k] += outcome(decs, k)[2]
    c0 = G["const"][payload]               # deterministic samples
    return run_id, [c0["C"][n] + C[n] for n in range(5)], {k: c0["Ck"][k] + Ck[k] for k in KS}, (c0["att"] + att) / G["n"]


def pct(xs, q):
    k = (len(xs) - 1) * q; f = math.floor(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summ(xs):
    xs = sorted(xs); q = lambda p: pct(xs, p)
    return {"mean": sum(xs) / len(xs), "p2.5": q(0.025), "p97.5": q(0.975), "min": xs[0], "max": xs[-1]}


def main():
    for p, h in ((VAL, VAL_SHA), (E2S, E2S_SHA), *((ROOT / k, v) for k, v in PRE.items())):
        if sha(p) != h: sys.exit(f"hash mismatch: {p}")
    ids_sha = next(l.split()[0] for l in IDS_MANIFEST.read_text().splitlines() if l.rstrip().endswith("e1_distributions/identities.txt"))
    if sha(IDS) != ids_sha: sys.exit("S2 identities.txt does not match its committed manifest")
    e2 = json.load(open(E2S))
    if sha(ROOT / "data/splits/final_train.csv") != e2["inputs_sha256"]["data/splits/final_train.csv"]: sys.exit("final_train hash mismatch")
    sys.path.insert(0, str(HERE.parent / "simulator")); from simulate import Simulator
    sim = Simulator(); order = sim.order; N = len(order); pos = {s: i for i, s in enumerate(order)}
    names = json.load(open(ROOT / "data/processed/feature_names.json"))["source_feature_order"]
    if not set(PERTURB) <= set(names): sys.exit(f"perturbed features missing: {sorted(set(PERTURB) - set(names))}")
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- 0a frozen scoring ----------
    pre = joblib.load(ROOT / "models/preprocessing/preprocessor.joblib")
    m1, m2 = XGBClassifier(), XGBClassifier(); m1.load_model(ROOT / "models/v1/model.json"); m2.load_model(ROOT / "models/v2/model.json")
    te = pd.read_csv(ROOT / "data/splits/strict_test_nonoverlap.csv"); te["id"] = te["id"].astype(str)
    te = te.set_index("id").loc[order].reset_index()
    def score(df):
        X = pre.transform(df[names]); return ((m1.predict_proba(X)[:, 1] >= 0.5).astype(int), (m2.predict_proba(X)[:, 1] >= 0.5).astype(int))
    o1, o2 = score(te)
    ok0a = all(int(o1[i]) == sim.pred[s]["v1"] and int(o2[i]) == sim.pred[s]["v2"] for i, s in enumerate(order))
    print("step 0a: frozen scoring reproduces X_m predictions:", ok0a, flush=True)
    if not ok0a: sys.exit("STOP 0a")

    # ---------- E4.1 / E4.2 from the 20 valid A1 logs (+ 0b) ----------
    rep = json.load(open(VAL))["runs"]; fz = {x["run_id"]: x for x in sim.freeze["identities"]}
    raw = ROOT / "results/raw/phase9_v3"; logs = []; bad = []
    for row in csv.DictReader(open(raw / "registry.csv")):
        if row["disposition"] != "VALID" or fz[row["run_id"]]["attacker"] != "A1": continue
        rid = row["run_id"]; per = {}
        for l in open(raw / rid / "requests.jsonl"):
            q = json.loads(l); per.setdefault(str(q["sample_id"]), {})[q["attempt_number"]] = int(q["prediction"])
        res = [outcome([d[a] for a in range(1, 6)], k) for d in per.values() for k in (5,)]
        C = [sum(1 for f, _, _ in res if 0 < f <= n) for n in range(1, 6)]
        if C != [rep[rid]["C"][str(n)] for n in range(1, 6)]: bad.append(rid)
        Ck = {k: sum(outcome([d[a] for a in range(1, 6)], k)[2] for d in per.values()) for k in KS}
        logs.append({"run_id": rid, "state": fz[rid]["rollout_state"], "C1": C[0], "C5": C[4],
                     **{f"C_k{k}": Ck[k] for k in KS}, **{f"gain_k{k}": Ck[k] - C[0] for k in KS},
                     "mean_attempts_early_stop": sum(r[1] for r in res) / len(res)})
    print(f"step 0b: log machinery reproduces C(1..5) for {len(logs) - len(bad)}/{len(logs)} valid A1 runs", flush=True)
    if bad or len(logs) != 20: sys.exit(f"STOP 0b {bad} n={len(logs)}")
    with open(OUT / "detection_cost_logs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(logs[0])); w.writeheader(); w.writerows(logs)

    # ---------- E4.3 variants ----------
    tr = pd.read_csv(ROOT / "data/splits/final_train.csv"); tr = tr[tr["label"] == 1].reset_index(drop=True)
    tr["_k"] = tr["attack_cat"].astype(str) + "|" + tr["proto"].astype(str) + "|" + tr["service"].astype(str) + "|" + tr["state"].astype(str)
    pools = tr.groupby("_k").indices
    rng = np.random.default_rng(SEED); donors = np.full((N, NVAR), -1, dtype=np.int64); unpert = []
    keys = (te["attack_cat"].astype(str) + "|" + te["proto"].astype(str) + "|" + te["service"].astype(str) + "|" + te["state"].astype(str)).tolist()
    for i, s in enumerate(order):
        pool = pools.get(keys[i], np.array([], dtype=np.int64))
        if len(pool) < NVAR: unpert.append(s)
        else: donors[i] = pool[rng.choice(len(pool), NVAR, replace=False)]
    V = te.loc[np.repeat(np.arange(N), NVAR)].reset_index(drop=True); flat = donors.reshape(-1); mask = flat >= 0
    for c in PERTURB:
        col = V[c].to_numpy(dtype=np.float64).copy(); col[mask] = tr[c].to_numpy(dtype=np.float64)[flat[mask]]; V[c] = col
    vmeta = [(s, v + 1, (str(tr.at[int(donors[i, v]), "id"]) if donors[i, v] >= 0 else None)) for i, s in enumerate(order) for v in range(NVAR)]
    v1, v2 = score(V)
    with open(OUT / "variants.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sample_id", "variant", "donor_id", "v1", "v2"])
        for (s, v, dn), a, b in zip(vmeta, v1, v2): w.writerow([s, v, dn or "", int(a), int(b)])
    dec = {s: ([int(v1[5 * i + a]) for a in range(5)], [int(v2[5 * i + a]) for a in range(5)]) for i, s in enumerate(order)}
    lab = lambda a, b: {(1, 0): "X12", (0, 1): "X21", (0, 0): "both", (1, 1): "det"}[(a, b)]
    orig = {s: lab(sim.pred[s]["v1"], sim.pred[s]["v2"]) for s in order}
    xd = [s for s in order if orig[s] in ("X12", "X21")]
    persist = {s: sum(lab(dec[s][0][a], dec[s][1][a]) == orig[s] for a in range(5)) for s in xd}
    new = [(s, a, lab(dec[s][0][a], dec[s][1][a])) for s in order if orig[s] not in ("X12", "X21") for a in range(5)
           if dec[s][0][a] != dec[s][1][a]]
    e43 = {"unperturbed_samples": len(unpert), "unperturbed_XD": sum(orig[s] in ("X12", "X21") for s in unpert),
           "variants": len(vmeta), "XD": len(xd),
           "XD_variant_persistence_share": sum(persist.values()) / (NVAR * len(xd)),
           "XD_samples_persisting_all5": sum(v == NVAR for v in persist.values()),
           "XD_samples_persisting_ge1": sum(v >= 1 for v in persist.values()),
           "new_disagreement_variants": len(new), "new_disagreement_samples": len({s for s, _, _ in new}),
           "new_disagreements_by_direction": {d: sum(1 for *_, x in new if x == d) for d in ("X12", "X21")}}

    # ---------- simulation (0c, E4.1/E4.2 simulated, E4.3 fresh-variant Delta(5)) ----------
    def prep(payload):
        rel, cC, cK, catt = [], [0] * 5, {k: 0 for k in KS}, 0
        for i, s in enumerate(order):
            d1, d2 = ([sim.pred[s]["v1"]] * 5, [sim.pred[s]["v2"]] * 5) if payload == "orig" else dec[s]
            if any(d1[a] != d2[a] for a in range(5)): rel.append((i, d1, d2)); continue
            first, used, _ = outcome(d1)
            catt += used
            for n in range(5): cC[n] += 0 < first <= n + 1
            for k in KS: cK[k] += outcome(d1, k)[2]
        return rel, {"C": cC, "Ck": cK, "att": catt}
    rel_o, c_o = prep("orig"); rel_v, c_v = prep("variant")
    g = {"basis": sim.basis, "n": N, "rel": {"orig": rel_o, "variant": rel_v}, "const": {"orig": c_o, "variant": c_v}}
    init(g)
    bad = [x["run_id"] for x in logs if sim_task((x["run_id"], STATES[x["state"]], "orig"))[1] != [rep[x["run_id"]]["C"][str(n)] for n in range(1, 6)]]
    print(f"step 0c: simulation reproduces C(1..5) for {20 - len(bad)}/20 valid A1 runs", flush=True)
    if bad: sys.exit(f"STOP 0c {bad}")
    lines = IDS.read_text().split("\n")
    pool = Pool(os.cpu_count() or 1, initializer=init, initargs=(g,))
    sims = {}
    for st in ("R1", "R2"):
        ids = [x for x in lines if x.startswith(f"SIM-P1-{st}-A1-")]
        if len(ids) != 10_000: sys.exit(f"identity count {st}")
        for payload in ("orig", "variant"):
            rows = []
            for rid, C, Ck, att in pool.imap(sim_task, [(r, STATES[st], payload) for r in ids], chunksize=100):
                rows.append({"run_id": rid, "C1": C[0], "C5": C[4], "D5": C[4] - C[0], **{f"gain_k{k}": Ck[k] - C[0] for k in KS},
                             "mean_attempts_early_stop": att})
            with open(OUT / f"sim_{st}_{payload}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            sims[f"{st}-{payload}"] = {k: summ([r[k] for r in rows]) for k in rows[0] if k != "run_id"}
            print(f"sim {st} {payload} done", flush=True)
    pool.close(); pool.join()

    # ---------- E4.4 campaign view (derived, exact per-sample probabilities under independent routing) ----------
    fam = te.set_index("id")["attack_cat"].astype(str).to_dict(); camp = {}
    for st, w in STATES.items():
        for att_name, n in (("A0", 1), ("A1", 5)):
            pf = {}
            for s in order:
                o = orig[s]
                p = {"det": 0.0, "both": 1.0, "X12": 1 - (1 - w) ** n, "X21": 1 - w ** n}[o]
                pf.setdefault(fam[s], []).append(p)
            camp[f"{st}-{att_name}"] = {f: {"n_samples": len(v), "p_sample": sum(v) / len(v),
                                            **{f"P_ge1_missed_m{m}": 1 - (1 - sum(v) / len(v)) ** m for m in MS}} for f, v in sorted(pf.items())}

    by_state = lambda k: {st: summ([x[k] for x in logs if x["state"] == st]) for st in sorted({x["state"] for x in logs})}
    summary = {"definitions": "DEV-R1-006", "seed": SEED, "perturbed_features": PERTURB,
               "logs": {"runs": len(logs), **{k: by_state(k) for k in [f"gain_k{k}" for k in KS] + ["C5", "mean_attempts_early_stop"]}},
               "simulated": sims, "near_duplicate": e43, "campaign_derived": camp,
               "notes": ["k = 1 gives zero retry gain by construction (lower bound)",
                         "campaign view assumes independent flows; derived, not observed"]}
    out = OUT / "e4_summary.json"; json.dump(summary, open(out, "w"), indent=1, sort_keys=True)
    print("logs gain by k (mean per state):", {k: {s: round(v['mean'], 2) for s, v in summary['logs'][k].items()} for k in [f"gain_k{k}" for k in KS]})
    print("logs mean attempts (early stop):", {s: round(v['mean'], 3) for s, v in summary['logs']['mean_attempts_early_stop'].items()})
    print("near-duplicate:", json.dumps(e43))
    for c, v in sims.items(): print(c, {k: round(v[k]["mean"], 2) for k in ("D5", "gain_k1", "gain_k2", "gain_k3", "gain_k5", "mean_attempts_early_stop")})
    print("wrote", out, sha(out))


if __name__ == "__main__":
    main()
