"""Revision r1, Phase 6, step 2: E3 routing-policy simulation and live-run predictions (pre-registration Section 5).
Run only after revision/prereg/E3_FREEZE.json is committed and timestamped.

Uses the validated simulator (revision/simulator/simulate.py, 40/40 exact) for routing keys and scoring.
Distributions use an exact shortcut: only X12 and X21 samples can change outcome, so C(n), K12 and K21 need only
their requests. The realised V2 share (needed for d in A0 cells) hashes every request.

Step 0 checks the shortcut against the full simulator on the first 3 identities of every cell. Any mismatch stops
the run before any distribution is computed.

Outputs (revision/results/e3_sim/):
  cell_<policy>_<state>_<attacker>.csv   one row per simulated identity
  live_predictions.json                  full-simulator prediction for each of the 24 live identities (12 primary,
                                         12 reserve), with SHA-256 of the per-request revision and decision sequences
  e3_summary.json                        per-cell mean, 2.5/97.5 percentiles, min, max
"""
import csv, hashlib, json, math, os, pathlib, sys
from multiprocessing import Pool

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE = ROOT / "revision/prereg/E3_FREEZE.json"; FREEZE_SHA = "4623d2e75ea6d2615f50601d237ecb8c1ec54ae031ee9c61e4fa7400178c4706"
IDS = ROOT / "revision/prereg/E3_SIM_IDENTITIES.txt"
SIMPY = ROOT / "revision/simulator/simulate.py"; SIMPY_SHA = "56f019dc873b558f7f95d990ae020b3e29a195de3d6ccf8cfb1281c7081074de"
VAL = ROOT / "revision/results/e1_validation/validation_report.json"; VAL_SHA = "57427065fc8c63447392a3c2f14e29fa832ff04ecac55d0ca2f03f00725cb496"
OUT = ROOT / "revision/results/e3_sim"
G = {}


def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def u(material): return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") / 2**64


def init(g): G.update(g)


def key(policy, run_id, sid, a, attacker):  # identical to Simulator.material, minus the basis|run_id| prefix
    if policy == "P2": return f"{run_id}-S{sid}-A{a}" if attacker == "A1-multi" else f"{run_id}-S{sid}"
    if policy == "P3": return G["vh"][sid]
    raise ValueError(policy)


def task(args):
    run_id, policy, attacker, w2, need_share = args
    pre = f"{G['basis']}|{run_id}|"; A = 1 if attacker == "A0" else 5
    first, k12, k21 = [], 0, 0
    for setname, sids in (("X12", G["x12"]), ("X21", G["x21"])):
        for sid in sids:
            ev = 0
            for a in range(1, A + 1):
                rv2 = u(pre + key(policy, run_id, sid, a, attacker)) < w2
                if a == 1 and rv2:
                    if setname == "X12": k12 += 1
                    else: k21 += 1
                if ev == 0 and (rv2 if setname == "X12" else not rv2): ev = a
            first.append(ev)
    share = None
    if need_share:
        if w2 in (0.0, 1.0): share = w2
        elif policy == "P3": share = sum(c for vh, c in G["vcount"].items() if u(pre + vh) < w2) / G["n"]
        else: share = sum(u(pre + key(policy, run_id, sid, 1, attacker)) < w2 for sid in G["order"]) / G["n"]
    return run_id, k12, k21, [sum(1 for f in first if 0 < f <= m) for m in range(1, A + 1)], share


def pct(xs, q):
    xs = sorted(xs); k = (len(xs) - 1) * q; f = math.floor(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summ(xs): return {"mean": sum(xs) / len(xs), "p2.5": pct(xs, 0.025), "p97.5": pct(xs, 0.975), "min": min(xs), "max": max(xs)}


def main():
    for p, h in ((FREEZE, FREEZE_SHA), (SIMPY, SIMPY_SHA), (VAL, VAL_SHA)):
        if sha(p) != h: sys.exit(f"hash mismatch: {p}")
    fz = json.load(open(FREEZE))
    if sha(IDS) != fz["sim_identities_sha256"]: sys.exit("simulated identity list does not match the freeze")
    lines = IDS.read_text().split("\n")[:-1]
    sys.path.insert(0, str(SIMPY.parent)); from simulate import Simulator
    sim = Simulator(); W = fz["weights"]
    NB, N12, N21, N = len(sim.sets["both"]), len(sim.sets["X12"]), len(sim.sets["X21"]), len(sim.order)
    M0, M3 = NB + N21, NB + N12
    vcount = {}
    for sid in sim.order: vcount[sim.vec_hash[sid]] = vcount.get(sim.vec_hash[sid], 0) + 1
    g = {"basis": sim.basis, "x12": sorted(sim.sets["X12"]), "x21": sorted(sim.sets["X21"]), "vh": sim.vec_hash,
         "vcount": vcount, "order": sim.order, "n": N}
    init(g); pool = Pool(os.cpu_count() or 1, initializer=init, initargs=(g,))
    cells = [tuple(c) for c in fz["sim_cells"]]
    ids = {}
    for p, s, a in cells:
        pre = f"SIM-{p}-{s}-{a}-"; ids[(p, s, a)] = [x for x in lines if x.startswith(pre)]
        if len(ids[(p, s, a)]) != fz["sim_per_cell"]: sys.exit(f"identity count mismatch {p}-{s}-{a}")

    def row(p, s, a, r):
        rid, k12, k21, C, share = r; C = [NB + c for c in C]
        if a == "A0":
            return {"run_id": rid, "C1": C[0], "K12": k12, "K21": k21, "share": share,
                    "d": None if share is None else C[0] - ((1 - share) * M0 + share * M3)}
        return {"run_id": rid, "C1": C[0], "C5": C[4], **{f"D{m}": C[m - 1] - C[0] for m in range(2, 6)}, "K12": k12, "K21": k21}

    # ---- step 0: shortcut == full simulator, first 3 identities of every cell ----
    bad = []
    for p, s, a in cells:
        for rid in ids[(p, s, a)][:3]:
            full = sim.run(rid, s, a, p)
            r = row(p, s, a, task((rid, p, a, W[s], a == "A0")))
            ok = r["C1"] == full["C"][1] and r["K12"] == full["K12"] and r["K21"] == full["K21"]
            if a == "A0": ok = ok and abs(r["share"] - full["realized_v2_share"]) < 1e-12 and abs(r["d"] - full["d"]) < 1e-9
            else: ok = ok and all(r["C1"] + (r[f"D{m}"] if m > 1 else 0) == full["C"][m] for m in range(1, 6))
            if not ok: bad.append(rid)
    print(f"step 0: shortcut equals full simulator for {3 * len(cells) - len(bad)}/{3 * len(cells)} identities", flush=True)
    if bad: sys.exit(f"STOP: shortcut mismatch {bad}")

    # ---- live predictions (full simulator, every request) ----
    OUT.mkdir(parents=True, exist_ok=True); live = {}
    for cell, grp in fz["live_identities"].items():
        p, s, a = cell.split("-", 2)
        for rid in grp["primary"] + grp["reserve"]:
            r = sim.run(rid, s, a, p, keep_requests=True); det = r.pop("requests_detail")
            r["revision_sequence_sha256"] = hashlib.sha256("".join("2" if x[4] == "v2" else "1" for x in det).encode()).hexdigest()
            r["decision_sequence_sha256"] = hashlib.sha256("".join(str(x[5]) for x in det).encode()).hexdigest()
            r["Delta5"] = r["C"][5] - r["C"][1]
            r["predicted_Delta5_zero"] = r["Delta5"] == 0
            live[rid] = r
    lp = OUT / "live_predictions.json"; json.dump(live, open(lp, "w"), indent=1, sort_keys=True)
    for rid in fz["live_execution_order"]:
        r = live[rid]; print(f"live {rid}: C1={r['C'][1]} C5={r['C'][5]} Delta5={r['Delta5']} K12={r['K12']} K21={r['K21']} share={r['realized_v2_share']:.4f}")
    print("wrote", lp, sha(lp), flush=True)

    # ---- distributions ----
    summary = {"cells": {}, "constants": {"N": N, "X12": N12, "X21": N21, "both": NB, "M_R0": M0, "M_R3": M3}}
    for p, s, a in cells:
        rows = [row(p, s, a, r) for r in pool.imap(task, [(rid, p, a, W[s], a == "A0") for rid in ids[(p, s, a)]], chunksize=50)]
        keys = [k for k in rows[0] if k != "run_id"]
        with open(OUT / f"cell_{p}_{s}_{a}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["run_id"] + keys); wr.writeheader(); wr.writerows(rows)
        summary["cells"][f"{p}-{s}-{a}"] = {k: summ([r[k] for r in rows]) for k in keys if rows[0][k] is not None}
        print(f"cell {p}-{s}-{a} done", flush=True)
    pool.close(); pool.join()
    summary["live_predictions_sha256"] = sha(lp)
    out = OUT / "e3_summary.json"; json.dump(summary, open(out, "w"), indent=1, sort_keys=True)
    for c, v in summary["cells"].items():
        print(c, {m: (round(x["mean"], 2), round(x["p2.5"], 2), round(x["p97.5"], 2)) for m, x in v.items() if m in ("C1", "C5", "D5", "d")})
    print("wrote", out, sha(out))


if __name__ == "__main__":
    main()
