"""Revision r1, Phase 4, S2 + E1 distributions (pre-registration Sections 3 and 8; DEV-R1-001, DEV-R1-002).

Exact shortcut: only the 149 disagreement samples can change outcome under routing, so C(n), K12, K21 need only
their requests. The realized V2 share (needed for d) is computed by hashing every request, in the mixed
single-attempt cells only. Request schedule = sample-major, attempts consecutive (validated in E1, 40/40).
Step 0 reproduces all 40 observed runs with this shortcut; any mismatch stops the script before any distribution.

Outputs (revision/results/e1_distributions/):
  identities.txt                 every simulated identity, written and hashed before computation
  cell_<state>_<attacker>.csv    one row per simulated identity
  detect_W<w2>.csv               single-attempt runs at extra canary weights (detectability)
  e1_summary.json                cell summaries, observed-run percentiles, Gate 4 vs simulator, canary sizing,
                                 detectability, rollout-schedule exposure
"""
import csv, hashlib, json, math, os, pathlib, sys
from multiprocessing import Pool

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "revision/results/e1_distributions"
VAL = ROOT / "revision/results/e1_validation/validation_report.json"
VAL_SHA = "57427065fc8c63447392a3c2f14e29fa832ff04ecac55d0ca2f03f00725cb496"
STATES = {"R0": 0.0, "R1": 0.1, "R2": 0.5, "R3": 1.0}
DETECT_W = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
NSIM = 10_000
SCHEDULE_W = (0.1, 0.5, 1.0); SCHEDULE_SPLITS = [(5, 0, 0), (3, 2, 0), (1, 2, 2)]

G = {}  # worker globals


def init(basis, n, x12_idx, x21_idx):
    G.update(basis=basis, n=n, x12=x12_idx, x21=x21_idx)


def u(prefix, seq):
    return int.from_bytes(hashlib.sha256(prefix + b"%d" % seq).digest()[:8], "big") / 2**64


def a0_task(args):
    run_id, w2, full = args
    pre = f"{G['basis']}|{run_id}|".encode()
    k12 = sum(u(pre, i + 1) < w2 for i in G["x12"])
    k21 = sum(u(pre, i + 1) < w2 for i in G["x21"])
    share = None
    if full:
        if w2 == 0.0: share = 0.0
        elif w2 == 1.0: share = 1.0
        else: share = sum(u(pre, s) < w2 for s in range(1, G["n"] + 1)) / G["n"]
    return run_id, k12, k21, share


def a1_task(args):
    run_id, w2 = args
    pre = f"{G['basis']}|{run_id}|".encode()
    first = []  # first attempt (1..5) at which the sample is evaded, or 0
    k12 = k21 = 0
    for setname, idxs in (("X12", G["x12"]), ("X21", G["x21"])):
        for i in idxs:
            ev = 0
            for a in range(1, 6):
                rv2 = u(pre, 5 * i + a) < w2
                if a == 1 and rv2:
                    if setname == "X12": k12 += 1
                    else: k21 += 1
                if ev == 0 and ((setname == "X12" and rv2) or (setname == "X21" and not rv2)):
                    ev = a
            first.append(ev)
    return run_id, k12, k21, [sum(1 for f in first if 0 < f <= n) for n in range(1, 6)]


def pct(xs, q):
    xs = sorted(xs); k = (len(xs) - 1) * q; f = math.floor(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summ(xs):
    return {"mean": sum(xs) / len(xs), "p2.5": pct(xs, 0.025), "p97.5": pct(xs, 0.975), "min": min(xs), "max": max(xs)}


def rank(xs, v):  # mid-rank percentile of v within xs
    lo = sum(1 for x in xs if x < v); eq = sum(1 for x in xs if x == v)
    return (lo + eq / 2) / len(xs)


def main():
    sys.path.insert(0, str(HERE.parent / "simulator"))
    from simulate import Simulator
    if hashlib.sha256(VAL.read_bytes()).hexdigest() != VAL_SHA:
        sys.exit("validation report hash mismatch")
    sim = Simulator()
    pos = {sid: i for i, sid in enumerate(sim.order)}
    x12 = sorted(pos[s] for s in sim.sets["X12"]); x21 = sorted(pos[s] for s in sim.sets["X21"])
    NB, N12, N21, NM = len(sim.sets["both"]), len(x12), len(x21), len(sim.order)
    M0, M3 = NB + N21, NB + N12
    OUT.mkdir(parents=True, exist_ok=True)
    workers = os.cpu_count() or 1
    pool = Pool(workers, initializer=init, initargs=(sim.basis, NM, x12, x21))

    def a0_row(rid, k12, k21, share):
        c1 = NB + k12 + (N21 - k21)
        d = None if share is None else c1 - ((1 - share) * M0 + share * M3)
        return {"run_id": rid, "C1": c1, "K12": k12, "K21": k21, "share": share, "d": d}

    def a1_row(rid, k12, k21, C):
        C = [NB + c for c in C]
        return {"run_id": rid, "C1": C[0], "C5": C[4], **{f"D{n}": C[n - 1] - C[0] for n in range(2, 6)}, "K12": k12, "K21": k21}

    # ---------- step 0: the shortcut must reproduce the 40 observed runs ----------
    rep = json.load(open(VAL))["runs"]; ident = {x["run_id"]: x for x in sim.freeze["identities"]}
    obs = {}; bad = []
    for rid in sorted(rep):
        st, at = ident[rid]["rollout_state"], ident[rid]["attacker"]; w2 = STATES[st]; R = rep[rid]
        if at == "A0":
            r = a0_row(*pool.apply(a0_task, ((rid, w2, True),)))
            ok = r["C1"] == R["C"]["1"] and r["K12"] == R["K12"] and r["K21"] == R["K21"] and \
                abs(r["share"] - R["realized_v2_share"]) < 1e-12 and abs(r["d"] - R["d"]) < 1e-9
        else:
            r = a1_row(*pool.apply(a1_task, ((rid, w2),)))
            ok = all(r["C1"] + (r[f"D{n}"] if n > 1 else 0) == R["C"][str(n)] for n in range(1, 6)) and \
                r["K12"] == R["K12"] and r["K21"] == R["K21"]
        obs[rid] = (st, at, r)
        if not ok: bad.append(rid)
    print(f"step 0: shortcut reproduces {40 - len(bad)}/40 observed runs")
    if bad or len(rep) != 40:
        sys.exit(f"STOP: shortcut mismatch for {bad}")

    # ---------- identities (written and hashed before any simulated outcome) ----------
    cells = [(s, a) for s in STATES for a in ("A0", "A1")]
    ids = {c: [f"SIM-P1-{c[0]}-{c[1]}-{i:05d}" for i in range(1, NSIM + 1)] for c in cells}
    dids = {w: [f"SIM-P1-W{w:.2f}-A0-{i:05d}" for i in range(1, NSIM + 1)] for w in DETECT_W}
    idf = OUT / "identities.txt"
    with open(idf, "w") as f:
        for c in cells: f.write("\n".join(ids[c]) + "\n")
        for w in DETECT_W: f.write("\n".join(dids[w]) + "\n")
    print("identities written:", sum(map(len, ids.values())) + sum(map(len, dids.values())),
          hashlib.sha256(idf.read_bytes()).hexdigest())

    # ---------- cell distributions ----------
    summary = {"cells": {}, "observed_percentiles": {}, "gate4_vs_simulator": {}}
    for st, at in cells:
        w2 = STATES[st]
        if at == "A0":
            rows = [a0_row(*r) for r in pool.imap(a0_task, [(rid, w2, True) for rid in ids[(st, at)]], chunksize=50)]
            keys = ["C1", "K12", "K21", "share", "d"]
        else:
            rows = [a1_row(*r) for r in pool.imap(a1_task, [(rid, w2) for rid in ids[(st, at)]], chunksize=200)]
            keys = ["C1", "C5", "D2", "D3", "D4", "D5", "K12", "K21"]
        with open(OUT / f"cell_{st}_{at}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["run_id"] + keys); wr.writeheader(); wr.writerows(rows)
        summary["cells"][f"{st}-{at}"] = {k: summ([r[k] for r in rows]) for k in keys}
        for rid, (ost, oat, r) in obs.items():
            if (ost, oat) == (st, at):
                summary["observed_percentiles"][rid] = {k: rank([x[k] for x in rows], r[k]) for k in keys if k != "share"}
        if at == "A0" and st in ("R1", "R2"):
            ds = [r["d"] for r in rows]; means = [sum(ds[i:i + 5]) / 5 for i in range(0, NSIM, 5)]
            dbar = sum(obs[rid][2]["d"] for rid in obs if obs[rid][0] == st and obs[rid][1] == "A0") / 5
            summary["gate4_vs_simulator"][st] = {"observed_dbar": dbar, "groups_of_5": len(means),
                "p_two_sided": sum(1 for m in means if abs(m) >= abs(dbar)) / len(means),
                "dbar_2.5": pct(means, 0.025), "dbar_97.5": pct(means, 0.975)}
        print(f"cell {st}-{at} done")

    # ---------- detectability at canary weights (single attempt) ----------
    summary["detectability"] = {}
    for w in DETECT_W:
        rows = [a0_row(*r) for r in pool.imap(a0_task, [(rid, w, False) for rid in dids[w]], chunksize=200)]
        with open(OUT / f"detect_W{w:.2f}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["run_id", "C1", "K12", "K21"]); wr.writeheader()
            wr.writerows({k: r[k] for k in ("run_id", "C1", "K12", "K21")} for r in rows)
        law_gain5 = N12 * ((1 - (1 - w) ** 5) - w) + N21 * ((1 - w ** 5) - (1 - w))
        summary["detectability"][f"{w:.2f}"] = {"share_runs_C1_le_M_R0": sum(r["C1"] <= M0 for r in rows) / NSIM,
                                               "expected_retry_gain_n5": law_gain5}
    pool.close(); pool.join()

    # ---------- analytic: canary sizing and rollout schedule ----------
    summary["canary_sizing_expected_gain"] = {f"{w:.2f}": {str(n): N12 * ((1 - (1 - w) ** n) - w) + N21 * ((1 - w ** n) - (1 - w))
                                                           for n in range(1, 11)} for w in DETECT_W}
    summary["schedule"] = {}
    for sp in SCHEDULE_SPLITS:
        p12 = 1 - math.prod((1 - w) ** k for w, k in zip(SCHEDULE_W, sp))
        p21 = 1 - math.prod(w ** k for w, k in zip(SCHEDULE_W, sp))
        summary["schedule"][str(sp)] = {"P_X12_reaches_V2": p12, "P_X21_reaches_V1": p21, "expected_C": NB + N12 * p12 + N21 * p21}
    summary["constants"] = {"N": NM, "X12": N12, "X21": N21, "both": NB, "M_R0": M0, "M_R3": M3, "NSIM": NSIM,
                            "schedule_weights": SCHEDULE_W, "workers": workers}
    out = OUT / "e1_summary.json"; json.dump(summary, open(out, "w"), indent=1, sort_keys=True)
    for k, v in summary["cells"].items():
        print(k, {m: (round(s["mean"], 2), round(s["p2.5"], 2), round(s["p97.5"], 2)) for m, s in v.items() if m in ("C1", "C5", "D5", "d")})
    print("gate4 vs simulator:", summary["gate4_vs_simulator"])
    print("detectability:", {w: round(v["share_runs_C1_le_M_R0"], 3) for w, v in summary["detectability"].items()})
    print("wrote", out, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
