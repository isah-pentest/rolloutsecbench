#!/usr/bin/env python3
"""
RolloutSecBench - EXPLORATORY (post hoc) routing-null check. NOT part of the confirmatory analysis.

Question: are the two unusual mixed-state observations properties of the frozen routing RULE, or of the
particular run identities that were executed?
  (1) Gate 4, R1-A0: mean d = -4.71 samples (binomial-null p = 0.0042)
  (2) R2-A1: Delta(5) = C(5) - C(1) varies far less across the 5 runs than binomial routing predicts
Method (offline; no traffic, no evidence written):
  Part A  re-derive every valid run with the frozen predict() of make_v3_freeze.py (same basis, same frozen
          order and predictions) and require exact equality with per_run.csv. This re-verifies the inputs.
  Part B  apply the same predict() to M hypothetical run identities per mixed cell ("EXPL-<cell>-#####",
          never executed) and build the empirical distribution of the statistics under the frozen rule:
            A0 cells: per-run d_r (Gate-4 definition, realized share) and the 5-run mean d
            A1 cells: per-run C(1), C(5), Delta(5), their SDs and correlation, and the 5-run SD of Delta(5)
          Empirical p-values for the observed 5-run statistics come from resampling 5 of the M runs
          (R draws, fixed seed).
If the empirical routing null reproduces the binomial one, the observations are attributable to the
particular realizations, not to the rule.

Inputs are verified: frozen input hashes (make_v3_freeze.FROZEN_INPUTS), analysis manifest for per_run.csv.
Outputs (--out-dir, new folder): routing_null.json, routing_null_runs.csv, provenance.json

Usage (repo root; Part B takes a few minutes per 1,000 identities):
  .venv-blueprint-v2/bin/python analysis/exploratory/routing_null_check.py \\
      --analysis-dir results/derived/phase9_v3/analysis_20261022T200702Z \\
      --out-dir results/derived/phase9_v3/exploratory/routing_null --m 2000
"""
import argparse
import csv
import hashlib
import json
import platform
import random
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
SEED = 2026102203
R_DRAWS = 200_000


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stop(msg):
    print(f"REFUSING: {msg}", file=sys.stderr)
    sys.exit(2)


def load_rule(root):
    sys.path.insert(0, str(root / "workloads"))
    import make_v3_freeze as m  # frozen predict(), route(), BASIS, STATES, ATTEMPTS
    import xm_loader
    xm_loader.ROOT = root
    for rel, h in m.FROZEN_INPUTS.items():
        if xm_loader._sha256(root / rel) != h:
            stop(f"frozen input hash mismatch: {rel}")
    samples, _ = xm_loader.load_xm(root)
    order = [s["sample_id"] for s in samples]
    frozen = {}
    with open(root / xm_loader.FROZEN_XM, newline="") as fh:
        for r in csv.DictReader(fh):
            frozen[r["id"]] = {"v1": int(r["v1_prediction"]), "v2": int(r["v2_prediction"])}
    return m, order, frozen


def stats_for(m, rid, state, attacker, order, frozen, M0, M3):
    o = m.predict(rid, state, attacker, order, frozen)
    w = o["realized_v2_share"]
    row = {"run_id": rid, "state": state, "attacker": attacker, "w2_obs": w, "C1": o["evasions_attempt1"]}
    if attacker == "A0":
        row["d"] = o["evasions_attempt1"] - ((1 - w) * M0 + w * M3)
    else:
        k = m.ATTEMPTS["A1"]
        row["CK"] = o["cumulative_evasions_by_n"][str(k)]
        row["DK"] = row["CK"] - row["C1"]
    return row


def resample_p(pool, stat, observed, tail, rng, draws=R_DRAWS, k=5):
    hits = 0
    for _ in range(draws):
        s = stat(rng.sample(pool, k))
        hits += (s <= observed + 1e-12) if tail == "low" else (abs(s) >= abs(observed) - 1e-12)
    return (hits + 1) / (draws + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--out-dir", default="results/derived/phase9_v3/exploratory/routing_null")
    ap.add_argument("--m", type=int, default=2000, help="hypothetical identities per mixed cell")
    a = ap.parse_args()
    adir, out = (ROOT / a.analysis_dir).resolve(), (ROOT / a.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        stop(f"output folder {out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    man = dict(reversed(l.split(None, 1)) for l in (adir / "ANALYSIS_MANIFEST.sha256").read_text().splitlines())
    man = {k.strip(): v for k, v in man.items()}
    if sha(adir / "per_run.csv") != man.get("per_run.csv"):
        stop("per_run.csv does not match the analysis manifest")
    m, order, frozen = load_rule(ROOT)
    M0 = sum(1 for s in order if frozen[s]["v1"] == 0)
    M3 = sum(1 for s in order if frozen[s]["v2"] == 0)

    # ---------------- Part A: re-derive the valid runs
    rows = list(csv.DictReader(open(adir / "per_run.csv")))
    K = m.ATTEMPTS["A1"]
    for r in rows:
        x = stats_for(m, r["run_id"], r["rollout_state"], r["attacker"], order, frozen, M0, M3)
        ok = x["C1"] == int(r["C1"]) and abs(x["w2_obs"] - float(r["realized_v2_share"])) < 1e-12
        if r["attacker"] == "A1":
            ok = ok and x["CK"] == int(r[f"C{K}"])
        if not ok:
            stop(f"{r['run_id']}: frozen rule does not reproduce per_run.csv")
    print(f"PART A PASS: all {len(rows)} valid runs re-derived exactly from the frozen rule")
    obs = {}
    for cell in ("R1-A0", "R2-A0", "R1-A1", "R2-A1"):
        st_, at = cell.split("-")
        xs = [stats_for(m, r["run_id"], st_, at, order, frozen, M0, M3) for r in rows if r["cell"] == cell]
        obs[cell] = xs

    # ---------------- Part B: empirical routing null
    sims, summary = [], {}
    rng = random.Random(SEED)
    for cell in ("R1-A0", "R2-A0", "R1-A1", "R2-A1"):
        st_, at = cell.split("-")
        pool = []
        for i in range(a.m):
            pool.append(stats_for(m, f"EXPL-{cell}-{i:05d}", st_, at, order, frozen, M0, M3))
            if (i + 1) % 250 == 0:
                print(f"  {cell}: {i + 1}/{a.m}", flush=True)
        sims += pool
        if at == "A0":
            ds = [p["d"] for p in pool]
            od = st.fmean(x["d"] for x in obs[cell])
            summary[cell] = {"single_run_sd_d": st.stdev(ds), "mean_d_null": st.fmean(ds),
                             "observed_mean_d": od,
                             "p_two_sided_5run_mean": resample_p(ds, st.fmean, od, "two", rng)}
        else:
            c1, ck, dk = [p["C1"] for p in pool], [p["CK"] for p in pool], [p["DK"] for p in pool]
            osd = st.stdev([x["DK"] for x in obs[cell]])
            summary[cell] = {"single_run_sd_C1": st.stdev(c1), f"single_run_sd_C{K}": st.stdev(ck),
                             f"single_run_sd_D{K}": st.stdev(dk), f"corr_C1_C{K}": st.correlation(c1, ck),
                             f"mean_D{K}_null": st.fmean(dk), f"observed_5run_sd_D{K}": osd,
                             f"observed_mean_D{K}": st.fmean(x["DK"] for x in obs[cell]),
                             f"p_5run_sd_D{K}_as_small": resample_p(dk, st.stdev, osd, "low", rng)}
    res = {"status": "EXPLORATORY, post hoc; not confirmatory", "m_per_cell": a.m, "resample_draws": R_DRAWS,
           "seed": SEED, "M_R0": M0, "M_R3": M3, "summary": summary}
    (out / "routing_null.json").write_text(json.dumps(res, indent=2) + "\n")
    with open(out / "routing_null_runs.csv", "w", newline="") as f:
        keys = sorted({k for s in sims for k in s})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sims)
    prov = {"generated_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__)),
            "python": platform.python_version(), "per_run_sha256": man["per_run.csv"],
            "make_v3_freeze_sha256": sha(ROOT / "workloads/make_v3_freeze.py"),
            "outputs_sha256": {p: sha(out / p) for p in ("routing_null.json", "routing_null_runs.csv")}}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
