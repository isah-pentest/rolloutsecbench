#!/usr/bin/env python3
"""
RolloutSecBench Phase-9 v3 - confirmatory analysis (READ-ONLY on all evidence).

Reads ONLY: the freeze manifest, registry.csv (VALID rows), each valid run's directory, and the frozen
offline predictions. Writes ONLY to a new directory results/derived/phase9_v3/analysis_<UTC>/ (never overwrites).

Stage 0  integrity     5 VALID runs per cell; registry/validation hashes; final_checksums verify; validator and
                       client hashes equal the freeze pins; observed outcome == pre-registered prediction per run
Stage 1  per run       ASR (attempt 1), C(n) n=1..K, retry gain D(n)=C(n)-C(1), realized V2 share
Stage 2  per cell      mean, SD, 95% t-CI of every quantity
Stage 3  exposure law  C(n) = |both-miss| + |X12|(1-(1-w)^n) + |X21|(1-w^n); expected curve and the 95% band of a
                       5-run mean, against observed means (A1 cells)
Stage 4  Gate 4        (freeze, primary) per mixed A0 cell: mean d_r, d_r = observed misses - (w1_obs*M_R0 + w2_obs*M_R3);
                       null K12~Bin(|X12|,w_obs), K21~Bin(|X21|,w_obs); two-sided Monte-Carlo p (1e6 draws, seed
                       2026101902); Holm over R1,R2 at alpha 0.05. Secondary (pre-registered): |d_r|/N > MDE per run.
                       Mechanism check: realized V2 share inside X12 and X21 vs global, per run.
Stage 5  latency       P50/P95/P99 per run, pooled by transport path (endpoint), all requests and excluding the
                       first 5 s of each run (warm-up sensitivity)            [--skip-latency to omit]
Stage 6  figure        exposure-law figure (expected vs observed C(n), R1 and R2)  [needs matplotlib]

All population constants (N, |X12|, |X21|, both-miss, misses per revision) are derived from the frozen file,
never hard-coded. Formal H2-H5 tests other than Gate 4 are NOT computed here: apply them exactly as specified
in the blueprint, from per_run.csv.

Usage (repository root):  python3 workloads/analyze_v3.py --freeze data/manifests/phase9_v3/phase9_v3_freeze.json
"""
import argparse
import csv
import hashlib
import json
import math
import statistics as stats
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
BASE = ROOT / "results/raw/phase9_v3"
FROZEN_XM = ROOT / "results/derived/malicious_predictions_strict.csv"
W = {"R0": 0.0, "R1": 0.1, "R2": 0.5, "R3": 1.0}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
MDE = 3.64705466658e-05


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ci(xs):
    n = len(xs)
    m = stats.fmean(xs)
    if n < 2:
        return m, 0.0, m, m
    s = stats.stdev(xs)
    h = T975.get(n - 1, 1.96) * s / math.sqrt(n)
    return m, s, m - h, m + h


def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, prev = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = prev
    return adj, [a < alpha for a in adj]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", default="data/manifests/phase9_v3/phase9_v3_freeze.json")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-checksums", action="store_true", help="skip re-hashing run files (faster; not for final)")
    ap.add_argument("--mc-draws", type=int, default=1_000_000)
    a = ap.parse_args()

    freeze_p = ROOT / a.freeze
    freeze = json.loads(freeze_p.read_text())
    recorded = (freeze_p.parent / (freeze_p.name + ".sha256")).read_text().split()[0]
    if sha(freeze_p) != recorded:
        raise SystemExit("FAIL: freeze file hash mismatch")
    out = ROOT / "results/derived/phase9_v3" / ("analysis_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True, exist_ok=False)
    ids = {i["run_id"]: i for i in freeze["identities"]}
    K = freeze["design"]["attempts"]["A1"]
    problems = []

    # ---------------- frozen population constants ----------------
    frozen, order = {}, []
    for r in csv.DictReader(open(FROZEN_XM)):
        frozen[r["id"]] = (int(r["v1_prediction"]), int(r["v2_prediction"]))
        order.append(r["id"])
    N = len(order)
    X12 = {s for s, (p1, p2) in frozen.items() if p1 == 1 and p2 == 0}   # only V2 misses
    X21 = {s for s, (p1, p2) in frozen.items() if p1 == 0 and p2 == 1}   # only V1 misses
    BM = sum(1 for p1, p2 in frozen.values() if p1 == 0 and p2 == 0)
    M1, M2 = BM + len(X21), BM + len(X12)
    const = {"N": N, "X12_only_v2_misses": len(X12), "X21_only_v1_misses": len(X21), "both_miss": BM,
             "v1_misses": M1, "v2_misses": M2, "MRD": (len(X12) + len(X21)) / N}
    print("population:", const)

    # ---------------- stage 0: integrity ----------------
    reg = list(csv.DictReader(open(BASE / "registry.csv")))
    valid = [r for r in reg if r["disposition"] == "VALID"]
    per_cell = Counter(r["cell"] for r in valid)
    cells = [f"{s}-{x}" for s in W for x in ("A0", "A1")]
    for c in cells:
        if per_cell[c] != freeze["design"]["required_valid_per_cell"]:
            problems.append(f"cell {c} has {per_cell[c]} valid runs")
    pins = freeze["pinned_code_sha256"]
    runs = []
    for r in valid:
        rid = r["run_id"]
        d = BASE / rid
        v = json.loads((d / "validation.json").read_text())
        man = json.loads((d / "run_manifest.json").read_text())
        disp = json.loads((d / "disposition.json").read_text())
        if v["disposition"] != "VALID" or disp["disposition"] != "VALID":
            problems.append(f"{rid}: validation/disposition not VALID")
        if sha(d / "validation.json") != r["validation_sha256"]:
            problems.append(f"{rid}: validation.json hash != registry")
        if v["validator_sha256"] != pins["workloads/validate_v3_run.py"]:
            problems.append(f"{rid}: validator hash != freeze pin")
        if man["client_sha256"] != pins["workloads/v3_client.py"]:
            problems.append(f"{rid}: client hash != freeze pin")
        if not a.skip_checksums:
            for line in (d / "final_checksums.sha256").read_text().splitlines():
                h, name = line.split(None, 1)
                if sha(d / name.strip()) != h:
                    problems.append(f"{rid}: checksum mismatch {name.strip()}")
        o, pr = v["outcomes_not_used_for_validity"], ids[rid]["predicted_outcome"]
        if o["evasions_attempt1"] != pr["evasions_attempt1"]:
            problems.append(f"{rid}: evasions differ from pre-registered prediction")
        if "cumulative_evasions_by_n" in pr and o["cumulative_evasions_by_n"] != pr["cumulative_evasions_by_n"]:
            problems.append(f"{rid}: C(n) differs from pre-registered prediction")
        if abs(o["realized_v2_share"] - pr["realized_v2_share"]) > 1e-12:
            problems.append(f"{rid}: V2 share differs from prediction")
        runs.append({"rid": rid, "cell": r["cell"], "state": ids[rid]["rollout_state"],
                     "attacker": ids[rid]["attacker"], "type": r["identity_type"], "endpoint": man["endpoint"],
                     "o": o, "dir": d})
        print(f"  integrity {rid}: ok" if not any(p.startswith(rid) for p in problems) else f"  integrity {rid}: PROBLEM")
    integrity = {"valid_runs": len(valid), "per_cell": dict(per_cell), "invalid_runs": [r["run_id"] for r in reg if r["disposition"] != "VALID"],
                 "checksums_verified": not a.skip_checksums, "problems": problems, "PASS": not problems,
                 "freeze_sha256": recorded, "population": const}
    (out / "integrity_report.json").write_text(json.dumps(integrity, indent=2) + "\n")
    print("INTEGRITY", "PASS" if not problems else f"FAIL {problems}")

    # ---------------- stage 1: per run ----------------
    rows = []
    for x in runs:
        o = x["o"]
        c = [o["cumulative_evasions_by_n"][str(n)] for n in range(1, K + 1)] if x["attacker"] == "A1" else [o["evasions_attempt1"]]
        row = {"run_id": x["rid"], "cell": x["cell"], "rollout_state": x["state"], "attacker": x["attacker"],
               "identity_type": x["type"], "endpoint": x["endpoint"], "realized_v2_share": o["realized_v2_share"],
               "evasions_attempt1": o["evasions_attempt1"], "ASR_attempt1": o["evasions_attempt1"] / N}
        for n in range(1, K + 1):
            row[f"C{n}"] = c[n - 1] if len(c) >= n else ""
            row[f"ASR_C{n}"] = c[n - 1] / N if len(c) >= n else ""
            row[f"D{n}"] = c[n - 1] - c[0] if len(c) >= n else ""
        rows.append(row)
    with open(out / "per_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["cell"], r["run_id"])))

    # ---------------- stage 2: per cell ----------------
    cell_rows = []
    for c in cells:
        rr = [r for r in rows if r["cell"] == c]
        keys = ["ASR_attempt1", "realized_v2_share"] + ([f"C{n}" for n in range(1, K + 1)] + [f"D{n}" for n in range(2, K + 1)]
                                                       if c.endswith("A1") else [])
        for k in keys:
            m, s, lo, hi = ci([float(r[k]) for r in rr])
            cell_rows.append({"cell": c, "quantity": k, "n_runs": len(rr), "mean": m, "sd": s, "ci95_lo": lo, "ci95_hi": hi})
    with open(out / "per_cell.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_rows[0]))
        w.writeheader()
        w.writerows(cell_rows)

    # ---------------- stage 3: exposure law ----------------
    law_rows = []
    for st in W:
        wv = W[st]
        obs = [r for r in rows if r["cell"] == f"{st}-A1"]
        for n in range(1, K + 1):
            q, rr_ = 1 - (1 - wv) ** n, 1 - wv ** n
            exp = BM + len(X12) * q + len(X21) * rr_
            sd1 = math.sqrt(len(X12) * q * (1 - q) + len(X21) * rr_ * (1 - rr_))
            se5 = sd1 / math.sqrt(len(obs)) if obs else float("nan")
            m, s, lo, hi = ci([float(r[f"C{n}"]) for r in obs])
            law_rows.append({"state": st, "w2": wv, "n": n, "expected_C": exp, "sd_single_run": sd1,
                             "se_mean_of_runs": se5, "band95_lo": exp - 1.96 * se5, "band95_hi": exp + 1.96 * se5,
                             "observed_mean_C": m, "observed_ci95_lo": lo, "observed_ci95_hi": hi,
                             "z_mean_vs_law": (m - exp) / se5 if se5 > 0 else 0.0})
    with open(out / "exposure_law.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(law_rows[0]))
        w.writeheader()
        w.writerows(law_rows)

    # ---------------- stage 4: Gate 4 ----------------
    try:
        import numpy as np
    except ImportError:
        np = None
    MR0 = [r["evasions_attempt1"] for r in rows if r["cell"] == "R0-A0"]
    MR3 = [r["evasions_attempt1"] for r in rows if r["cell"] == "R3-A0"]
    if set(MR0) != {M1} or set(MR3) != {M2}:
        problems.append("R0/R3 A0 misses differ from frozen revision misses")
    m_r0, m_r3 = stats.fmean(MR0), stats.fmean(MR3)
    gate = {"rule": freeze["gate4"]["primary_rule"], "mc_draws": a.mc_draws, "seed": 2026101902, "cells": {}}
    pvals, labels = [], []
    rng = np.random.default_rng(2026101902) if np is not None else None   # one frozen seed; R1 then R2 drawn in order
    for st in ("R1", "R2"):
        rr = [x for x in runs if x["cell"] == f"{st}-A0"]
        per = []
        for x in rr:
            k12 = k21 = 0
            for line in open(x["dir"] / "requests.jsonl"):
                j = json.loads(line)
                if j["selected_revision"] == "v2":
                    k12 += j["sample_id"] in X12
                    k21 += j["sample_id"] in X21
            wo = x["o"]["realized_v2_share"]
            misses = x["o"]["evasions_attempt1"]
            d = misses - ((1 - wo) * m_r0 + wo * m_r3)
            per.append({"run_id": x["rid"], "w2_obs": wo, "misses": misses, "d_samples": d, "d_asr": d / N,
                        "K12_obs": k12, "K21_obs": k21, "v2_share_in_X12": k12 / len(X12), "v2_share_in_X21": k21 / len(X21),
                        "exceeds_superseded_MDE": abs(d / N) > MDE})
        dbar = stats.fmean(p["d_samples"] for p in per)
        if np is not None:
            sims = np.zeros(a.mc_draws)
            for p in per:
                wo = p["w2_obs"]
                sims += (rng.binomial(len(X12), wo, a.mc_draws) - rng.binomial(len(X21), wo, a.mc_draws)
                         - (len(X12) - len(X21)) * wo)
            sims /= len(per)
            pval = float((np.sum(np.abs(sims) >= abs(dbar) - 1e-12) + 1) / (a.mc_draws + 1))
            null_sd = float(np.std(sims))
        else:
            pval, null_sd = float("nan"), float("nan")
        gate["cells"][f"{st}-A0"] = {"runs": per, "mean_d_samples": dbar, "mean_d_asr": dbar / N,
                                     "null_sd_of_mean_d": null_sd, "p_two_sided_mc": pval,
                                     "secondary_MDE_exceedances": sum(p["exceeds_superseded_MDE"] for p in per)}
        pvals.append(pval)
        labels.append(f"{st}-A0")
    if np is not None:
        adj, rej = holm(pvals)
        for lab, pa, rj in zip(labels, adj, rej):
            gate["cells"][lab].update({"p_holm": pa, "reject_additivity_at_0.05": rj})
    gate["note"] = "single generator seeded 2026101902 (freeze); R1 draws first, then R2"
    (out / "gate4.json").write_text(json.dumps(gate, indent=2) + "\n")
    for lab in labels:
        g = gate["cells"][lab]
        print(f"GATE4 {lab}: mean d = {g['mean_d_samples']:+.2f} samples, p_mc = {g['p_two_sided_mc']:.4f}, "
              f"p_holm = {g.get('p_holm', float('nan')):.4f}, MDE exceedances {g['secondary_MDE_exceedances']}/5")

    # ---------------- stage 5: latency ----------------
    if not a.skip_latency:
        lat_rows, pooled = [], defaultdict(list)
        for x in runs:
            L, Lw = [], []
            for line in open(x["dir"] / "requests.jsonl"):
                j = json.loads(line)
                L.append(j["latency_ms"])
                if j["scheduled_offset_s"] >= 5.0:
                    Lw.append(j["latency_ms"])
            for tag, arr in (("all", L), ("excl_first_5s", Lw)):
                if not arr:
                    continue
                arr.sort()
                q = lambda p: arr[min(len(arr) - 1, int(p * len(arr)))]
                rec = {"run_id": x["rid"], "cell": x["cell"], "endpoint": x["endpoint"], "subset": tag, "n": len(arr),
                       "p50_ms": q(.50), "p95_ms": q(.95), "p99_ms": q(.99), "max_ms": arr[-1]}
                lat_rows.append(rec)
                pooled[(x["endpoint"], tag)].append(rec)
            print(f"  latency {x['rid']} done")
        with open(out / "latency_per_run.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(lat_rows[0]))
            w.writeheader()
            w.writerows(lat_rows)
        with open(out / "latency_by_transport_path.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["endpoint", "subset", "runs", "median_of_run_p50", "median_of_run_p95", "median_of_run_p99"])
            for (ep, tag), recs in sorted(pooled.items()):
                w.writerow([ep, tag, len(recs), stats.median(r["p50_ms"] for r in recs),
                            stats.median(r["p95_ms"] for r in recs), stats.median(r["p99_ms"] for r in recs)])

    # ---------------- stage 6: figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 8, "font.family": "serif"})
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3), constrained_layout=True)
        for ax, st in zip(axes, ("R1", "R2")):
            lr = [r for r in law_rows if r["state"] == st]
            ns = [r["n"] for r in lr]
            ax.fill_between(ns, [r["band95_lo"] for r in lr], [r["band95_hi"] for r in lr], color="0.85",
                            label="law: 95% band (mean of 5 runs)")
            ax.plot(ns, [r["expected_C"] for r in lr], "k-", lw=1, label="law: expected C(n)")
            for r in [r for r in rows if r["cell"] == f"{st}-A1"]:
                ax.plot(ns, [r[f"C{n}"] for n in ns], "o", ms=2, color="0.55", alpha=.6)
            ax.errorbar(ns, [r["observed_mean_C"] for r in lr],
                        yerr=[[r["observed_mean_C"] - r["observed_ci95_lo"] for r in lr],
                              [r["observed_ci95_hi"] - r["observed_mean_C"] for r in lr]],
                        fmt="s", ms=3.5, color="C0", capsize=2, lw=1, label="observed mean, 95% CI")
            ax.set_title(f"{st} (w$_2$ = {W[st]})")
            ax.set_xlabel("attempts n")
            ax.set_xticks(ns)
        axes[0].set_ylabel("cumulative evasions C(n)")
        axes[1].legend(loc="lower right", fontsize=6.5, frameon=False)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_exposure_law.{ext}", dpi=300)
        print("figure written")
    except ImportError:
        print("matplotlib not available: figure skipped")

    integrity["problems"] = problems
    integrity["PASS"] = not problems
    (out / "integrity_report.json").write_text(json.dumps(integrity, indent=2) + "\n")
    with open(out / "ANALYSIS_MANIFEST.sha256", "w") as f:
        f.write(f"{sha(Path(__file__))}  analyze_v3.py(script)\n{recorded}  phase9_v3_freeze.json\n"
                f"{sha(BASE / 'registry.csv')}  registry.csv\n")
        for p in sorted(out.iterdir()):
            if p.is_file() and p.name != "ANALYSIS_MANIFEST.sha256":
                f.write(f"{sha(p)}  {p.name}\n")
    print(f"\nOUTPUT {out}\nINTEGRITY {'PASS' if not problems else 'FAIL'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
