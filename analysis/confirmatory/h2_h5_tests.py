#!/usr/bin/env python3
"""
RolloutSecBench - confirmatory H2, H4, H5 tests and descriptive H3, Phase-9 v3.

Governing plan: scientific statistical audit v1.0 (Flags 1-5), with amendment A1-B5-L16 (A1 budget K = 5).
The test specifications below were fixed and committed BEFORE this script was first run on the analysis of
record; the commit timestamp of this file is the evidence of that order.

READ-ONLY on evidence. Inputs, all verified before any statistic is computed:
  * per_run.csv, gate4.json, integrity_report.json  against the analysis-of-record ANALYSIS_MANIFEST.sha256
  * integrity_report.json: PASS with checksums verified; 5 valid runs in each of the 8 cells
  * stable cells R0/R3: realized V2 share exactly 0 / 1 (otherwise K12, K21 are not determined)
Unit of analysis: the run (audit Flag 2). Outcomes are sample-level evasion shares (audit Flag 4):
  A0 -> C(1)/N,  A1 -> C(K)/N.

SPECIFICATION
  H2 (confirmatory)  Omnibus effect of the categorical rollout factor R0..R3 (audit Flag 1.1) on run-level
                     C(1)/N over the 20 A0 runs. Null: the SHARP null that rollout state does not change the
                     distribution of run outcomes (the null under which a permutation test is exact; a
                     mean-only null is not testable exactly here because R0/R3 have zero variance).
                     Statistic: between-state sum of squares. p: Monte-Carlo permutation of run labels
                     across states (B draws, seed SEED_H2), (b+1)/(B+1).
                     NOTE: R0 and R3 are deterministic and differ by |X12|-|X21| samples, so this null is
                     false by construction and its rejection is guaranteed by the design.
                     Descriptive companions: mixed-state mean minus the configured-weight mixture, 95% t-CI.
  H4 (confirmatory)  Omnibus over n in {1..K} within R1 and R2 (audit Flag 3). Because C_r(n) is
                     non-decreasing in n, C_r(n) = C_r(1) for all n  <=>  Delta_r(K) = 0. Statistic: number of
                     the 10 mixed-state A1 runs with Delta_r(K) > 0; exact one-sided sign test (H4 is
                     directional; Delta_r(K) >= 0 by construction; zero gains count as ties and are dropped).
                     Effect sizes: Delta(n) = C(n) - C(1), n = 2..K, per state, mean with 95% t-CI over runs.
  H5 (confirmatory)  Rollout x Attacker interaction in the 4x2 design on run-level outcomes (40 runs).
                     At R0 and R3 one revision serves every attempt, so the retry effect
                     delta_s = mean(A1) - mean(A0) is zero by construction (verified: every R0 run equals
                     M_R0 and every R3 run equals M_R3, in both attacker cells). No interaction therefore
                     means delta_R1 = delta_R2 = 0. Test: two-sided Welch t-test of A1 vs A0 runs at R1 and
                     at R2 (independent runs), combined by Fisher's method, chi-square with 4 df.
                     (Pre-run validation: permutation of the interaction SS, Freedman-Lane, and studentized
                     permutation were liberal, 7-10% at nominal 5%, under the v3 variance structure;
                     Welch + Fisher held 4.1-5.5%.)
  Holm               Holm-Bonferroni step-down over (p_H2, p_H4, p_H5), FWER alpha = 0.05 (audit Flag 5).
  H3 (descriptive)   OLS  ASR_{r,d} = alpha + beta * E_{r,d} + gamma * direction_d  over the 20 A0 runs x 2
                     directions (audit Flag 1.2), with
                       ASR_{r,12} = K12/|X12|,          E_{r,12} = w_V2,r
                       ASR_{r,21} = (|X21| - K21)/|X21|, E_{r,21} = w_V1,r = 1 - w_V2,r
                     Reported: coefficients, classical and run-cluster-robust (CR1) SEs. No p-value enters Holm.
  Not computed here: a sample-level mixed model (secondary; needs request-level evidence and may not converge).

Uses the Python standard library and numpy only.

Outputs (--out-dir, new folder): h2_h5_results.json, h2_h5_values.tex (LaTeX macros), h4_effects.csv,
h3_points.csv, h2_h5_provenance.json.

Usage (repo root):
  .venv-blueprint-v2/bin/python analysis/confirmatory/h2_h5_tests.py \\
      --analysis-dir results/derived/phase9_v3/analysis_20261022T200702Z \\
      --out-dir results/derived/phase9_v3/confirmatory
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
STATES = ["R0", "R1", "R2", "R3"]
W_CONF = {"R0": 0.0, "R1": 0.1, "R2": 0.5, "R3": 1.0}
B_PERM = 100_000
SEED_H2 = 2026102201
ALPHA = 0.05
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stop(msg):
    print(f"REFUSING TO COMPUTE: {msg}", file=sys.stderr)
    sys.exit(2)


def tci(xs):
    m = st.fmean(xs)
    if len(xs) < 2:
        return m, 0.0, m, m
    s = st.stdev(xs)
    h = T975[len(xs) - 1] * s / math.sqrt(len(xs))
    return m, s, m - h, m + h


def holm(ps, alpha=ALPHA):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, prev = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = prev
    return adj, [a <= alpha for a in adj]


def ss_between(values, groups):
    g = st.fmean(values)
    out = 0.0
    for lab in set(groups):
        v = [x for x, k in zip(values, groups) if k == lab]
        out += len(v) * (st.fmean(v) - g) ** 2
    return out


def _betacf(a, b, x):
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return h


def betainc_reg(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    return betainc_reg(df / 2.0, 0.5, df / (df + t * t))


def welch(a0, a1):
    """Two-sided Welch t-test of mean(a1) - mean(a0); returns t, Satterthwaite df, p."""
    m0, m1, v0, v1 = st.fmean(a0), st.fmean(a1), st.variance(a0), st.variance(a1)
    q0, q1 = v0 / len(a0), v1 / len(a1)
    if q0 + q1 == 0:
        stop("Welch test undefined: zero variance in both groups of a mixed state")
    t = (m1 - m0) / math.sqrt(q0 + q1)
    df = (q0 + q1) ** 2 / ((q0 ** 2 / (len(a0) - 1) if q0 else 0) + (q1 ** 2 / (len(a1) - 1) if q1 else 0))
    return t, df, t_two_sided_p(t, df)


def fisher_combine_two(p1, p2):
    """Fisher's method for two independent p-values: X = -2 ln(p1 p2) ~ chi2(4); sf = e^{-X/2}(1 + X/2)."""
    x = -2.0 * (math.log(p1) + math.log(p2))
    return math.exp(-x / 2.0) * (1.0 + x / 2.0), x


def binom_upper_tail(k, n):
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def ols(X, y):
    """Plain OLS with classical and run-cluster-robust (CR1) standard errors; X rows include intercept."""
    import numpy as np
    X, y = np.asarray(X, float), np.asarray(y, float)
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    n, p = X.shape
    s2 = float(e @ e) / (n - p)
    se = np.sqrt(np.diag(XtX_inv) * s2)
    return b, se, e, XtX_inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--out-dir", default="results/derived/phase9_v3/confirmatory")
    a = ap.parse_args()
    adir = (ROOT / a.analysis_dir).resolve()
    out = (ROOT / a.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        stop(f"output folder {out} is not empty; results are never overwritten")
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- integrity
    man = {}
    for line in (adir / "ANALYSIS_MANIFEST.sha256").read_text().splitlines():
        h, name = line.split(None, 1)
        man[name.strip()] = h
    for name in ("per_run.csv", "gate4.json", "integrity_report.json"):
        if sha(adir / name) != man.get(name):
            stop(f"{name} does not match the analysis manifest")
    integ = json.loads((adir / "integrity_report.json").read_text())
    if not (integ.get("PASS") and integ.get("checksums_verified")):
        stop("analysis of record lacks a full integrity PASS")
    pop = integ["population"]
    N, n12, n21 = pop["N"], pop["X12_only_v2_misses"], pop["X21_only_v1_misses"]
    bm = pop["both_miss"]
    rows = list(csv.DictReader(open(adir / "per_run.csv")))
    cells = {}
    for r in rows:
        cells.setdefault((r["rollout_state"], r["attacker"]), []).append(r)
    for s_ in STATES:
        for at in ("A0", "A1"):
            if len(cells.get((s_, at), [])) != 5:
                stop(f"cell {s_}-{at} does not have 5 valid runs")
    K = max(int(k[1:]) for k in rows[0] if k.startswith("C") and k[1:].isdigit())
    if not all(r[f"C{K}"] != "" for r in rows if r["attacker"] == "A1"):
        stop(f"A1 runs lack C({K})")
    gate4 = json.loads((adir / "gate4.json").read_text())

    def outcome(r):
        return int(r["C1"] if r["attacker"] == "A0" else r[f"C{K}"]) / N

    checks = []

    # ---------------- H2
    a0 = [r for r in rows if r["attacker"] == "A0"]
    y2 = [outcome(r) for r in a0]
    g2 = [r["rollout_state"] for r in a0]
    obs2 = ss_between(y2, g2)
    rng = random.Random(SEED_H2)
    lab = g2[:]
    b_ = 0
    for _ in range(B_PERM):
        rng.shuffle(lab)
        b_ += ss_between(y2, lab) >= obs2 - 1e-18
    p_h2 = (b_ + 1) / (B_PERM + 1)
    h2_desc = {}
    for s_ in STATES:
        v = [outcome(r) for r in cells[(s_, "A0")]]
        m, sd, lo, hi = tci(v)
        mix = (bm + n12 * W_CONF[s_] + n21 * (1 - W_CONF[s_])) / N
        h2_desc[s_] = {"mean": m, "sd": sd, "ci95": [lo, hi], "configured_mixture": mix,
                       "mean_minus_mixture": m - mix, "ci95_minus_mixture": [lo - mix, hi - mix]}
    checks.append(f"H2: {len(a0)} A0 runs, {B_PERM} permutations")

    # ---------------- H4
    mixed_a1 = [r for r in rows if r["attacker"] == "A1" and r["rollout_state"] in ("R1", "R2")]
    gains = [int(r[f"C{K}"]) - int(r["C1"]) for r in mixed_a1]
    for r in mixed_a1:
        cs = [int(r[f"C{n}"]) for n in range(1, K + 1)]
        if any(cs[i] > cs[i + 1] for i in range(K - 1)):
            stop(f"{r['run_id']}: C(n) is not non-decreasing")
    nz = [g for g in gains if g != 0]
    k_pos = sum(g > 0 for g in nz)
    p_h4 = binom_upper_tail(k_pos, len(nz)) if nz else 1.0
    h4_eff = []
    for s_ in ("R1", "R2"):
        rr = cells[(s_, "A1")]
        for n in range(2, K + 1):
            v = [int(r[f"C{n}"]) - int(r["C1"]) for r in rr]
            m, sd, lo, hi = tci(v)
            w = W_CONF[s_]
            law = n12 * ((1 - (1 - w) ** n) - w) + n21 * ((1 - w ** n) - (1 - w))
            h4_eff.append({"state": s_, "n": n, "mean_delta_samples": m, "sd": sd, "ci95_lo": lo, "ci95_hi": hi,
                           "law_expected_delta": law})
    checks.append(f"H4: {len(mixed_a1)} mixed-state A1 runs; C(n) non-decreasing in every run")

    # ---------------- H5 (structural zeros at R0/R3; Welch per mixed state; Fisher combination)
    m_r0, m_r3 = bm + n21, bm + n12
    for s_, m_ in (("R0", m_r0), ("R3", m_r3)):
        for at in ("A0", "A1"):
            vals = {int(r["C1"] if at == "A0" else r[f"C{K}"]) for r in cells[(s_, at)]}
            if vals != {m_}:
                stop(f"{s_}-{at}: outcomes {sorted(vals)} differ from the deterministic value {m_}")
    h5_states = {}
    for s_ in ("R1", "R2"):
        a0v = [outcome(r) for r in cells[(s_, "A0")]]
        a1v = [outcome(r) for r in cells[(s_, "A1")]]
        t, df, p = welch(a0v, a1v)
        h5_states[s_] = {"delta": st.fmean(a1v) - st.fmean(a0v), "delta_samples": N * (st.fmean(a1v) - st.fmean(a0v)),
                         "welch_t": t, "df": df, "p_two_sided": p}
    p_h5, chi2_h5 = fisher_combine_two(h5_states["R1"]["p_two_sided"], h5_states["R2"]["p_two_sided"])
    h5_cells = {f"{s_}-{at}": st.fmean([outcome(r) for r in cells[(s_, at)]]) for s_ in STATES for at in ("A0", "A1")}
    h5_retry = {s_: h5_cells[f"{s_}-A1"] - h5_cells[f"{s_}-A0"] for s_ in STATES}
    checks.append("H5: R0/R3 outcomes equal their deterministic values in both attacker cells (structural zeros)")

    # ---------------- Holm
    adj, rej = holm([p_h2, p_h4, p_h5])

    # ---------------- H3 (descriptive)
    import numpy as np
    k_by_run = {}
    for cell in ("R1-A0", "R2-A0"):
        for r in gate4["cells"][cell]["runs"]:
            k_by_run[r["run_id"]] = (r["K12_obs"], r["K21_obs"])
    pts = []
    for r in a0:
        w2 = float(r["realized_v2_share"])
        if r["rollout_state"] in ("R0", "R3"):
            if w2 not in (0.0, 1.0):
                stop(f"{r['run_id']}: stable-state realized share is {w2}, not 0 or 1")
            k12, k21 = round(n12 * w2), round(n21 * w2)
        else:
            if r["run_id"] not in k_by_run:
                stop(f"{r['run_id']}: K12/K21 missing from gate4.json")
            k12, k21 = k_by_run[r["run_id"]]
        misses = int(r["C1"])
        if misses != bm + k12 + (n21 - k21):
            stop(f"{r['run_id']}: misses do not decompose as both_miss + K12 + (|X21| - K21)")
        pts.append({"run_id": r["run_id"], "state": r["rollout_state"], "direction": "X12", "E": w2,
                    "ASR": k12 / n12, "dir21": 0})
        pts.append({"run_id": r["run_id"], "state": r["rollout_state"], "direction": "X21", "E": 1 - w2,
                    "ASR": (n21 - k21) / n21, "dir21": 1})
    X = [[1.0, p["E"], p["dir21"]] for p in pts]
    y = [p["ASR"] for p in pts]
    bcoef, se, e, XtX_inv = ols(X, y)
    Xa = np.asarray(X)
    meat = np.zeros((3, 3))
    runs_ = sorted({p["run_id"] for p in pts})
    for rid in runs_:
        idx = [i for i, p in enumerate(pts) if p["run_id"] == rid]
        sg = Xa[idx].T @ e[idx]
        meat += np.outer(sg, sg)
    G, nobs = len(runs_), len(pts)
    cr1 = (G / (G - 1)) * ((nobs - 1) / (nobs - 3)) * (XtX_inv @ meat @ XtX_inv)
    se_cr = np.sqrt(np.diag(cr1))
    tq = T975.get(G - 1, 2.093)  # df = G - 1 for cluster-robust intervals (G = 20 -> 2.093)
    h3 = {"n_points": nobs, "n_clusters": G,
          "coef": {"alpha": float(bcoef[0]), "beta_exposure": float(bcoef[1]), "gamma_direction21": float(bcoef[2])},
          "se_classical": {"alpha": float(se[0]), "beta_exposure": float(se[1]), "gamma_direction21": float(se[2])},
          "se_cluster_CR1": {"alpha": float(se_cr[0]), "beta_exposure": float(se_cr[1]),
                             "gamma_direction21": float(se_cr[2])},
          "beta_ci95_cluster": [float(bcoef[1] - tq * se_cr[1]), float(bcoef[1] + tq * se_cr[1])],
          "t_quantile_used": tq}
    checks.append(f"H3: {nobs} points from {G} A0 runs; misses decompose exactly in every run")

    # ---------------- outputs
    res = {"plan": "statistical audit v1.0 Flags 1-5; amendment A1-B5-L16 (K=%d)" % K, "unit": "run",
           "population": {"N": N, "X12": n12, "X21": n21, "both_miss": bm},
           "H2": {"statistic": "between-state SS of run-level C(1)/N (A0)", "observed": obs2, "B": B_PERM,
                  "seed": SEED_H2, "p": p_h2, "descriptive": h2_desc},
           "H4": {"statistic": f"count of mixed-state A1 runs with Delta(K={K})>0", "n_runs": len(gains),
                  "n_nonzero": len(nz), "k_positive": k_pos, "p_one_sided_exact": p_h4,
                  "min_attainable_p": 1 / 2 ** len(nz) if nz else 1.0, "gains_samples": gains},
           "H5": {"test": "Welch t (A1 vs A0) at R1 and R2, Fisher combination, chi2(4)", "chi2": chi2_h5,
                  "p": p_h5, "per_state": h5_states, "cell_means": h5_cells, "retry_effect_by_state": h5_retry,
                  "structural_zero_states": ["R0", "R3"]},
           "holm": {"family": ["H2", "H4", "H5"], "p_raw": [p_h2, p_h4, p_h5], "p_adj": adj, "reject": rej,
                    "alpha": ALPHA},
           "H3_descriptive": h3}
    (out / "h2_h5_results.json").write_text(json.dumps(res, indent=2) + "\n")
    with open(out / "h4_effects.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(h4_eff[0]))
        w_.writeheader()
        w_.writerows(h4_eff)
    with open(out / "h3_points.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(pts[0]))
        w_.writeheader()
        w_.writerows(pts)

    def pv(p):
        return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"

    def pvt(p):
        if p >= 1e-4:
            return f"{p:.4f}"
        m, e = f"{p:.1e}".split("e")
        return f"\\ensuremath{{{m}\\times10^{{{int(e)}}}}}"

    def mac(name, val):
        return f"\\newcommand{{\\{name}}}{{{val}}}"

    def sn(x, d):
        s = f"{x:.{d}f}"
        return "\\ensuremath{-}" + s[1:] if s.startswith("-") else s

    tex = ["% generated by h2_h5_tests.py - do not edit by hand",
           mac("HtwoP", pvt(p_h2)), mac("HtwoPadj", pvt(adj[0])), mac("HtwoReject", "rejected" if rej[0] else "not rejected"),
           mac("HfourK", str(k_pos)), mac("HfourN", str(len(nz))), mac("HfourP", pvt(p_h4)), mac("HfourPadj", pvt(adj[1])),
           mac("HfourReject", "rejected" if rej[1] else "not rejected"),
           mac("HfiveP", pvt(p_h5)), mac("HfivePadj", pvt(adj[2])), mac("HfiveReject", "rejected" if rej[2] else "not rejected"),
           mac("HpermB", f"{B_PERM:,}".replace(",", "{,}")),
           mac("HthreeBeta", sn(h3["coef"]["beta_exposure"], 3)),
           mac("HthreeBetaLo", sn(h3["beta_ci95_cluster"][0], 3)), mac("HthreeBetaHi", sn(h3["beta_ci95_cluster"][1], 3)),
           mac("HthreeGamma", sn(h3["coef"]["gamma_direction21"], 3)), mac("HthreeAlpha", sn(h3["coef"]["alpha"], 3))]
    words = {2: "Two", 3: "Three", 4: "Four", 5: "Five"}
    for e_ in h4_eff:
        tag = ("Rone" if e_["state"] == "R1" else "Rtwo") + "D" + words[e_["n"]]
        tex += [mac(f"Hfour{tag}", sn(e_["mean_delta_samples"], 1)), mac(f"Hfour{tag}Lo", sn(e_["ci95_lo"], 1)),
                mac(f"Hfour{tag}Hi", sn(e_["ci95_hi"], 1)), mac(f"Hfour{tag}Law", sn(e_["law_expected_delta"], 1))]
    (out / "h2_h5_values.tex").write_text("\n".join(tex) + "\n")

    prov = {"generated_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__)),
            "python": platform.python_version(), "numpy": np.__version__, "analysis_dir": str(adir),
            "inputs_sha256": {k: man[k] for k in ("per_run.csv", "gate4.json", "integrity_report.json")},
            "checks": checks,
            "outputs_sha256": {p: sha(out / p) for p in ("h2_h5_results.json", "h2_h5_values.tex",
                                                         "h4_effects.csv", "h3_points.csv")}}
    (out / "h2_h5_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")

    for c in checks:
        print("CHECK PASS", c)
    print(f"H2  p = {pv(p_h2)}   Holm {pv(adj[0])}  {'REJECT' if rej[0] else 'retain'}")
    print(f"H4  {k_pos}/{len(nz)} runs with Delta({K})>0, p = {pv(p_h4)}   Holm {pv(adj[1])}  {'REJECT' if rej[1] else 'retain'}")
    print(f"H5  p = {pv(p_h5)}   Holm {pv(adj[2])}  {'REJECT' if rej[2] else 'retain'}")
    for s_, v in h5_states.items():
        print(f"    {s_}: retry effect {v['delta_samples']:+.1f} samples, Welch t = {v['welch_t']:.2f}, "
              f"df = {v['df']:.2f}, p = {pv(v['p_two_sided'])}")
    for e_ in h4_eff:
        print(f"    Delta({e_['n']}) {e_['state']}: {e_['mean_delta_samples']:.1f} "
              f"[{e_['ci95_lo']:.1f}, {e_['ci95_hi']:.1f}]  law {e_['law_expected_delta']:.1f}")
    print(f"H3  beta = {h3['coef']['beta_exposure']:.4f}  CR1 95% CI [{h3['beta_ci95_cluster'][0]:.4f}, "
          f"{h3['beta_ci95_cluster'][1]:.4f}]  gamma = {h3['coef']['gamma_direction21']:.4f}")
    for s_ in STATES:
        d = h2_desc[s_]
        print(f"    {s_}-A0 mean - configured mixture: {N * d['mean_minus_mixture']:+.2f} samples")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
