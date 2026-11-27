"""Revision r1, Phase 4, S3 + S4 (pre-registration Section 8): robustness of the H2-H5 decisions.

Run-level outcomes are taken from the E1 validation report (simulator reproduction of all 40 valid runs,
revision/results/e1_validation/validation_report.json) and cross-checked against the original analysis
outputs before any alternative test is computed. Original tests are recomputed and must equal the reported values.

Alternatives (fixed in the pre-registration):
  H2  Kruskal-Wallis (tie-corrected, chi-square 3 df)
  H4  Wilcoxon signed-rank (exact, one-sided, sign-flip enumeration); one-sample t (one-sided)
      [simulated distribution of Delta(5): reported with the E1 distributions, see DEV-R1-001]
  H5  Stouffer combination of the per-state Welch tests; exact Mann-Whitney per state + Fisher;
      parametric bootstrap of the Welch statistic per state (100,000 draws, seed 11242026) + Fisher
  H3  CR2 (Bell-McCaffrey) and wild cluster bootstrap (Rademacher, 9,999 draws, seed 11242026) intervals for beta
Each alternative p-value is also Holm-adjusted with the other two original confirmatory p-values.
Standard library + numpy only. Writes revision/results/s3_s4_robustness/.
"""
import csv, hashlib, itertools, json, math, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "revision/results/s3_s4_robustness"; OUT.mkdir(parents=True, exist_ok=True)
SEED, N, X12, X21, BOTH = 11242026, 38263, 92, 57, 596
VAL = ROOT / "revision/results/e1_validation/validation_report.json"
VAL_SHA = "57427065fc8c63447392a3c2f14e29fa832ff04ecac55d0ca2f03f00725cb496"
ORIG = ROOT / "results/derived/phase9_v3/confirmatory/h2_h5_results.json"
H3PTS = ROOT / "results/derived/phase9_v3/confirmatory/h3_points.csv"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


if sha(VAL) != VAL_SHA:
    sys.exit("validation report hash mismatch")
runs = json.load(open(VAL))["runs"]
orig = json.load(open(ORIG))


def cell(rid):
    s, a = rid.split("-")[1:3]
    return s, a


A0 = {s: [] for s in ("R0", "R1", "R2", "R3")}; A1 = {s: [] for s in A0}; GAIN = {s: [] for s in A0}
for rid in sorted(runs):
    s, a = cell(rid); C = {int(k): v for k, v in runs[rid]["C"].items()}
    if a == "A0": A0[s].append(C[1] / N)
    else: A1[s].append(C[5] / N); GAIN[s].append(C[5] - C[1])

# ---------- distribution helpers (no scipy) ----------
def betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1; c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300); h = d
    for m in range(1, 400):
        m2 = 2 * m; aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300); c = 1 + aa / c if abs(c) > 1e-300 else 1e300
        h *= d * c; aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300); c = 1 + aa / c if abs(c) > 1e-300 else 1e300
        de = d * c; h *= de
        if abs(de - 1) < 1e-15: break
    return h


def betainc(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    return math.exp(lb) * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - math.exp(lb) * betacf(b, a, 1 - x) / b


def t_sf(t, df):  # P(T > t)
    x = df / (df + t * t); p = 0.5 * betainc(df / 2, 0.5, x)
    return p if t > 0 else 1 - p


def norm_sf(z): return 0.5 * math.erfc(z / math.sqrt(2))


def norm_isf(p):
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if norm_sf(mid) > p else (lo, mid)
    return (lo + hi) / 2


def chi2_sf(x, k):  # integer k
    if k % 2 == 0:
        return math.exp(-x / 2) * sum((x / 2) ** i / math.factorial(i) for i in range(k // 2))
    s = math.erfc(math.sqrt(x / 2)); term = math.sqrt(2 * x / math.pi) * math.exp(-x / 2); acc = 0.0
    for i in range(1, (k + 1) // 2):
        acc += term; term *= x / (2 * i + 1)
    return s + acc


def fisher(ps): return chi2_sf(-2 * sum(math.log(p) for p in ps), 2 * len(ps))


def welch(a0, a1):
    m0, m1 = np.mean(a0), np.mean(a1); v0, v1 = np.var(a0, ddof=1), np.var(a1, ddof=1); n0, n1 = len(a0), len(a1)
    se2 = v0 / n0 + v1 / n1; t = (m1 - m0) / math.sqrt(se2)
    df = se2 ** 2 / ((v0 / n0) ** 2 / (n0 - 1) + (v1 / n1) ** 2 / (n1 - 1))
    return t, df, 2 * t_sf(abs(t), df)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0] * len(ps); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def ranks(x):
    x = np.asarray(x, float); order = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]: j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r


# ---------- recompute the originals and check ----------
checks = {}
t1, d1, p1 = welch(A0["R1"], A1["R1"]); t2, d2, p2 = welch(A0["R2"], A1["R2"])
p_h5 = fisher([p1, p2])
checks["H5_p_equals_original"] = math.isclose(p_h5, orig["H5"]["p"], rel_tol=1e-6)
gains = GAIN["R1"] + GAIN["R2"]
checks["H4_gains_equal_original"] = sorted(gains) == sorted(orig["H4"]["gains_samples"])
P_ORIG = {"H2": orig["H2"]["p"], "H4": orig["H4"]["p_one_sided_exact"], "H5": orig["H5"]["p"]}

# ---------- H2: Kruskal-Wallis ----------
vals = sum((A0[s] for s in A0), []); grp = sum(([s] * len(A0[s]) for s in A0), [])
r = ranks(vals); n = len(vals); Hs = 0.0
for s in A0:
    rs = r[[i for i, g in enumerate(grp) if g == s]]; Hs += len(rs) * (rs.mean() - (n + 1) / 2) ** 2
H = 12 / (n * (n + 1)) * Hs
_, counts = np.unique(vals, return_counts=True); H /= 1 - (counts ** 3 - counts).sum() / (n ** 3 - n)
h2_kw = {"H": H, "df": 3, "p": chi2_sf(H, 3)}

# ---------- H4: Wilcoxon exact (sign-flip), one-sample t ----------
g = np.array(gains, float); rg = ranks(np.abs(g)); w_obs = rg[g > 0].sum()
ws = [sum(rg[i] for i in range(len(g)) if s_[i]) for s_ in itertools.product([0, 1], repeat=len(g))]
h4_w = {"W_plus": float(w_obs), "p_one_sided_exact": sum(w >= w_obs for w in ws) / len(ws)}
tt = g.mean() / (g.std(ddof=1) / math.sqrt(len(g))); h4_t = {"t": tt, "df": len(g) - 1, "p_one_sided": t_sf(tt, len(g) - 1)}

# ---------- H5: Stouffer, Mann-Whitney exact + Fisher, parametric bootstrap + Fisher ----------
z = [math.copysign(norm_isf(p / 2), t) for p, t in ((p1, t1), (p2, t2))]
Z = sum(z) / math.sqrt(2); h5_st = {"z_states": z, "Z": Z, "p_two_sided": 2 * norm_sf(abs(Z))}


def mw_exact(a0, a1):
    allv = np.array(a0 + a1); rk = ranks(allv); n1 = len(a1); idx = range(len(allv))
    obs = rk[len(a0):].sum() - n1 * (n1 + 1) / 2; mu = len(a0) * n1 / 2
    us = [rk[list(c)].sum() - n1 * (n1 + 1) / 2 for c in itertools.combinations(idx, n1)]
    return obs, sum(abs(u - mu) >= abs(obs - mu) - 1e-12 for u in us) / len(us)


mw = {s: mw_exact(A0[s], A1[s]) for s in ("R1", "R2")}
h5_mw = {"U": {s: mw[s][0] for s in mw}, "p_states": {s: mw[s][1] for s in mw}, "p_fisher": fisher([mw["R1"][1], mw["R2"][1]])}
rng = np.random.default_rng(SEED); pb = {}
for s, tobs in (("R1", t1), ("R2", t2)):
    sd0, sd1 = np.std(A0[s], ddof=1), np.std(A1[s], ddof=1); Bp = 100_000
    x0 = rng.normal(0, sd0, (Bp, 5)); x1 = rng.normal(0, sd1, (Bp, 5))
    se = np.sqrt(x0.var(axis=1, ddof=1) / 5 + x1.var(axis=1, ddof=1) / 5); ts = (x1.mean(1) - x0.mean(1)) / se
    pb[s] = (np.sum(np.abs(ts) >= abs(tobs)) + 1) / (Bp + 1)
h5_pb = {"p_states": pb, "p_fisher": fisher(list(pb.values())), "draws_per_state": 100_000}

# ---------- H3: CR1 (original), CR2, wild cluster bootstrap ----------
pts = []
for rid in sorted(runs):
    s, a = cell(rid)
    if a != "A0": continue
    w = runs[rid]["realized_v2_share"]; k12, k21 = runs[rid]["K12"], runs[rid]["K21"]
    pts.append((rid, w, k12 / X12, 0)); pts.append((rid, 1 - w, (X21 - k21) / X21, 1))
with open(H3PTS) as f:
    ref = {(r_["run_id"], int(r_["dir21"])): (float(r_["E"]), float(r_["ASR"])) for r_ in csv.DictReader(f)}
checks["H3_points_equal_original"] = all(abs(ref[(p[0], p[3])][0] - p[1]) < 1e-12 and abs(ref[(p[0], p[3])][1] - p[2]) < 1e-12 for p in pts)
X = np.array([[1, p[1], p[3]] for p in pts]); y = np.array([p[2] for p in pts]); cl = [p[0] for p in pts]
XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
G = sorted(set(cl)); nobs, k = X.shape; tq = 2.093  # t_{0.975, G-1 = 19}, as in the original analysis
meat1 = sum(np.outer(X[[i for i, c in enumerate(cl) if c == g_]].T @ e[[i for i, c in enumerate(cl) if c == g_]],
                     X[[i for i, c in enumerate(cl) if c == g_]].T @ e[[i for i, c in enumerate(cl) if c == g_]]) for g_ in G)
V1 = XtXi @ meat1 @ XtXi * (len(G) / (len(G) - 1)) * ((nobs - 1) / (nobs - k))
meat2 = np.zeros((k, k))
for g_ in G:
    ix = [i for i, c in enumerate(cl) if c == g_]; Xg = X[ix]; Hgg = Xg @ XtXi @ Xg.T
    vals_, vecs = np.linalg.eigh(np.eye(len(ix)) - Hgg)
    Ag = vecs @ np.diag([1 / math.sqrt(v) if v > 1e-12 else 0.0 for v in vals_]) @ vecs.T
    u = Xg.T @ Ag @ e[ix]; meat2 += np.outer(u, u)
V2 = XtXi @ meat2 @ XtXi
ci = lambda V: [float(b[1] - tq * math.sqrt(V[1, 1])), float(b[1] + tq * math.sqrt(V[1, 1]))]
checks["H3_CR1_equals_original"] = all(math.isclose(a_, b_, rel_tol=1e-6) for a_, b_ in zip(ci(V1), orig["H3"]["beta_ci95_cluster"])) if "H3" in orig else None
rng2 = np.random.default_rng(SEED); Bw = 9_999; tstar = np.empty(Bw); gid = np.array([G.index(c) for c in cl])
for i_ in range(Bw):
    wv = rng2.choice([-1.0, 1.0], size=len(G))[gid]; ys = X @ b + e * wv; bs = XtXi @ X.T @ ys; es = ys - X @ bs
    m_ = sum(np.outer(X[gid == j].T @ es[gid == j], X[gid == j].T @ es[gid == j]) for j in range(len(G)))
    Vs = XtXi @ m_ @ XtXi * (len(G) / (len(G) - 1)) * ((nobs - 1) / (nobs - k))
    tstar[i_] = (bs[1] - b[1]) / math.sqrt(Vs[1, 1])
se1 = math.sqrt(V1[1, 1]); qlo, qhi = np.percentile(tstar, [2.5, 97.5])
h3 = {"beta": float(b[1]), "gamma": float(b[2]), "alpha": float(b[0]), "ci95_CR1": ci(V1), "ci95_CR2": ci(V2),
      "ci95_wild_cluster_bootstrap_t": [float(b[1] - qhi * se1), float(b[1] - qlo * se1)], "wild_draws": Bw}

# ---------- Holm with the other two original p-values ----------
def holm_sub(h, p):
    ps = dict(P_ORIG); ps[h] = p; keys = ["H2", "H4", "H5"]; adj = holm([ps[k_] for k_ in keys]); return adj[keys.index(h)]


alts = {"H2 Kruskal-Wallis": ("H2", h2_kw["p"]), "H4 Wilcoxon exact": ("H4", h4_w["p_one_sided_exact"]),
        "H4 one-sample t": ("H4", h4_t["p_one_sided"]), "H5 Stouffer": ("H5", h5_st["p_two_sided"]),
        "H5 Mann-Whitney exact + Fisher": ("H5", h5_mw["p_fisher"]), "H5 parametric bootstrap + Fisher": ("H5", h5_pb["p_fisher"])}
table = {name: {"hypothesis": h, "p": p, "p_holm": holm_sub(h, p), "reject_at_0.05": holm_sub(h, p) < 0.05,
                "original_p": P_ORIG[h]} for name, (h, p) in alts.items()}
res = {"checks": checks, "robustness_table": table, "H2_kruskal_wallis": h2_kw, "H4_wilcoxon": h4_w, "H4_t": h4_t,
       "H5_stouffer": h5_st, "H5_mann_whitney": h5_mw, "H5_parametric_bootstrap": h5_pb, "H3": h3,
       "notes": ["H4 simulated distribution of Delta(5) is reported with the E1 distributions (DEV-R1-001).",
                 "Exact Mann-Whitney with n = 5 per group has a minimum two-sided p of 2/252 = 0.0079 per state."],
       "numpy": np.__version__, "seed": SEED}
if not all(v for v in checks.values() if v is not None):
    print(json.dumps(checks, indent=1)); sys.exit("a cross-check against the original analysis failed; nothing written")
out = OUT / "s3_s4_results.json"; json.dump(res, open(out, "w"), indent=2, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("checks:", checks)
for name, v in table.items():
    print(f"{name:34s} p = {v['p']:.3g}  Holm = {v['p_holm']:.3g}  reject = {v['reject_at_0.05']}  (original p = {v['original_p']:.3g})")
print("H3 beta = %.4f  CR1 %s  CR2 %s  wild-t %s" % (h3["beta"], [round(x, 4) for x in h3["ci95_CR1"]],
      [round(x, 4) for x in h3["ci95_CR2"]], [round(x, 4) for x in h3["ci95_wild_cluster_bootstrap_t"]]))
print("wrote", out, sha(out))
