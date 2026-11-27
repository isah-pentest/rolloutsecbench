"""Revision r1, Phase 4, S1 (pre-registered, Section 8): duplicate-aware malicious revision disagreement.

Reports, per row and per unique feature vector:
  MRD with an exact Clopper-Pearson 95% interval (rows: as in the manuscript; vectors: one observation per vector);
  row-level MRD with a cluster-bootstrap 95% percentile interval (unique vector = cluster, 10,000 resamples,
  numpy.random.default_rng(11242026));
  the directional split and its exact two-sided binomial (McNemar) p-value.
Inputs are checked against the phase9_v3 freeze before use. Writes revision/results/s1_duplicates/.
"""
import csv, hashlib, json, math, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "revision/results/s1_duplicates"; OUT.mkdir(parents=True, exist_ok=True)
SEED, B = 11242026, 10_000


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


freeze = json.load(open(ROOT / "data/manifests/phase9_v3/phase9_v3_freeze.json"))
for rel, want in freeze["frozen_inputs_sha256"].items():
    if sha(ROOT / rel) != want: sys.exit(f"input hash mismatch: {rel}")


def log_binom_pmf(k, n, p):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


def tail_ge(x, n, p):  # P(X >= x)
    if x <= 0: return 1.0
    return min(1.0, sum(math.exp(log_binom_pmf(k, n, p)) for k in range(x, n + 1)))


def clopper_pearson(x, n, a=0.05):
    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if x == 0 else solve(lambda p: tail_ge(x, n, p) < a / 2)
    upper = 1.0 if x == n else solve(lambda p: 1 - tail_ge(x + 1, n, p) > a / 2)
    return lower, upper


def mcnemar_exact(b, c):  # two-sided exact binomial test of b vs c, p = 0.5
    n, k = b + c, max(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


names = json.load(open(ROOT / "data/processed/feature_names.json"))
if not isinstance(names, list):
    names = next(v for v in names.values() if isinstance(v, list) and len(v) == 42)
vec = {}
with open(ROOT / "data/splits/strict_test_nonoverlap.csv", newline="") as f:
    for r in csv.DictReader(f): vec[r["id"]] = tuple(r[n] for n in names)
cluster = {}  # vector -> [rows, set label]
with open(ROOT / "results/derived/malicious_predictions_strict.csv", newline="") as f:
    for r in csv.DictReader(f):
        lab = {(1, 1): "det", (1, 0): "X12", (0, 1): "X21", (0, 0): "both"}[(int(r["v1_prediction"]), int(r["v2_prediction"]))]
        c = cluster.setdefault(vec[r["id"]], [0, lab])
        if c[1] != lab: sys.exit("conflicting labels within one feature vector")
        c[0] += 1
rows = np.array([c[0] for c in cluster.values()]); labs = np.array([c[1] for c in cluster.values()])
in_xd = np.isin(labs, ["X12", "X21"])
n_rows, n_vec = int(rows.sum()), len(rows)
xd_rows, xd_vec = int(rows[in_xd].sum()), int(in_xd.sum())
x12_rows, x21_rows = int(rows[labs == "X12"].sum()), int(rows[labs == "X21"].sum())
x12_vec, x21_vec = int((labs == "X12").sum()), int((labs == "X21").sum())

rng = np.random.default_rng(SEED)
pv = np.full(n_vec, 1.0 / n_vec)
boot = np.empty(B); done = 0
while done < B:
    m = min(500, B - done)
    counts = rng.multinomial(n_vec, pv, size=m)
    boot[done:done + m] = (counts[:, in_xd] @ rows[in_xd]) / (counts @ rows)
    done += m

res = {
    "rows": {"n": n_rows, "XD": xd_rows, "X12": x12_rows, "X21": x21_rows, "MRD": xd_rows / n_rows,
             "clopper_pearson_95": clopper_pearson(xd_rows, n_rows),
             "cluster_bootstrap_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
             "mcnemar_exact_p": mcnemar_exact(x12_rows, x21_rows)},
    "unique_vectors": {"n": n_vec, "XD": xd_vec, "X12": x12_vec, "X21": x21_vec, "MRD": xd_vec / n_vec,
                       "clopper_pearson_95": clopper_pearson(xd_vec, n_vec),
                       "mcnemar_exact_p": mcnemar_exact(x12_vec, x21_vec)},
    "bootstrap": {"B": B, "generator": "numpy.random.default_rng", "seed": SEED, "cluster": "unique 42-feature vector",
                  "numpy": np.__version__},
}
out = OUT / "s1_results.json"; json.dump(res, open(out, "w"), indent=2, sort_keys=True)
print(json.dumps(res, indent=1))
print("wrote", out, sha(out))
