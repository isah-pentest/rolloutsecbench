"""Revision r1, Phase 5b, E2 on CIC-IDS2017 (CNS2022 corrected release): pre-registration Section 4 as amended by
Amendment 1 (A1.7), DEV-R1-003, DEV-R1-004. Run with ~/.venvs/rsb-revision/bin/python.

Inputs: the splits, preprocessor and feature list written by cic_prepare.py, each checked against the hashes in
revision/results/cic/prepare_summary.json, which is itself pinned here.
Grid: 4 families x {M1, M2} x seeds {101, 202, 303} = 24 pairs. Each pair is written to revision/results/e2_cic/pairs/
when it finishes. An interrupted run resumes and skips finished pairs. Summary files: e2_cic_pairs.csv and
e2_cic_summary.json, with min/q1/median/q3/max (numpy linear) by family and by mechanism, plus byte-identical
prediction groups.
There is no step 0 on this dataset (no published CIC-IDS2017 pair exists to reproduce).
"""
import csv, hashlib, json, math, pathlib, sys, time
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "revision/results/cic/prepare_summary.json"
SUMMARY_SHA = "91edb9c9adf914db53149680a1a0fdf4f05d2b4a26004233152473952530fefa"
PROTO_PATH = ROOT / "data/manifests/model_revision_protocol_v1.0.json"
PROTO_SHA = "f47d8bad1dd7064414e532a793e74782864db5b244bc91eb2baf2a831f91e29c"
OUT = ROOT / "revision/results/e2_cic"; (OUT / "pairs").mkdir(parents=True, exist_ok=True); (OUT / "preds").mkdir(exist_ok=True)
SEEDS = [101, 202, 303]; FAMILIES = ["xgb", "lgbm", "rf", "mlp"]; MECHS = ["M1", "M2"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


if sha(SUMMARY) != SUMMARY_SHA: sys.exit("prepare_summary.json hash mismatch")
if sha(PROTO_PATH) != PROTO_SHA: sys.exit("protocol hash mismatch")
S = json.load(open(SUMMARY))
for k, v in S["outputs"].items():
    if sha(ROOT / v["path"]) != v["sha256"]: sys.exit(f"hash mismatch: {k}")
PROTO = json.load(open(PROTO_PATH))
REQ = PROTO["model_acceptance_policy"]["v2"]["relative_noninferiority_requirements"]
TOL_F1, TOL_AUC = REQ["f1"]["absolute_tolerance"], REQ["auroc"]["absolute_tolerance"]
THR = PROTO["revision_v1"]["threshold_policy"]["threshold"]
feats = json.load(open(ROOT / S["outputs"]["feature_names"]["path"]))
if feats != S["features"]: sys.exit("feature list mismatch")
proto_cols = S["protocol_column"]
PRE = joblib.load(ROOT / S["outputs"]["preprocessor"]["path"])
dt = {c: str for c in proto_cols}
tr = pd.read_csv(ROOT / S["outputs"]["final_train"]["path"], dtype=dt)
va = pd.read_csv(ROOT / S["outputs"]["validation"]["path"], dtype=dt)
st_str = pd.read_csv(ROOT / S["outputs"]["strict_test"]["path"], dtype=str)  # exact text for unique-vector keys
st = pd.read_csv(ROOT / S["outputs"]["strict_test"]["path"], dtype=dt)
for nm, d in (("final_train", tr), ("validation", va), ("strict_test", st)):
    if len(d) != S["outputs"][nm]["rows"]: sys.exit(f"row count mismatch: {nm}")
mal = (st["label"] == 1).to_numpy()
te = st[mal].reset_index(drop=True); te_str = st_str[mal].reset_index(drop=True)
if len(te) != S["strict_test_binary_counts"]["1"]: sys.exit("malicious strict-test count mismatch")
VK = te_str[feats].agg("|".join, axis=1).tolist(); FAM = te["fine_label"].astype(str).tolist()
ytr, yva = tr["label"].to_numpy(), va["label"].to_numpy()
X = (PRE.transform(tr[feats]), PRE.transform(va[feats]), PRE.transform(te[feats]))
del tr, st, st_str
print(f"loaded: train {X[0].shape}, validation {X[1].shape}, malicious strict test {X[2].shape}", flush=True)


def make(family, seed, variant):
    if family == "xgb":
        p = dict(PROTO["revision_v2" if variant == "M2" else "revision_v1"]["hyperparameters"]); p["random_state"] = seed
        return XGBClassifier(**p)
    if family == "lgbm":
        p = dict(n_estimators=300, learning_rate=0.05, num_leaves=31)
        if variant == "M2": p.update(num_leaves=23, learning_rate=0.04, n_estimators=450, reg_lambda=1.5)
        return LGBMClassifier(**p, random_state=seed, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1)
    if family == "rf":
        p = dict(n_estimators=300, max_depth=None, min_samples_leaf=1)
        if variant == "M2": p.update(max_depth=20, min_samples_leaf=2)
        return RandomForestClassifier(**p, random_state=seed, n_jobs=-1)
    if family == "mlp":
        p = dict(hidden_layer_sizes=(64, 32), alpha=1e-4, early_stopping=True, max_iter=200)
        if variant == "M2": p.update(alpha=1e-3)
        return make_pipeline(StandardScaler(with_mean=False), MLPClassifier(**p, random_state=seed))
    raise ValueError(family)


def fit_eval(model):
    t0 = time.time(); model.fit(X[0], ytr)
    pva = model.predict_proba(X[1])[:, 1]; pte = model.predict_proba(X[2])[:, 1]
    return {"f1": float(f1_score(yva, (pva >= THR).astype(int))), "auroc": float(roc_auc_score(yva, pva)),
            "pred": (pte >= THR).astype(int), "fit_s": round(time.time() - t0, 1)}


def binom_p(b, c):
    n, k = b + c, max(b, c); return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def estimands(p1, p2):
    lab = np.where((p1 == 1) & (p2 == 0), "X12", np.where((p1 == 0) & (p2 == 1), "X21", np.where((p1 == 0) & (p2 == 0), "both", "det")))
    per_vec = {}
    for k_, l_ in zip(VK, lab):
        if per_vec.setdefault(k_, l_) != l_: sys.exit("conflicting labels within a feature vector")
    vl = np.array(list(per_vec.values())); n12, n21, nb = int((lab == "X12").sum()), int((lab == "X21").sum()), int((lab == "both").sum())
    xd = [f for f, l_ in zip(FAM, lab) if l_ in ("X12", "X21")]
    top = max(sorted(set(xd)), key=xd.count) if xd else None
    gain = lambda w: n12 * ((1 - (1 - w) ** 5) - w) + n21 * ((1 - w ** 5) - (1 - w))
    return {"X12": n12, "X21": n21, "both_miss": nb, "MRD_row": (n12 + n21) / len(lab),
            "McNemar_p_row": binom_p(n12, n21), "X12_vec": int((vl == "X12").sum()), "X21_vec": int((vl == "X21").sum()),
            "n_vec": len(vl), "MRD_vec": float(np.isin(vl, ["X12", "X21"]).mean()),
            "McNemar_p_vec": binom_p(int((vl == "X12").sum()), int((vl == "X21").sum())),
            "top_family": top, "top_family_share_XD": (xd.count(top) / len(xd)) if xd else None,
            "expected_gain_w0.1_n5": gain(0.1), "expected_gain_w0.5_n5": gain(0.5)}


cache = {}
for fam in FAMILIES:
    for s in SEEDS:
        for m in MECHS:
            pid = f"cic-{fam}-{m}-s{s}"; pf = OUT / "pairs" / f"{pid}.json"
            if pf.exists(): print(pid, "already done"); continue
            if (fam, s) not in cache: cache[(fam, s)] = fit_eval(make(fam, s, "M1"))
            v1 = cache[(fam, s)]
            v2 = fit_eval(make(fam, s + 1000, "M1") if m == "M1" else make(fam, s, "M2"))
            gate = v2["f1"] >= v1["f1"] - TOL_F1 and v2["auroc"] >= v1["auroc"] - TOL_AUC
            rec = {"pair": pid, "dataset": "CIC-IDS2017 (CNS2022)", "family": fam, "mechanism": m, "seed": s, "gate1_pass": bool(gate),
                   "v1_val_f1": v1["f1"], "v2_val_f1": v2["f1"], "v1_val_auroc": v1["auroc"], "v2_val_auroc": v2["auroc"],
                   "fit_s": [v1["fit_s"], v2["fit_s"]], **estimands(v1["pred"], v2["pred"])}
            pr = OUT / "preds" / f"{pid}.csv"
            pd.DataFrame({"id": te["id"], "v1": v1["pred"], "v2": v2["pred"]}).to_csv(pr, index=False)
            rec["preds_sha256"] = sha(pr); json.dump(rec, open(pf, "w"), indent=1)
            print(f"{pid}: gate={gate} X12/X21={rec['X12']}/{rec['X21']} MRD={rec['MRD_row']:.4%} "
                  f"gain(0.1)={rec['expected_gain_w0.1_n5']:.1f} fit_s={rec['fit_s']}", flush=True)

recs = [json.load(open(p)) for p in sorted((OUT / "pairs").glob("*.json"))]
if len(recs) != 24: sys.exit(f"expected 24 pairs, found {len(recs)}")
df = pd.DataFrame(recs); df.drop(columns=["fit_s"]).to_csv(OUT / "e2_cic_pairs.csv", index=False)
passing = df[df["gate1_pass"]]
EST = ["MRD_row", "MRD_vec", "X12", "X21", "both_miss", "X12_vec", "X21_vec", "top_family_share_XD", "expected_gain_w0.1_n5", "expected_gain_w0.5_n5"]


def q(x):
    x = x.dropna().astype(float)
    return {"n": 0} if len(x) == 0 else {"n": int(len(x)), "min": float(x.min()), "q1": float(np.percentile(x, 25)),
                                          "median": float(np.median(x)), "q3": float(np.percentile(x, 75)), "max": float(x.max())}


groups = df.groupby("preds_sha256")["pair"].apply(sorted)
summary = {"prepare_summary_sha256": SUMMARY_SHA, "pairs": len(df), "gate1_pass": int(df["gate1_pass"].sum()),
           "gate1_fail": sorted(df.loc[~df["gate1_pass"], "pair"]), "malicious_strict_test_rows": len(te),
           "quartile_method": "numpy.percentile, linear interpolation",
           "all_passing": {k: q(passing[k]) for k in EST},
           "by_family": {f: {k: q(g[k]) for k in EST} for f, g in passing.groupby("family")},
           "by_mechanism": {m: {k: q(g[k]) for k in EST} for m, g in passing.groupby("mechanism")},
           "zero_disagreement_pairs": sorted(df.loc[(df["X12"] == 0) & (df["X21"] == 0), "pair"]),
           "byte_identical_prediction_groups": [g for g in groups if len(g) > 1],
           "distinct_prediction_files": int(df["preds_sha256"].nunique()),
           "direction": {"X12_gt_X21": int((passing["X12"] > passing["X21"]).sum()), "X21_gt_X12": int((passing["X21"] > passing["X12"]).sum()),
                         "equal": int((passing["X12"] == passing["X21"]).sum())},
           "McNemar_row_p_lt_0.05": int((passing["McNemar_p_row"] < 0.05).sum()),
           "McNemar_vec_p_lt_0.05": int((passing["McNemar_p_vec"] < 0.05).sum())}
out = OUT / "e2_cic_summary.json"; json.dump(summary, open(out, "w"), indent=1, sort_keys=True)
for k in ("gate1_pass", "gate1_fail", "zero_disagreement_pairs", "byte_identical_prediction_groups", "distinct_prediction_files",
          "direction", "McNemar_row_p_lt_0.05", "McNemar_vec_p_lt_0.05"):
    print(k, "=", summary[k])
for g in ("by_family", "by_mechanism"):
    for name, v in summary[g].items():
        for k in ("MRD_row", "expected_gain_w0.1_n5"):
            if v[k]["n"]: print(f"{g:12s} {name:5s} {k:22s} min {v[k]['min']:.4g} q1 {v[k]['q1']:.4g} med {v[k]['median']:.4g} q3 {v[k]['q3']:.4g} max {v[k]['max']:.4g}")
print("wrote", out, sha(out))
