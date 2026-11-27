"""Revision r1, Phase 5, E2: generalization across revision pairs on UNSW-NB15 (pre-registration Section 4;
DEV-R1-003, DEV-R1-004). Run with ~/.venvs/rsb-revision/bin/python.

Step 0 retrains the original V1 and V2 (frozen protocol, seed 42) and requires their predictions on the
38,263 strict-test malicious samples to equal the frozen predictions exactly; otherwise nothing else runs.
Grid: 4 families x 4 mechanisms x seeds {101, 202, 303} = 48 pairs. Each pair is written to
revision/results/e2/pairs/<pair>.json as soon as it finishes (an interrupted run resumes; finished pairs are
skipped). Summary: revision/results/e2/e2_pairs.csv and e2_summary.json.
"""
import csv, hashlib, json, math, pathlib, sys, time
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "revision/results/e2"; (OUT / "pairs").mkdir(parents=True, exist_ok=True); (OUT / "preds").mkdir(exist_ok=True)
PINNED = {
    "data/manifests/model_revision_protocol_v1.0.json": "f47d8bad1dd7064414e532a793e74782864db5b244bc91eb2baf2a831f91e29c",
    "models/preprocessing/preprocessor.joblib": "4ea29c9353505b423ce37b077f3ef821f76da426230518d771ebe217e3c2408f",
    "data/processed/feature_names.json": "2d17e517d2e4ec044f421af2e84020c6b189b8f35a744d281bb6a126650d320a",
}
SEEDS = [101, 202, 303]; FAMILIES = ["xgb", "lgbm", "rf", "mlp"]; MECHS = ["M1", "M2", "M3", "M4"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


FREEZE = json.load(open(ROOT / "data/manifests/phase9_v3/phase9_v3_freeze.json"))  # same inputs check as S1
for rel, want in {**PINNED, **FREEZE["frozen_inputs_sha256"]}.items():
    if sha(ROOT / rel) != want: sys.exit(f"hash mismatch: {rel}")
for rel in ("data/splits/strict_test_nonoverlap.csv", "results/derived/malicious_predictions_strict.csv"):
    if rel not in FREEZE["frozen_inputs_sha256"]: sys.exit(f"not pinned by the freeze: {rel}")
inputs_sha = {rel: sha(ROOT / rel) for rel in ("data/splits/final_train.csv", "data/splits/validation.csv")}
PROTO = json.load(open(ROOT / "data/manifests/model_revision_protocol_v1.0.json"))
TOL_F1 = PROTO["model_acceptance_policy"]["v2"]["relative_noninferiority_requirements"]["f1"]["absolute_tolerance"]
TOL_AUC = PROTO["model_acceptance_policy"]["v2"]["relative_noninferiority_requirements"]["auroc"]["absolute_tolerance"]
THR = PROTO["revision_v1"]["threshold_policy"]["threshold"]
names = json.load(open(ROOT / "data/processed/feature_names.json"))
if not isinstance(names, list): names = next(v for v in names.values() if isinstance(v, list) and len(v) == 42)
PRE = joblib.load(ROOT / "models/preprocessing/preprocessor.joblib")
CAT = list(next(t[2] for t in PRE.transformers_ if t[0] == "categorical"))
NUM = list(next(t[2] for t in PRE.transformers_ if t[0] == "numeric"))
CT = [c for c in names if c.startswith("ct_")]

tr = pd.read_csv(ROOT / "data/splits/final_train.csv"); va = pd.read_csv(ROOT / "data/splits/validation.csv")
te = pd.read_csv(ROOT / "data/splits/strict_test_nonoverlap.csv")
te = te[te["label"] == 1].reset_index(drop=True)
frozen = pd.read_csv(ROOT / "results/derived/malicious_predictions_strict.csv")
if len(te) != 38263 or sorted(te["id"].astype(str)) != sorted(frozen["id"].astype(str)): sys.exit("strict-test malicious set mismatch")
need = {"v1_prediction", "v2_prediction", "v1_probability", "v2_probability"}
if not need <= set(frozen.columns): sys.exit(f"prediction columns missing: {sorted(need - set(frozen.columns))}; columns = {list(frozen.columns)}")
frozen = frozen.set_index(frozen["id"].astype(str)).loc[te["id"].astype(str)]
vec_key = {}
with open(ROOT / "data/splits/strict_test_nonoverlap.csv", newline="") as f:
    for r in csv.DictReader(f): vec_key[r["id"]] = "|".join(r[n] for n in names)
VK = [vec_key[str(i)] for i in te["id"]]; FAM = te["attack_cat"].astype(str).tolist()
ytr, yva = tr["label"].to_numpy(), va["label"].to_numpy()


def build_pre(cat, num):  # identical design to training/fit_preprocessor.build_preprocessor
    return ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=True, dtype=np.float32), cat),
                              ("numeric", "passthrough", num)], remainder="drop", sparse_threshold=1.0, verbose_feature_names_out=False)


PRE_M4 = build_pre([c for c in CAT if c not in CT], [c for c in NUM if c not in CT]).fit(tr[[c for c in names if c not in CT]])
X = {"full": (PRE.transform(tr[names]), PRE.transform(va[names]), PRE.transform(te[names]))}
X["m4"] = tuple(PRE_M4.transform(d[[c for c in names if c not in CT]]) for d in (tr, va, te))


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


def fit_eval(model, xs, rows=None):
    Xtr, Xva, Xte = xs; Xf, yf = (Xtr, ytr) if rows is None else (Xtr[rows], ytr[rows])
    t0 = time.time(); model.fit(Xf, yf)
    pva = model.predict_proba(Xva)[:, 1]; pte = model.predict_proba(Xte)[:, 1]
    return {"f1": float(f1_score(yva, (pva >= THR).astype(int))), "auroc": float(roc_auc_score(yva, pva)),
            "pred": (pte >= THR).astype(int), "prob": pte, "fit_s": round(time.time() - t0, 1)}


def binom_p(b, c):
    n, k = b + c, max(b, c); return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def estimands(p1, p2):
    lab = np.where((p1 == 1) & (p2 == 0), "X12", np.where((p1 == 0) & (p2 == 1), "X21", np.where((p1 == 0) & (p2 == 0), "both", "det")))
    per_vec = {}
    for k_, l_ in zip(VK, lab):
        if per_vec.setdefault(k_, l_) != l_: sys.exit("conflicting labels within a feature vector")
    vl = np.array(list(per_vec.values())); n12, n21, nb = int((lab == "X12").sum()), int((lab == "X21").sum()), int((lab == "both").sum())
    xd = [f for f, l_ in zip(FAM, lab) if l_ in ("X12", "X21")]
    top = max(set(xd), key=xd.count) if xd else None
    gain = lambda w: n12 * ((1 - (1 - w) ** 5) - w) + n21 * ((1 - w ** 5) - (1 - w))
    return {"X12": n12, "X21": n21, "both_miss": nb, "MRD_row": (n12 + n21) / len(lab),
            "McNemar_p_row": binom_p(n12, n21), "X12_vec": int((vl == "X12").sum()), "X21_vec": int((vl == "X21").sum()),
            "n_vec": len(vl), "MRD_vec": float(np.isin(vl, ["X12", "X21"]).mean()),
            "McNemar_p_vec": binom_p(int((vl == "X12").sum()), int((vl == "X21").sum())),
            "top_family": top, "top_family_share_XD": (xd.count(top) / len(xd)) if xd else None,
            "expected_gain_w0.1_n5": gain(0.1), "expected_gain_w0.5_n5": gain(0.5)}


# ---------- step 0: reproduce the published V1 and V2 ----------
r1 = fit_eval(make("xgb", 42, "M1"), X["full"]); r2 = fit_eval(make("xgb", 42, "M2"), X["full"])
same = bool((r1["pred"] == frozen["v1_prediction"].to_numpy()).all() and (r2["pred"] == frozen["v2_prediction"].to_numpy()).all())
dp = max(float(np.abs(r1["prob"] - frozen["v1_probability"].to_numpy()).max()), float(np.abs(r2["prob"] - frozen["v2_probability"].to_numpy()).max()))
step0 = {"predictions_identical": same, "max_abs_prob_diff": dp, "original_pair": estimands(r1["pred"], r2["pred"]),
         "val_f1": [r1["f1"], r2["f1"]], "val_auroc": [r1["auroc"], r2["auroc"]]}
json.dump(step0, open(OUT / "step0_reproduction.json", "w"), indent=1)
print(f"step 0: predictions identical = {same}, max |dp| = {dp:.3g}, original pair X12/X21 = "
      f"{step0['original_pair']['X12']}/{step0['original_pair']['X21']}, val F1 {r1['f1']:.4f}/{r2['f1']:.4f}", flush=True)
if not same or dp >= 1e-6: sys.exit("STOP: the revision environment does not reproduce the published models")

# ---------- grid ----------
cache = {}
for fam in FAMILIES:
    for s in SEEDS:
        for m in MECHS:
            pid = f"{fam}-{m}-s{s}"; pf = OUT / "pairs" / f"{pid}.json"
            if pf.exists(): print(pid, "already done"); continue
            if m == "M3":
                idx = np.arange(len(ytr))
                a = train_test_split(idx, train_size=0.9, stratify=ytr, random_state=s)[0]
                b = train_test_split(idx, train_size=0.9, stratify=ytr, random_state=s + 5000)[0]
                v1 = fit_eval(make(fam, s, "M1"), X["full"], np.sort(a)); v2 = fit_eval(make(fam, s, "M1"), X["full"], np.sort(b))
            else:
                if (fam, s) not in cache: cache[(fam, s)] = fit_eval(make(fam, s, "M1"), X["full"])
                v1 = cache[(fam, s)]
                v2 = fit_eval(make(fam, s + 1000 if m == "M1" else s, m if m == "M2" else "M1"), X["m4" if m == "M4" else "full"])
            gate = v2["f1"] >= v1["f1"] - TOL_F1 and v2["auroc"] >= v1["auroc"] - TOL_AUC
            rec = {"pair": pid, "family": fam, "mechanism": m, "seed": s, "gate1_pass": bool(gate),
                   "v1_val_f1": v1["f1"], "v2_val_f1": v2["f1"], "v1_val_auroc": v1["auroc"], "v2_val_auroc": v2["auroc"],
                   "fit_s": [v1["fit_s"], v2["fit_s"]], **estimands(v1["pred"], v2["pred"])}
            pr = OUT / "preds" / f"{pid}.csv"
            pd.DataFrame({"id": te["id"], "v1": v1["pred"], "v2": v2["pred"]}).to_csv(pr, index=False)
            rec["preds_sha256"] = sha(pr); json.dump(rec, open(pf, "w"), indent=1)
            print(f"{pid}: gate={gate} X12/X21={rec['X12']}/{rec['X21']} MRD={rec['MRD_row']:.4%} "
                  f"gain(0.1)={rec['expected_gain_w0.1_n5']:.1f}", flush=True)

# ---------- summary ----------
recs = [json.load(open(p)) for p in sorted((OUT / "pairs").glob("*.json"))]
if len(recs) != 48: sys.exit(f"expected 48 pairs, found {len(recs)}")
with open(OUT / "e2_pairs.csv", "w", newline="") as f:
    keys = [k for k in recs[0] if k != "fit_s"]; w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(recs)
passing = [r for r in recs if r["gate1_pass"]]


def dist(rs, k):
    v = sorted(r[k] for r in rs)
    return {"n": len(v), "min": v[0], "median": float(np.median(v)), "max": v[-1]} if v else {"n": 0}


summary = {"pairs": len(recs), "gate1_pass": len(passing), "gate1_fail": [r["pair"] for r in recs if not r["gate1_pass"]],
           "inputs_sha256": inputs_sha, "step0": {k: step0[k] for k in ("predictions_identical", "max_abs_prob_diff")},
           "passing": {k: dist(passing, k) for k in ("MRD_row", "MRD_vec", "X12", "X21", "expected_gain_w0.1_n5", "expected_gain_w0.5_n5")},
           "by_family": {fm: {k: dist([r for r in passing if r["family"] == fm], k) for k in ("MRD_row", "expected_gain_w0.1_n5")} for fm in FAMILIES},
           "by_mechanism": {m: {k: dist([r for r in passing if r["mechanism"] == m], k) for k in ("MRD_row", "expected_gain_w0.1_n5")} for m in MECHS},
           "pairs_with_X12_gt_X21": sum(r["X12"] > r["X21"] for r in passing),
           "pairs_with_McNemar_row_p_lt_0.05": sum(r["McNemar_p_row"] < 0.05 for r in passing),
           "original_pair_MRD_row_rank": sum(r["MRD_row"] < step0["original_pair"]["MRD_row"] for r in passing) / max(len(passing), 1)}
out = OUT / "e2_summary.json"; json.dump(summary, open(out, "w"), indent=1, sort_keys=True)
print(json.dumps(summary["passing"], indent=1)); print("wrote", out, sha(out))
