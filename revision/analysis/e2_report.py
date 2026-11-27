"""Revision r1, Phase 5, E2 report (pre-registration Section 4, "Summaries"). Reads the finished E2 outputs only;
trains nothing. Adds what e2_summary.json lacks: quartiles (numpy linear interpolation) by family and by mechanism,
the recorded ct_ feature list (Section 4, M4), and a duplicate-model check (pairs whose prediction files are
byte-identical, e.g. seed-invariant configurations). Output: revision/results/e2/e2_report.json.
"""
import hashlib, json, pathlib, sys
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
E2 = ROOT / "revision/results/e2"
EXPECT = {"e2_summary.json": "02c94f55787f84885460399813f6277147e9217e96d175c61a6afe7b9e60b38d",
          "e2_pairs.csv": "8dafb37178670d6b17db1ca3eda100431540a5b3c6615b1e793d47518fd0e5e6",
          "step0_reproduction.json": "8b0dbd049075b741246da2c8b399a037c7127144d6c631953beb15d4858c8d0b"}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for f, h in EXPECT.items():
    if sha(E2 / f) != h: sys.exit(f"hash mismatch: {f}")
recs = [json.load(open(p)) for p in sorted((E2 / "pairs").glob("*.json"))]
if len(recs) != 48: sys.exit("expected 48 pair records")
for r in recs:
    if sha(E2 / "preds" / f"{r['pair']}.csv") != r["preds_sha256"]: sys.exit(f"prediction file changed: {r['pair']}")
df = pd.DataFrame(recs); step0 = json.load(open(E2 / "step0_reproduction.json"))
EST = ["MRD_row", "MRD_vec", "X12", "X21", "both_miss", "X12_vec", "X21_vec", "top_family_share_XD",
       "expected_gain_w0.1_n5", "expected_gain_w0.5_n5"]


def q(s):
    s = s.dropna().astype(float)
    if len(s) == 0: return {"n": 0}
    return {"n": int(len(s)), "min": float(s.min()), "q1": float(np.percentile(s, 25)), "median": float(np.median(s)),
            "q3": float(np.percentile(s, 75)), "max": float(s.max())}


passing = df[df["gate1_pass"]]
names = json.load(open(ROOT / "data/processed/feature_names.json"))
if not isinstance(names, list): names = next(v for v in names.values() if isinstance(v, list) and len(v) == 42)
groups = df.groupby("preds_sha256")["pair"].apply(sorted)
orig = step0["original_pair"]
report = {
    "inputs_sha256": EXPECT,
    "quartile_method": "numpy.percentile, linear interpolation",
    "gate1": {"pass": int(df["gate1_pass"].sum()), "fail": sorted(df.loc[~df["gate1_pass"], "pair"])},
    "all_passing": {k: q(passing[k]) for k in EST},
    "by_family": {f: {k: q(g[k]) for k in EST} for f, g in passing.groupby("family")},
    "by_mechanism": {m: {k: q(g[k]) for k in EST} for m, g in passing.groupby("mechanism")},
    "ct_features_dropped_in_M4": [c for c in names if c.startswith("ct_")],
    "zero_disagreement_pairs": sorted(df.loc[(df["X12"] == 0) & (df["X21"] == 0), "pair"]),
    "byte_identical_prediction_groups": [g for g in groups if len(g) > 1],
    "distinct_prediction_files": int(df["preds_sha256"].nunique()),
    "direction": {"X12_gt_X21": int((passing["X12"] > passing["X21"]).sum()),
                  "X21_gt_X12": int((passing["X21"] > passing["X12"]).sum()),
                  "equal": int((passing["X12"] == passing["X21"]).sum())},
    "McNemar_row_p_lt_0.05": int((passing["McNemar_p_row"] < 0.05).sum()),
    "McNemar_vec_p_lt_0.05": int((passing["McNemar_p_vec"] < 0.05).sum()),
    "original_pair": {k: orig[k] for k in ("X12", "X21", "MRD_row", "MRD_vec", "expected_gain_w0.1_n5", "expected_gain_w0.5_n5")},
    "original_pair_position": {k: {"below": int((passing[k] < orig[k]).sum()), "equal": int((passing[k] == orig[k]).sum()),
                                   "above": int((passing[k] > orig[k]).sum())}
                               for k in ("MRD_row", "MRD_vec", "expected_gain_w0.1_n5", "expected_gain_w0.5_n5")},
}
out = E2 / "e2_report.json"; json.dump(report, open(out, "w"), indent=1, sort_keys=True)
print("ct_ features:", report["ct_features_dropped_in_M4"])
print("zero-disagreement pairs:", report["zero_disagreement_pairs"])
print("byte-identical prediction groups:", report["byte_identical_prediction_groups"])
print("distinct prediction files:", report["distinct_prediction_files"], "of 48")
print("direction:", report["direction"], "| McNemar p<0.05 row/vec:", report["McNemar_row_p_lt_0.05"], report["McNemar_vec_p_lt_0.05"])
print("original pair position:", report["original_pair_position"])
for k in ("MRD_row", "expected_gain_w0.1_n5"):
    for g in ("by_family", "by_mechanism"):
        for name, v in report[g].items():
            print(f"{g:12s} {name:5s} {k:22s} min {v[k]['min']:.4g} q1 {v[k]['q1']:.4g} med {v[k]['median']:.4g} q3 {v[k]['q3']:.4g} max {v[k]['max']:.4g}")
print("wrote", out, sha(out))
