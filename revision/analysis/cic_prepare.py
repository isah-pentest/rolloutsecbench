"""Revision r1, Phase 5b: prepare CIC-IDS2017 (CNS2022 corrected release) exactly as fixed in Amendment 1 (A1.3-A1.6).
Run with ~/.venvs/rsb-revision/bin/python. Trains no model.

Reads the five day CSVs straight from the pinned zip (Monday..Friday), applies the label rule (A1.4), the
column and row rules (A1.5) and the split (A1.6), fits the preprocessor on final train only, and writes:
  data/external/cicids2017_corrected/derived/{final_train,validation,strict_test}.csv.gz, preprocessor.joblib (gitignored)
  revision/results/cic/prepare_summary.json (committed): every count, every dropped column, every output hash.
Stops, without writing splits, whenever Amendment 1 says "stop and amend".

Split detail (A1.6 "stratified random split by fine label"): one numpy Generator(seed 11242026); fine labels in sorted
order; each label's row indices are permuted; the first round(0.70 n) go to final train, the next round(0.15 n) to
validation, the rest to test. Labels with fewer than 3 rows go wholly to final train.
"""
import hashlib, io, json, pathlib, sys, zipfile
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ROOT = pathlib.Path(__file__).resolve().parents[2]
ZIP = ROOT / "data/external/cicids2017_corrected/raw/CICIDS2017_improved.zip"
ZIP_SHA = "97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883"
DER = ROOT / "data/external/cicids2017_corrected/derived"; RES = ROOT / "revision/results/cic"
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
SEED = 11242026
IDENT = {"flowid", "srcip", "sourceip", "dstip", "destinationip", "srcport", "sourceport", "dstport", "destinationport", "timestamp", "id"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def stop(msg): sys.exit("STOP (amend before continuing): " + msg)


norm = lambda c: str(c).strip().lower().replace(" ", "").replace("_", "")
if sha(ZIP) != ZIP_SHA: stop("zip hash mismatch")
S = {"zip_sha256": ZIP_SHA, "seed": SEED}

# ---- A1.3: archive must hold exactly the five day CSVs ----
with zipfile.ZipFile(ZIP) as z:
    names = sorted(z.namelist())
    if sorted(f"{d}.csv" for d in DAYS) != names: stop(f"unexpected archive contents {names}")
    frames = []
    for d in DAYS:
        with z.open(f"{d}.csv") as f: df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df); print(f"read {d}: {len(df)} rows, {df.shape[1]} columns", flush=True)
cols0 = list(frames[0].columns)
if any(list(f.columns) != cols0 for f in frames): stop("day files have different columns")
df = pd.concat(frames, ignore_index=True); del frames
S["raw_rows"] = len(df); S["raw_columns"] = cols0

# ---- A1.4 labels ----
lab_cols = [c for c in cols0 if norm(c) == "label"]
if len(lab_cols) != 1: stop(f"label columns found: {lab_cols}")
LAB = lab_cols[0]
fine = df[LAB].astype(str).str.strip()
low = fine.str.lower()
y = np.where((low == "benign") | low.str.contains("attempted"), 0, 1)
S["fine_label_counts_raw"] = fine.value_counts().sort_index().to_dict()

# ---- A1.5 columns (i) and (ii) ----
drop_ident = [c for c in cols0 if norm(c) in IDENT]
drop_label = [c for c in cols0 if "label" in norm(c) or "attempted" in norm(c)]
keep = [c for c in cols0 if c not in drop_ident and c not in drop_label]
proto = [c for c in keep if norm(c) == "protocol"]
num = [c for c in keep if c not in proto]
nonnum = [c for c in num if not pd.api.types.is_numeric_dtype(df[c])]
if nonnum: stop(f"non-numeric retained columns: {nonnum}")
S["dropped_identifier_time"] = drop_ident; S["dropped_label_like"] = drop_label; S["protocol_column"] = proto

# ---- A1.5 rows with missing or infinite retained values ----
bad = df[num].isna().any(axis=1).to_numpy() | ~np.isfinite(df[num].to_numpy(dtype=np.float64)).all(axis=1)
if proto: bad |= df[proto[0]].isna().to_numpy()
S["rows_removed_missing_or_infinite"] = int(bad.sum())
S["rows_removed_by_fine_label"] = fine[bad].value_counts().sort_index().to_dict()
df = df.loc[~bad, keep].reset_index(drop=True); fine = fine[~bad].reset_index(drop=True); y = y[~bad]
df.insert(0, "id", np.arange(len(df))); df["fine_label"] = fine; df["label"] = y
S["rows_after_cleaning"] = len(df)
S["fine_label_counts_clean"] = fine.value_counts().sort_index().to_dict()
S["binary_counts_clean"] = {"0": int((y == 0).sum()), "1": int((y == 1).sum())}

# ---- A1.6 split ----
rng = np.random.default_rng(SEED); part = np.empty(len(df), dtype=object)
for lab in sorted(fine.unique()):
    idx = np.flatnonzero(fine.to_numpy() == lab); idx = rng.permutation(idx); n = len(idx)
    if n < 3: part[idx] = "train"; continue
    a = round(0.70 * n); b = a + round(0.15 * n)
    part[idx[:a]] = "train"; part[idx[a:b]] = "validation"; part[idx[b:]] = "test"
S["labels_with_fewer_than_3_rows"] = sorted(l for l, c in fine.value_counts().items() if c < 3)
tr, va, te = (df[part == p].reset_index(drop=True) for p in ("train", "validation", "test"))

# ---- A1.5 (iii): constant on final train ----
const = [c for c in proto + num if tr[c].nunique(dropna=False) <= 1]
S["dropped_constant_on_train"] = const
proto = [c for c in proto if c not in const]; num = [c for c in num if c not in const]; feats = proto + num
S["features"] = feats; S["n_features"] = len(feats)

# ---- strict test: remove test rows whose full retained feature vector occurs in final train ----
seen = tr[feats].drop_duplicates()
m = te[feats].merge(seen, on=feats, how="left", indicator=True)["_merge"].to_numpy()
st = te[m == "left_only"].reset_index(drop=True)
S["test_rows"] = len(te); S["strict_test_rows"] = len(st); S["test_rows_removed_overlap"] = int((m == "both").sum())
for nm, d in (("final_train", tr), ("validation", va), ("strict_test", st)):
    S[f"{nm}_fine_label_counts"] = d["fine_label"].value_counts().sort_index().to_dict()
    S[f"{nm}_binary_counts"] = {"0": int((d["label"] == 0).sum()), "1": int((d["label"] == 1).sum())}

# ---- preprocessor (same design as training/fit_preprocessor.build_preprocessor), final train only ----
for d in (tr, va, st):
    for c in proto: d[c] = d[c].astype(str)
pre = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=True, dtype=np.float32), proto),
                         ("numeric", "passthrough", num)], remainder="drop", sparse_threshold=1.0, verbose_feature_names_out=False)
pre.fit(tr[feats]); S["transformed_feature_count"] = int(pre.transform(tr[feats].head(1)).shape[1])

# ---- write ----
DER.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
out = {}
for nm, d in (("final_train", tr), ("validation", va), ("strict_test", st)):
    p = DER / f"{nm}.csv.gz"; d[["id"] + feats + ["fine_label", "label"]].to_csv(p, index=False, compression={"method": "gzip", "mtime": 0})
    out[nm] = {"path": str(p.relative_to(ROOT)), "sha256": sha(p), "rows": len(d)}
joblib.dump(pre, DER / "preprocessor.joblib"); out["preprocessor"] = {"path": str((DER / "preprocessor.joblib").relative_to(ROOT)), "sha256": sha(DER / "preprocessor.joblib")}
json.dump(feats, open(DER / "feature_names.json", "w"), indent=1); out["feature_names"] = {"path": str((DER / "feature_names.json").relative_to(ROOT)), "sha256": sha(DER / "feature_names.json")}
S["outputs"] = out
p = RES / "prepare_summary.json"; json.dump(S, open(p, "w"), indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))
for k in ("raw_rows", "rows_removed_missing_or_infinite", "rows_after_cleaning", "binary_counts_clean", "dropped_identifier_time",
          "dropped_label_like", "dropped_constant_on_train", "n_features", "transformed_feature_count", "test_rows", "strict_test_rows",
          "strict_test_binary_counts", "labels_with_fewer_than_3_rows"):
    print(k, "=", S[k])
print("fine labels (clean):", S["fine_label_counts_clean"])
for k, v in out.items(): print(k, v["sha256"], v.get("rows", ""))
print("wrote", p, sha(p))
