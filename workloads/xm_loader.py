#!/usr/bin/env python3
"""
RolloutSecBench - frozen X_m request loader (replaces experiments/run_experiment.py::fast_load_samples).

Builds one inference payload per malicious test sample EXACTLY as the frozen offline
evaluation saw it, and fails hard instead of falling back:

  * population   = the 38,263 ids in results/derived/malicious_predictions_strict.csv
                   (asserted equal as a set; every row must have label == 1)
  * source rows  = data/splits/strict_test_nonoverlap.csv, SHA-256 verified
  * features     = the 42 raw columns in feature_names.json["source_feature_order"],
                   in that order; categorical columns keep their ORIGINAL STRING value
                   (e.g. 'udp', '-', 'INT'); numeric columns get the type pandas.read_csv
                   would infer for the whole column (int64 -> int, otherwise float)
  * no defaults  = an empty, missing or unparsable numeric value aborts the load

Standard library only.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRICT_TEST = "data/splits/strict_test_nonoverlap.csv"
STRICT_TEST_SHA256 = "8081409731eb1780f13ff13374009594e1912743eddc2046513097e5d6830729"
FEATURE_NAMES = "data/processed/feature_names.json"
FEATURE_NAMES_SHA256 = "2d17e517d2e4ec044f421af2e84020c6b189b8f35a744d281bb6a126650d320a"
FROZEN_XM = "results/derived/malicious_predictions_strict.csv"
FROZEN_XM_SHA256 = "1a2b0e104eb497d7ea50e5dadd9834e5795e66ae24fb74d32d6bc19f67b0c04f"
SERVING_PROTOCOL = "data/manifests/phase5_serving_protocol_v1.0.json"
N_M = 38263


class LoaderError(RuntimeError):
    pass


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(rel, expected):
    path = ROOT / rel
    if not path.is_file():
        raise LoaderError(f"missing frozen input: {rel}")
    got = _sha256(path)
    if got != expected:
        raise LoaderError(f"SHA-256 mismatch for {rel}: expected {expected}, found {got}")
    return path


def _is_int_literal(s):
    s = s.strip()
    if s[:1] in "+-":
        s = s[1:]
    return s.isdigit()


def _as_float(name, s, row_id):
    try:
        v = float(s)
    except ValueError:
        raise LoaderError(f"non-numeric value {s!r} in numeric column {name!r} (id={row_id})")
    if math.isnan(v) or math.isinf(v):
        raise LoaderError(f"non-finite value {s!r} in column {name!r} (id={row_id})")
    return v


def load_xm(root=None):
    """Return (samples, provenance). samples: list of dicts with sample_id, ground_truth,
    attack_family, features (dict in source_feature_order)."""
    global ROOT
    if root is not None:
        ROOT = Path(root).resolve()

    test_path = _verify(STRICT_TEST, STRICT_TEST_SHA256)
    fn_path = _verify(FEATURE_NAMES, FEATURE_NAMES_SHA256)
    xm_path = _verify(FROZEN_XM, FROZEN_XM_SHA256)

    names = json.loads(fn_path.read_text())["source_feature_order"]
    proto_path = ROOT / SERVING_PROTOCOL
    if proto_path.is_file():
        contract = json.loads(proto_path.read_text())["input_contract"]
        categorical = set(contract["categorical_features"])
        excluded = set(contract.get("excluded_from_model_input", []))
        if excluded & set(names):
            raise LoaderError(f"source_feature_order contains excluded fields: {sorted(excluded & set(names))}")
    else:
        raise LoaderError(f"missing serving protocol {SERVING_PROTOCOL} (needed for categorical list)")

    with open(xm_path, newline="") as fh:
        frozen_ids = [r["id"] for r in csv.DictReader(fh)]
    if len(frozen_ids) != N_M or len(set(frozen_ids)) != N_M:
        raise LoaderError(f"frozen X_m has {len(frozen_ids)} rows / {len(set(frozen_ids))} unique ids, expected {N_M}")
    frozen_set = set(frozen_ids)

    with open(test_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    header = rows[0].keys() if rows else []
    missing = [n for n in names if n not in header]
    if missing:
        raise LoaderError(f"strict test file lacks feature columns: {missing}")

    # dtype inference over the WHOLE file, as pandas.read_csv did offline
    int_cols = {n for n in names if n not in categorical
                and all(_is_int_literal(r[n]) for r in rows)}

    by_id = {}
    for r in rows:
        if r["id"] in by_id:
            raise LoaderError(f"duplicate id in strict test file: {r['id']}")
        by_id[r["id"]] = r

    samples = []
    for sid in frozen_ids:  # frozen order
        r = by_id.get(sid)
        if r is None:
            raise LoaderError(f"frozen X_m id {sid} not found in strict test file")
        if r["label"].strip() != "1":
            raise LoaderError(f"frozen X_m id {sid} has label {r['label']!r} in strict test file")
        feats = {}
        for n in names:
            raw = r[n]
            if n in categorical:
                if raw == "":
                    raise LoaderError(f"empty categorical {n!r} (id={sid})")
                feats[n] = raw
            elif n in int_cols:
                feats[n] = int(raw)
            else:
                feats[n] = _as_float(n, raw, sid)
        samples.append({"sample_id": sid, "ground_truth": 1,
                        "attack_family": r["attack_cat"], "features": feats})

    if {s["sample_id"] for s in samples} != frozen_set or len(samples) != N_M:
        raise LoaderError("loaded population differs from frozen X_m")

    provenance = {
        "population": "frozen X_m (malicious_predictions_strict.csv ids)",
        "n_samples": len(samples),
        "strict_test_sha256": STRICT_TEST_SHA256,
        "feature_names_sha256": FEATURE_NAMES_SHA256,
        "frozen_xm_sha256": FROZEN_XM_SHA256,
        "categorical_features": sorted(categorical),
        "integer_features": sorted(int_cols),
        "float_features": sorted(set(names) - categorical - int_cols),
    }
    return samples, provenance


if __name__ == "__main__":
    s, p = load_xm()
    print(json.dumps(p, indent=2))
    print("first payload features:", json.dumps(s[0]["features"])[:400])
