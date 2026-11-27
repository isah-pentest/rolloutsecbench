"""Revision r1, Phase 6 live, pre-engineering check (read-only, computes no outcome).

Pre-registration Section 5 defines the P3 key from "the 42 raw feature values as written in the strict-test CSV,
in feature_names.json order, joined with '|'". A live router never sees the CSV. It sees the JSON features
that v3_client.py sends: values typed by xm_loader.load_xm, serialised with json.dumps and parsed again.
This script checks, for every X_m sample, whether
    canon(features) = "|".join(v if isinstance(v, str) else repr(v) for v in features in feature order)
computed on the JSON round-tripped payload equals the pre-registered CSV-text key. It also checks that the
feature order the client sends matches the order the simulator used.
  100% equal -> the router can implement P3 exactly as pre-registered; no deviation.
  any mismatch -> STOP; the P3 key definition needs a recorded deviation before any router code.
"""
import csv, hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workloads"))
import xm_loader  # noqa: E402

fn = json.load(open(ROOT / "data/processed/feature_names.json"))
client_order = fn["source_feature_order"] if isinstance(fn, dict) else fn
sim_order = fn if isinstance(fn, list) else next(v for v in fn.values() if isinstance(v, list) and len(v) == 42)
print("feature orders identical (client vs simulator):", client_order == sim_order, "| n =", len(client_order))

samples, prov = xm_loader.load_xm(ROOT)
raw = {}
with open(ROOT / "data/splits/strict_test_nonoverlap.csv", newline="") as f:
    for r in csv.DictReader(f): raw[r["id"]] = r


def canon(feats, order):
    return "|".join(feats[n] if isinstance(feats[n], str) else repr(feats[n]) for n in order)


mism, types = [], {}
for s in samples:
    sid = str(s["sample_id"])
    wire = json.loads(json.dumps(s["features"]))            # what the router receives
    for n in client_order: types.setdefault(type(wire[n]).__name__, set()).add(n)
    a = canon(wire, sim_order); b = "|".join(raw[sid][n] for n in sim_order)
    if a != b:
        diff = [(n, repr(wire[n]), raw[sid][n]) for n in sim_order if (wire[n] if isinstance(wire[n], str) else repr(wire[n])) != raw[sid][n]]
        mism.append((sid, diff[:3]))
print("samples:", len(samples), "| mismatching keys:", len(mism))
print("wire value types:", {t: len(c) for t, c in types.items()})
for sid, d in mism[:10]: print("  example", sid, d)
keys_csv = {hashlib.sha256("|".join(raw[str(s["sample_id"])][n] for n in sim_order).encode()).hexdigest() for s in samples}
print("unique P3 keys (CSV text):", len(keys_csv))
print("RESULT:", "EXACT - implement P3 as pre-registered" if not mism and client_order == sim_order else "STOP - deviation needed")
