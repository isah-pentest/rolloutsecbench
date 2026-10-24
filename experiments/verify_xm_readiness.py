#!/usr/bin/env python3
"""
RolloutSecBench - X_m dataset readiness check (replaces experiments/verify_full_dataset_readiness.py).

Offline, no network. Confirms that the request population the v3 execution will send is exactly
the frozen X_m, with correctly typed features, BEFORE any traffic:
  1. xm_loader.load_xm() succeeds (hashes, population, labels, no fallbacks)
  2. per-family counts equal the frozen malicious_disagreement_by_attack_family_strict.csv
  3. categorical features are non-numeric strings exactly as in the source file; no value is '0.0'
  4. every payload has the 42 source features, in source_feature_order, and nothing else
Writes results/derived/phase9_v3_preflight/xm_readiness.json and exits non-zero on any failure.

Usage (repository root):  python3 experiments/verify_xm_readiness.py
"""
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workloads"))
import xm_loader  # noqa: E402

FAMILY_CSV = ROOT / "results/derived/malicious_disagreement_by_attack_family_strict.csv"


def main():
    checks, failures = {}, []

    def check(name, ok, detail=""):
        checks[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            failures.append(f"{name}: {detail}")

    try:
        samples, prov = xm_loader.load_xm(ROOT)
        check("loader", True, f"{len(samples):,} samples")
    except xm_loader.LoaderError as e:
        print(f"READINESS FAIL - loader refused: {e}", file=sys.stderr)
        return 2

    fam = Counter(s["attack_family"] for s in samples)
    with open(FAMILY_CSV, newline="") as fh:
        frozen_fam = {r["attack_cat"]: int(r["malicious_count"]) for r in csv.DictReader(fh)}
    check("family_counts_match_frozen", dict(fam) == frozen_fam,
          f"loaded {dict(sorted(fam.items()))} vs frozen {dict(sorted(frozen_fam.items()))}")

    names = json.loads((ROOT / xm_loader.FEATURE_NAMES).read_text())["source_feature_order"]
    cats = set(prov["categorical_features"])
    order_ok = all(list(s["features"]) == names for s in samples)
    check("payload_keys_exact_source_order", order_ok, f"{len(names)} features expected per payload")

    bad_cat = [(s["sample_id"], n, s["features"][n]) for s in samples for n in cats
               if not isinstance(s["features"][n], str) or s["features"][n] in ("", "0.0", "0")]
    check("categoricals_are_original_strings", not bad_cat, f"violations: {bad_cat[:5]}")
    cat_values = {n: dict(Counter(s["features"][n] for s in samples).most_common(6)) for n in sorted(cats)}

    bad_num = [(s["sample_id"], n) for s in samples for n in names if n not in cats
               and not isinstance(s["features"][n], (int, float))]
    check("numerics_are_numbers", not bad_num, f"violations: {bad_num[:5]}")

    verdict = "PASS" if not failures else "FAIL"
    out = ROOT / "results/derived/phase9_v3_preflight"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "artifact": "RolloutSecBench",
        "check": "X_m dataset readiness (offline)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "checks": checks,
        "loader_provenance": prov,
        "categorical_value_examples": cat_values,
        "loader_sha256": xm_loader._sha256(Path(xm_loader.__file__)),
        "script_sha256": xm_loader._sha256(Path(__file__)),
    }
    (out / "xm_readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    for k, v in checks.items():
        print(f"  [{'PASS' if v['pass'] else 'FAIL'}] {k}  {v['detail'][:110]}")
    print("  categorical examples:", json.dumps(cat_values)[:300])
    print(f"X_m READINESS: {verdict}  -> {out / 'xm_readiness.json'}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
