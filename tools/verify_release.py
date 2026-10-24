#!/usr/bin/env python3
"""
RolloutSecBench - release verification (standard library only, read-only).

Checks, in order, and reports PASS / FAIL / SKIP for each:

  1. release    every file listed in RELEASE_MANIFEST.sha256 is present and unmodified
  2. published  every artifact listed in environment/published_hashes.sha256 that is present locally
                matches the hash printed in the paper (Table S1, Figure 1, frozen predictions);
                absent artifacts (dataset, evidence archive) are reported as SKIP, not as failures
  3. prefixes   artifacts identified in Figure 1 only by an abbreviated hash (models, freeze manifest)
                match that prefix when present
  4. pinned     if the freeze manifest is present, every file in its pinned_code_sha256 matches the
                repository copy byte for byte (the code that ran the confirmatory campaign)
  5. routing    the routing rule implemented by workloads/make_v3_freeze.py (used by the analysis) and
                by workloads/validate_v3_run.py equals the rule specified in the frozen router protocol,
                on 20,000 pseudo-random (run_id, request_sequence, weight) triples

Usage (repository root):  python3 tools/verify_release.py [--strict]
  --strict  treat SKIP as failure (use once the dataset and the evidence archive are in place)
Exit status 0 only if nothing FAILED (and, with --strict, nothing was skipped).
"""
import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path.cwd()
FIG1_PREFIXES = {  # abbreviated identities on the provenance rail of Figure 1
    "V1 model": "6c0aa79a", "V2 model": "ea664630", "freeze manifest": "67d0ac98"}
FREEZE = ROOT / "data/manifests/phase9_v3/phase9_v3_freeze.json"
PROTOCOL = ROOT / "data/manifests/phase7/router/phase7_router_protocol_v1.0.json"

results = []


def report(check, status, detail):
    results.append((check, status, detail))
    print(f"[{status:4}] {check:9} {detail}")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_sums(path):
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            h, rel = line.split(None, 1)
            out.append((h, rel.lstrip("*")))
    return out


def check_sums(name, path, missing_is_skip):
    if not Path(path).exists():
        report(name, "SKIP", f"{path} not present")
        return
    bad = 0
    for h, rel in read_sums(path):
        p = ROOT / rel
        if not p.exists():
            if missing_is_skip:
                report(name, "SKIP", f"{rel} not present")
            else:
                report(name, "FAIL", f"{rel} missing")
                bad += 1
            continue
        if sha(p) != h:
            report(name, "FAIL", f"{rel} hash differs")
            bad += 1
        else:
            report(name, "PASS", rel)
    return bad


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(mod)
    return mod


def check_prefixes():
    if not FREEZE.exists():
        for k, v in FIG1_PREFIXES.items():
            report("prefixes", "SKIP", f"{k} ({v}...): freeze manifest not present")
        return
    fh = sha(FREEZE)
    report("prefixes", "PASS" if fh.startswith(FIG1_PREFIXES["freeze manifest"]) else "FAIL",
           f"freeze manifest sha256 {fh[:16]}... (Figure 1: {FIG1_PREFIXES['freeze manifest']}...)")
    for rev, key in (("v1", "V1 model"), ("v2", "V2 model")):
        p = ROOT / f"models/{rev}/model_manifest.json"
        if not p.exists():
            report("prefixes", "SKIP", f"{key}: models/{rev}/model_manifest.json not present")
            continue
        m = json.loads(p.read_text())
        cands = {"model artifact": m.get("artifacts", {}).get("model", {}).get("sha256"),
                 "model manifest": sha(p)}
        hit = [k for k, h in cands.items() if h and h.startswith(FIG1_PREFIXES[key])]
        report("prefixes", "PASS" if hit else "FAIL",
               f"{key}: Figure 1 prefix {FIG1_PREFIXES[key]}... "
               + (f"matches the {hit[0]} hash" if hit else "matches neither the model artifact nor its manifest"))


def check_pinned():
    if not FREEZE.exists():
        report("pinned", "SKIP", "freeze manifest not present (part of the evidence archive)")
        return
    fr = json.loads(FREEZE.read_text())
    pinned = fr.get("pinned_code_sha256", {})
    if not pinned:
        report("pinned", "FAIL", "freeze manifest has no pinned_code_sha256")
        return
    for rel, h in sorted(pinned.items()):
        p = ROOT / rel
        if not p.exists():
            report("pinned", "FAIL", f"{rel} missing")
        elif sha(p) != h:
            report("pinned", "FAIL", f"{rel} differs from the code pinned at freeze")
        else:
            report("pinned", "PASS", rel)


def check_routing():
    if not PROTOCOL.exists():
        report("routing", "FAIL", f"{PROTOCOL.relative_to(ROOT)} missing")
        return
    proto = json.loads(PROTOCOL.read_text())["routing"]
    basis = proto["route_seed_basis_sha256"]
    if (proto.get("digest_bytes"), proto.get("uniform_denominator"), proto.get("material")) != (
            "first_8_big_endian_unsigned", "2^64", "route_seed_basis_sha256|run_id|request_sequence"):
        report("routing", "FAIL", "protocol routing specification is not the one this check implements")
        return

    def spec(run_id, seq, w2):
        d = hashlib.sha256(f"{basis}|{run_id}|{seq}".encode("utf-8")).digest()
        return "v2" if int.from_bytes(d[:8], "big", signed=False) / 2 ** 64 < w2 else "v1"

    try:
        freeze_mod = load_module(ROOT / "workloads/make_v3_freeze.py", "make_v3_freeze")
    except Exception as e:  # noqa: BLE001
        report("routing", "FAIL", f"cannot import workloads/make_v3_freeze.py: {type(e).__name__}: {e}")
        return
    if freeze_mod.BASIS != basis:
        report("routing", "FAIL", "make_v3_freeze.BASIS differs from the protocol basis")
        return
    try:
        val_mod = load_module(ROOT / "workloads/validate_v3_run.py", "validate_v3_run")
    except Exception as e:  # noqa: BLE001
        report("routing", "FAIL", f"cannot import workloads/validate_v3_run.py: {type(e).__name__}: {e}")
        return
    rng = random.Random(20261023)
    weights = [0.0, 0.1, 0.5, 1.0]
    mism_f = mism_v = 0
    for _ in range(20_000):
        rid = f"V3-R{rng.randrange(4)}-A{rng.randrange(2)}-P{rng.randrange(1, 6):02d}"
        seq = rng.randrange(1, 191_316)
        w = rng.choice(weights)
        ref = spec(rid, seq, w)
        mism_f += freeze_mod.route(rid, seq, w) != ref
        mism_v += val_mod.route(basis, rid, seq, w)[1] != ref
    report("routing", "PASS" if mism_f == 0 else "FAIL",
           f"make_v3_freeze.route (analysis) vs protocol: {mism_f} mismatches in 20,000 triples")
    report("routing", "PASS" if mism_v == 0 else "FAIL",
           f"validate_v3_run.route (validator) vs protocol: {mism_v} mismatches in 20,000 triples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    check_sums("release", ROOT / "RELEASE_MANIFEST.sha256", missing_is_skip=False)
    check_sums("published", ROOT / "environment/published_hashes.sha256", missing_is_skip=True)
    check_prefixes()
    check_pinned()
    check_routing()
    n_fail = sum(s == "FAIL" for _, s, _ in results)
    n_skip = sum(s == "SKIP" for _, s, _ in results)
    n_pass = sum(s == "PASS" for _, s, _ in results)
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if n_fail or (a.strict and n_skip):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
