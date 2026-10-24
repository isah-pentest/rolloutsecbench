#!/usr/bin/env python3
"""
RolloutSecBench - X_m serving-equivalence gate (extends Phase-5 / Section K to the attack population).

Sends every one of the 38,263 frozen X_m samples DIRECTLY to each backend (bypassing the
router), using xm_loader.load_xm(), and compares the served response with the frozen
offline prediction in results/derived/malicious_predictions_strict.csv.

PASS iff, for BOTH revisions:
  * every request returns HTTP 200 and the expected model_version
  * discrete prediction == frozen prediction for all 38,263 samples (zero tolerance)
  * |served probability - frozen probability| < 1e-4 for all samples (Section K epsilon)

Outputs (results/derived/phase9_xm_equivalence/):
  xm_equivalence_<rev>.csv   one row per sample
  xm_equivalence_summary.json  verdict, counts, max/median deltas, input hashes, script hash

Usage (backends reached via kubectl port-forward, see run instructions):
  python3 workloads/xm_equivalence_gate.py --v1-url http://127.0.0.1:18101/predict \
                                           --v2-url http://127.0.0.1:18102/predict
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xm_loader  # noqa: E402

EPSILON = 1e-4


def post(url, request_id, features, timeout):
    body = json.dumps({"request_id": request_id, "features": features}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    last = None
    for _ in range(3):  # transport retries only; the request is idempotent
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, {"error": e.read().decode(errors="replace")[:300]}
        except Exception as e:  # connection errors
            last = e
            time.sleep(0.5)
    return 0, {"error": repr(last)}


def run_revision(rev, url, samples, frozen, out_dir, workers, timeout):
    def one(i_s):
        i, s = i_s
        status, resp = post(url, f"XMEQ-{rev}-{i:05d}", s["features"], timeout)
        return s, status, resp

    rows, deltas = [], []
    counts = {"http_non200": 0, "wrong_model_version": 0, "decision_mismatch": 0, "prob_over_eps": 0}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for s, status, resp in ex.map(one, enumerate(samples)):
            f_pred, f_prob = frozen[s["sample_id"]][rev]
            ok = status == 200
            pred = resp.get("prediction") if ok else None
            prob = resp.get("probability") if ok else None
            mv = (resp.get("model_version") or "").lower() if ok else ""
            if not ok:
                counts["http_non200"] += 1
            elif mv != rev:
                counts["wrong_model_version"] += 1
            d = abs(float(prob) - f_prob) if ok else None
            if ok:
                deltas.append(d)
                if int(pred) != f_pred:
                    counts["decision_mismatch"] += 1
                if d >= EPSILON:
                    counts["prob_over_eps"] += 1
            rows.append([s["sample_id"], s["attack_family"], status, mv, pred, f_pred, prob, f_prob, d,
                         resp.get("error", "") if not ok else ""])
    path = out_dir / f"xm_equivalence_{rev}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "attack_family", "http_status", "model_version", "served_prediction",
                    "frozen_prediction", "served_probability", "frozen_probability", "abs_prob_delta", "error"])
        w.writerows(rows)
    passed = all(v == 0 for v in counts.values()) and len(deltas) == len(samples)
    return {
        "endpoint": url,
        "n": len(samples),
        **counts,
        "max_abs_prob_delta": max(deltas) if deltas else None,
        "median_abs_prob_delta": statistics.median(deltas) if deltas else None,
        "verdict": "PASS" if passed else "FAIL",
        "per_sample_csv": str(path.relative_to(xm_loader.ROOT)),
        "per_sample_csv_sha256": xm_loader._sha256(path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-url", required=True)
    ap.add_argument("--v2-url", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--root", default=None, help="repository root (default: parent of this file's folder)")
    args = ap.parse_args()

    try:
        samples, prov = xm_loader.load_xm(args.root)
    except xm_loader.LoaderError as e:
        print(f"GATE NOT RUN - loader refused: {e}", file=sys.stderr)
        return 2

    frozen = {}
    with open(xm_loader.ROOT / xm_loader.FROZEN_XM, newline="") as fh:
        for r in csv.DictReader(fh):
            frozen[r["id"]] = {"v1": (int(r["v1_prediction"]), float(r["v1_probability"])),
                               "v2": (int(r["v2_prediction"]), float(r["v2_probability"]))}

    out_dir = xm_loader.ROOT / "results" / "derived" / "phase9_xm_equivalence"
    out_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    results = {}
    for rev, url in (("v1", args.v1_url), ("v2", args.v2_url)):
        t0 = time.time()
        results[rev] = run_revision(rev, url, samples, frozen, out_dir, args.workers, args.timeout)
        results[rev]["seconds"] = round(time.time() - t0, 1)
        print(f"{rev}: {results[rev]['verdict']}  {json.dumps({k: results[rev][k] for k in ('http_non200','wrong_model_version','decision_mismatch','prob_over_eps','max_abs_prob_delta')})}")

    verdict = "PASS" if all(r["verdict"] == "PASS" for r in results.values()) else "FAIL"
    summary = {
        "artifact": "RolloutSecBench",
        "gate": "X_m serving equivalence (Section K extended to the attack population)",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "epsilon_probability": EPSILON,
        "decision_tolerance": 0,
        "loader": prov,
        "loader_sha256": xm_loader._sha256(Path(xm_loader.__file__)),
        "script_sha256": xm_loader._sha256(Path(__file__)),
        "revisions": results,
        "verdict": verdict,
    }
    (out_dir / "xm_equivalence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"X_m SERVING EQUIVALENCE: {verdict}  -> {out_dir / 'xm_equivalence_summary.json'}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
