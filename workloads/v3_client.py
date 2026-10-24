#!/usr/bin/env python3
"""
RolloutSecBench Phase-9 v3 traffic client (one run per invocation). Standard library only.

Contract (verified against the deployed router serving/router/phase7/router_phase7_v1.py, sha256 358bab02...):
  request  : experiment_id, run_id, request_id, request_sequence, replicate_id, sample_id, ground_truth,
             attack_family, rollout_state, connection_mode, features (dict), attempt_number
  response : request_id, selected_revision, prediction, probability, model_sha256, preprocessor_sha256,
             routing_uniform_value, configured_v2_weight, processing_ms

Scientific mode   : population = frozen X_m (workloads/xm_loader.py), A0 = 1 attempt/sample,
                    A1 = <budget> attempts/sample, ALL attempts always sent (no early stop).
Qualification mode: population = first N BENIGN rows of the frozen strict split, A0 semantics only;
                    used before the freeze to qualify the full path without malicious traffic.

Fail-closed properties
  * the run directory must not exist; it is never overwritten
  * no request is retried (a retry would change the frozen attempt sequence); failures are recorded
  * request_sequence runs 1..N without gaps; offered load is paced to a fixed schedule
  * the client never decides validity: that is validate_v3_run.py's job, from objective criteria only

Usage
  python3 workloads/v3_client.py --mode SCIENCE --run-id V3-R1-A0-P01 --rollout-state R1 \
      --attacker A0 --budget 1 --qps 16 --endpoint http://127.0.0.1:18000/predict
  python3 workloads/v3_client.py --mode QUALIFICATION --run-id V3Q-R2-001 --rollout-state R2 \
      --attacker A0 --budget 1 --qps 16 --qual-n 2000 --endpoint http://127.0.0.1:18000/predict
"""
import argparse
import csv
import hashlib
import http.client
import json
import os
import queue
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xm_loader  # noqa: E402

CLIENT_VERSION = "phase9-v3-client-1.0"
CONNECTION_MODE = "PERSISTENT"
STATES = {"R0": 0.0, "R1": 0.1, "R2": 0.5, "R3": 1.0}   # configured V2 weight


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def qualification_population(root, n):
    """First n benign rows of the frozen strict split, typed exactly like xm_loader does."""
    root = Path(root)
    xm_loader.ROOT = root
    test = xm_loader._verify(xm_loader.STRICT_TEST, xm_loader.STRICT_TEST_SHA256)
    names = json.loads((root / xm_loader.FEATURE_NAMES).read_text())["source_feature_order"]
    contract = json.loads((root / xm_loader.SERVING_PROTOCOL).read_text())["input_contract"]
    cats = set(contract["categorical_features"])
    with open(test, newline="") as fh:
        rows = list(csv.DictReader(fh))
    int_cols = {c for c in names if c not in cats and all(xm_loader._is_int_literal(r[c]) for r in rows)}
    out = []
    for r in rows:
        if r["label"].strip() != "0":
            continue
        feats = {c: (r[c] if c in cats else int(r[c]) if c in int_cols else xm_loader._as_float(c, r[c], r["id"]))
                 for c in names}
        out.append({"sample_id": r["id"], "ground_truth": 0, "attack_family": r["attack_cat"], "features": feats})
        if len(out) == n:
            break
    if len(out) != n:
        raise SystemExit(f"only {len(out)} benign rows available, {n} requested")
    return out


class Conn(threading.local):
    """One persistent HTTP connection per worker thread."""
    def get(self, host, port):
        c = getattr(self, "c", None)
        if c is None:
            c = self.c = http.client.HTTPConnection(host, port, timeout=30)
        return c

    def drop(self):
        c = getattr(self, "c", None)
        if c is not None:
            try:
                c.close()
            except Exception:
                pass
        self.c = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("SCIENCE", "QUALIFICATION"), required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rollout-state", choices=sorted(STATES), required=True)
    ap.add_argument("--attacker", choices=("A0", "A1"), required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--qps", type=float, required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--qual-n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-root", default="results/raw/phase9_v3")
    ap.add_argument("--root", default=str(HERE.parent))
    a = ap.parse_args()

    root = Path(a.root).resolve()
    if a.attacker == "A0" and a.budget != 1:
        raise SystemExit("A0 requires --budget 1")
    if a.attacker == "A1" and a.budget < 2:
        raise SystemExit("A1 requires --budget >= 2")
    if a.mode == "QUALIFICATION" and a.attacker != "A0":
        raise SystemExit("qualification uses A0 semantics only")
    prefix = "V3Q-" if a.mode == "QUALIFICATION" else "V3-"
    if not a.run_id.startswith(prefix):
        raise SystemExit(f"{a.mode} run ids must start with {prefix}")

    run_dir = (root / a.out_root / a.run_id)
    if run_dir.exists():
        print("FAIL_CLOSED=run-directory-already-exists"); return 2

    if a.mode == "SCIENCE":
        population, prov = xm_loader.load_xm(root)
    else:
        population = qualification_population(root, a.qual_n)
        prov = {"population": f"first {a.qual_n} benign rows of strict split",
                "strict_test_sha256": xm_loader.STRICT_TEST_SHA256}

    u = urllib.parse.urlparse(a.endpoint)
    host, port, path = u.hostname, u.port or 80, u.path or "/predict"
    experiment_id = f"V3-EXP-{a.rollout_state}-{a.attacker}"
    attempts = a.budget
    expected = len(population) * attempts

    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "client_version": CLIENT_VERSION,
        "client_sha256": xm_loader._sha256(Path(__file__)),
        "loader_sha256": xm_loader._sha256(Path(xm_loader.__file__)),
        "mode": a.mode, "run_id": a.run_id, "experiment_id": experiment_id,
        "rollout_state": a.rollout_state, "configured_v2_weight": STATES[a.rollout_state],
        "attacker_type": a.attacker, "attempts_per_sample": attempts,
        "population_size": len(population), "expected_requests": expected,
        "target_qps": a.qps, "endpoint": a.endpoint, "connection_mode": CONNECTION_MODE,
        "workers": a.workers, "population_provenance": prov,
        "status": "STARTED", "started_utc": utc(),
    }
    atomic_json(run_dir / "run_manifest.json", manifest)

    work = queue.Queue()
    seq = 0
    for s in population:
        for att in range(1, attempts + 1):
            seq += 1
            work.put((seq, att, s))
    for _ in range(a.workers):
        work.put(None)

    out_lock = threading.Lock()
    out = open(run_dir / "requests.jsonl", "x", encoding="utf-8")
    counters = {"written": 0, "non200": 0, "transport_errors": 0}
    conn = Conn()
    start = time.perf_counter()

    def worker():
        while True:
            item = work.get()
            if item is None:
                return
            seq, att, s = item
            rid = f"{a.run_id}-REQ-{seq:08d}"
            payload = {
                "experiment_id": experiment_id, "run_id": a.run_id, "request_id": rid,
                "request_sequence": seq, "replicate_id": a.run_id, "sample_id": s["sample_id"],
                "ground_truth": s["ground_truth"], "attack_family": s["attack_family"],
                "rollout_state": a.rollout_state, "connection_mode": CONNECTION_MODE,
                "features": s["features"], "attempt_number": att,
            }
            target = start + (seq - 1) / a.qps
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            t_send = time.perf_counter()
            ts_send = utc()
            status, body, err = None, {}, None
            try:
                c = conn.get(host, port)
                c.request("POST", path, body=json.dumps(payload),
                          headers={"Content-Type": "application/json", "Connection": "keep-alive"})
                r = c.getresponse()
                raw = r.read()
                status = r.status
                try:
                    body = json.loads(raw)
                except Exception:
                    body = {"_raw": raw[:300].decode(errors="replace")}
            except Exception as e:      # no retry: record and move on
                err = f"{type(e).__name__}:{e}"
                conn.drop()
            t_recv = time.perf_counter()
            rec = {
                "run_id": a.run_id, "request_id": rid, "request_sequence": seq,
                "sample_id": s["sample_id"], "ground_truth": s["ground_truth"],
                "attack_family": s["attack_family"], "attempt_number": att,
                "rollout_state": a.rollout_state,
                "scheduled_offset_s": round(target - start, 6), "send_offset_s": round(t_send - start, 6),
                "latency_ms": round((t_recv - t_send) * 1000, 3), "timestamp_send_utc": ts_send,
                "http_status": status, "transport_error": err,
                "selected_revision": body.get("selected_revision"),
                "prediction": body.get("prediction"), "probability": body.get("probability"),
                "routing_uniform_value": body.get("routing_uniform_value"),
                "configured_v2_weight": body.get("configured_v2_weight"),
                "model_sha256": body.get("model_sha256"), "preprocessor_sha256": body.get("preprocessor_sha256"),
                "error_detail": None if status == 200 else str(body.get("detail", body.get("_raw", "")))[:300],
            }
            line = json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n"
            with out_lock:
                out.write(line)
                counters["written"] += 1
                if status != 200:
                    counters["non200"] += 1
                if err:
                    counters["transport_errors"] += 1
                if counters["written"] % 2048 == 0:
                    out.flush(); os.fsync(out.fileno())
                    print(f"  {counters['written']:,}/{expected:,} requests "
                          f"({counters['non200']} non-200)", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out.flush(); os.fsync(out.fileno()); out.close()
    elapsed = time.perf_counter() - start

    summary = {
        "run_id": a.run_id, "mode": a.mode, "expected_requests": expected,
        "written_requests": counters["written"], "non_200": counters["non200"],
        "transport_errors": counters["transport_errors"], "elapsed_s": round(elapsed, 3),
        "achieved_qps": round(counters["written"] / elapsed, 4) if elapsed else None,
        "client_completed_utc": utc(),
        "note": "Validity is decided only by validate_v3_run.py from pre-specified objective criteria.",
    }
    atomic_json(run_dir / "run_summary.json", summary)
    manifest.update(status="CLIENT_COMPLETED", ended_utc=utc())
    atomic_json(run_dir / "run_manifest.json", manifest)
    with open(run_dir / "checksums.sha256", "x") as f:
        for name in ("run_manifest.json", "requests.jsonl", "run_summary.json"):
            f.write(f"{xm_loader._sha256(run_dir / name)}  {name}\n")
    print(f"CLIENT_DONE run={a.run_id} written={counters['written']:,}/{expected:,} "
          f"non200={counters['non200']} elapsed={elapsed:.1f}s qps={summary['achieved_qps']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
