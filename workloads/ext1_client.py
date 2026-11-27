#!/usr/bin/env python3
"""
RolloutSecBench EXT1 traffic client (revision r1, E3 routing policies; one run per invocation).
Derived from workloads/v3_client.py (pinned a742c9ad...). Request contents, pacing, persistence, no-retry and
fail-closed behaviour are unchanged. Additions:
  * every request carries session_id (pre-registration Section 5):
        A0, A1-one : <run_id>-S<sample_id>
        A1-multi   : <run_id>-S<sample_id>-A<attempt>
  * --policy (P1|P2|P3) is recorded in the manifest. The policy is set on the router, never by the client.
  * attackers A0 (1 attempt), A1-one and A1-multi (5 attempts, sample-major, attempts consecutive)
  * run ids: SCIENCE "EXT1-", QUALIFICATION "EXT1Q-"; default output root results/raw/ext1
Qualification mode: the first N benign rows of the strict split (as in v3_client), with any attacker type, so the
route reconstruction of every policy can be dry-run on benign traffic.
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
import http.client
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xm_loader  # noqa: E402
import v3_client  # noqa: E402  (qualification_population, atomic_json, utc, Conn reused unchanged)

CLIENT_VERSION = "ext1-client-1.0"
CONNECTION_MODE = "PERSISTENT"
STATES = {"R0": 0.0, "R1": 0.1, "R2": 0.5, "R3": 1.0}
ATTEMPTS = {"A0": 1, "A1-one": 5, "A1-multi": 5}


def session_id(run_id, sample_id, attempt, attacker):
    return f"{run_id}-S{sample_id}-A{attempt}" if attacker == "A1-multi" else f"{run_id}-S{sample_id}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("SCIENCE", "QUALIFICATION"), required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--policy", choices=("P1", "P2", "P3"), required=True)
    ap.add_argument("--rollout-state", choices=sorted(STATES), required=True)
    ap.add_argument("--attacker", choices=sorted(ATTEMPTS), required=True)
    ap.add_argument("--qps", type=float, required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--qual-n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-root", default="results/raw/ext1")
    ap.add_argument("--root", default=str(HERE.parent))
    a = ap.parse_args()

    root = Path(a.root).resolve()
    prefix = "EXT1Q-" if a.mode == "QUALIFICATION" else "EXT1-"
    if not a.run_id.startswith(prefix):
        raise SystemExit(f"{a.mode} run ids must start with {prefix}")
    run_dir = root / a.out_root / a.run_id
    if run_dir.exists():
        print("FAIL_CLOSED=run-directory-already-exists"); return 2

    if a.mode == "SCIENCE":
        population, prov = xm_loader.load_xm(root)
    else:
        population = v3_client.qualification_population(root, a.qual_n)
        prov = {"population": f"first {a.qual_n} benign rows of strict split",
                "strict_test_sha256": xm_loader.STRICT_TEST_SHA256}

    u = urllib.parse.urlparse(a.endpoint)
    host, port, path = u.hostname, u.port or 80, u.path or "/predict"
    experiment_id = f"EXT1-EXP-{a.policy}-{a.rollout_state}-{a.attacker}"
    attempts = ATTEMPTS[a.attacker]
    expected = len(population) * attempts

    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "client_version": CLIENT_VERSION,
        "client_sha256": xm_loader._sha256(Path(__file__)),
        "v3_client_sha256": xm_loader._sha256(Path(v3_client.__file__)),
        "loader_sha256": xm_loader._sha256(Path(xm_loader.__file__)),
        "mode": a.mode, "run_id": a.run_id, "experiment_id": experiment_id, "routing_policy": a.policy,
        "rollout_state": a.rollout_state, "configured_v2_weight": STATES[a.rollout_state],
        "attacker_type": a.attacker, "attempts_per_sample": attempts,
        "session_rule": "<run_id>-S<sample_id>-A<attempt>" if a.attacker == "A1-multi" else "<run_id>-S<sample_id>",
        "population_size": len(population), "expected_requests": expected,
        "target_qps": a.qps, "endpoint": a.endpoint, "connection_mode": CONNECTION_MODE,
        "workers": a.workers, "population_provenance": prov,
        "status": "STARTED", "started_utc": v3_client.utc(),
    }
    v3_client.atomic_json(run_dir / "run_manifest.json", manifest)

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
    conn = v3_client.Conn()
    start = time.perf_counter()

    def worker():
        while True:
            item = work.get()
            if item is None:
                return
            seq, att, s = item
            rid = f"{a.run_id}-REQ-{seq:08d}"
            sess = session_id(a.run_id, s["sample_id"], att, a.attacker)
            payload = {
                "experiment_id": experiment_id, "run_id": a.run_id, "request_id": rid,
                "request_sequence": seq, "replicate_id": a.run_id, "sample_id": s["sample_id"],
                "ground_truth": s["ground_truth"], "attack_family": s["attack_family"],
                "rollout_state": a.rollout_state, "connection_mode": CONNECTION_MODE,
                "features": s["features"], "attempt_number": att, "session_id": sess,
            }
            target = start + (seq - 1) / a.qps
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            t_send = time.perf_counter()
            ts_send = v3_client.utc()
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
                "attack_family": s["attack_family"], "attempt_number": att, "session_id": sess,
                "rollout_state": a.rollout_state,
                "scheduled_offset_s": round(target - start, 6), "send_offset_s": round(t_send - start, 6),
                "latency_ms": round((t_recv - t_send) * 1000, 3), "timestamp_send_utc": ts_send,
                "http_status": status, "transport_error": err,
                "selected_revision": body.get("selected_revision"),
                "prediction": body.get("prediction"), "probability": body.get("probability"),
                "routing_uniform_value": body.get("routing_uniform_value"),
                "configured_v2_weight": body.get("configured_v2_weight"),
                "routing_policy": body.get("routing_policy"),
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
        "client_completed_utc": v3_client.utc(),
        "note": "Validity is decided only by validate_ext1_run.py from pre-specified objective criteria.",
    }
    v3_client.atomic_json(run_dir / "run_summary.json", summary)
    manifest.update(status="CLIENT_COMPLETED", ended_utc=v3_client.utc())
    v3_client.atomic_json(run_dir / "run_manifest.json", manifest)
    with open(run_dir / "checksums.sha256", "x") as f:
        for name in ("run_manifest.json", "requests.jsonl", "run_summary.json"):
            f.write(f"{xm_loader._sha256(run_dir / name)}  {name}\n")
    print(f"CLIENT_DONE run={a.run_id} written={counters['written']:,}/{expected:,} "
          f"non200={counters['non200']} elapsed={elapsed:.1f}s qps={summary['achieved_qps']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
