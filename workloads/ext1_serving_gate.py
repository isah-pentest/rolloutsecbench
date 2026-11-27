#!/usr/bin/env python3
"""
RolloutSecBench EXT1 serving-equivalence gate (revision r1, DEV-R1-005). Standard library only. Computes no outcome.

Sends every X_m sample (workloads/xm_loader.py, frozen order and typing) directly to the rebuilt V1 and V2 model
backends, using the backend request the router sends ({"request_id", "features"}). PASS requires, for both revisions
and all 38,263 samples:
  HTTP 200; model_version equal to the revision; model_sha256 = the frozen hash of that revision;
  preprocessor_sha256 = the frozen preprocessor hash; prediction equal to the frozen offline prediction;
  |probability - frozen| < 1e-4 (the campaign's V5 tolerance)
Writes revision/results/ext1_rebuild/serving_gate.json (fails if it already exists).

  python3 workloads/ext1_serving_gate.py --v1 http://127.0.0.1:18101 --v2 http://127.0.0.1:18102
"""
import argparse, csv, hashlib, http.client, json, queue, sys, threading, urllib.parse
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xm_loader  # noqa: E402
import validate_v3_run as v3v  # noqa: E402

SHA = {"v1": v3v.V1_SHA, "v2": v3v.V2_SHA}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True); ap.add_argument("--v2", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--root", default=str(HERE.parent))
    a = ap.parse_args()
    root = Path(a.root).resolve()
    out = root / "revision/results/ext1_rebuild/serving_gate.json"
    if out.exists(): sys.exit("FAIL_CLOSED: serving_gate.json already exists")
    samples, prov = xm_loader.load_xm(root)
    ref = v3v.frozen_predictions(root, "SCIENCE")
    work = queue.Queue()
    for rev, ep in (("v1", a.v1), ("v2", a.v2)):
        for s in samples: work.put((rev, ep, s))
    for _ in range(a.workers): work.put(None)
    lock = threading.Lock(); fails = {}; examples = {}; n = {"v1": 0, "v2": 0}; maxdp = {"v1": 0.0, "v2": 0.0}
    local = threading.local()

    def fail(code, detail):
        with lock:
            fails[code] = fails.get(code, 0) + 1; examples.setdefault(code, detail)

    def worker():
        while True:
            it = work.get()
            if it is None: return
            rev, ep, s = it
            u = urllib.parse.urlparse(ep)
            key = (u.hostname, u.port)
            c = getattr(local, "c", {}).get(key)
            if c is None:
                c = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
                local.__dict__.setdefault("c", {})[key] = c
            rid = f"EXT1-GATE-{rev}-{s['sample_id']}"
            try:
                c.request("POST", "/predict", body=json.dumps({"request_id": rid, "features": s["features"]}),
                          headers={"Content-Type": "application/json"})
                r = c.getresponse(); body = json.loads(r.read()); status = r.status
            except Exception as e:
                local.c.pop(key, None); fail("transport", f"{rev} {s['sample_id']}: {type(e).__name__}"); continue
            sid = str(s["sample_id"])
            if status != 200: fail("http", f"{rev} {sid}: {status}"); continue
            if body.get("request_id") != rid: fail("request_id", f"{rev} {sid}")
            if str(body.get("model_version")).lower() != rev: fail("model_version", f"{rev} {sid}: {body.get('model_version')}")
            if body.get("model_sha256") != SHA[rev]: fail("model_sha256", f"{rev} {sid}: {body.get('model_sha256')}")
            if body.get("preprocessor_sha256") != v3v.PRE_SHA: fail("preprocessor_sha256", f"{rev} {sid}")
            fp, fb = ref[sid][rev]
            if int(body["prediction"]) != fp: fail("prediction", f"{rev} {sid}: {body['prediction']} vs {fp}")
            dp = abs(float(body["probability"]) - fb)
            if dp >= v3v.EPS: fail("probability", f"{rev} {sid}: |dp|={dp:.2e}")
            with lock:
                n[rev] += 1; maxdp[rev] = max(maxdp[rev], dp)
                done = n["v1"] + n["v2"]
                if done % 5000 == 0: print(f"  {done:,}/{2 * len(samples):,}", flush=True)

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    [t.start() for t in ts]; [t.join() for t in ts]
    passed = not fails and n["v1"] == n["v2"] == len(samples)
    rep = {"gate": "EXT1 serving equivalence (DEV-R1-005)", "gate_sha256": xm_loader._sha256(Path(__file__)),
           "utc": datetime.now(timezone.utc).isoformat(), "samples": len(samples), "checked": n,
           "max_abs_prob_diff": maxdp, "tolerance": v3v.EPS, "failures": fails, "first_examples": examples,
           "decision": "PASS" if passed else "FAIL", "frozen_reference_sha256": xm_loader.FROZEN_XM_SHA256}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as f: json.dump(rep, f, indent=2, sort_keys=True); f.write("\n")
    print(json.dumps({k: rep[k] for k in ("checked", "max_abs_prob_diff", "failures", "decision")}))
    print("wrote", out, xm_loader._sha256(out))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
