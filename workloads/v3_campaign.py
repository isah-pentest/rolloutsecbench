#!/usr/bin/env python3
"""
RolloutSecBench Phase-9 v3 campaign runner: executes the frozen identities sequentially, one at a
time, resumably, with silent live monitoring and complete per-run evidence capture.

Per identity
  pre-run   P0 disk space, P1 all pods Ready (records pod UIDs + restart counts), P2 router /health
            protocol hash, router deployment env, source configmap hashes and image IDs equal the freeze,
            P3 router port-forward alive (managed by this runner). Pre-run failures are retried; if they
            persist the campaign PAUSES without consuming the identity (no request has been sent).
  run       client subprocess (consumption boundary = first request); monitor thread samples pods and
            node-runtime CPU/memory every 5 s into system_metrics.csv
  post-run  P1 again (pod UIDs and restart counts must be unchanged), router evidence extracted for this
            run_id, validate_v3_run.py (V1-V7), final disposition = VALID only if all criteria pass
  replace   INVALID -> next ordered reserve of the same cell is executed immediately after
  stop      any INVALID pauses the campaign after queuing its reserve (protects reserves from a
            recurring defect); --continue-after-invalid pauses only after 2 consecutive INVALIDs

Evidence per run (results/raw/phase9_v3/<run_id>/):
  run_manifest.json, requests.jsonl, run_summary.json, checksums.sha256       (client)
  router_evidence.jsonl, system_metrics.csv, client.log, prepost_state.json    (campaign)
  validation.json, disposition.json, final_checksums.sha256                    (validator + campaign)
Campaign level: registry.csv (append-only), campaign_status.json, campaign.log, activations.jsonl

Usage (repository root):
  tmux new -s v3      # then inside tmux:
  python3 workloads/v3_campaign.py --freeze data/manifests/phase9_v3/phase9_v3_freeze.json
  # detach: Ctrl-b d     reattach: tmux attach -t v3
  # watch silently from anywhere:  python3 workloads/v3_campaign.py --status
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import xm_loader  # noqa: E402

NS = "rolloutsecbench"
BASE = ROOT / "results/raw/phase9_v3"
LOGS = BASE / "_campaign"
REG = BASE / "registry.csv"
STATUS = BASE / "campaign_status.json"
REG_FIELDS = ["run_id", "cell", "identity_type", "reserve_order", "activated_after", "disposition",
              "reasons", "started_utc", "ended_utc", "elapsed_s", "validation_sha256"]
PORT = 18000
INTERVAL = 5.0
PRECHECK_TRIES = 3


def utc():
    return datetime.now(timezone.utc).isoformat()


def log(msg):
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"{utc()}  {msg}"
    print(line, flush=True)
    with open(LOGS / "campaign.log", "a") as f:
        f.write(line + "\n")


def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:
        return 99, "", repr(e)


def sha(p):
    return xm_loader._sha256(Path(p))


# ------------------------------------------------------------------ cluster probes
class Cluster:
    def __init__(self, freeze, test_mode, endpoint):
        self.f = freeze
        self.test = test_mode
        self.endpoint = endpoint
        rt = freeze.get("runtime_at_freeze") or {}
        self.node = rt.get("node")
        self.pf = None

    def pods(self):
        if self.test:
            return {"router": {"uid": "t", "ready": True, "restarts": 0}}, None
        rc, out, err = sh(["kubectl", "-n", NS, "get", "pods", "-o", "json"])
        if rc != 0:
            return {}, f"kubectl get pods failed: {err.strip()[:200]}"
        res = {}
        for p in json.loads(out)["items"]:
            name = p["metadata"]["name"]
            role = next((r for r in ("router", "model-v1", "model-v2") if name.startswith(r + "-")), None)
            if not role:
                continue
            cs = (p["status"].get("containerStatuses") or [{}])[0]
            res[role] = {"pod": name, "uid": p["metadata"]["uid"], "ready": bool(cs.get("ready")),
                         "restarts": int(cs.get("restartCount", 0)), "imageID": cs.get("imageID")}
        return res, None

    def stats(self):
        if self.test or not self.node:
            return {}
        rc, out, _ = sh(["docker", "exec", self.node, "crictl", "stats", "-o", "json"], timeout=20)
        if rc != 0:
            return {}
        vals = {}
        try:
            for s in json.loads(out).get("stats", []):
                pod = s.get("attributes", {}).get("labels", {}).get("io.kubernetes.pod.name", "")
                role = next((r for r in ("router", "model-v1", "model-v2") if pod.startswith(r + "-")), None)
                if not role:
                    continue
                cpu = (s.get("cpu") or {}).get("usageCoreNanoSeconds", {}).get("value")
                mem = (s.get("memory") or {}).get("workingSetBytes", {}).get("value")
                if cpu is not None:
                    vals[f"{role}_cpu_ns"] = cpu
                if mem is not None:
                    vals[f"{role}_mem_bytes"] = mem
        except Exception:
            pass
        return vals

    def health(self):
        try:
            return json.loads(urllib.request.urlopen(self.endpoint + "/health", timeout=5).read())
        except Exception as e:
            return {"error": repr(e)}

    def ensure_port_forward(self):
        if self.test:
            return True
        if self.pf is None or self.pf.poll() is not None:
            sh(["pkill", "-f", f"port-forward svc/router {PORT}:8000"])
            LOGS.mkdir(parents=True, exist_ok=True)
            pflog = open(LOGS / "port_forward.log", "a")
            pflog.write(f"--- {utc()} (re)starting port-forward\n"); pflog.flush()
            self.pf = subprocess.Popen(["kubectl", "-n", NS, "port-forward", "svc/router", f"{PORT}:8000"],
                                       stdout=pflog, stderr=subprocess.STDOUT)
            time.sleep(3)
        return "status" in self.health()

    def precheck(self):
        """Returns (ok, details). Never sends a /predict request."""
        d = {"time": utc()}
        free = shutil.disk_usage(ROOT).free / 2 ** 30
        d["disk_free_gib"] = round(free, 1)
        if free < 5:
            return False, {**d, "fail": "P0 disk free < 5 GiB"}
        pods, err = self.pods()
        d["pods"] = pods
        if err or not all(pods.get(r, {}).get("ready") for r in (("router",) if self.test else ("router", "model-v1", "model-v2"))):
            return False, {**d, "fail": f"P1 pods not all Ready: {err or {k: v.get('ready') for k, v in pods.items()}}"}
        if not self.ensure_port_forward():
            return False, {**d, "fail": "P3 router not reachable through port-forward"}
        h = self.health()
        d["router_health"] = h
        rt = self.f.get("runtime_at_freeze")
        if rt and not self.test:
            if h.get("protocol_sha256") != rt["router_protocol_sha256"]:
                return False, {**d, "fail": "P2 router protocol hash differs from freeze"}
            rc, out, _ = sh(["kubectl", "-n", NS, "get", "deploy", "router", "-o", "json"])
            c = json.loads(out)["spec"]["template"]["spec"]["containers"][0] if rc == 0 else {}
            env = {e["name"]: e.get("value") for e in c.get("env", [])}
            if env != rt["router_env"]:
                return False, {**d, "fail": "P2 router deployment env differs from freeze"}
            rc, out, _ = sh(["kubectl", "-n", NS, "get", "configmap", "-o", "json"])
            cm = {m["metadata"]["name"]: {k: hashlib.sha256(v.encode()).hexdigest() for k, v in m.get("data", {}).items()}
                  for m in json.loads(out)["items"] if m["metadata"]["name"].startswith("phase7-router")} if rc == 0 else None
            if cm != rt["router_source_configmaps_sha256"]:
                return False, {**d, "fail": "P2 router source/protocol configmap differs from freeze"}
            frozen_ids = {i for p in rt["pods"] for i in p["imageID"]}
            live_ids = {v.get("imageID") for v in pods.values()}
            if not live_ids <= frozen_ids:
                return False, {**d, "fail": f"P2 image IDs differ from freeze: {live_ids - frozen_ids}"}
        return True, d

    def prior_science_records(self, test_file=None):
        """Number of router-evidence records already carrying a science run id (must be 0 at first start)."""
        needle = '"run_id": "V3-'
        if self.test:
            return sum(needle in l for l in open(test_file)) if test_file and Path(test_file).exists() else 0
        path = "/var/lib/rolloutsecbench/phase6-router-evidence/" + Path(
            self.f["runtime_at_freeze"]["router_env"]["ROUTER_EVIDENCE_PATH"]).name
        rc, out, _ = sh(["docker", "exec", self.node, "sh", "-c", f"grep -cF '{needle}' {path} || true"], timeout=600)
        return int(out.strip() or 0) if rc == 0 else -1

    def router_evidence(self, run_id, dest, test_file=None):
        needle = f'"run_id": "{run_id}"'
        if self.test:
            src = Path(test_file)
            with open(src) as fi, open(dest, "x") as fo:
                for line in fi:
                    if needle in line:
                        fo.write(line)
            return True
        path = "/var/lib/rolloutsecbench/phase6-router-evidence/" + Path(
            (self.f["runtime_at_freeze"]["router_env"]["ROUTER_EVIDENCE_PATH"])).name
        # streamed straight to disk (an A1 run has ~191k router records)
        with open(dest, "x") as fo:
            try:
                rc = subprocess.run(["docker", "exec", self.node, "grep", "-F", needle, path],
                                    stdout=fo, stderr=subprocess.DEVNULL, timeout=900).returncode
            except Exception:
                rc = 99
        return rc in (0, 1)   # 1 = no match; the validator then fails V7 explicitly


# ------------------------------------------------------------------ monitor
def monitor(cluster, stop, out_path, client_pid_getter):
    fields = ["timestamp_utc", "elapsed_s", "client_alive", "port_forward_alive"]
    roles = ("router", "model-v1", "model-v2")
    for r in roles:
        fields += [f"{r}_pod", f"{r}_uid", f"{r}_ready", f"{r}_restarts", f"{r}_cpu_ns", f"{r}_mem_bytes"]
    fields += ["probe_error"]
    t0 = time.monotonic()
    with open(out_path, "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        while not stop.is_set():
            row = {"timestamp_utc": utc(), "elapsed_s": f"{time.monotonic() - t0:.1f}"}
            pid = client_pid_getter()
            row["client_alive"] = bool(pid and os.path.exists(f"/proc/{pid}"))
            row["port_forward_alive"] = cluster.test or (cluster.pf is not None and cluster.pf.poll() is None)
            pods, err = cluster.pods()
            for r, v in pods.items():
                row.update({f"{r}_pod": v.get("pod"), f"{r}_uid": v.get("uid"), f"{r}_ready": v.get("ready"),
                            f"{r}_restarts": v.get("restarts")})
            row.update(cluster.stats())
            row["probe_error"] = err or ""
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
            stop.wait(INTERVAL)


# ------------------------------------------------------------------ registry / status
def read_registry():
    if not REG.exists():
        return []
    with open(REG, newline="") as f:
        return list(csv.DictReader(f))


def append_registry(row):
    new = not REG.exists()
    with open(REG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in REG_FIELDS})
        f.flush()
        os.fsync(f.fileno())


def write_status(freeze, state, note=""):
    reg = read_registry()
    valid = {}
    for r in reg:
        if r["disposition"] == "VALID":
            valid[r["cell"]] = valid.get(r["cell"], 0) + 1
    remaining_req = sum(i["expected_requests"] for i in freeze["identities"]
                        if i["run_id"] in state["queue"])
    qps = freeze["design"]["target_qps"]
    st = {"updated_utc": utc(), "state": state["phase"], "note": note,
          "current_run": state.get("current"), "completed_runs": len(reg),
          "valid_per_cell": dict(sorted(valid.items())),
          "invalid_runs": [r["run_id"] for r in reg if r["disposition"] != "VALID"],
          "queued_runs": len(state["queue"]),
          "estimated_hours_remaining": round(remaining_req / qps / 3600 + 0.05 * len(state["queue"]), 1)}
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2) + "\n")
    os.replace(tmp, STATUS)


# ------------------------------------------------------------------ one run
def execute(identity, freeze, cluster, args, activated_after):
    rid, cell = identity["run_id"], identity["cell"]
    rd = BASE / rid
    LOGS.mkdir(parents=True, exist_ok=True)
    tmp_metrics = LOGS / f"{rid}.system_metrics.csv"
    tmp_clientlog = LOGS / f"{rid}.client.log"
    pre_pods, _ = cluster.pods()
    started = utc()
    t0 = time.time()
    cmd = [sys.executable, str(HERE / "v3_client.py"), "--mode", "SCIENCE", "--run-id", rid,
           "--rollout-state", identity["rollout_state"], "--attacker", identity["attacker"],
           "--budget", str(identity["attempts_per_sample"]), "--qps", str(freeze["design"]["target_qps"]),
           "--workers", str(freeze["design"]["client_workers"]),
           "--endpoint", args.endpoint + "/predict", "--root", str(ROOT)]
    stop = threading.Event()
    holder = {}
    mon = threading.Thread(target=monitor, args=(cluster, stop, tmp_metrics, lambda: holder.get("pid")), daemon=True)
    mon.start()
    with open(tmp_clientlog, "x") as lf:
        # own session: a Ctrl-C in the tmux pane reaches only the runner (which then finishes the run)
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        holder["pid"] = proc.pid
        rc = proc.wait()
    stop.set()
    mon.join(timeout=INTERVAL + 15)
    post_pods, _ = cluster.pods()
    ended = utc()

    reasons = []
    if not rd.is_dir():
        reasons.append(f"CLIENT_PRODUCED_NO_RUN_DIRECTORY rc={rc}")
        rd.mkdir(parents=True, exist_ok=True)
    for src, name in ((tmp_metrics, "system_metrics.csv"), (tmp_clientlog, "client.log")):
        if src.exists():
            shutil.move(str(src), rd / name)
    stable = {r: (pre_pods.get(r, {}).get("uid"), pre_pods.get(r, {}).get("restarts")) ==
                 (post_pods.get(r, {}).get("uid"), post_pods.get(r, {}).get("restarts"))
              for r in set(pre_pods) | set(post_pods)}
    if not all(stable.values()):
        reasons.append(f"P1_POD_CHANGED_DURING_RUN {[k for k, v in stable.items() if not v]}")
    (rd / "prepost_state.json").write_text(json.dumps(
        {"pre": pre_pods, "post": post_pods, "stable": stable, "client_rc": rc}, indent=2) + "\n")

    ev = rd / "router_evidence.jsonl"
    if not ev.exists() and not cluster.router_evidence(rid, ev, args.test_router_evidence):
        reasons.append("ROUTER_EVIDENCE_EXTRACTION_FAILED")
    vdisp = "INVALID"
    if (rd / "requests.jsonl").exists() and (rd / "run_summary.json").exists():
        vr = subprocess.run([sys.executable, str(HERE / "validate_v3_run.py"), "--run-dir", str(rd),
                             "--route-basis", freeze["routing"]["basis_sha256"], "--router-evidence", str(ev),
                             "--root", str(ROOT)], capture_output=True, text=True)
        try:
            v = json.loads((rd / "validation.json").read_text())
            vdisp = v["disposition"]
            reasons += [f"{k}x{n}" for k, n in v["failed_criteria"].items()]
        except Exception as e:
            reasons.append(f"VALIDATOR_FAILED {e!r} {vr.stderr[-200:]}")
    else:
        reasons.append("CLIENT_EVIDENCE_INCOMPLETE")
    disposition = "VALID" if (vdisp == "VALID" and not reasons) else "INVALID"

    disp = {"run_id": rid, "cell": cell, "identity_type": identity["identity_type"],
            "activated_after": activated_after, "disposition": disposition, "reasons": reasons,
            "started_utc": started, "ended_utc": ended, "freeze_sha256": args.freeze_sha,
            "predicted_outcome_from_freeze": identity["predicted_outcome"]}
    (rd / "disposition.json").write_text(json.dumps(disp, indent=2) + "\n")
    with open(rd / "final_checksums.sha256", "x") as f:
        for p in sorted(rd.iterdir()):
            if p.is_file() and p.name != "final_checksums.sha256":
                f.write(f"{sha(p)}  {p.name}\n")
    append_registry({"run_id": rid, "cell": cell, "identity_type": identity["identity_type"],
                     "reserve_order": identity.get("reserve_order") or "", "activated_after": activated_after or "",
                     "disposition": disposition, "reasons": "; ".join(reasons), "started_utc": started,
                     "ended_utc": ended, "elapsed_s": round(time.time() - t0, 1),
                     "validation_sha256": sha(rd / "validation.json") if (rd / "validation.json").exists() else ""})
    return disposition, reasons


# ------------------------------------------------------------------ main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", default="data/manifests/phase9_v3/phase9_v3_freeze.json")
    ap.add_argument("--endpoint", default=f"http://127.0.0.1:{PORT}")
    ap.add_argument("--status", action="store_true", help="print campaign status and exit (read-only)")
    ap.add_argument("--continue-after-invalid", action="store_true",
                    help="do not pause after a single INVALID (still pauses after 2 consecutive INVALIDs)")
    ap.add_argument("--max-runs", type=int, default=0, help="stop after this many runs (0 = all)")
    ap.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--test-router-evidence", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.status:
        print(STATUS.read_text() if STATUS.exists() else "no campaign status yet")
        return 0

    fp = (ROOT / args.freeze) if not Path(args.freeze).is_absolute() else Path(args.freeze)
    recorded = (fp.parent / (fp.name + ".sha256")).read_text().split()[0]
    args.freeze_sha = sha(fp)
    if args.freeze_sha != recorded:
        raise SystemExit("FAIL_CLOSED: freeze file does not match its recorded SHA-256")
    freeze = json.loads(fp.read_text())
    for rel, h in freeze["pinned_code_sha256"].items():
        if sha(ROOT / rel) != h:
            raise SystemExit(f"FAIL_CLOSED: pinned code changed since freeze: {rel}")
    for rel, h in freeze["frozen_inputs_sha256"].items():
        if sha(ROOT / rel) != h:
            raise SystemExit(f"FAIL_CLOSED: frozen input changed: {rel}")

    BASE.mkdir(parents=True, exist_ok=True)
    lockf = open(BASE / ".campaign.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("FAIL_CLOSED: another campaign runner is active")
    # an orphaned client (runner killed with -9) may still be writing a run directory
    live = []
    for pd in Path("/proc").iterdir():
        try:
            argv = (pd / "cmdline").read_bytes().split(b"\0")
        except Exception:
            continue
        # match the interpreter's script argument exactly (not shells whose text merely mentions it)
        if len(argv) > 2 and argv[1].endswith(b"v3_client.py") and b"SCIENCE" in argv:
            live.append(pd.name)
    if live:
        raise SystemExit(f"FAIL_CLOSED: a science client is still running (pid {live}); "
                         "wait for it to finish or stop it, then resume")

    ids = {i["run_id"]: i for i in freeze["identities"]}
    reserves = {}
    for i in freeze["identities"]:
        if i["identity_type"] == "RESERVE":
            reserves.setdefault(i["cell"], []).append(i)
    for c in reserves:
        reserves[c].sort(key=lambda x: x["reserve_order"])

    # reconstruct state from the append-only registry (resume)
    reg = read_registry()
    done = {r["run_id"]: r for r in reg}
    used_reserves = {r["run_id"] for r in reg if r["identity_type"] == "RESERVE"}
    valid = {}
    for r in reg:
        if r["disposition"] == "VALID":
            valid[r["cell"]] = valid.get(r["cell"], 0) + 1
    # a run directory without a registry row = crashed mid-run: identity consumed, recorded INVALID
    for d in sorted(BASE.glob("V3-*")):
        if d.is_dir() and d.name not in done:
            log(f"RECOVER {d.name}: run directory without registry row -> INVALID (interrupted)")
            ident = ids[d.name]
            append_registry({"run_id": d.name, "cell": ident["cell"], "identity_type": ident["identity_type"],
                             "reserve_order": ident.get("reserve_order") or "", "disposition": "INVALID",
                             "reasons": "INTERRUPTED_EXECUTION_EVIDENCE_INCOMPLETE", "ended_utc": utc()})
            done[d.name] = {"disposition": "INVALID", "cell": ident["cell"], "identity_type": ident["identity_type"]}
            if ident["identity_type"] == "RESERVE":
                used_reserves.add(d.name)

    queue = [rid for rid in freeze["design"]["execution_order_primaries"] if rid not in done]
    # invalid runs whose replacement has not yet been executed
    pending_repl = []
    for r in reg + [dict(run_id=k, **v) for k, v in done.items() if k not in {x["run_id"] for x in reg}]:
        if r["disposition"] != "VALID":
            pending_repl.append(r)
    activations_path = BASE / "activations.jsonl"
    activated = [json.loads(l) for l in open(activations_path)] if activations_path.exists() else []
    replaced = {a["invalid_run_id"] for a in activated}
    pre_queue = []
    for r in pending_repl:
        if r["run_id"] not in replaced:
            nxt = next((x for x in reserves[r["cell"]] if x["run_id"] not in used_reserves
                        and x["run_id"] not in {a["reserve_run_id"] for a in activated}), None)
            if nxt and valid.get(r["cell"], 0) < 5:
                rec = {"time": utc(), "invalid_run_id": r["run_id"], "reserve_run_id": nxt["run_id"], "cell": r["cell"]}
                with open(activations_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                activated.append(rec)
                pre_queue.append(nxt["run_id"])
    for a in activated:   # activated but not yet executed reserves go first
        if a["reserve_run_id"] not in done and a["reserve_run_id"] not in pre_queue:
            pre_queue.append(a["reserve_run_id"])
    queue = pre_queue + queue
    state = {"phase": "RUNNING", "queue": queue}

    cluster = Cluster(freeze, args.test_mode, args.endpoint)
    stop_req = {"flag": False}

    def on_signal(sig, frm):
        stop_req["flag"] = True
        log("signal received: will stop after the current run completes")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    if not reg:
        n_prior = cluster.prior_science_records(args.test_router_evidence)
        if n_prior != 0:
            raise SystemExit(f"FAIL_CLOSED: router evidence already holds {n_prior} science (V3-) records before the "
                             "first run (or could not be read: -1); science run ids must be unused")
    log(f"campaign start/resume: {len(done)} runs recorded, {len(queue)} queued, freeze {args.freeze_sha[:16]}")
    executed = 0
    consecutive_invalid = 0
    while state["queue"]:
        if stop_req["flag"]:
            state["phase"] = "STOPPED_BY_OPERATOR"
            break
        if args.max_runs and executed >= args.max_runs:
            state["phase"] = "STOPPED_MAX_RUNS"
            break
        rid = state["queue"][0]
        ident = ids[rid]
        if valid.get(ident["cell"], 0) >= 5:
            log(f"SKIP {rid}: cell {ident['cell']} already has 5 valid runs")
            state["queue"].pop(0)
            continue
        ok, details = False, {}
        for attempt in range(1, PRECHECK_TRIES + 1):
            ok, details = cluster.precheck()
            if ok:
                break
            log(f"PRECHECK {rid} attempt {attempt} failed: {details.get('fail')}")
            time.sleep(30)
        if not ok:
            state["phase"] = "PAUSED_PRECHECK"
            write_status(freeze, state, f"precheck failed before {rid}: {details.get('fail')} (identity NOT consumed)")
            log(f"PAUSED: precheck failed before {rid}; no request sent; fix the cluster and re-run to resume")
            return 3
        state["current"] = rid
        write_status(freeze, state, f"running {rid}")
        log(f"RUN {rid} ({ident['cell']}, {ident['identity_type']}, {ident['expected_requests']:,} requests)")
        activated_after = next((a["invalid_run_id"] for a in activated if a["reserve_run_id"] == rid), None)
        disposition, reasons = execute(ident, freeze, cluster, args, activated_after)
        executed += 1
        state["queue"].pop(0)
        state["current"] = None
        log(f"DONE {rid}: {disposition} {reasons if reasons else ''}")
        if disposition == "VALID":
            valid[ident["cell"]] = valid.get(ident["cell"], 0) + 1
            consecutive_invalid = 0
        else:
            nxt = next((x for x in reserves[ident["cell"]] if x["run_id"] not in used_reserves
                        and x["run_id"] not in {a["reserve_run_id"] for a in activated}), None)
            if nxt:
                rec = {"time": utc(), "invalid_run_id": rid, "reserve_run_id": nxt["run_id"], "cell": ident["cell"]}
                with open(activations_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                activated.append(rec)
                used_reserves.add(nxt["run_id"])
                state["queue"].insert(0, nxt["run_id"])
                log(f"ACTIVATE reserve {nxt['run_id']} after objective INVALID of {rid}")
            else:
                nxt = None
                log(f"RESERVE EXHAUSTED for cell {ident['cell']}: coverage incomplete")
            consecutive_invalid += 1
            if not args.continue_after_invalid or consecutive_invalid >= 2:
                state["phase"] = "PAUSED_AFTER_INVALID"
                write_status(freeze, state, f"{rid} INVALID {reasons}; " + (f"reserve {nxt['run_id']} queued first" if nxt
                             else f"RESERVES EXHAUSTED for {ident['cell']}") + "; paused to protect reserves")
                log("PAUSED after INVALID run (reserve already queued first); inspect, then re-run to resume")
                return 4
        write_status(freeze, state)
    if not state["queue"]:
        state["phase"] = "COMPLETE"
    write_status(freeze, state)
    log(f"campaign {state['phase']}: valid per cell {dict(sorted(valid.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
