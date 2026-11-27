#!/usr/bin/env python3
"""
RolloutSecBench EXT1 campaign runner (revision r1, E3 live cells). Derived from workloads/v3_campaign.py
(pinned 698a3104...). Run lifecycle, evidence capture, fail-closed checks, reserve rule and pausing are the same.
Differences:
  * before each run the router's ROUTING_POLICY is set to the identity's policy (kubectl set env + rollout).
    Only the router pod restarts; the model pods are untouched. Every switch is logged in policy_switches.jsonl.
  * precheck P2 is policy-aware: /health protocol hash and routing_policy, Deployment env (= frozen env with
    ROUTING_POLICY = the identity's policy), ext1 ConfigMap hashes and image IDs must equal the EXT1 live freeze.
  * client ext1_client.py, validator validate_ext1_run.py; run ids EXT1-; 2 valid runs per cell
    (pre-registration Section 5); output results/raw/ext1.
  * --qualify runs benign EXT1Q- dry runs under each policy, with the same prechecks and validator, before the freeze.

  tmux new -s ext1
  python3 workloads/ext1_campaign.py --freeze data/manifests/ext1/ext1_live_freeze.json
  python3 workloads/ext1_campaign.py --status
"""
import argparse, csv, fcntl, hashlib, json, os, shutil, signal, subprocess, sys, threading, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import xm_loader  # noqa: E402

NS = "rolloutsecbench"
BASE = ROOT / "results/raw/ext1"
LOGS = BASE / "_campaign"
REG = BASE / "registry.csv"
STATUS = BASE / "campaign_status.json"
REG_FIELDS = ["run_id", "cell", "identity_type", "reserve_order", "activated_after", "disposition",
              "reasons", "started_utc", "ended_utc", "elapsed_s", "validation_sha256"]
PORT = 18000
INTERVAL = 5.0
PRECHECK_TRIES = 3
EVIDENCE_DIR = "/var/lib/rolloutsecbench/phase6-router-evidence/"
QUALIFY = [("P1", "R2", "A0"), ("P2", "R2", "A1-one"), ("P2", "R2", "A1-multi"), ("P3", "R2", "A1-one")]


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


class Cluster:
    def __init__(self, freeze, args):
        self.f = freeze or {}
        self.test = args.test_mode
        self.test_policy_cmd = args.test_policy_cmd
        self.endpoint = args.endpoint
        rt = self.f.get("runtime_at_freeze") or {}
        self.node = rt.get("node") or args.node
        self.evidence_file = EVIDENCE_DIR + "ext1_router_requests.jsonl"
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
            if p["metadata"].get("deletionTimestamp"):
                continue
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

    def ensure_port_forward(self, restart=False):
        if self.test:
            return "status" in self.health()
        if restart and self.pf is not None:
            self.pf.terminate(); self.pf = None
        if self.pf is None or self.pf.poll() is not None:
            sh(["pkill", "-f", f"port-forward svc/router {PORT}:8000"])
            LOGS.mkdir(parents=True, exist_ok=True)
            pflog = open(LOGS / "port_forward.log", "a")
            pflog.write(f"--- {utc()} (re)starting port-forward\n"); pflog.flush()
            self.pf = subprocess.Popen(["kubectl", "-n", NS, "port-forward", "svc/router", f"{PORT}:8000"],
                                       stdout=pflog, stderr=subprocess.STDOUT)
            time.sleep(3)
        return "status" in self.health()

    def deployed_policy(self):
        if self.test:
            return self.health().get("routing_policy")
        rc, out, _ = sh(["kubectl", "-n", NS, "get", "deploy", "router", "-o", "json"])
        if rc != 0:
            return None
        env = {e["name"]: e.get("value") for e in json.loads(out)["spec"]["template"]["spec"]["containers"][0].get("env", [])}
        return env.get("ROUTING_POLICY")

    def ensure_policy(self, policy):
        """Set ROUTING_POLICY on the router Deployment if it differs; never touches the model Deployments."""
        cur = self.deployed_policy()
        if cur == policy and self.health().get("routing_policy") == policy:
            return True, "unchanged"
        if self.test:
            rc = subprocess.run(self.test_policy_cmd.format(policy=policy), shell=True).returncode
            time.sleep(3)
            ok = rc == 0 and self.health().get("routing_policy") == policy
        else:
            rc1, _, e1 = sh(["kubectl", "-n", NS, "set", "env", "deploy/router", f"ROUTING_POLICY={policy}"])
            rc2, _, e2 = sh(["kubectl", "-n", NS, "rollout", "status", "deploy/router", "--timeout=300s"], timeout=330)
            ok = rc1 == 0 and rc2 == 0 and self.ensure_port_forward(restart=True) and self.health().get("routing_policy") == policy
            if not ok:
                return False, f"policy switch failed: {e1.strip()[:120]} {e2.strip()[:120]}"
        rec = {"time": utc(), "from": cur, "to": policy, "ok": ok}
        LOGS.mkdir(parents=True, exist_ok=True)
        with open(BASE / "policy_switches.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"POLICY {cur} -> {policy}: {'ok' if ok else 'FAILED'}")
        return ok, "switched"

    def precheck(self, policy):
        """Returns (ok, details). Never sends a /predict request."""
        d = {"time": utc(), "policy": policy}
        free = shutil.disk_usage(ROOT).free / 2 ** 30
        d["disk_free_gib"] = round(free, 1)
        if free < 5:
            return False, {**d, "fail": "P0 disk free < 5 GiB"}
        ok, how = self.ensure_policy(policy)
        if not ok:
            return False, {**d, "fail": f"P2 {how}"}
        pods, err = self.pods()
        d["pods"] = pods
        roles = ("router",) if self.test else ("router", "model-v1", "model-v2")
        if err or not all(pods.get(r, {}).get("ready") for r in roles):
            return False, {**d, "fail": f"P1 pods not all Ready: {err or {k: v.get('ready') for k, v in pods.items()}}"}
        if not self.ensure_port_forward():
            return False, {**d, "fail": "P3 router not reachable through port-forward"}
        h = self.health()
        d["router_health"] = h
        if h.get("routing_policy") != policy:
            return False, {**d, "fail": f"P2 router reports policy {h.get('routing_policy')} != {policy}"}
        rt = self.f.get("runtime_at_freeze")
        if rt and not self.test:
            if h.get("protocol_sha256") != rt["router_protocol_sha256"]:
                return False, {**d, "fail": "P2 router protocol hash differs from freeze"}
            rc, out, _ = sh(["kubectl", "-n", NS, "get", "deploy", "router", "-o", "json"])
            c = json.loads(out)["spec"]["template"]["spec"]["containers"][0] if rc == 0 else {}
            env = {e["name"]: e.get("value") for e in c.get("env", [])}
            if env != {**rt["router_env"], "ROUTING_POLICY": policy}:
                return False, {**d, "fail": "P2 router deployment env differs from freeze"}
            if c.get("args") != rt["router_args"]:
                return False, {**d, "fail": "P2 router module/args differ from freeze"}
            rc, out, _ = sh(["kubectl", "-n", NS, "get", "configmap", "-o", "json"])
            cm = {m["metadata"]["name"]: {k: hashlib.sha256(v.encode()).hexdigest() for k, v in m.get("data", {}).items()}
                  for m in json.loads(out)["items"] if m["metadata"]["name"].startswith("ext1-router")} if rc == 0 else None
            if cm != rt["router_source_configmaps_sha256"]:
                return False, {**d, "fail": "P2 ext1 router source/protocol configmap differs from freeze"}
            frozen_ids = {i for p in rt["pods"] for i in p["imageID"]}
            live_ids = {v.get("imageID") for v in pods.values()}
            if not live_ids <= frozen_ids:
                return False, {**d, "fail": f"P2 image IDs differ from freeze: {live_ids - frozen_ids}"}
        return True, d

    def prior_records(self, needle, test_file=None):
        if self.test:
            return sum(needle in l for l in open(test_file)) if test_file and Path(test_file).exists() else 0
        rc, out, _ = sh(["docker", "exec", self.node, "sh", "-c", f"grep -cF '{needle}' {self.evidence_file} 2>/dev/null || true"], timeout=600)
        return int(out.strip() or 0) if rc == 0 else -1

    def router_evidence(self, run_id, dest, test_file=None):
        needle = f'"run_id": "{run_id}"'
        if self.test:
            with open(test_file) as fi, open(dest, "x") as fo:
                for line in fi:
                    if needle in line:
                        fo.write(line)
            return True
        with open(dest, "x") as fo:
            try:
                rc = subprocess.run(["docker", "exec", self.node, "grep", "-F", needle, self.evidence_file],
                                    stdout=fo, stderr=subprocess.DEVNULL, timeout=900).returncode
            except Exception:
                rc = 99
        return rc in (0, 1)


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
    remaining_req = sum(i["expected_requests"] for i in freeze["identities"] if i["run_id"] in state["queue"])
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


def run_once(rid, policy, state_, attacker, mode, cluster, args, qps, workers, basis, qual_n=None):
    """Client + monitor + post-run checks + evidence extraction + validator. Returns (vdisp, reasons, rd)."""
    rd = BASE / rid
    LOGS.mkdir(parents=True, exist_ok=True)
    tmp_metrics, tmp_clientlog = LOGS / f"{rid}.system_metrics.csv", LOGS / f"{rid}.client.log"
    pre_pods, _ = cluster.pods()
    cmd = [sys.executable, str(HERE / "ext1_client.py"), "--mode", mode, "--run-id", rid, "--policy", policy,
           "--rollout-state", state_, "--attacker", attacker, "--qps", str(qps), "--workers", str(workers),
           "--endpoint", args.endpoint + "/predict", "--root", str(ROOT)]
    if qual_n:
        cmd += ["--qual-n", str(qual_n)]
    stop, holder = threading.Event(), {}
    mon = threading.Thread(target=monitor, args=(cluster, stop, tmp_metrics, lambda: holder.get("pid")), daemon=True)
    mon.start()
    with open(tmp_clientlog, "x") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        holder["pid"] = proc.pid
        rc = proc.wait()
    stop.set()
    mon.join(timeout=INTERVAL + 15)
    post_pods, _ = cluster.pods()
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
        {"pre": pre_pods, "post": post_pods, "stable": stable, "client_rc": rc, "policy": policy}, indent=2) + "\n")
    ev = rd / "router_evidence.jsonl"
    if not ev.exists() and not cluster.router_evidence(rid, ev, args.test_router_evidence):
        reasons.append("ROUTER_EVIDENCE_EXTRACTION_FAILED")
    vdisp = "INVALID"
    if (rd / "requests.jsonl").exists() and (rd / "run_summary.json").exists():
        vr = subprocess.run([sys.executable, str(HERE / "validate_ext1_run.py"), "--run-dir", str(rd),
                             "--route-basis", basis, "--router-evidence", str(ev), "--root", str(ROOT)],
                            capture_output=True, text=True)
        try:
            v = json.loads((rd / "validation.json").read_text())
            vdisp = v["disposition"]
            reasons += [f"{k}x{n}" for k, n in v["failed_criteria"].items()]
        except Exception as e:
            reasons.append(f"VALIDATOR_FAILED {e!r} {vr.stderr[-200:]}")
    else:
        reasons.append("CLIENT_EVIDENCE_INCOMPLETE")
    return vdisp, reasons, rd


def qualify(args):
    """Benign EXT1Q- dry runs under every policy (route reconstruction under each policy's key), before the freeze."""
    basis = json.loads((ROOT / "data/manifests/ext1/ext1_router_protocol_v1.0.json").read_text())["routing"]["route_seed_basis_sha256"]
    cluster = Cluster({}, args)
    results = []
    for pol, st, att in QUALIFY:
        rid = f"EXT1Q-{pol}-{st}-{att}-{args.qualify_tag}"
        ok, d = cluster.precheck(pol)
        if not ok:
            log(f"QUALIFY precheck failed before {rid}: {d.get('fail')}"); return 3
        log(f"QUALIFY {rid} (n={args.qual_n})")
        vdisp, reasons, rd = run_once(rid, pol, st, att, "QUALIFICATION", cluster, args, args.qps, args.workers, basis, args.qual_n)
        disposition = "VALID" if vdisp == "VALID" and not reasons else "INVALID"
        results.append({"run_id": rid, "policy": pol, "disposition": disposition, "reasons": reasons,
                        "validation_sha256": sha(rd / "validation.json") if (rd / "validation.json").exists() else None})
        log(f"QUALIFY {rid}: {disposition} {reasons}")
    out = BASE / f"qualification_{args.qualify_tag}.json"
    out.write_text(json.dumps({"utc": utc(), "runs": results,
                               "decision": "PASS" if all(r["disposition"] == "VALID" for r in results) else "FAIL"}, indent=2) + "\n")
    print("qualification:", "PASS" if all(r["disposition"] == "VALID" for r in results) else "FAIL", out, sha(out))
    return 0


def execute(identity, freeze, cluster, args, activated_after):
    rid, cell = identity["run_id"], identity["cell"]
    started, t0 = utc(), time.time()
    vdisp, reasons, rd = run_once(rid, identity["policy"], identity["rollout_state"], identity["attacker"], "SCIENCE",
                                  cluster, args, freeze["design"]["target_qps"], freeze["design"]["client_workers"],
                                  freeze["routing"]["basis_sha256"])
    ended = utc()
    disposition = "VALID" if (vdisp == "VALID" and not reasons) else "INVALID"
    disp = {"run_id": rid, "cell": cell, "identity_type": identity["identity_type"], "policy": identity["policy"],
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", default="data/manifests/ext1/ext1_live_freeze.json")
    ap.add_argument("--endpoint", default=f"http://127.0.0.1:{PORT}")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--continue-after-invalid", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--qualify", action="store_true", help="benign EXT1Q- dry runs under every policy (before the freeze)")
    ap.add_argument("--qualify-tag", default="Q01")
    ap.add_argument("--qual-n", type=int, default=400)
    ap.add_argument("--qps", type=float, default=16.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--node", default="rolloutsecbench-qualification-control-plane")
    ap.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--test-router-evidence", help=argparse.SUPPRESS)
    ap.add_argument("--test-policy-cmd", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.status:
        print(STATUS.read_text() if STATUS.exists() else "no campaign status yet")
        return 0
    BASE.mkdir(parents=True, exist_ok=True)
    lockf = open(BASE / ".campaign.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("FAIL_CLOSED: another EXT1 runner is active")
    if args.qualify:
        return qualify(args)

    fp = (ROOT / args.freeze) if not Path(args.freeze).is_absolute() else Path(args.freeze)
    recorded = (fp.parent / (fp.name + ".sha256")).read_text().split()[0]
    args.freeze_sha = sha(fp)
    if args.freeze_sha != recorded:
        raise SystemExit("FAIL_CLOSED: freeze file does not match its recorded SHA-256")
    freeze = json.loads(fp.read_text())
    for rel, h in {**freeze["pinned_code_sha256"], **freeze["frozen_inputs_sha256"]}.items():
        if sha(ROOT / rel) != h:
            raise SystemExit(f"FAIL_CLOSED: pinned file changed since freeze: {rel}")
    target_valid = freeze["design"]["valid_runs_per_cell"]

    live = []
    for pd in Path("/proc").iterdir():
        try:
            argv = (pd / "cmdline").read_bytes().split(b"\0")
        except Exception:
            continue
        if len(argv) > 2 and argv[1].endswith(b"ext1_client.py") and b"SCIENCE" in argv:
            live.append(pd.name)
    if live:
        raise SystemExit(f"FAIL_CLOSED: a science client is still running (pid {live})")

    ids = {i["run_id"]: i for i in freeze["identities"]}
    reserves = {}
    for i in freeze["identities"]:
        if i["identity_type"] == "RESERVE":
            reserves.setdefault(i["cell"], []).append(i)
    for c in reserves:
        reserves[c].sort(key=lambda x: x["reserve_order"])

    reg = read_registry()
    done = {r["run_id"]: r for r in reg}
    used_reserves = {r["run_id"] for r in reg if r["identity_type"] == "RESERVE"}
    valid = {}
    for r in reg:
        if r["disposition"] == "VALID":
            valid[r["cell"]] = valid.get(r["cell"], 0) + 1
    for d in sorted(BASE.glob("EXT1-*")):
        if d.is_dir() and d.name not in done and d.name in ids:
            log(f"RECOVER {d.name}: run directory without registry row -> INVALID (interrupted)")
            ident = ids[d.name]
            append_registry({"run_id": d.name, "cell": ident["cell"], "identity_type": ident["identity_type"],
                             "reserve_order": ident.get("reserve_order") or "", "disposition": "INVALID",
                             "reasons": "INTERRUPTED_EXECUTION_EVIDENCE_INCOMPLETE", "ended_utc": utc()})
            done[d.name] = {"disposition": "INVALID", "cell": ident["cell"], "identity_type": ident["identity_type"]}
            if ident["identity_type"] == "RESERVE":
                used_reserves.add(d.name)

    queue = [rid for rid in freeze["design"]["execution_order_primaries"] if rid not in done]
    activations_path = BASE / "activations.jsonl"
    activated = [json.loads(l) for l in open(activations_path)] if activations_path.exists() else []
    replaced = {a["invalid_run_id"] for a in activated}
    pre_queue = []
    for rid_, r in done.items():
        if r["disposition"] != "VALID" and rid_ not in replaced:
            nxt = next((x for x in reserves[r["cell"]] if x["run_id"] not in used_reserves
                        and x["run_id"] not in {a["reserve_run_id"] for a in activated}), None)
            if nxt and valid.get(r["cell"], 0) < target_valid:
                rec = {"time": utc(), "invalid_run_id": rid_, "reserve_run_id": nxt["run_id"], "cell": r["cell"]}
                with open(activations_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                activated.append(rec)
                pre_queue.append(nxt["run_id"])
    for a in activated:
        if a["reserve_run_id"] not in done and a["reserve_run_id"] not in pre_queue:
            pre_queue.append(a["reserve_run_id"])
    state = {"phase": "RUNNING", "queue": pre_queue + queue}

    cluster = Cluster(freeze, args)
    stop_req = {"flag": False}

    def on_signal(sig, frm):
        stop_req["flag"] = True
        log("signal received: will stop after the current run completes")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    if not reg:
        n_prior = cluster.prior_records('"run_id": "EXT1-', args.test_router_evidence)
        if n_prior != 0:
            raise SystemExit(f"FAIL_CLOSED: router evidence already holds {n_prior} EXT1- science records (-1 = unreadable)")
    log(f"EXT1 campaign start/resume: {len(done)} recorded, {len(state['queue'])} queued, freeze {args.freeze_sha[:16]}")
    executed = consecutive_invalid = 0
    while state["queue"]:
        if stop_req["flag"]:
            state["phase"] = "STOPPED_BY_OPERATOR"; break
        if args.max_runs and executed >= args.max_runs:
            state["phase"] = "STOPPED_MAX_RUNS"; break
        rid = state["queue"][0]
        ident = ids[rid]
        if valid.get(ident["cell"], 0) >= target_valid:
            log(f"SKIP {rid}: cell {ident['cell']} already has {target_valid} valid runs")
            state["queue"].pop(0); continue
        ok, details = False, {}
        for attempt in range(1, PRECHECK_TRIES + 1):
            ok, details = cluster.precheck(ident["policy"])
            if ok:
                break
            log(f"PRECHECK {rid} attempt {attempt} failed: {details.get('fail')}")
            time.sleep(30)
        if not ok:
            state["phase"] = "PAUSED_PRECHECK"
            write_status(freeze, state, f"precheck failed before {rid}: {details.get('fail')} (identity NOT consumed)")
            log(f"PAUSED: precheck failed before {rid}; no request sent")
            return 3
        state["current"] = rid
        write_status(freeze, state, f"running {rid}")
        log(f"RUN {rid} ({ident['cell']}, {ident['identity_type']}, {ident['expected_requests']:,} requests)")
        activated_after = next((a["invalid_run_id"] for a in activated if a["reserve_run_id"] == rid), None)
        disposition, reasons = execute(ident, freeze, cluster, args, activated_after)
        executed += 1
        state["queue"].pop(0); state["current"] = None
        log(f"DONE {rid}: {disposition} {reasons if reasons else ''}")
        if disposition == "VALID":
            valid[ident["cell"]] = valid.get(ident["cell"], 0) + 1
            consecutive_invalid = 0
        else:
            nxt = next((x for x in reserves.get(ident["cell"], []) if x["run_id"] not in used_reserves
                        and x["run_id"] not in {a["reserve_run_id"] for a in activated}), None)
            if nxt:
                rec = {"time": utc(), "invalid_run_id": rid, "reserve_run_id": nxt["run_id"], "cell": ident["cell"]}
                with open(activations_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                activated.append(rec); used_reserves.add(nxt["run_id"])
                state["queue"].insert(0, nxt["run_id"])
                log(f"ACTIVATE reserve {nxt['run_id']} after objective INVALID of {rid}")
            else:
                log(f"RESERVE EXHAUSTED for cell {ident['cell']}: coverage incomplete")
            consecutive_invalid += 1
            if not args.continue_after_invalid or consecutive_invalid >= 2:
                state["phase"] = "PAUSED_AFTER_INVALID"
                write_status(freeze, state, f"{rid} INVALID {reasons}; paused to protect reserves")
                log("PAUSED after INVALID run (reserve already queued first); inspect, then re-run to resume")
                return 4
        write_status(freeze, state)
    if not state["queue"]:
        state["phase"] = "COMPLETE"
    write_status(freeze, state)
    log(f"EXT1 campaign {state['phase']}: valid per cell {dict(sorted(valid.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
