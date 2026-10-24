#!/usr/bin/env python3
"""
RolloutSecBench Phase-9 v3 - finalize the evidence of a run whose CLIENT completed but whose campaign
RUNNER process terminated before post-run processing (DEV-002). One-off, documented recovery tool.

It performs exactly the runner's post-run steps (v3_campaign.execute), using the runner's own functions:
  move client.log + system_metrics.csv into the run dir -> pod state (P1) -> router evidence extraction ->
  validate_v3_run.py (V1-V7) -> disposition.json -> final_checksums.sha256 -> registry row.
The run is ALWAYS recorded INVALID (the runner did not supervise it to completion); validator failures are
appended to the reasons. Reserve activation is then done by the runner itself on resume (pre-registered).

Refuses if: a runner holds the campaign lock, the run already has a registry row or validation.json,
or the client did not complete (run_summary.json missing).

Usage (repository root, runner NOT running):
  python3 tools/v3_preflight/finalize_interrupted_run.py --run-id V3-R2-A1-P02 \
      --freeze data/manifests/phase9_v3/phase9_v3_freeze.json --pre-state-from V3-R0-A0-P02
"""
import argparse
import fcntl
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "workloads"))
import v3_campaign as C  # noqa: E402  (pinned runner; imported, not modified)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--pre-state-from", required=True,
                    help="last VALID run before the interrupted one; its post-run pod state is the pre-state")
    ap.add_argument("--reason", default="RUNNER_PROCESS_TERMINATED_DURING_RUN (post-run evidence finalized by "
                                        "finalize_interrupted_run.py - DEV-002)")
    ap.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--test-router-evidence", help=argparse.SUPPRESS)
    a = ap.parse_args()

    rid = a.run_id
    fp = Path(a.freeze)
    freeze = json.loads(fp.read_text())
    freeze_sha = C.sha(fp)
    ids = {i["run_id"]: i for i in freeze["identities"]}
    ident = ids[rid]
    rd = C.BASE / rid

    lockf = open(C.BASE / ".campaign.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("REFUSED: a campaign runner is active")
    if any(r["run_id"] == rid for r in C.read_registry()):
        raise SystemExit("REFUSED: run already has a registry row")
    if (rd / "validation.json").exists() or (rd / "disposition.json").exists():
        raise SystemExit("REFUSED: run already validated")
    if not (rd / "run_summary.json").exists():
        raise SystemExit("REFUSED: client did not complete (no run_summary.json)")

    man = json.loads((rd / "run_manifest.json").read_text())
    summ = json.loads((rd / "run_summary.json").read_text())
    cluster = C.Cluster(freeze, a.test_mode, "http://127.0.0.1:18080")

    reasons = [a.reason]
    for src, name in ((C.LOGS / f"{rid}.system_metrics.csv", "system_metrics.csv"),
                      (C.LOGS / f"{rid}.client.log", "client.log")):
        if src.exists() and not (rd / name).exists():
            shutil.move(str(src), rd / name)

    pre = json.loads((C.BASE / a.pre_state_from / "prepost_state.json").read_text())["post"]
    post, _ = cluster.pods()
    stable = {r: (pre.get(r, {}).get("uid"), pre.get(r, {}).get("restarts")) ==
                 (post.get(r, {}).get("uid"), post.get(r, {}).get("restarts")) for r in set(pre) | set(post)}
    if not a.test_mode and not all(stable.values()):
        reasons.append(f"P1_POD_CHANGED_DURING_RUN {[k for k, v in stable.items() if not v]}")
    (rd / "prepost_state.json").write_text(json.dumps(
        {"pre": pre, "pre_source": f"post-run state of {a.pre_state_from}", "post": post, "stable": stable,
         "client_rc": None, "note": "finalized post hoc (DEV-002); runner terminated during the run"}, indent=2) + "\n")

    ev = rd / "router_evidence.jsonl"
    if not ev.exists() and not cluster.router_evidence(rid, ev, a.test_router_evidence):
        reasons.append("ROUTER_EVIDENCE_EXTRACTION_FAILED")
    subprocess.run([sys.executable, str(ROOT / "workloads/validate_v3_run.py"), "--run-dir", str(rd),
                    "--route-basis", freeze["routing"]["basis_sha256"], "--router-evidence", str(ev),
                    "--root", str(ROOT)])
    v = json.loads((rd / "validation.json").read_text())
    reasons += [f"{k}x{n}" for k, n in v["failed_criteria"].items()]

    disp = {"run_id": rid, "cell": ident["cell"], "identity_type": ident["identity_type"],
            "activated_after": None, "disposition": "INVALID", "reasons": reasons,
            "started_utc": man.get("started_utc"), "ended_utc": summ.get("client_completed_utc"),
            "freeze_sha256": freeze_sha, "predicted_outcome_from_freeze": ident["predicted_outcome"],
            "finalized_by": {"tool": "tools/v3_preflight/finalize_interrupted_run.py",
                             "sha256": C.sha(Path(__file__)), "deviation": "DEV-002", "utc": C.utc()}}
    (rd / "disposition.json").write_text(json.dumps(disp, indent=2) + "\n")
    with open(rd / "final_checksums.sha256", "x") as f:
        for p in sorted(rd.iterdir()):
            if p.is_file() and p.name != "final_checksums.sha256":
                f.write(f"{C.sha(p)}  {p.name}\n")
    C.append_registry({"run_id": rid, "cell": ident["cell"], "identity_type": ident["identity_type"],
                       "reserve_order": ident.get("reserve_order") or "", "activated_after": "",
                       "disposition": "INVALID", "reasons": "; ".join(reasons),
                       "started_utc": man.get("started_utc"), "ended_utc": summ.get("client_completed_utc"),
                       "elapsed_s": summ.get("elapsed_s"), "validation_sha256": C.sha(rd / "validation.json")})
    C.log(f"FINALIZED {rid} post hoc (DEV-002): INVALID {reasons}")
    print(f"FINALIZED {rid}: INVALID")
    for r in reasons:
        print("  -", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
