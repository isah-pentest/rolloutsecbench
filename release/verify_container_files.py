#!/usr/bin/env python3
"""
RolloutSecBench - list every file under /app in the running router and model containers, hash it inside
the container, and compare it with the file at the same path in this repository (/app/<path> -> <path>).
Read-only (kubectl exec runs a Python one-liner that only reads files). Run from the repository root:

    python3 release/verify_container_files.py [--context kind-rolloutsecbench-qualification]

Writes kubernetes/campaign/CONTAINER_FILES.json and prints, per file:
  MATCH        same bytes as the repository file
  DIFFERS      the repository file has other bytes
  NOT-IN-REPO  no repository file at that path (to resolve: which source produced it?)
  mounted      ConfigMap mount (already verified by export_campaign_k8s.py)
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

NS = "rolloutsecbench"
LISTER = ("import hashlib,os,json;r={}\n"
          "for d,_,fs in os.walk('/app'):\n"
          " if '__pycache__' in d: continue\n"
          " for f in fs:\n"
          "  p=os.path.join(d,f)\n"
          "  try: r[p]=hashlib.sha256(open(p,'rb').read()).hexdigest()\n"
          "  except Exception as e: r[p]='ERROR '+str(e)\n"
          "print(json.dumps(r))")


def kubectl(ctx, *a):
    r = subprocess.run(["kubectl", "--context", ctx, "-n", NS, *a], capture_output=True, text=True, timeout=120)
    if r.returncode:
        sys.exit(f"kubectl failed: {' '.join(a)}\n{r.stderr[:400]}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="kind-rolloutsecbench-qualification")
    a = ap.parse_args()
    report = {}
    for dep in ("router", "model-v1", "model-v2"):
        spec = json.loads(kubectl(a.context, "get", "deployment", dep, "-o", "json"))["spec"]["template"]["spec"]
        mounts = {m["mountPath"] for c in spec["containers"] for m in c.get("volumeMounts", [])}
        files = json.loads(kubectl(a.context, "exec", f"deploy/{dep}", "--", "python", "-c", LISTER))
        rows = []
        for p, h in sorted(files.items()):
            rel = p[len("/app/"):]
            if p in mounts:
                st = "mounted"
            else:
                local = Path(rel)
                if not local.is_file():
                    st = "NOT-IN-REPO"
                else:
                    st = "MATCH" if hashlib.sha256(local.read_bytes()).hexdigest() == h else "DIFFERS"
            rows.append({"path": p, "sha256": h, "status": st})
            print(f"{dep:9} {st:12} {p}")
        report[dep] = rows
    out = Path("kubernetes/campaign/CONTAINER_FILES.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    bad = [r for v in report.values() for r in v if r["status"] in ("DIFFERS", "NOT-IN-REPO")]
    print(f"\nwrote {out}; {len(bad)} file(s) DIFFERS or NOT-IN-REPO")


if __name__ == "__main__":
    main()
