#!/usr/bin/env python3
"""
RolloutSecBench - export the Kubernetes objects of the confirmatory campaign and verify them against
the runtime recorded in the freeze manifest (runtime_at_freeze). Read-only on the cluster.

Run from the repository root, with the campaign cluster running and selected as kubectl context:

    python3 release/export_campaign_k8s.py [--context kind-rolloutsecbench-qualification] [--out kubernetes/campaign]

Writes one cleaned JSON manifest per object (usable with kubectl apply -f) (status, uid, resourceVersion, timestamps, managedFields and
last-applied annotations removed) to --out, plus EXPORT_VERIFICATION.json. Reports, for each check,
whether the live object equals what the freeze recorded:
  * router Deployment: image, command, environment
  * ConfigMaps phase7-router-*: SHA-256 of every data entry (as hashed by make_v3_freeze.py)
  * model pods: image and imageID
It never modifies the cluster. If a check differs, the export is still written but marked DIFFERS, and
it must not be released as "the manifests that ran the campaign" without explanation.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
FREEZE = ROOT / "data/manifests/phase9_v3/phase9_v3_freeze.json"
NS = "rolloutsecbench"
DROP_META = {"uid", "resourceVersion", "creationTimestamp", "generation", "managedFields", "selfLink"}


def kubectl(ctx, *args):
    cmd = ["kubectl"] + (["--context", ctx] if ctx else []) + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        sys.exit(f"kubectl failed: {' '.join(cmd)}\n{r.stderr.strip()[:500]}")
    return r.stdout


def clean(obj):
    obj.pop("status", None)
    md = obj.get("metadata", {})
    for k in DROP_META:
        md.pop(k, None)
    ann = md.get("annotations") or {}
    for k in list(ann):
        if k.startswith("kubectl.kubernetes.io/last-applied") or k.startswith("deployment.kubernetes.io/"):
            ann.pop(k)
    if not ann:
        md.pop("annotations", None)
    if obj.get("kind") == "Service":
        spec = obj.get("spec", {})
        for k in ("clusterIP", "clusterIPs", "internalTrafficPolicy", "ipFamilies", "ipFamilyPolicy", "sessionAffinity"):
            spec.pop(k, None)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="kind-rolloutsecbench-qualification")
    ap.add_argument("--out", default="kubernetes/campaign")
    a = ap.parse_args()
    fz = json.loads(FREEZE.read_text())
    rt = fz.get("runtime_at_freeze") or {}
    out = ROOT / a.out
    if out.exists() and any(out.iterdir()):
        sys.exit(f"REFUSING: {out} is not empty")
    out.mkdir(parents=True, exist_ok=True)

    objs = {}
    for kind in ("deployment", "service", "configmap"):
        for item in json.loads(kubectl(a.context, "-n", NS, "get", kind, "-o", "json"))["items"]:
            name = item["metadata"]["name"]
            if kind == "configmap" and name == "kube-root-ca.crt":
                continue
            objs[(item["kind"], name)] = item
    pods = json.loads(kubectl(a.context, "-n", NS, "get", "pods", "-o", "json"))["items"]
    checks = []

    def check(name, live, frozen):
        checks.append({"check": name, "status": "MATCH" if live == frozen else "DIFFERS",
                       "live": live, "frozen": frozen})

    dep = objs.get(("Deployment", "router"))
    if dep:
        c = dep["spec"]["template"]["spec"]["containers"][0]
        check("router image", c.get("image"), rt.get("router_image"))
        check("router command", c.get("command", []) + c.get("args", []), rt.get("router_command"))
        check("router env", {e["name"]: e.get("value") for e in c.get("env", [])}, rt.get("router_env"))
    else:
        checks.append({"check": "router Deployment", "status": "MISSING"})
    for cm_name, frozen in (rt.get("router_source_configmaps_sha256") or {}).items():
        cm = objs.get(("ConfigMap", cm_name))
        live = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in (cm or {}).get("data", {}).items()} if cm else None
        check(f"ConfigMap {cm_name} data hashes", live, frozen)
    frozen_pods = {p["name"].rsplit("-", 2)[0]: p for p in rt.get("pods", [])}
    for p in pods:
        app = p["metadata"]["name"].rsplit("-", 2)[0]
        if app in frozen_pods:
            st = p["status"].get("containerStatuses", [])
            check(f"pod {app} image", [s.get("image") for s in st], frozen_pods[app]["image"])
            check(f"pod {app} imageID", [s.get("imageID") for s in st], frozen_pods[app]["imageID"])

    for (kind, name), obj in sorted(objs.items()):
        # JSON is a valid kubectl manifest format and preserves ConfigMap data byte for byte
        (out / f"{kind.lower()}-{name}.json").write_text(json.dumps(clean(obj), indent=2, sort_keys=False) + "\n")
    report = {"context": a.context, "namespace": NS, "freeze": str(FREEZE.relative_to(ROOT)),
              "objects": sorted(f"{k}/{n}" for k, n in objs), "checks": checks,
              "all_match": all(c["status"] == "MATCH" for c in checks)}
    (out / "EXPORT_VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n")
    for c in checks:
        print(f"[{c['status']:7}] {c['check']}")
    print(f"\n{len(objs)} objects written to {out}; all checks match the freeze: {report['all_match']}")
    return 0 if report["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
