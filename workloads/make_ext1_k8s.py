#!/usr/bin/env python3
"""
Revision r1, Phase 6 live: build the EXT1 Kubernetes objects from the published campaign export. Deterministic.

Inputs (hash-checked): kubernetes/campaign/deployment-router.json (the published router Deployment),
serving/router/ext1/router_ext1_v1.py, data/manifests/ext1/ext1_router_protocol_v1.0.json.
Outputs (kubernetes/ext1/):
  configmap-ext1-router-source.json    data: router_ext1_v1.py
  configmap-ext1-router-protocol.json  data: ext1_router_protocol_v1.0.json
  deployment-router-ext1.json          the published Deployment with ONLY these changes:
      module serving.router.phase7.router_phase7_v1 -> serving.router.phase7.router_ext1_v1 (mounted from ConfigMap,
      same image, same namespace-package location); env PHASE7_PROTOCOL_PATH -> EXT1_PROTOCOL_PATH;
      ROUTER_EVIDENCE_PATH -> /evidence/ext1_router_requests.jsonl (separate file, same hostPath volume);
      ROUTING_POLICY=P1 added; phase7 ConfigMap volumes/mounts -> ext1 ConfigMaps; phase annotations.
Model Deployments, Services, image, probes, resources and model hashes are untouched.
"""
import copy, hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = {"kubernetes/campaign/deployment-router.json": None,
       "serving/router/ext1/router_ext1_v1.py": "f56a88cafe3bab237530d5f4c86494223731a76d8dc9b2edc05e724fb6c510dc",
       "data/manifests/ext1/ext1_router_protocol_v1.0.json": "af8c8243662e13cf0c29d5a97bc19878287d91da52655764c757c9954cd2bfde"}
sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
for p, h in SRC.items():
    if h and sha(p) != h: sys.exit(f"hash mismatch: {p}")
OUT = ROOT / "kubernetes/ext1"; OUT.mkdir(parents=True, exist_ok=True)


def cm(name, fname, rel):
    return {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": name, "namespace": "rolloutsecbench"},
            "data": {fname: (ROOT / rel).read_text(encoding="utf-8")}}


objs = {
    "configmap-ext1-router-source.json": cm("ext1-router-source", "router_ext1_v1.py", "serving/router/ext1/router_ext1_v1.py"),
    "configmap-ext1-router-protocol.json": cm("ext1-router-protocol", "ext1_router_protocol_v1.0.json",
                                              "data/manifests/ext1/ext1_router_protocol_v1.0.json"),
}
d = copy.deepcopy(json.loads((ROOT / "kubernetes/campaign/deployment-router.json").read_text()))
tpl = d["spec"]["template"]
tpl["metadata"]["annotations"] = {"rolloutsecbench/phase": "ext1", "rolloutsecbench/router-protocol": "ext1-v1.0"}
c = tpl["spec"]["containers"][0]
if c["args"][0] != "serving.router.phase7.router_phase7_v1:app": sys.exit("unexpected router module in published Deployment")
c["args"][0] = "serving.router.phase7.router_ext1_v1:app"
env = []
for e in c["env"]:
    if e["name"] == "PHASE7_PROTOCOL_PATH":
        env.append({"name": "EXT1_PROTOCOL_PATH", "value": "/ext1/protocol/ext1_router_protocol_v1.0.json"})
    elif e["name"] == "ROUTER_EVIDENCE_PATH":
        env.append({"name": "ROUTER_EVIDENCE_PATH", "value": "/evidence/ext1_router_requests.jsonl"})
    else:
        env.append(e)
env.append({"name": "ROUTING_POLICY", "value": "P1"})
c["env"] = env
mounts = []
for m in c["volumeMounts"]:
    if m["name"] == "phase7-router-source":
        mounts.append({"mountPath": "/app/serving/router/phase7/router_ext1_v1.py", "name": "ext1-router-source",
                       "readOnly": True, "subPath": "router_ext1_v1.py"})
    elif m["name"] == "phase7-router-protocol":
        mounts.append({"mountPath": "/ext1/protocol/ext1_router_protocol_v1.0.json", "name": "ext1-router-protocol",
                       "readOnly": True, "subPath": "ext1_router_protocol_v1.0.json"})
    else:
        mounts.append(m)
c["volumeMounts"] = mounts
vols = []
for v in tpl["spec"]["volumes"]:
    if v["name"] in ("phase7-router-source", "phase7-router-protocol"):
        n = v["name"].replace("phase7", "ext1")
        vols.append({"configMap": {"defaultMode": 420, "name": n}, "name": n})
    else:
        vols.append(v)
tpl["spec"]["volumes"] = vols
objs["deployment-router-ext1.json"] = d
for name, o in objs.items():
    (OUT / name).write_text(json.dumps(o, indent=2, sort_keys=True) + "\n")
    print(sha(f"kubernetes/ext1/{name}"), f"kubernetes/ext1/{name}")
