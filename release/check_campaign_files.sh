#!/usr/bin/env bash
# RolloutSecBench: read-only checks on the newly released campaign files before they are committed.
# Run from the repository root. Prints findings; changes nothing.
set -uo pipefail
F="environment/requirements.lock environment/analysis_environment.lock kubernetes/campaign/*.json kubernetes/campaign/kind-cluster.yaml"
echo "== 1. local paths, editable installs, private addresses"
grep -nE 'file://|/home/|/mnt/c|C:\\\\|^-e |@ git\+|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.' $F || echo "none"
echo "== 2. hostPath volumes and volume sources in the Deployments"
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("kubernetes/campaign/deployment-*.json")):
    spec = json.load(open(f))["spec"]["template"]["spec"]
    for v in spec.get("volumes", []):
        src = {k: v[k] for k in v if k != "name"}
        print(f"{f}: volume {v['name']}: {json.dumps(src)}")
    for c in spec["containers"]:
        for m in c.get("volumeMounts", []):
            print(f"{f}: mount {m['name']} -> {m['mountPath']}{' (ro)' if m.get('readOnly') else ''}")
        print(f"{f}: imagePullPolicy {c.get('imagePullPolicy')}")
PY
echo "== 3. files the router image needs (PHASE6_PROTOCOL_PATH)"
ls -l data/manifests/phase6_router_validation_protocol_v2.1.json 2>&1
git ls-files --error-unmatch data/manifests/phase6_router_validation_protocol_v2.1.json >/dev/null 2>&1 && echo "tracked" || echo "NOT tracked"
grep -n "PHASE6_PROTOCOL_PATH\|phase6_router_validation" serving/router/phase7/router_phase7_v1.py | head
grep -n "COPY\|ADD" serving/docker/Dockerfile serving/router/Dockerfile* 2>/dev/null
ls serving/router/ 2>&1
echo "== 4. secret-like strings"
grep -niE 'password|secret|token|api[_-]?key|BEGIN [A-Z ]*PRIVATE' $F | grep -v '"name": "kube' || echo "none"
echo "== 5. hashes of the new files"
sha256sum $F release/export_campaign_k8s.py release/write_kind_config.sh
