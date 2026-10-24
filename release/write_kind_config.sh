#!/usr/bin/env bash
# RolloutSecBench: write kubernetes/campaign/kind-cluster.yaml, the kind definition of the campaign cluster,
# reconstructed from the live node. It records only what the live cluster shows:
#   cluster name rolloutsecbench-qualification, one control-plane node,
#   kind default networking (kindnet CNI, podSubnet 10.244.0.0/16, serviceSubnet 10.96.0.0/16),
#   no host bind mounts (the node mounts only /lib/modules, which kind adds itself),
#   node image kindest/node:v1.37.0, pinned by the digest of the image the live node runs.
# Read-only on Docker. Run from the repository root.
set -euo pipefail
OUT=kubernetes/campaign/kind-cluster.yaml
NODE=rolloutsecbench-qualification-control-plane
D=docker; command -v docker >/dev/null || D=docker.exe
[[ -e "$OUT" ]] && { echo "REFUSING: $OUT exists"; exit 1; }
IMG_ID=$($D inspect "$NODE" --format '{{.Image}}')
TAG=$($D inspect "$NODE" --format '{{.Config.Image}}')
DIGEST=$($D image inspect "$IMG_ID" --format '{{range .RepoDigests}}{{println .}}{{end}}' | grep '^kindest/node@sha256:' | head -1 || true)
[[ -n "$DIGEST" ]] || { echo "No kindest/node repo digest found for $IMG_ID ($TAG); stop and report this."; exit 1; }
SUBNETS=$($D exec "$NODE" cat /kind/kubeadm.conf | grep -E 'podSubnet|serviceSubnet' | tr -d ' ' | sort -u)
echo "$SUBNETS" | grep -qx 'podSubnet:10.244.0.0/16' || { echo "podSubnet differs: $SUBNETS"; exit 1; }
echo "$SUBNETS" | grep -qx 'serviceSubnet:10.96.0.0/16' || { echo "serviceSubnet differs: $SUBNETS"; exit 1; }
PORTS=$($D inspect "$NODE" --format '{{json .HostConfig.PortBindings}}')
cat > "$OUT" <<YAML
# RolloutSecBench campaign cluster (kind). Reconstructed on $(date -u +%Y-%m-%d) from the live node
# $NODE, which ran the confirmatory campaign; not the file used to create it.
# Live node: image $TAG, kind default networking (kindnet), podSubnet 10.244.0.0/16,
# serviceSubnet 10.96.0.0/16, no host bind mounts. Host port bindings of the live node:
# $PORTS
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: rolloutsecbench-qualification
networking:
  podSubnet: 10.244.0.0/16
  serviceSubnet: 10.96.0.0/16
nodes:
  - role: control-plane
    image: $DIGEST
YAML
echo "wrote $OUT  (node image $DIGEST)"
