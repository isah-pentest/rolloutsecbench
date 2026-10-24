#!/usr/bin/env bash
# RolloutSecBench: record the workstation and software versions of the campaign environment, read-only.
# Run from the repository root in WSL, with Docker Desktop and the campaign cluster running:
#
#   bash release/collect_environment.sh > release_environment.txt 2>&1
#
# Collects hardware and OS versions only. It does not record host names, user names, serial numbers,
# network addresses or any file contents. Nothing is changed on the host, in Docker or in the cluster.
set -uo pipefail
CTX=kind-rolloutsecbench-qualification
NS=rolloutsecbench
D=docker; command -v docker >/dev/null || D=docker.exe
PS="powershell.exe -NoProfile -NonInteractive -Command"
sec() { echo; echo "== $*"; }

echo "collected_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

sec "host hardware (Windows)"
$PS "\$c=Get-CimInstance Win32_ComputerSystem; 'manufacturer: '+\$c.Manufacturer; 'model: '+\$c.Model; 'installed_memory_bytes: '+\$c.TotalPhysicalMemory" 2>&1 | tr -d '\r'
$PS "Get-CimInstance Win32_Processor | % { 'cpu: '+\$_.Name.Trim(); 'physical_cores: '+\$_.NumberOfCores; 'logical_processors: '+\$_.NumberOfLogicalProcessors; 'max_clock_mhz: '+\$_.MaxClockSpeed }" 2>&1 | tr -d '\r'
$PS "Get-CimInstance Win32_PhysicalMemory | % { 'memory_module: '+[math]::Round(\$_.Capacity/1GB)+' GB, '+\$_.ConfiguredClockSpeed+' MT/s, SMBIOS type '+\$_.SMBIOSMemoryType }" 2>&1 | tr -d '\r'
$PS "Get-CimInstance Win32_VideoController | % { 'gpu: '+\$_.Name+', driver '+\$_.DriverVersion }" 2>&1 | tr -d '\r'
$PS "Get-PhysicalDisk | % { 'disk: '+\$_.MediaType+', '+\$_.BusType+', '+[math]::Round(\$_.Size/1e12,2)+' TB' }" 2>&1 | tr -d '\r'

sec "host operating system (Windows)"
$PS "\$o=Get-CimInstance Win32_OperatingSystem; 'os: '+\$o.Caption; 'os_version: '+\$o.Version; 'os_build: '+\$o.BuildNumber" 2>&1 | tr -d '\r'
$PS "'power_scheme: '+((powercfg /getactivescheme) -replace '^.*\(|\).*$','')" 2>&1 | tr -d '\r'

sec "WSL 2"
wsl.exe --version 2>&1 | tr -d '\0\r'
echo "kernel: $(uname -r)"
. /etc/os-release 2>/dev/null && echo "distribution: $PRETTY_NAME"
echo "wsl_logical_processors: $(nproc)"
echo "wsl_memory_total_kib: $(awk '/MemTotal/ {print $2}' /proc/meminfo)"
echo "wslconfig (memory/processors limits, if set):"
WINPROFILE=$(wslpath "$($PS '$env:USERPROFILE' 2>/dev/null | tr -d '\r')" 2>/dev/null)
[[ -n "$WINPROFILE" && -f "$WINPROFILE/.wslconfig" ]] && grep -iE '^\s*(memory|processors|swap)\s*=' "$WINPROFILE/.wslconfig" || echo "  (no .wslconfig limits)"

sec "container and Kubernetes tooling"
$D version --format 'docker_client: {{.Client.Version}}
docker_server: {{.Server.Version}}
docker_server_os: {{.Server.Os}}/{{.Server.Arch}}' 2>&1
$D info --format 'docker_desktop: {{.OperatingSystem}}
docker_cpus: {{.NCPU}}
docker_memory_bytes: {{.MemTotal}}
cgroup: {{.CgroupDriver}} v{{.CgroupVersion}}' 2>&1
echo "kind: $(kind version 2>&1)"
kubectl version --client -o yaml 2>&1 | grep -E 'gitVersion' | sed 's/^ */kubectl_client_/'
kubectl --context $CTX version -o yaml 2>&1 | sed -n '/serverVersion/,$p' | grep gitVersion | sed 's/^ */kubernetes_server_/'
kubectl --context $CTX get nodes -o jsonpath='{range .items[*]}node_kubelet: {.status.nodeInfo.kubeletVersion}{"\n"}node_runtime: {.status.nodeInfo.containerRuntimeVersion}{"\n"}node_os: {.status.nodeInfo.osImage}{"\n"}node_kernel: {.status.nodeInfo.kernelVersion}{"\n"}node_cpu_capacity: {.status.capacity.cpu}{"\n"}node_memory_capacity: {.status.capacity.memory}{"\n"}{end}' 2>&1

sec "campaign containers: Python and packages actually installed"
for dep in router model-v1 model-v2; do
  echo "-- $dep"
  kubectl --context $CTX -n $NS get deploy $dep -o jsonpath='image: {.spec.template.spec.containers[0].image}{"\n"}cpu/memory requests: {.spec.template.spec.containers[0].resources}{"\n"}replicas: {.spec.replicas}{"\n"}' 2>&1
  kubectl --context $CTX -n $NS exec deploy/$dep -- python -c "import sys,platform; print('python', sys.version.split()[0], platform.machine())" 2>&1
  kubectl --context $CTX -n $NS exec deploy/$dep -- python -m pip list --format=freeze 2>&1 | grep -iE '^(xgboost|scikit-learn|numpy|pandas|scipy|joblib|fastapi|uvicorn|starlette|pydantic|httpx)=='
done

sec "local Python environments"
for py in .venv*/bin/python; do
  [[ -x "$py" ]] || continue
  echo "-- $py"
  "$py" -c "import sys; print('python', sys.version.split()[0])"
  "$py" -m pip list --format=freeze 2>/dev/null | grep -iE '^(xgboost|scikit-learn|numpy|pandas|scipy|statsmodels|matplotlib|joblib|pyarrow)=='
done

sec "versions recorded in the frozen model manifests"
grep -hoE '"(python|numpy|pandas|scikit_learn|sklearn|xgboost|scipy|joblib)"\s*:\s*"[^"]+"' models/v1/*.json models/v2/*.json 2>/dev/null | sort -u

sec "analysis of record provenance"
grep -hoE '"(python|numpy|matplotlib|scipy|statsmodels)"\s*:\s*"[^"]+"' results/derived/phase9_v3/analysis_20261022T200702Z/*.json 2>/dev/null | sort -u

sec "TeX (for the manuscript build)"
pdflatex --version 2>/dev/null | head -1 || echo "pdflatex not on WSL PATH"
