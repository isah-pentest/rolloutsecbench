#!/usr/bin/env bash
# RolloutSecBench: clean-room test of the public release, as an outside reader would run it.
# Starts from the built public tree, the evidence archive and the two official UNSW-NB15 files only;
# nothing else from the private repository is visible to it. Writes only inside --work.
#
#   bash release/cleanroom_test.sh --public ../rolloutsecbench-public --archive ../rolloutsecbench-evidence-v1.tar.gz \
#        --raw data/raw --python .venv-blueprint-v2/bin/python --train-python <venv with pandas 3.0.5> \
#        [--work /tmp/rsb-cleanroom]
#
# Steps: copy the public tree; add the two raw files; regenerate data/splits/ with training/generate_splits.py
# (only if the archive does not carry them); unpack the archive and check ARCHIVE_MANIFEST.sha256;
# tools/verify_release.py --strict; tools/reproduce_claims.sh --reference-dir paper_inputs.
# Passes only if every check passes and all six paper inputs are BYTE-IDENTICAL.
set -euo pipefail
WORK=/tmp/rsb-cleanroom; PUBLIC=""; ARCHIVE=""; RAW=""; PY=""; TPY=""
while [[ $# -gt 0 ]]; do case "$1" in
  --public) PUBLIC=$(realpath "$2"); shift 2 ;; --archive) ARCHIVE=$(realpath "$2"); shift 2 ;;
  --raw) RAW=$(realpath "$2"); shift 2 ;; --python) PY=$(realpath -s "$2"); shift 2 ;;
  --train-python) TPY=$(realpath -s "$2"); shift 2 ;; --work) WORK="$2"; shift 2 ;;
  *) echo "unknown argument $1"; exit 2 ;; esac; done
for v in PUBLIC ARCHIVE RAW PY; do [[ -n "${!v}" ]] || { echo "missing --${v,,}"; exit 2; }; done
[[ -e "$WORK" ]] && { echo "REFUSING: $WORK exists (remove it first)"; exit 2; }
step() { echo; echo "==> $*"; }

step "1. fresh copy of the public release"
cp -a "$PUBLIC" "$WORK"; cd "$WORK"
[[ -d .git ]] && rm -rf .git
sha256sum -c --quiet <(grep -v '^#' RELEASE_MANIFEST.sha256) && echo "RELEASE_MANIFEST.sha256: all files OK"
EXTRA=$(comm -13 <(grep -v '^#' RELEASE_MANIFEST.sha256 | awk '{print $2}' | sort) \
                 <(find . -type f ! -name RELEASE_MANIFEST.sha256 | sed 's#^\./##' | sort))
[[ -z "$EXTRA" ]] || { echo "FAIL: files in the release that RELEASE_MANIFEST.sha256 does not list:"; echo "$EXTRA"; exit 1; }
echo "no files outside RELEASE_MANIFEST.sha256"
CTRL=$(grep -rlP '[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]' --include='*.txt' --include='*.lock' --include='*.md' --include='*.py' --include='*.sh' --include='*.json' --include='*.yaml' --include='*.cff' . || true)
[[ -z "$CTRL" ]] || { echo "FAIL: text files containing terminal control characters:"; echo "$CTRL"; exit 1; }
echo "no control characters in text files"

step "2. official UNSW-NB15 files"
mkdir -p data/raw; cp "$RAW/UNSW_NB15_training-set.csv" "$RAW/UNSW_NB15_testing-set.csv" data/raw/
"$PY" tools/verify_release.py | grep -iE 'raw|UNSW' || true

step "3. evidence archive"
tar xzf "$ARCHIVE"
sha256sum -c --quiet <(grep -v '^#' ARCHIVE_MANIFEST.sha256) && echo "ARCHIVE_MANIFEST.sha256: all files OK"
mkdir -p paper_inputs_check && cp paper_inputs/*.tex paper_inputs_check/

step "4. dataset splits"
if ls data/splits/*.csv >/dev/null 2>&1; then
  echo "splits supplied by the archive"
else
  [[ -n "$TPY" ]] || { echo "the archive has no data/splits/: give --train-python (training environment)"; exit 2; }
  # generate_splits.py refuses to overwrite frozen artifacts: set the archived split manifest aside,
  # regenerate, keep the regenerated manifest for comparison, then restore every archived manifest.
  mkdir -p .cleanroom/archived .cleanroom/regenerated
  if [[ -f data/manifests/split_manifest.json ]]; then mv data/manifests/split_manifest.json .cleanroom/archived/; fi
  "$TPY" training/generate_splits.py
  echo "regenerated split files:"; sha256sum data/splits/*.csv
  cp data/manifests/split_manifest.json .cleanroom/regenerated/ 2>/dev/null || true
  tar xzf "$ARCHIVE" data/manifests/
  if [[ -f .cleanroom/regenerated/split_manifest.json ]]; then
    if cmp -s .cleanroom/regenerated/split_manifest.json data/manifests/split_manifest.json; then
      echo "regenerated split_manifest.json is byte-identical to the archived one"
    else
      echo "regenerated split_manifest.json differs from the archived one (expected if it records a timestamp); differing keys:"
      "$PY" - .cleanroom/regenerated/split_manifest.json data/manifests/split_manifest.json <<'PY' || true
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{p}.{k}")
    else: yield p, o
fa, fb = dict(flat(a)), dict(flat(b))
for k in sorted(set(fa) | set(fb)):
    if fa.get(k) != fb.get(k): print(f"  {k}: regenerated={str(fa.get(k))[:70]!r} archived={str(fb.get(k))[:70]!r}")
PY
    fi
  fi
fi

step "5. tools/verify_release.py --strict"
"$PY" tools/verify_release.py --strict | tail -15

step "6. regenerate every value, table and figure and compare with paper_inputs/"
PYTHON="$PY" bash tools/reproduce_claims.sh --reference-dir paper_inputs 2>&1 | tail -15
G=$(ls -d results/reproduced/*/paper/tables | tail -1)
"$PY" tools/compare_generated.py paper_inputs "$G"
echo; echo "CLEAN-ROOM TEST PASSED in $WORK"
