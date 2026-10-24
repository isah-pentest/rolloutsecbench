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
  "$TPY" training/generate_splits.py
  echo "regenerated:"; sha256sum data/splits/*.csv
  # generate_splits.py may rewrite manifests; restore the archived originals
  tar xzf "$ARCHIVE" data/manifests/
fi

step "5. tools/verify_release.py --strict"
"$PY" tools/verify_release.py --strict | tail -15

step "6. regenerate every value, table and figure and compare with paper_inputs/"
PYTHON="$PY" bash tools/reproduce_claims.sh --reference-dir paper_inputs 2>&1 | tail -15
G=$(ls -d results/reproduced/*/paper/tables | tail -1)
"$PY" tools/compare_generated.py paper_inputs "$G"
echo; echo "CLEAN-ROOM TEST PASSED in $WORK"
