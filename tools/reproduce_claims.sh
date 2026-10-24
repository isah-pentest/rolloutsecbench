#!/usr/bin/env bash
# RolloutSecBench - regenerate every number, table and figure of the paper from the frozen evidence.
#
# Offline: sends no traffic and never modifies evidence. Each analysis script verifies its own inputs
# against recorded SHA-256 values and refuses to write if they differ, and make_claims_tex.py refuses
# if any qualitative statement in the paper is not supported by the evidence.
#
# Requires (see REPRODUCE.md):
#   * the UNSW-NB15 files and frozen offline artifacts in data/, models/, results/derived/
#   * the evidence archive unpacked at the repository root (data/manifests/phase9_v3/, results/raw/phase9_v3/)
#   * Python 3.11+ with numpy and matplotlib (requirements.txt)
#
# Usage (repository root):
#   bash tools/reproduce_claims.sh [--reference-dir DIR] [--m 1000] [--allow-skips]
#     --reference-dir  a directory of generated files to compare against (e.g. the archive's
#                      results/derived/phase9_v3/claims); macros are compared value by value
#     --m              hypothetical run identities per mixed cell in the exploratory routing check
#                      (the paper reports 1,000)
#     --allow-skips    run even if tools/verify_release.py reports skipped (absent) artifacts
set -euo pipefail

PY="${PYTHON:-python3}"
M=1000
REF=""
STRICT="--strict"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reference-dir) REF="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --allow-skips) STRICT=""; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="results/reproduced/${STAMP}"
mkdir -p "${OUT}"
LOG="${OUT}/reproduce.log"
exec > >(tee -a "${LOG}") 2>&1

step() { echo; echo "==> $*"; }

step "0/9 integrity: release manifest, published hashes, pinned code, routing rule"
"${PY}" tools/verify_release.py ${STRICT}

step "1/9 confirmatory analysis of the VALID runs (registry, per-run outcomes, Gate 4, latency, Figure 3)"
before="$(ls -d results/derived/phase9_v3/analysis_* 2>/dev/null | sort | tail -n 1 || true)"
"${PY}" workloads/analyze_v3.py --freeze data/manifests/phase9_v3/phase9_v3_freeze.json
A="$(ls -d results/derived/phase9_v3/analysis_* | sort | tail -n 1)"
if [[ "${A}" == "${before}" ]]; then echo "analysis directory was not created" >&2; exit 1; fi
echo "analysis directory: ${A}"

step "2/9 H2-H5 tests and H3 regression"
"${PY}" analysis/confirmatory/h2_h5_tests.py --analysis-dir "${A}" --out-dir "${OUT}/confirmatory"

step "3/9 exploratory routing-rule check (${M} identities per mixed cell)"
"${PY}" analysis/exploratory/routing_null_check.py --analysis-dir "${A}" --out-dir "${OUT}/routing_null" --m "${M}"

step "4/9 Gate-4 values and Table 6"
"${PY}" analysis/tables/make_gate4_tex.py --analysis-dir "${A}" --out-dir "${OUT}/gate4"

step "5/9 Tables 1, S1 and 2"
"${PY}" analysis/tables/make_table1_dataset.py --out-dir "${OUT}/table1"
"${PY}" analysis/tables/make_table2_models.py --out-dir "${OUT}/table2"

step "6/9 Figure 2 (malicious revision disagreement)"
"${PY}" analysis/figures/make_fig2_disagreement.py --repo . --out-dir "${OUT}/figures"

step "7/9 Figure 4 (rollout x attacker interaction)"
"${PY}" analysis/figures/make_fig5_interaction.py --analysis-dir "${A}" --out-dir "${OUT}/fig5"

step "8/9 every number and qualitative claim of the paper (claims_values.tex, Tables 4 and 7)"
"${PY}" analysis/claims/make_claims_tex.py --analysis-dir "${A}" \
    --confirmatory-dir "${OUT}/confirmatory" --routing-null-dir "${OUT}/routing_null" \
    --table1-dir "${OUT}/table1" --table2-dir "${OUT}/table2" --out-dir "${OUT}/claims"

step "9/9 collect the files the paper inputs"
P="${OUT}/paper"
mkdir -p "${P}/tables" "${P}/images"
cp "${OUT}/gate4/gate4_values.tex" "${OUT}/gate4/table_gate4.tex" "${P}/tables/"
cp "${OUT}/confirmatory/h2_h5_values.tex" "${P}/tables/"
cp "${OUT}/claims/claims_values.tex" "${OUT}/claims/run_accounting.tex" "${OUT}/claims/latency_table.tex" "${P}/tables/"
cp "${OUT}/figures/fig2_malicious_disagreement.pdf" "${OUT}/fig5/fig5_interaction.pdf" "${P}/images/"
cp "${A}/fig_exposure_law.pdf" "${P}/images/" 2>/dev/null || echo "note: ${A}/fig_exposure_law.pdf not produced (matplotlib missing?)"
if [[ -f paper/figures/fig1_architecture.tex ]] && command -v pdflatex >/dev/null; then
  (cd paper/figures && pdflatex -interaction=nonstopmode -halt-on-error fig1_architecture.tex >/dev/null) \
    && cp paper/figures/fig1_architecture.pdf "${P}/images/" || echo "note: Figure 1 did not compile"
fi
(cd "${P}" && find . -type f | sort | xargs sha256sum) > "${OUT}/GENERATED.sha256"

if [[ -n "${REF}" ]]; then
  step "compare generated LaTeX values with ${REF}"
  "${PY}" tools/compare_generated.py "${REF}" "${P}/tables"
fi

echo
echo "DONE. Generated paper inputs: ${P}  (checksums: ${OUT}/GENERATED.sha256; log: ${LOG})"
