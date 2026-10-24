#!/usr/bin/env bash
# RolloutSecBench: insert the evidence-archive DOI into REPRODUCE.md, release/public/CITATION.cff and
# docs/RELEASE_DECISIONS.md. Run from the repository root:  bash release/set_doi.sh 10.5281/zenodo.NNNNNNN
set -euo pipefail
DOI="${1:-}"
RX='^10\.[0-9]{4,9}/[A-Za-z0-9._/:-]+$'
[[ "$DOI" =~ $RX ]] || { echo "usage: $0 10.5281/zenodo.NNNNNNN"; exit 2; }
grep -q '<EVIDENCE_ARCHIVE_DOI>' REPRODUCE.md || { echo "REPRODUCE.md has no DOI placeholder (already set?)"; exit 1; }
python3 - "$DOI" <<'PY'
import re, sys
doi = sys.argv[1]
p = "REPRODUCE.md"; s = open(p).read()
s = s.replace("`<EVIDENCE_ARCHIVE_DOI>` <!-- RELEASE-BLOCKER: replace with the archive DOI -->",
              f"[https://doi.org/{doi}](https://doi.org/{doi}) (`rolloutsecbench-evidence-v1.tar.gz`, SHA-256 "
              "`17b61615728b68b27c689b48258a1b78a3d095f14f5fc11a0d8eca8e34dd12ca`)")
assert "<EVIDENCE_ARCHIVE_DOI>" not in s, "placeholder left in REPRODUCE.md"
open(p, "w").write(s)
p = "release/public/CITATION.cff"; s = open(p).read()
if "\ndoi:" not in s:
    s = s.replace("license: MIT\n", f"license: MIT\nidentifiers:\n  - type: doi\n    value: \"{doi}\"\n    description: \"Evidence archive\"\n")
open(p, "w").write(s)
p = "docs/RELEASE_DECISIONS.md"; s = open(p).read()
s = re.sub(r"\| DOI \|[^\n]*\n", f"| DOI | Done: evidence archive deposited at https://doi.org/{doi} | — |\n", s)
open(p, "w").write(s)
print("DOI set:", doi)
PY
grep -n "doi.org" REPRODUCE.md | head -2; grep -n -A3 "identifiers" release/public/CITATION.cff
