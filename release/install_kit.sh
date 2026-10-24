#!/usr/bin/env bash
# Install or update the release kit in the private repository.
# Usage (repository root):  bash <kit>/release/install_kit.sh <kit-directory> [--update]
#   without --update: never overwrites an existing file (reports COLLISION)
#   with --update   : replaces files that a previous kit installed, keeping the old copy as <file>.kitbak;
#                     files that are not kit files are never touched
set -euo pipefail
KIT="${1:?usage: install_kit.sh <kit-directory> [--update]}"
UPDATE="${2:-}"
[[ -d .git ]] || { echo "run from the repository root" >&2; exit 2; }
[[ "$(cd "$KIT" && pwd -P)" != "$(pwd -P)/_release_kit" ]] || { echo "pass the downloaded kit folder, not _release_kit" >&2; exit 2; }
rm -rf _release_kit && mkdir _release_kit && cp -r "$KIT"/. _release_kit/
# kit-owned files: created by the kit, not present in the research repository before it
FILES="REPRODUCE.md docs/REPRODUCTION_MAP.md docs/IDENTIFIERS.md docs/RELEASE_DECISIONS.md
environment/published_hashes.sha256 release/build_public_release.py release/allowlist.txt release/denylist.txt
release/inspect_local_repo.py release/inspect_details.py release/install_kit.sh
release/export_campaign_k8s.py release/write_kind_config.sh release/check_campaign_files.sh release/verify_container_files.py release/collect_environment.sh release/build_evidence_archive.py release/archive_allowlist.txt release/cleanroom_test.sh
release/public/README.md release/public/requirements.txt release/public/LICENSE release/public/CITATION.cff
tools/verify_release.py tools/reproduce_claims.sh tools/compare_generated.py paper/figures/fig1_architecture.tex"
for f in $FILES; do
  [[ -f "$KIT/$f" ]] || continue
  if [[ -e "$f" ]]; then
    if cmp -s "$KIT/$f" "$f"; then echo "same      $f"
    elif [[ "$UPDATE" == "--update" ]]; then cp "$f" "$f.kitbak" && cp "$KIT/$f" "$f" && echo "updated   $f  (previous copy: $f.kitbak)"
    else echo "COLLISION $f  (kept; rerun with --update to replace kit files)"; fi
  else
    mkdir -p "$(dirname "$f")" && cp "$KIT/$f" "$f" && echo "added     $f"
  fi
done
