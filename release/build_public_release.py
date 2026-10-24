#!/usr/bin/env python3
"""
RolloutSecBench - build the public release tree from the private research repository.

Run from the PRIVATE repository root:

    python3 release/build_public_release.py --out ../rolloutsecbench-public

Sources of released files
  * release/allowlist.txt  repository files, copied byte for byte at the same path
  * release/public/        public versions of repository front matter (README.md, requirements.txt, ...),
                           copied to the release root under the same relative path; used where the
                           private file of that name must stay private (e.g. the pre-execution README)

The build REFUSES to finish if
  * an allowlisted file is missing (unless --allow-missing);
  * a file matches release/denylist.txt (internal documents, evidence, data, credentials, environments);
  * an allowlisted file is not committed, or differs from its committed version (unless --allow-uncommitted),
    so that the release corresponds to one commit;
  * a file pinned in the freeze manifest differs from its pinned hash;
  * a release/public file would replace an allowlisted file;
  * any released text file contains a secret-like line.
It writes RELEASE_MANIFEST.sha256 (source commit + one line per released file), which
tools/verify_release.py checks.
"""
import argparse
import fnmatch
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
OVERLAY = ROOT / "release/public"
SECRET = re.compile(
    r"(-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|xox[abp]-[A-Za-z0-9-]{10,}"
    r"|(?i:(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]))")
TEXT_EXT = {".py", ".sh", ".md", ".txt", ".json", ".csv", ".tex", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".cff", ""}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    r = subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT)
    return r.stdout.rstrip("\n") if r.returncode == 0 else None


def patterns(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]


def denied(rel, deny, by_name=True):
    # a denylist line "!<exact path>" exempts that one file (recorded in docs/RELEASE_DECISIONS.md)
    low = rel.lower()
    if "!" + low in (p.lower() for p in deny):
        return []
    return [p for p in deny if not p.startswith("!") and (fnmatch.fnmatch(low, p.lower())
            or (by_name and fnmatch.fnmatch(Path(low).name, p.lower())))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze", default="data/manifests/phase9_v3/phase9_v3_freeze.json")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--allow-uncommitted", action="store_true", help="for dry runs only")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        sys.exit(f"REFUSING: {out} exists; choose a new directory")
    allow = patterns(HERE / "allowlist.txt")
    deny = patterns(HERE / "denylist.txt")
    fz = ROOT / a.freeze
    pinned = json.loads(fz.read_text()).get("pinned_code_sha256", {}) if fz.exists() else {}
    head = git("rev-parse", "HEAD")
    tracked = set((git("ls-files") or "").splitlines())
    dirty = {l[3:] for l in (git("status", "--porcelain") or "").splitlines()}

    problems, plan, explicit = [], {}, set()  # plan: released path -> source path
    for pat in allow:
        hits = sorted(p for p in ROOT.glob(pat) if p.is_file())
        if not hits:
            (print if a.allow_missing else problems.append)(f"allowlisted but missing: {pat}")
        wildcard = any(ch in pat for ch in "*?[")
        for p in hits:
            rel = p.relative_to(ROOT).as_posix()
            if wildcard and denied(rel, deny):
                print(f"excluded by denylist: {rel}")   # a glob may match files that must stay private
                continue
            plan[rel] = p
            if not wildcard:
                explicit.add(rel)
    if OVERLAY.is_dir():
        for p in sorted(q for q in OVERLAY.rglob("*") if q.is_file()):
            rel = p.relative_to(OVERLAY).as_posix()
            if p.name.endswith((".kitbak", ".pyc")) or "__pycache__" in p.parts:
                print(f"skipped local file in overlay: release/public/{rel}")
                continue
            if rel in plan:
                problems.append(f"release/public/{rel} would replace the allowlisted file {rel}")
            else:
                plan[rel] = p

    for rel, src in sorted(plan.items()):
        srel = src.relative_to(ROOT).as_posix()
        # release/public files replace private front matter of the same name, so for them only full-path
        # patterns apply (e.g. *blueprint*), not file-name patterns such as README.md
        d = denied(srel, deny, by_name=not srel.startswith("release/public/"))
        if d:
            problems.append(f"{srel}: matches denylist pattern {d[0]}")
            continue
        if not a.allow_uncommitted:
            if srel not in tracked:
                problems.append(f"{srel}: not committed (git add + commit it, or remove it from the allowlist)")
            elif srel in dirty:
                problems.append(f"{srel}: differs from its committed version")
        if srel in pinned and sha(src) != pinned[srel]:
            problems.append(f"{srel}: differs from the hash pinned in the freeze manifest")
        if src.suffix in TEXT_EXT and src.stat().st_size < 5_000_000:
            for i, line in enumerate(src.read_text(errors="replace").splitlines(), 1):
                if SECRET.search(line):
                    problems.append(f"{srel}:{i}: secret-like content")
    if problems:
        print("REFUSING to build the release:")
        for x in problems:
            print("  " + x)
        return 1

    out.mkdir(parents=True)
    lines = [f"# RolloutSecBench public release; source commit {head or 'unknown'}"]
    for rel, src in sorted(plan.items()):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        h = sha(dst)
        if h != sha(src):
            sys.exit(f"copy of {rel} is not byte-identical")
        lines.append(f"{h}  {rel}")
    (out / "RELEASE_MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    n_pin = sum(1 for r, s in plan.items() if s.relative_to(ROOT).as_posix() in pinned)
    print(f"release written to {out}: {len(plan)} files from commit {head}, {n_pin} verified against the freeze manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
