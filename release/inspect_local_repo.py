#!/usr/bin/env python3
"""
RolloutSecBench - inspect the PRIVATE working repository before building the public release.
Read-only, standard library only. Run from the repository root:

    python3 release/inspect_local_repo.py > release_inspection.txt

It reports, without changing anything:
  1. git state: HEAD, branch, uncommitted and untracked files, and whether HEAD equals the commit
     recorded in the freeze manifest (git_head_at_freeze)
  2. pinned code: every file in the freeze manifest's pinned_code_sha256, hashed and compared
  3. frozen inputs: the freeze manifest's frozen_inputs_sha256, plus the paper's published hashes
  4. evidence: VALID/INVALID runs in the registry, per-run directories and their checksum files
  5. analysis of record: analysis_* directories and their ANALYSIS_MANIFEST.sha256 verification
  6. code inventory: every source file, classified as RELEASE (allowlisted), PIPELINE CANDIDATE
     (offline pipeline, serving app, container or Kubernetes files not yet allowlisted), INTERNAL
     (matches the denylist) or OTHER, so the allowlist can be completed
  7. paper inputs: whether each script the paper depends on is present
"""
import fnmatch
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
FREEZE = ROOT / "data/manifests/phase9_v3/phase9_v3_freeze.json"
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".mypy_cache", ".ruff_cache", "_release_kit"}


def skipped(parts):
    return any(x in SKIP_DIRS or x.startswith(".venv") or x.endswith(".egg-info") for x in parts)
CODE_EXT = {".py", ".sh", ".yaml", ".yml", ".json", ".toml", ".cfg", ".txt", ".md", ".tex", ".ipynb", ""}
PIPELINE_HINTS = ("split", "preprocess", "train", "evaluat", "accept", "disagree", "phase1", "phase2",
                  "phase3", "phase4", "phase5", "serving", "model_server", "dockerfile", "k8s", "kube",
                  "deploy", "manifest", "kind", "helm", "requirements", "environment")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT)
    return r.stdout.strip() if r.returncode == 0 else None


def pats(name):
    p = HERE / name
    return [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")] if p.exists() else []


def match(rel, plist):
    low = rel.lower()
    return any(fnmatch.fnmatch(low, x.lower()) or fnmatch.fnmatch(Path(low).name, x.lower()) for x in plist)


def section(t):
    print(f"\n=== {t}")


def main():
    freeze = json.loads(FREEZE.read_text()) if FREEZE.exists() else {}
    section("1. git state")
    head = git("rev-parse", "HEAD")
    print("HEAD:", head, "| branch:", git("rev-parse", "--abbrev-ref", "HEAD"))
    print("HEAD at freeze:", freeze.get("git_head_at_freeze"),
          "| equal" if head and head == freeze.get("git_head_at_freeze") else "| DIFFERENT or unknown")
    st = git("status", "--porcelain") or ""
    print(f"uncommitted/untracked entries: {len(st.splitlines())}")
    for l in st.splitlines()[:200]:
        print("   ", l)

    section("2. pinned code (freeze manifest)")
    if not freeze:
        print("freeze manifest not found at", FREEZE.relative_to(ROOT))
    for rel, h in sorted(freeze.get("pinned_code_sha256", {}).items()):
        p = ROOT / rel
        print(f"  {'MATCH' if p.exists() and sha(p) == h else ('MISSING' if not p.exists() else 'DIFFERS')}  {rel}")

    section("3. frozen inputs and published hashes")
    for rel, h in sorted(freeze.get("frozen_inputs_sha256", {}).items()):
        p = ROOT / rel
        print(f"  {'MATCH' if p.exists() and sha(p) == h else ('MISSING' if not p.exists() else 'DIFFERS')}  {rel}")
    pub = ROOT / "environment/published_hashes.sha256"
    if pub.exists():
        for line in pub.read_text().split("\n"):
            if line.strip() and not line.startswith("#"):
                h, rel = line.split(None, 1)
                p = ROOT / rel
                print(f"  {'MATCH' if p.exists() and sha(p) == h else ('MISSING' if not p.exists() else 'DIFFERS')}  {rel} (paper)")

    section("4. evidence")
    reg = ROOT / "results/raw/phase9_v3/registry.csv"
    if reg.exists():
        import csv
        rows = list(csv.DictReader(open(reg, newline="")))
        cols = rows[0].keys() if rows else []
        dcol = next((c for c in cols if "disposition" in c.lower() or c.lower() in ("status", "validity")), None)
        print(f"registry rows: {len(rows)}; columns: {', '.join(cols)}")
        if dcol:
            from collections import Counter
            print("dispositions:", dict(Counter(r[dcol] for r in rows)))
    else:
        print("registry not found")
    runs = sorted(p for p in (ROOT / "results/raw/phase9_v3").glob("V3-*") if p.is_dir())
    print(f"run directories: {len(runs)}")
    for r in runs:
        files = sorted(x.name for x in r.iterdir())
        size = sum(x.stat().st_size for x in r.rglob("*") if x.is_file())
        print(f"  {r.name}: {len(files)} files, {size/1e6:.1f} MB, checksum files: "
              f"{[f for f in files if 'sha256' in f.lower() or 'checksum' in f.lower()]}")

    section("5. analysis of record")
    for a in sorted((ROOT / "results/derived/phase9_v3").glob("analysis_*")):
        man = a / "ANALYSIS_MANIFEST.sha256"
        if not man.exists():
            print(f"  {a.name}: no manifest")
            continue
        # analyze_v3.py records three external inputs by name, then every file of the directory
        external = {"analyze_v3.py(script)": ROOT / "workloads/analyze_v3.py",
                    "phase9_v3_freeze.json": FREEZE,
                    "registry.csv": ROOT / "results/raw/phase9_v3/registry.csv"}
        bad, notes = 0, []
        for line in man.read_text().splitlines():
            h, rel = line.split(None, 1)
            rel = rel.lstrip("*")
            p = external.get(rel, a / rel)
            ok = p.exists() and sha(p) == h
            if not ok:
                bad += 1
                notes.append(f"{rel}: {'missing' if not p.exists() else 'differs'}")
        print(f"  {a.name}: {'VERIFIED' if not bad else f'{bad} entries differ or missing'}"
              + (f"  [{'; '.join(notes)}]" if notes else ""))
    rec = ROOT / "results/derived/phase9_v3/README_ANALYSIS_OF_RECORD.txt"
    if rec.exists():
        print("  analysis of record note:", " | ".join(rec.read_text().split("\n")[:6]))

    section("6. code inventory")
    allow, deny = pats("allowlist.txt"), pats("denylist.txt")
    allowed = {q.relative_to(ROOT).as_posix() for a in allow for q in ROOT.glob(a) if q.is_file()}
    classes = {"RELEASE": [], "INTERNAL": [], "PIPELINE CANDIDATE": [], "OTHER": []}
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or skipped(p.relative_to(ROOT).parts):
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(("data/raw/", "data/splits/", "data/processed/", "results/", "models/")):
            continue
        if p.suffix.lower() not in CODE_EXT and p.name.lower() != "dockerfile":
            continue
        if match(rel, deny):
            classes["INTERNAL"].append(rel)
        elif rel in allowed:
            classes["RELEASE"].append(rel)
        elif any(h in rel.lower() for h in PIPELINE_HINTS):
            classes["PIPELINE CANDIDATE"].append(rel)
        else:
            classes["OTHER"].append(rel)
    for k, v in classes.items():
        print(f"-- {k} ({len(v)})")
        for rel in v:
            print("   ", rel)

    section("7. scripts the paper depends on")
    for rel in ["workloads/analyze_v3.py", "analysis/confirmatory/h2_h5_tests.py",
                "analysis/exploratory/routing_null_check.py", "analysis/tables/make_gate4_tex.py",
                "analysis/tables/make_table1_dataset.py", "analysis/tables/make_table2_models.py",
                "analysis/claims/make_claims_tex.py", "analysis/figures/make_fig5_interaction.py",
                "analysis/figures/make_fig2_disagreement.py"]:
        print(f"  {'present' if (ROOT / rel).exists() else 'MISSING'}  {rel}")
    print("  (git tracking of each: see inspect_details.py)")


if __name__ == "__main__":
    main()
