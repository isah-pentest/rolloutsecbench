#!/usr/bin/env python3
"""
RolloutSecBench - second-stage, read-only inspection for the public release (standard library only).
Run from the PRIVATE repository root:

    python3 _release_kit/release/inspect_details.py > release_details.txt 2>&1

It prints what is needed to finish release/allowlist.txt and to confirm that the released tree can
reproduce every claim:
  A. freeze manifest: git commit, deployed runtime (router image/command/env/configmaps, pod images),
     pinned code, frozen inputs
  B. Kubernetes and deviation YAML: kinds, names, images, configmaps, services/ports
  C. every allowlisted and REVIEW-marked file: exists, tracked and clean in git, size, SHA-256
  D. content scan of those files: internal-document references, absolute paths, addresses, e-mail
     addresses, secret-like strings, and local imports that would not resolve in the release
  E. full text of the REVIEW-marked files (truncated at 250 lines each) and of the repository's
     README / ARTIFACT-EVALUATION / reproduce.sh / Makefile
  F. evidence that would go to the archive: sizes, the analysis-of-record note, git log
"""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
FREEZE = ROOT / "data/manifests/phase9_v3/phase9_v3_freeze.json"
INTERNAL = re.compile(r"blueprint|statistical audit|audit flag|\bflag [0-9]\b|section [A-Z]{1,2}\b|"
                      r"forensic|authorization|DRAFT|candidate[_ -]?0\d\d", re.I)
ABSPATH = re.compile(r"(/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+|/mnt/[a-z]/|[A-Za-z]:\\\\Users|OneDrive)")
IPV4 = re.compile(r"\b(?!127\.0\.0\.1|0\.0\.0\.0)(\d{1,3}\.){3}\d{1,3}\b")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SECRET = re.compile(r"(-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|"
                    r"(?i:(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]))")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    r = subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT)
    return r.stdout.rstrip() if r.returncode == 0 else ""


def section(t):
    print(f"\n{'=' * 100}\n=== {t}\n{'=' * 100}")


def candidates():
    allow, review = [], []
    for line in (HERE / "allowlist.txt").read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("# REVIEW"):
            review.append(s[len("# REVIEW"):].split()[0])
        elif not s.startswith("#"):
            allow.append(s)
    files = {}
    for pat in allow:
        hits = sorted(q for q in ROOT.glob(pat) if q.is_file())
        for q in hits:
            files[q.relative_to(ROOT).as_posix()] = "ALLOW"
        if not hits:
            files[pat] = "ALLOW(missing)"
    for rel in review:
        files[rel] = "REVIEW" if (ROOT / rel).is_file() else "REVIEW(missing)"
    return files


def a_freeze():
    section("A. freeze manifest")
    if not FREEZE.exists():
        print("not found:", FREEZE)
        return
    fz = json.loads(FREEZE.read_text())
    print("sha256:", sha(FREEZE))
    side = FREEZE.with_name(FREEZE.name + ".sha256")
    if side.exists():
        print("sidecar:", side.read_text().split()[0], "| equal" if side.read_text().split()[0] == sha(FREEZE) else "| DIFFERS")
    print("freeze:", fz.get("freeze"), "| status:", fz.get("status"), "| created:", fz.get("created_utc"))
    print("git_head_at_freeze:", fz.get("git_head_at_freeze"))
    head = fz.get("git_head_at_freeze")
    if head:
        print("commits since freeze:", git("rev-list", "--count", f"{head}..HEAD") or "unknown")
        changed = git("diff", "--name-only", head, "HEAD", "--", "workloads", "serving", "kubernetes", "analysis",
                      "training", "configs", "tools", "experiments/verify_xm_readiness.py")
        print("files changed since freeze in code directories:\n   " + ("\n   ".join(changed.splitlines()) or "(none)"))
    rt = fz.get("runtime_at_freeze") or {}
    print("\nruntime_at_freeze:")
    for k in ("router_image", "router_command", "router_env", "router_protocol_sha256", "router_source_configmaps_sha256", "node"):
        print(f"  {k}: {json.dumps(rt.get(k), indent=None)}")
    for p in rt.get("pods", []):
        print(f"  pod {p.get('name')}: image={p.get('image')} imageID={p.get('imageID')}")
    print("\npinned_code_sha256:")
    for rel, h in sorted(fz.get("pinned_code_sha256", {}).items()):
        print(f"  {h}  {rel}")
    print("frozen_inputs_sha256:")
    for rel, h in sorted(fz.get("frozen_inputs_sha256", {}).items()):
        print(f"  {h}  {rel}")
    print("routing:", json.dumps(fz.get("routing")))


def b_yaml():
    section("B. Kubernetes and deviation YAML (kinds, names, images, configmaps, ports)")
    files = sorted(list((ROOT / "kubernetes").rglob("*.yaml")) + list((ROOT / "kubernetes").rglob("*.yml"))
                   + list((ROOT / "data/manifests/phase9_v3/deviations").glob("*.yaml")))
    keys = re.compile(r"^\s*-?\s*(kind|name|image|configMap|configMapRef|nodePort|port|targetPort|containerPort|"
                      r"namespace|replicas|command|args|claimName|hostPath|path)\s*:\s*(.*)$")
    for f in files:
        print(f"\n--- {f.relative_to(ROOT)}  ({sha(f)[:16]}...)")
        for line in f.read_text(errors="replace").splitlines():
            m = keys.match(line)
            if m:
                print(f"    {m.group(1)}: {m.group(2)[:140]}")


def c_status(files):
    section("C. release candidates: git tracking and hashes")
    tracked = set(git("ls-files").splitlines())
    dirty = {l[3:] for l in git("status", "--porcelain").splitlines()}
    for rel, kind in sorted(files.items()):
        p = ROOT / rel
        if not p.is_file():
            print(f"  {kind:16} MISSING                         {rel}")
            continue
        state = "untracked" if rel not in tracked else ("MODIFIED" if rel in dirty else "committed")
        print(f"  {kind:16} {state:10} {p.stat().st_size:>9} B {sha(p)[:16]}  {rel}")


def local_modules():
    mods = set()
    for p in ROOT.rglob("*.py"):
        if any(x.startswith(".venv") or x in {".git", "_release_kit", "__pycache__"} for x in p.relative_to(ROOT).parts):
            continue
        mods.add(p.stem)
        mods.add(".".join(p.relative_to(ROOT).with_suffix("").parts))
    return mods


def d_scan(files):
    section("D. content scan of release candidates")
    local = local_modules()
    released = {Path(r).stem for r in files} | {".".join(Path(r).with_suffix("").parts) for r in files}
    for rel in sorted(files):
        p = ROOT / rel
        if not p.is_file() or p.stat().st_size > 3_000_000:
            continue
        text = p.read_text(errors="replace")
        hits = []
        for i, line in enumerate(text.splitlines(), 1):
            for name, rx in (("internal-ref", INTERNAL), ("abs-path", ABSPATH), ("ip", IPV4), ("email", EMAIL), ("SECRET", SECRET)):
                m = rx.search(line)
                if m:
                    hits.append(f"{name}@{i}: {m.group(0)[:60]}")
        unresolved = []
        if p.suffix == ".py":
            try:
                tree = ast.parse(text)
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [a.name for a in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                        names = [node.module]
                    for n in names:
                        top = n.split(".")[0]
                        if (n in local or top in local) and n not in released and top not in released:
                            unresolved.append(n)
            except SyntaxError as e:
                unresolved.append(f"SYNTAX ERROR {e}")
        if hits or unresolved:
            print(f"\n--- {rel}")
            for h in hits[:40]:
                print("    " + h)
            if len(hits) > 40:
                print(f"    ... {len(hits) - 40} more")
            for u in sorted(set(unresolved)):
                print(f"    local import NOT in release: {u}")


def e_text(files):
    section("E. full text of REVIEW files and repository front matter")
    show = [r for r, k in sorted(files.items()) if k == "REVIEW"]
    for extra in ("README.md", "ARTIFACT-EVALUATION.md", "reproduce.sh", "Makefile", "pyproject.toml", "CITATION.cff",
                  "data/download_dataset.sh"):
        if (ROOT / extra).is_file() and extra not in show:
            show.append(extra)
    for rel in show:
        lines = (ROOT / rel).read_text(errors="replace").splitlines()
        print(f"\n----- BEGIN {rel} ({len(lines)} lines) -----")
        for i, l in enumerate(lines[:250], 1):
            print(f"{i:4}| {l}")
        if len(lines) > 250:
            print(f"     ... truncated ({len(lines) - 250} more lines)")
        print(f"----- END {rel} -----")
    print("\nLICENSE first lines:")
    if (ROOT / "LICENSE").is_file():
        print("   " + "\n   ".join((ROOT / "LICENSE").read_text(errors="replace").splitlines()[:4]))


def f_evidence():
    section("F. evidence for the archive")

    def du(p):
        p = ROOT / p
        if not p.exists():
            return "absent"
        files = [x for x in p.rglob("*") if x.is_file()] if p.is_dir() else [p]
        return f"{len(files)} files, {sum(x.stat().st_size for x in files) / 1e9:.2f} GB"
    for rel in ("data/raw", "data/splits", "data/processed", "data/manifests", "models", "results/derived",
                "results/derived/phase9_v3", "results/raw/phase9_v3", "results/raw", "environment"):
        print(f"  {rel:32} {du(rel)}")
    rec = ROOT / "results/derived/phase9_v3/README_ANALYSIS_OF_RECORD.txt"
    if rec.exists():
        print("\nREADME_ANALYSIS_OF_RECORD.txt:\n   " + "\n   ".join(rec.read_text(errors="replace").splitlines()[:40]))
    d = ROOT / "results/derived/phase9_v3"
    if d.exists():
        print("\nresults/derived/phase9_v3 contents:")
        for x in sorted(d.iterdir()):
            n = len([y for y in x.rglob("*") if y.is_file()]) if x.is_dir() else 1
            print(f"   {x.name}{'/' if x.is_dir() else ''}  ({n} files)")
    print("\ngit log (last 15):\n   " + "\n   ".join(git("log", "--oneline", "-15").splitlines()))


def main():
    files = candidates()
    a_freeze()
    b_yaml()
    c_status(files)
    d_scan(files)
    e_text(files)
    f_evidence()


if __name__ == "__main__":
    main()
