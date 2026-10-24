#!/usr/bin/env python3
"""
RolloutSecBench - build the evidence archive from the private repository (standard library only).

Run from the private repository root:

    python3 release/build_evidence_archive.py --list                  # inspect only; writes nothing
    python3 release/build_evidence_archive.py --out ../rolloutsecbench-evidence-v1.tar.gz

Selection: release/archive_allowlist.txt. data/raw/ is never archived; data/splits/ only with
--include-splits. Every file is copied byte for byte: nothing is rewritten, so every hash recorded in
the manifests, run checksums and the paper still verifies. Tar entries carry no owner names
(uid/gid 0, empty uname/gname) and are written in sorted order.

The archive contains ARCHIVE_MANIFEST.sha256 (SHA-256 of every file, plus the source commit), and
the script writes <out>.sha256 and <out>.manifest.sha256 next to the archive.

Both modes report, without changing anything:
  * files that are not committed in git or are modified (results under results/ are expected to be
    ignored or committed; the report lists them so that nothing unreviewed is published),
  * files that contain absolute local paths or the local user name (left as they are, because
    rewriting them would break recorded hashes; listed so that the authors can disclose them),
  * files whose header carries UNSW-NB15 feature columns (they redistribute dataset rows),
  * secret-like strings (refuses to write the archive if any are found).
"""
import argparse
import fnmatch
import hashlib
import os
import re
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path.cwd()
LIST = ROOT / "release/archive_allowlist.txt"
NEVER = ("data/raw/", ".venv", "__pycache__", ".kitbak", ".pyc", ".DS_Store")
LOCAL = re.compile(rb"/home/[A-Za-z0-9_.-]+|/mnt/c/|[A-Z]:\\\\Users|pentestnet|isah_")
SECRET = re.compile(rb"(?i)(BEGIN [A-Z ]*PRIVATE KEY|aws_secret|api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9]{16}|ghp_[A-Za-z0-9]{30}|password\s*[:=]\s*['\"][^'\"]{4})")
UNSW = (b"sbytes", b"dbytes", b"sttl", b"dttl", b"sload", b"dload", b"spkts", b"dpkts", b"smean",
        b"dmean", b"ct_srv_src", b"synack", b"ackdat", b"sinpkt")


def glob_to_re(pat):
    out, i = "", 0
    while i < len(pat):
        if pat[i:i + 3] == "**/":
            out += "(?:.*/)?"; i += 3
        elif pat[i:i + 2] == "**":
            out += ".*"; i += 2
        elif pat[i] == "*":
            out += "[^/]*"; i += 1
        elif pat[i] == "?":
            out += "[^/]"; i += 1
        else:
            out += re.escape(pat[i]); i += 1
    return re.compile(out + r"\Z")


def git_files():
    tracked = set(subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout.decode().split("\0"))
    dirty = set()
    for line in subprocess.run(["git", "status", "--porcelain", "-z", "--untracked-files=all"],
                               capture_output=True, check=True).stdout.decode().split("\0"):
        if len(line) > 3:
            dirty.add(line[3:])
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return tracked, dirty, head


def select(include_splits):
    all_files = []
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x != ".git" and not x.startswith(".venv") and x != "__pycache__"]
        all_files += [Path(d, f).relative_to(ROOT).as_posix() for f in files]
    plan, missing = {}, []
    for raw in LIST.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        optional = line.startswith("?")
        line = line.lstrip("?").strip()
        dest_dir = None
        if "<=" in line:
            dest_dir, line = [x.strip() for x in line.split("<=", 1)]
        rx = glob_to_re(line)
        hits = sorted(f for f in all_files if rx.match(f))
        hits = [f for f in hits if not any(n in f for n in NEVER)]
        if not include_splits:
            hits = [f for f in hits if not f.startswith("data/splits/")]
        if not hits and not (optional or (line.startswith("data/splits/") and not include_splits)):
            missing.append(line)
        for f in hits:
            dest = f"{dest_dir}/{Path(f).name}" if dest_dir else f
            plan[dest] = f
    return plan, missing


def scan(path):
    h = hashlib.sha256()
    local = secret = False
    head = b""
    with open(path, "rb") as fh:
        prev = b""
        while chunk := fh.read(1 << 22):
            h.update(chunk)
            if not head:
                head = chunk[:65536]
            window = prev + chunk
            local = local or bool(LOCAL.search(window))
            secret = secret or bool(SECRET.search(window))
            prev = chunk[-256:]
    first = head.split(b"\n", 1)[0].lower()
    dataset_cols = sum(1 for c in UNSW if re.search(rb"(^|[,\"'\s{])" + c + rb"([,\"'\s:]|$)", first))
    if dataset_cols < 5:   # JSON lines: look inside the first record too
        dataset_cols = max(dataset_cols, sum(1 for c in UNSW if b'"' + c + b'"' in head[:16384]))
    return h.hexdigest(), local, secret, dataset_cols >= 5


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="inspect and report; write nothing")
    g.add_argument("--out", help="archive to write (.tar.gz)")
    ap.add_argument("--include-splits", action="store_true", help="also archive data/splits/ (UNSW-NB15 rows)")
    a = ap.parse_args()

    plan, missing = select(a.include_splits)
    tracked, dirty, head = git_files()
    rows, groups = [], defaultdict(lambda: [0, 0])
    local_files, secret_files, data_files, uncommitted = [], [], [], []
    for dest, src in sorted(plan.items()):
        size = os.path.getsize(src)
        digest, local, secret, data = scan(src)
        rows.append((dest, src, size, digest))
        key = "/".join(dest.split("/")[:3]) if dest.startswith("results/raw/") else "/".join(dest.split("/")[:2])
        groups[key][0] += 1; groups[key][1] += size
        if local: local_files.append(dest)
        if secret: secret_files.append(dest)
        if data: data_files.append(dest)
        if src not in tracked or src in dirty:
            uncommitted.append(src)

    total = sum(r[2] for r in rows)
    print(f"source commit {head}; {len(rows)} files, {human(total)}")
    for k, (n, s) in sorted(groups.items()):
        print(f"  {k:60} {n:7} files {human(s):>10}")
    for m in missing:
        print(f"MISSING (required pattern matched nothing): {m}")
    print(f"\nnot tracked by git or modified (expected for the large evidence under results/; review the rest): {len(uncommitted)}")
    uncommitted.sort(key=lambda f: f.startswith("results/"))
    for f in uncommitted[:25]: print(f"  {f}")
    print(f"\ncontain an absolute local path or the local user name (kept byte for byte): {len(local_files)}")
    for f in local_files[:25]: print(f"  {f}")
    if len(local_files) > 25: print(f"  ... and {len(local_files) - 25} more")
    print(f"\ncarry UNSW-NB15 feature columns (redistribute dataset rows): {len(data_files)}")
    for f in data_files[:25]: print(f"  {f}")
    if len(data_files) > 25: print(f"  ... and {len(data_files) - 25} more")
    print(f"\nsecret-like strings: {len(secret_files)}")
    for f in secret_files: print(f"  {f}")

    manifest = f"# RolloutSecBench evidence archive; source commit {head}\n" + "".join(
        f"{digest}  {dest}\n" for dest, src, size, digest in rows)
    if a.list:
        print("\n--list: nothing written")
        return 1 if (missing or secret_files) else 0
    if missing or secret_files:
        sys.exit("REFUSING to write the archive: missing required files or secret-like strings (see above)")

    out = Path(a.out).resolve()
    if out.exists():
        sys.exit(f"REFUSING: {out} exists")
    mf = out.with_name(out.name + ".manifest.tmp")
    mf.write_text(manifest)

    def reset(ti):
        ti.uid = ti.gid = 0; ti.uname = ti.gname = ""
        ti.mode = 0o644 if ti.isfile() else 0o755
        return ti

    with tarfile.open(out, "w:gz", compresslevel=6, format=tarfile.PAX_FORMAT) as tar:
        tar.add(mf, arcname="ARCHIVE_MANIFEST.sha256", filter=reset)
        for dest, src, size, digest in rows:
            tar.add(src, arcname=dest, recursive=False, filter=reset)
    mf.unlink()
    Path(str(out) + ".manifest.sha256").write_text(manifest)
    arch = hashlib.sha256(out.read_bytes()).hexdigest() if out.stat().st_size < (1 << 31) else None
    if arch is None:
        h = hashlib.sha256()
        with open(out, "rb") as fh:
            while c := fh.read(1 << 24): h.update(c)
        arch = h.hexdigest()
    Path(str(out) + ".sha256").write_text(f"{arch}  {out.name}\n")
    print(f"\nwrote {out} ({human(out.stat().st_size)}), sha256 {arch}")
    print(f"wrote {out.name}.sha256 and {out.name}.manifest.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
