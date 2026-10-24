#!/usr/bin/env python3
"""
RolloutSecBench - compare regenerated paper inputs with a reference set (standard library only).

For every *.tex file in either directory:
  * BYTE-IDENTICAL  the two files have the same SHA-256;
  * files that define LaTeX macros (\\newcommand / \\providecommand / \\def) are otherwise compared
    macro by macro, so comments, ordering and generation timestamps are ignored. A macro present
    in only one of the two files is reported by name and counts as a failure: the reference set
    must define every value the paper prints, and nothing else;
  * other files (tables) are otherwise compared line by line after stripping LaTeX comments; the
    differing lines are printed, and a table whose rows are the same but in another order is
    reported as ROWS-REORDERED (a failure, with the moved rows listed as unchanged in content);
  * a file present in only one directory is a failure.
Exit status 0 only if every file is identical in content.

Usage:  python3 tools/compare_generated.py REFERENCE_DIR GENERATED_DIR
"""
import difflib
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

MACRO = re.compile(r"\\(?:re)?(?:newcommand|providecommand)\{?\\([A-Za-z]+)\}?\{((?:[^{}]|\{[^{}]*\})*)\}")
DEF = re.compile(r"\\def\\([A-Za-z]+)\{((?:[^{}]|\{[^{}]*\})*)\}")


def strip_comments(text):
    return [re.sub(r"(?<!\\)%.*$", "", l).rstrip() for l in text.splitlines()]


def macros(text):
    body = "\n".join(strip_comments(text))
    out = {m.group(1): m.group(2).strip() for m in MACRO.finditer(body)}
    out.update({m.group(1): m.group(2).strip() for m in DEF.finditer(body)})
    return out


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(ref, gen):
    ref, gen = Path(ref), Path(gen)
    rfiles = {p.name: p for p in ref.rglob("*.tex")}
    gfiles = {p.name: p for p in gen.rglob("*.tex")}
    if not set(rfiles) & set(gfiles):
        print("no .tex files in common")
        return 2
    bad = 0
    for name in sorted(set(rfiles) | set(gfiles)):
        if name not in gfiles:
            print(f"[MISSING] {name} exists only in the reference set")
            bad += 1
            continue
        if name not in rfiles:
            print(f"[NOT-IN-REFERENCE] {name} was generated but the reference set has no copy")
            bad += 1
            continue
        rp, gp = rfiles[name], gfiles[name]
        if sha(rp) == sha(gp):
            print(f"[BYTE-IDENTICAL] {name}  {sha(gp)[:16]}")
            continue
        rt, gt = rp.read_text(), gp.read_text()
        rm, gm = macros(rt), macros(gt)
        if rm or gm:
            missing = sorted(set(rm) - set(gm))
            extra = sorted(set(gm) - set(rm))
            diff = sorted(k for k in set(rm) & set(gm) if rm[k] != gm[k])
            for k in missing:
                print(f"  {name}: \\{k} missing in regenerated set")
            for k in extra:
                print(f"  {name}: \\{k} = {gm[k]!r} generated but absent from the reference")
            for k in diff:
                print(f"  {name}: \\{k}  reference={rm[k]!r}  regenerated={gm[k]!r}")
            ok = not (missing or diff or extra)
            print(f"[{'IDENTICAL' if ok else 'DIFFERS'}] {name}: {len(rm)} reference macros, "
                  f"{len(diff)} differ, {len(missing)} missing, {len(extra)} only in the regenerated set")
            bad += not ok
        else:
            rl = [l for l in strip_comments(rt) if l]
            gl = [l for l in strip_comments(gt) if l]
            if rl == gl:
                print(f"[IDENTICAL] {name} (table, compared line by line)")
                continue
            reordered = Counter(rl) == Counter(gl)
            print(f"[{'ROWS-REORDERED' if reordered else 'DIFFERS'}] {name} (table, compared line by line)")
            for d in difflib.unified_diff(rl, gl, "reference", "regenerated", n=0, lineterm=""):
                print("    " + d)
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
