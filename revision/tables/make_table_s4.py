#!/usr/bin/env python3
"""
RolloutSecBench - Supplementary Table S4: SHA-256 identities of the extension analyses (pre-registration, amendment,
freezes, timestamp proofs, recorded deviations, and every analysis manifest), read from the repository and its git
history. Nothing is typed in: every hash is computed from the file, every commit is read with git, and every
timestamp-proof block height is read with the OpenTimestamps client.

Refuses (exit 2, nothing written) unless every listed file is tracked and unchanged from HEAD, every .ots proof
covers the file it accompanies (the proof's file hash equals the file's SHA-256), and every deviation heading found in
revision/prereg/DEVIATIONS.md has a commit that introduced it.

Outputs (--out-dir, default revision/tables/s4): tableS4_extension.tex (for the Supplementary Materials),
tableS4_provenance.json.
Usage (repository root): python revision/tables/make_table_s4.py [--ots ~/.venv-ots/bin/ots]
"""
import argparse, hashlib, json, platform, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

KEY_FILES = [
    ("Extension pre-registration v1.0", "revision/prereg/EXTENSION_PREREGISTRATION_v1.0.md"),
    ("Inputs of the pre-registration", "revision/prereg/INPUTS.sha256"),
    ("Simulator validation (40 valid runs)", "revision/results/e1_validation/validation_report.json"),
    ("Amendment 1 (CIC-IDS2017)", "revision/prereg/AMENDMENT_1_CICIDS2017.md"),
    ("Routing-policy identity freeze", "revision/prereg/E3_FREEZE.json"),
    ("Routing-policy simulated identities", "revision/prereg/E3_SIM_IDENTITIES.txt"),
    ("Live freeze of the routing-policy experiment", "data/manifests/ext1/ext1_live_freeze.json"),
    ("Recorded deviations", "revision/prereg/DEVIATIONS.md"),
]
MANIFEST_GLOB = "revision/manifests"


class Refuse(Exception):
    pass


def need(c, m):
    if not c:
        raise Refuse(m)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(repo, *a):
    r = subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)
    if r.returncode:
        raise Refuse(f"git {' '.join(a)}: {r.stderr.strip()[:200]}")
    return r.stdout


def clean(repo, rel):
    need(subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", rel], capture_output=True).returncode == 0,
         f"{rel} is not tracked")
    need(subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", rel], capture_output=True).returncode == 0,
         f"{rel} differs from HEAD")


def first_commit(repo, rel):
    out = git(repo, "log", "--reverse", "--format=%h %cI", "--", rel).split("\n")[0].strip()
    need(out, f"no commit found for {rel}")
    h, t = out.split()
    return h, t


def ots_info(ots, path):
    r = subprocess.run([ots, "info", str(path)], capture_output=True, text=True)
    need(r.returncode == 0, f"ots info failed for {path}: {r.stderr.strip()[:200]}")
    m = re.search(r"File sha256 hash:\s*([0-9a-f]{64})", r.stdout)
    need(m, f"no file hash in ots info for {path}")
    blocks = sorted({int(b) for b in re.findall(r"BitcoinBlockHeaderAttestation\((\d+)\)", r.stdout)})
    return m.group(1), blocks


def tt(s):
    return r"\texttt{" + s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("#", r"\#").replace("%", r"\%") + "}"


def esc(s):
    m = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
         "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    out = "".join(m.get(ch, ch) for ch in s)
    need(all(ord(ch) < 128 for ch in out), f"non-ASCII text in DEVIATIONS.md heading: {s!r}")
    return out


def day(iso):
    return iso[:10] + " " + iso[11:16] + " (UTC" + iso[19:] + ")"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, default=Path("revision/tables/s4"))
    ap.add_argument("--ots", default=str(Path.home() / ".venv-ots/bin/ots"))
    a = ap.parse_args()
    repo = a.repo.resolve()
    out = a.out_dir if a.out_dir.is_absolute() else repo / a.out_dir
    rec = {"key_files": [], "deviations": [], "manifests": []}
    try:
        for label, rel in KEY_FILES:
            clean(repo, rel)
            h = sha(repo / rel)
            c, t = first_commit(repo, rel)
            item = {"label": label, "path": rel, "sha256": h, "commit": c, "committed": t}
            if rel != "revision/prereg/DEVIATIONS.md":
                last = git(repo, "log", "-1", "--format=%h", "--", rel).strip()
                need(last == c, f"{rel} was changed after its first commit ({c} -> {last}); it is not a frozen file")
            o = repo / (rel + ".ots")
            if o.exists():
                clean(repo, rel + ".ots")
                oh, blocks = ots_info(a.ots, o)
                need(oh == h, f"{rel}.ots covers {oh}, not the file ({h})")
                oc, ot = first_commit(repo, rel + ".ots")
                item.update({"ots_blocks": blocks, "ots_commit": oc, "ots_committed": ot})
            rec["key_files"].append(item)

        dev_text = (repo / "revision/prereg/DEVIATIONS.md").read_text()
        heads = re.findall(r"^## (DEV-R1-\d{3}) \(([^)]*)\)(?::\s*(.*))?$", dev_text, flags=re.M)
        need(heads, "no DEV-R1 headings found")
        seen = set()
        for did, when, title in heads:
            need(did not in seen, f"{did} appears twice in DEVIATIONS.md")
            seen.add(did)
            out_c = git(repo, "log", "--reverse", "--format=%h %cI", f"-S{did}", "--",
                        "revision/prereg/DEVIATIONS.md").split("\n")[0].strip()
            need(out_c, f"no commit introduced {did}")
            c, t = out_c.split()
            rec["deviations"].append({"id": did, "recorded": when.strip(), "title": (title or "").strip(),
                                      "commit": c, "committed": t})

        mans = [m for m in git(repo, "ls-files", MANIFEST_GLOB).split() if m.endswith(".sha256")]
        need(mans, "no manifests found")
        for m in sorted(mans):
            clean(repo, m)
            c, t = first_commit(repo, m)
            lines = [l for l in (repo / m).read_text().splitlines() if l.strip()]
            rec["manifests"].append({"path": m, "sha256": sha(repo / m), "entries": len(lines), "commit": c,
                                     "committed": t})
    except Refuse as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return 2

    rows_k = []
    for k in rec["key_files"]:
        ots = ""
        if "ots_blocks" in k:
            ots = "Bitcoin block " + ", ".join(str(b) for b in k["ots_blocks"]) if k["ots_blocks"] else "pending"
        rows_k.append(f"{k['label']}\\newline {{\\tiny {tt(k['path'])}}} & {tt(k['sha256'])} & {tt(k['commit'])}, {day(k['committed'])} & {ots} \\\\")
    try:
        rows_d = [f"{d['id']} & {esc(d['recorded'])}{(': ' + esc(d['title'])) if d['title'] else ''} & {tt(d['commit'])}, {day(d['committed'])} \\\\"
                  for d in rec["deviations"]]
    except Refuse as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return 2
    rows_m = [f"{tt(m['path'].replace('revision/manifests/', ''))} & {m['entries']} & {tt(m['sha256'])} & {tt(m['commit'])}, {day(m['committed'])} \\\\"
              for m in rec["manifests"]]
    tex = rf"""% generated by make_table_s4.py from the repository and its git history - do not edit by hand
\begin{{table}}[H]
\renewcommand{{\thetable}}{{S4}}\renewcommand{{\theHtable}}{{S4}}%
\caption{{SHA-256 identities of the extension analyses: (a)~pre-registration, amendment, freezes and timestamp proofs; (b)~recorded deviations; (c)~analysis manifests.}}
\label{{tab:S4-extension}}
\centering\scriptsize
\begin{{adjustbox}}{{max width=\linewidth}}
\begin{{tabular}}{{p{{0.27\linewidth}}lll}}
\toprule
\multicolumn{{4}}{{l}}{{\textbf{{(a) Pre-registration, amendment, freezes and timestamp proofs}}}} \\
\textbf{{Item and file}} & \textbf{{SHA-256}} & \textbf{{First committed}} & \textbf{{OpenTimestamps}} \\
\midrule
{chr(10).join(rows_k)}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}

\medskip
\begin{{adjustbox}}{{max width=\linewidth}}
\begin{{tabular}}{{lp{{0.55\linewidth}}l}}
\toprule
\multicolumn{{3}}{{l}}{{\textbf{{(b) Recorded deviations}} (file \texttt{{revision/prereg/DEVIATIONS.md}})}} \\
\textbf{{Deviation}} & \textbf{{Recorded}} & \textbf{{Introduced in commit}} \\
\midrule
{chr(10).join(rows_d)}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}

\medskip
\begin{{adjustbox}}{{max width=\linewidth}}
\begin{{tabular}}{{lrll}}
\toprule
\multicolumn{{4}}{{l}}{{\textbf{{(c) Analysis manifests}} (directory \texttt{{revision/manifests/}}; each lists the SHA-256 of its inputs, scripts and outputs)}} \\
\textbf{{Manifest}} & \textbf{{Entries}} & \textbf{{SHA-256}} & \textbf{{First committed}} \\
\midrule
{chr(10).join(rows_m)}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table}}
"""
    out.mkdir(parents=True, exist_ok=True)
    t = out / "tableS4_extension.tex"
    t.write_text(tex)
    prov = {"artifact": "RolloutSecBench", "table": "Supplementary Table S4",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": {"path": "revision/tables/make_table_s4.py", "sha256": sha(Path(__file__))},
            "python": platform.python_version(), "git_head": git(repo, "rev-parse", "HEAD").strip(),
            "record": rec, "outputs": {t.name: sha(t)}}
    (out / "tableS4_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"OK: {len(rec['key_files'])} key files, {len(rec['deviations'])} deviations, {len(rec['manifests'])} manifests")
    for p in (t, out / "tableS4_provenance.json"):
        print(f"wrote {p.relative_to(repo) if p.is_relative_to(repo) else p}  {sha(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
