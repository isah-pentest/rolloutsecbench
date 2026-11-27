#!/usr/bin/env python3
"""
RolloutSecBench - CIC-IDS2017 dataset table (companion to the UNSW-NB15 dataset table) and dataset macros.

Every number comes from committed files; nothing is typed in:
  revision/results/cic/prepare_summary.json   written by revision/analysis/cic_prepare.py (pinned below; the same
                                              hash is pinned by e2_cic.py)
  revision/results/e2_cic/e2_cic_pairs.csv    distinct vectors of the CIC malicious strict test (n_vec), via the
                                              committed manifest revision/manifests/cic/phase5b_e2_cic.sha256
  revision/results/e2/e2_pairs.csv            distinct vectors of the UNSW-NB15 malicious strict test (n_vec), pinned
  revision/results/s1_duplicates/s1_results.json  UNSW-NB15 distinct vectors (tracked, unchanged from HEAD)
Refuses (exit 2, nothing written) on any hash mismatch or internal inconsistency:
  * the clean rows equal final train + validation + test, per fine label and in total;
  * test - strict test = test_rows_removed_overlap; binary counts equal the sums of fine-label counts;
  * the binary rule (0 = Benign or any label containing "Attempted") reproduces every recorded binary count;
  * n_vec is identical in all 24 CIC pairs and in all 48 UNSW-NB15 pairs, and the UNSW value equals S1.
Outputs (--out-dir, default revision/tables/dataset): table_dataset_cic.tex (complete float),
dataset_cic_values.tex (macros), table_dataset_cic_provenance.json.
Usage (repository root): python revision/tables/make_table_dataset_cic.py
"""
import argparse, csv, hashlib, json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

PREP = ("revision/results/cic/prepare_summary.json", "91edb9c9adf914db53149680a1a0fdf4f05d2b4a26004233152473952530fefa")
UNSW_PAIRS = ("revision/results/e2/e2_pairs.csv", "8dafb37178670d6b17db1ca3eda100431540a5b3c6615b1e793d47518fd0e5e6")
CIC_PAIRS = "revision/results/e2_cic/e2_cic_pairs.csv"
CIC_MANIFEST = "revision/manifests/cic/phase5b_e2_cic.sha256"
S1 = "revision/results/s1_duplicates/s1_results.json"


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


def git_clean(repo, rel):
    t = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", rel], capture_output=True).returncode
    d = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", rel], capture_output=True).returncode
    return t == 0 and d == 0


def manifest_hash(repo, manifest, rel):
    for line in (repo / manifest).read_text().splitlines():
        p = line.split(None, 1)
        if len(p) == 2:
            path = p[1].strip().lstrip("*")
            if path == rel or rel.endswith("/" + path) or path.endswith(rel):
                return p[0]
    raise Refuse(f"{rel} not listed in {manifest}")


def n(x):
    return f"{int(x):,}".replace(",", "{,}")


def pct(a, b):
    return f"{100 * a / b:.1f}\\%"


TEX_MAP = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "–": "--", "—": "---",
           "�": "?"}


def tex(s):
    out = []
    for ch in str(s):
        if ch in TEX_MAP:
            out.append(TEX_MAP[ch])
        elif ord(ch) < 128:
            out.append(ch)
        else:
            raise Refuse(f"non-ASCII character {ch!r} in label {s!r}; add it to TEX_MAP explicitly")
    return "".join(out)


def is_benign(label):
    low = label.strip().lower()
    return low == "benign" or "attempted" in low


def nvec(repo, rel):
    vals = set()
    with open(repo / rel, newline="") as fh:
        for r in csv.DictReader(fh):
            vals.add(int(r["n_vec"]))
    need(len(vals) == 1, f"{rel}: n_vec differs between pairs {sorted(vals)}")
    return vals.pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, default=Path("revision/tables/dataset"))
    a = ap.parse_args()
    repo = a.repo.resolve()
    out = a.out_dir if a.out_dir.is_absolute() else repo / a.out_dir
    inputs, checks = {}, []
    try:
        need(sha(repo / PREP[0]) == PREP[1], f"{PREP[0]} differs from pinned {PREP[1]}")
        need(sha(repo / UNSW_PAIRS[0]) == UNSW_PAIRS[1], f"{UNSW_PAIRS[0]} differs from pinned {UNSW_PAIRS[1]}")
        need(git_clean(repo, CIC_MANIFEST), f"{CIC_MANIFEST} not tracked or differs from HEAD")
        want = manifest_hash(repo, CIC_MANIFEST, CIC_PAIRS)
        need(sha(repo / CIC_PAIRS) == want, f"{CIC_PAIRS} differs from {CIC_MANIFEST}")
        need(git_clean(repo, S1), f"{S1} not tracked or differs from HEAD")
        inputs = {PREP[0]: PREP[1], UNSW_PAIRS[0]: UNSW_PAIRS[1], CIC_PAIRS: want, CIC_MANIFEST: sha(repo / CIC_MANIFEST),
                  S1: sha(repo / S1)}
        checks.append("inputs equal pinned hashes / committed manifest / HEAD")
        S = json.loads((repo / PREP[0]).read_text())

        clean = {k: int(v) for k, v in S["fine_label_counts_clean"].items()}
        tr = {k: int(v) for k, v in S["final_train_fine_label_counts"].items()}
        va = {k: int(v) for k, v in S["validation_fine_label_counts"].items()}
        st = {k: int(v) for k, v in S["strict_test_fine_label_counts"].items()}
        labels = sorted(clean)
        te = {k: clean[k] - tr.get(k, 0) - va.get(k, 0) for k in labels}
        need(all(v >= 0 for v in te.values()), "negative test count derived from clean - train - validation")
        need(sum(te.values()) == int(S["test_rows"]), "clean - train - validation does not equal test_rows")
        need(all(st.get(k, 0) <= te[k] for k in labels) and set(st) <= set(labels), "strict test exceeds test")
        need(sum(st.values()) == int(S["strict_test_rows"]), "strict test fine counts do not sum to strict_test_rows")
        need(int(S["test_rows"]) - int(S["strict_test_rows"]) == int(S["test_rows_removed_overlap"]),
             "test_rows - strict_test_rows differs from test_rows_removed_overlap")
        need(int(S["raw_rows"]) - int(S["rows_removed_missing_or_infinite"]) == int(S["rows_after_cleaning"]),
             "raw rows - removed rows differs from rows_after_cleaning")
        need(sum(clean.values()) == int(S["rows_after_cleaning"]), "clean fine counts do not sum to rows_after_cleaning")
        checks.append("clean = final train + validation + test per fine label; test - strict = removed overlap")

        def binary(d):
            return {"0": sum(v for k, v in d.items() if is_benign(k)), "1": sum(v for k, v in d.items() if not is_benign(k))}
        for name, d, key in (("clean", clean, "binary_counts_clean"), ("final train", tr, "final_train_binary_counts"),
                             ("validation", va, "validation_binary_counts"), ("strict test", st, "strict_test_binary_counts")):
            b = binary(d)
            need(b["0"] == int(S[key]["0"]) and b["1"] == int(S[key]["1"]), f"binary rule does not reproduce {key}")
        checks.append("binary rule (0 = Benign or *Attempted*) reproduces every recorded binary count")

        cic_nvec, unsw_nvec = nvec(repo, CIC_PAIRS), nvec(repo, UNSW_PAIRS[0])
        s1 = json.loads((repo / S1).read_text())
        need(unsw_nvec == int(s1["unique_vectors"]["n"]), f"UNSW n_vec {unsw_nvec} differs from S1 {s1['unique_vectors']['n']}")
        checks.append("n_vec identical across pairs on each dataset; UNSW value equals S1")
    except Refuse as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return 2
    except KeyError as e:
        print(f"REFUSING: key {e} missing from an input", file=sys.stderr)
        return 2

    bt, bv, bs = binary(tr), binary(va), binary(st)
    bte = binary(te)
    brm = {k: bte[k] - bs[k] for k in ("0", "1")}
    tot = lambda b: b["0"] + b["1"]
    attacks = sorted((k for k in labels if not is_benign(k)), key=lambda k: (-tr.get(k, 0), k))
    attempted = sorted(k for k in labels if "attempted" in k.lower())
    n_att = sum(clean[k] for k in attempted)
    few = list(S.get("labels_with_fewer_than_3_rows", []))
    try:
        rows = []
        for k in attacks:
            rem = te[k] - st.get(k, 0)
            rows.append(f"{tex(k)} & {n(tr.get(k, 0))} & {n(va.get(k, 0))} & {n(te[k])} & {n(st.get(k, 0))} & {n(rem)} \\\\")
        drop_id = ", ".join(r"\texttt{" + tex(c) + "}" for c in S["dropped_identifier_time"]) or "none"
        drop_lab = ", ".join(r"\texttt{" + tex(c) + "}" for c in S["dropped_label_like"]) or "none"
        drop_const = ", ".join(r"\texttt{" + tex(c) + "}" for c in S["dropped_constant_on_train"]) or "none"
        few_txt = ", ".join(tex(c) for c in few) if few else "none"
    except Refuse as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return 2

    body = rf"""% generated by make_table_dataset_cic.py from prepare_summary.json - do not edit by hand
\begin{{table}}[!tbp]
\caption{{CIC-IDS2017 (corrected CNS2022 release): characteristics and splits.}}
\label{{tab:dataset-cic}}
\begin{{adjustwidth}}{{-\extralength}}{{0cm}}
\centering\small
\begin{{adjustbox}}{{max width=\linewidth}}
\begin{{tabular}}{{lrrrrr}}
\toprule
 & \textbf{{Final train}} & \textbf{{Validation}} & \textbf{{Test}} & \textbf{{Strict test}} & \textbf{{Removed by}} \\
 & & & & \textbf{{(evaluation)}} & \textbf{{strict filter}} \\
\midrule
Rows & {n(tot(bt))} & {n(tot(bv))} & {n(tot(bte))} & {n(tot(bs))} & {n(tot(brm))} \\
Benign (incl.\ Attempted) & {n(bt['0'])} & {n(bv['0'])} & {n(bte['0'])} & {n(bs['0'])} & {n(brm['0'])} \\
Malicious & {n(bt['1'])} & {n(bv['1'])} & {n(bte['1'])} & \textbf{{{n(bs['1'])}}} & {n(brm['1'])} \\
Malicious share & {pct(bt['1'], tot(bt))} & {pct(bv['1'], tot(bv))} & {pct(bte['1'], tot(bte))} & {pct(bs['1'], tot(bs))} & --- \\
Distinct malicious vectors & --- & --- & --- & {n(cic_nvec)} & --- \\
\midrule
\multicolumn{{6}}{{l}}{{\textit{{Attack-label counts (label-1 rows)}}}} \\
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}

\vspace{{4pt}}
\begin{{tabular}}{{p{{0.25\linewidth}}p{{0.70\linewidth}}}}
\toprule
\multicolumn{{2}}{{l}}{{\textbf{{Preparation, split and leakage audit}}}} \\
\midrule
Release & Corrected CNS2022 release (\texttt{{CICIDS2017\_improved.zip}}, SHA-256 \texttt{{{PREP_ZIP_SHORT(S)}}}\ldots), five day files read Monday to Friday; {n(S['raw_rows'])} rows. \\
Label rule & Binary label 0 for Benign and for every label containing ``Attempted'' ({n(n_att)} rows), 1 otherwise; the attack label is the label text. \\
Dropped columns & Identifiers and time: {drop_id}. Label columns: {drop_lab}. Constant on final train: {drop_const}. \\
Rows removed & {n(S['rows_removed_missing_or_infinite'])} rows with a missing or infinite feature value, before the split. \\
Features & {int(S['n_features'])} features (\texttt{{Protocol}} one-hot, all others numeric; {int(S['transformed_feature_count'])} model inputs); preprocessor fit on final train only. \\
Split & No official split exists. Random split stratified by label (seed {int(S['seed'])}): 70\% final train, 15\% validation, 15\% test; labels with fewer than 3 rows go wholly to final train ({few_txt}). \\
Strict test rule & Test rows whose complete feature vector occurs in final train are removed ({n(S['test_rows_removed_overlap'])} rows, {pct(S['test_rows_removed_overlap'], S['test_rows'])} of the test split), as for UNSW-NB15. \\
Duplicates & Exact within-split duplicates are retained, not deduplicated. \\
\bottomrule
\end{{tabular}}
\end{{adjustwidth}}
\end{{table}}
"""
    macros = {"CICRawRows": n(S["raw_rows"]), "CICRowsRemovedBad": n(S["rows_removed_missing_or_infinite"]),
              "CICTestRows": n(S["test_rows"]), "CICStrictRows": n(S["strict_test_rows"]),
              "CICRemoved": n(S["test_rows_removed_overlap"]),
              "CICRemovedPct": f"{100 * int(S['test_rows_removed_overlap']) / int(S['test_rows']):.1f}",
              "CICNm": n(bs["1"]), "CICNmVec": n(cic_nvec), "CICNFeat": str(int(S["n_features"])),
              "CICAttempted": n(n_att), "CICNAttackLabels": str(len(attacks)), "UNSWNmVecCheck": n(unsw_nvec)}
    out.mkdir(parents=True, exist_ok=True)
    t = out / "table_dataset_cic.tex"
    t.write_text(body)
    v = out / "dataset_cic_values.tex"
    v.write_text("% generated by make_table_dataset_cic.py - do not edit by hand\n"
                 + "".join(f"\\newcommand{{\\{k}}}{{{x}}}\n" for k, x in sorted(macros.items())))
    prov = {"artifact": "RolloutSecBench", "table": "CIC-IDS2017 dataset characteristics and splits",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": {"path": "revision/tables/make_table_dataset_cic.py", "sha256": sha(Path(__file__))},
            "python": platform.python_version(), "inputs": inputs, "checks_passed": checks,
            "outputs": {p.name: sha(p) for p in (t, v)}}
    (out / "table_dataset_cic_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"OK: {len(checks)} checks passed")
    for p in (t, v, out / "table_dataset_cic_provenance.json"):
        print(f"wrote {p.relative_to(repo) if p.is_relative_to(repo) else p}  {sha(p)}")
    for k, x in sorted(macros.items()):
        print(f"  {k} = {x}")
    return 0


def PREP_ZIP_SHORT(S):
    return str(S["zip_sha256"])[:16]


if __name__ == "__main__":
    sys.exit(main())
