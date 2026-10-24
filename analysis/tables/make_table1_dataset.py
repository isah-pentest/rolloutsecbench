#!/usr/bin/env python3
"""
RolloutSecBench - Table I (dataset characteristics and splits) and Table S1 (artifact hashes).

Nothing is hard-coded: every number is recomputed from the frozen split files or read from the frozen Phase-1
records, and the script REFUSES TO WRITE if they disagree.

Reads (read-only)
  data/manifests/dataset_manifest.json      source, license, citation, partition, split policy, temporal assessment
  data/manifests/split_manifest.json        split files, row/label counts, seed, strict-test policy, SHA-256 of splits
  data/manifests/preprocessing_manifest.json feature space, encoder policy
  data/processed/feature_names.json         the 42 model features (source_feature_order)
  results/derived/dataset_audit.json        cross-partition overlap and label-ambiguity audit
  data/splits/{final_train,validation,strict_test_nonoverlap,official_test_full}.csv

Checks before writing (any failure -> exit 2, nothing written)
  * every split file's SHA-256 equals split_manifest; raw partitions equal the split/dataset manifests
  * recomputed rows and binary-label counts equal split_manifest
  * official_test_full minus strict_test (per family, per label) sums to the audited excluded-row count
  * strict test shares 0 exact 42-feature vectors with final train and with validation

Recomputed here (from the 42 features in source_feature_order; files read with utf-8-sig to drop a BOM)
  rows, benign/malicious, attack-family counts, rows removed by the strict filter per family,
  within-split exact duplicate feature vectors (occurrences beyond the first), cross-split shared vectors.

Outputs (--out-dir): table1_dataset.tex, tableS1_hashes.tex, table1_data.csv, table1_provenance.json
Usage (repo root):
  .venv-blueprint-v2/bin/python analysis/tables/make_table1_dataset.py --out-dir results/derived/phase9_v3/tables/table1
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
DM = ROOT / "data/manifests/dataset_manifest.json"
SM = ROOT / "data/manifests/split_manifest.json"
PM = ROOT / "data/manifests/preprocessing_manifest.json"
FN = ROOT / "data/processed/feature_names.json"
AUD = ROOT / "results/derived/dataset_audit.json"
SPLITS = [("final_train", "Final train"), ("validation", "Validation"),
          ("strict_test_nonoverlap", "Strict test (primary)"), ("official_test_full", "Official test (secondary)")]
BENIGN = "Normal"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stop(msg):
    print(f"REFUSING TO WRITE TABLE I: {msg}", file=sys.stderr)
    sys.exit(2)


def tex(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("$", r"\$"))


def n(x):
    return f"{x:,}".replace(",", "{,}")


def pct(a, b):
    return f"{100 * a / b:.1f}\\%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/derived/phase9_v3/tables/table1")
    a = ap.parse_args()
    out = (ROOT / a.out_dir).resolve()

    dm, sm, pm = (json.loads(p.read_text()) for p in (DM, SM, PM))
    aud = json.loads(AUD.read_text())
    feats = json.loads(FN.read_text())["source_feature_order"]
    if len(feats) != sm["feature_representation"]["feature_count"]:
        stop("feature_names.json and split_manifest disagree on the feature count")

    # ---------------- hashes vs frozen records
    hashes, checks = {}, []
    for key, _ in SPLITS:
        art = sm["artifacts"][key]
        p = ROOT / art["relative_path"]
        h = sha(p)
        if h != art["sha256"]:
            stop(f"{art['relative_path']} SHA-256 differs from split_manifest")
        hashes[art["relative_path"]] = h
    si = sm["source_integrity"]
    for k in ("train_raw", "test_raw"):
        p = ROOT / si[f"{k}_path"]
        h = sha(p)
        if h != si[f"{k}_sha256"]:
            stop(f"{si[k + '_path']} SHA-256 differs from split_manifest")
        hashes[si[f"{k}_path"]] = h
    if sha(AUD) != si["dataset_audit_sha256"]:
        stop("dataset_audit.json differs from the hash recorded in split_manifest")
    for p in (SM, PM, AUD, FN, ROOT / pm["artifacts"]["preprocessor"]["path"], DM):
        hashes[str(p.relative_to(ROOT))] = sha(p)
    if hashes[pm["artifacts"]["preprocessor"]["path"]] != pm["artifacts"]["preprocessor"]["sha256"]:
        stop("preprocessor hash differs from preprocessing_manifest")
    if hashes["data/processed/feature_names.json"] != pm["artifacts"]["feature_names"]["sha256"]:
        stop("feature_names.json hash differs from preprocessing_manifest")
    dm_refs = {"split_manifest": si["dataset_manifest_sha256"],
               "preprocessing_manifest": pm["provenance"]["dataset_manifest_sha256"]}
    checks.append("split, raw, audit, preprocessor and feature-name hashes match their frozen records")

    # ---------------- recompute from split files
    stats, vectors = {}, {}
    for key, _ in SPLITS:
        path = ROOT / sm["artifacts"][key]["relative_path"]
        lab, fam, seen = Counter(), Counter(), Counter()
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rd = csv.DictReader(fh)
            missing = [f for f in feats if f not in rd.fieldnames]
            if missing:
                stop(f"{path.name} lacks model features {missing[:3]}")
            for r in rd:
                lab[r["label"].strip()] += 1
                fam[r["attack_cat"].strip()] += 1
                seen[tuple(r[f] for f in feats)] += 1
        rows = sum(lab.values())
        art = sm["artifacts"][key]
        if rows != art["rows"] or {k: v for k, v in lab.items()} != {k: int(v) for k, v in art["binary_label_counts"].items()}:
            stop(f"{path.name}: recomputed rows/labels differ from split_manifest")
        stats[key] = {"rows": rows, "benign": lab["0"], "malicious": lab["1"], "families": fam,
                      "dup": sum(c - 1 for c in seen.values() if c > 1)}
        vectors[key] = set(seen)
    checks.append("recomputed row and binary-label counts match split_manifest for all four splits")

    shared = {(x, y): len(vectors[x] & vectors[y]) for i, (x, _) in enumerate(SPLITS) for (y, _) in SPLITS[i + 1:]}
    if shared[("final_train", "strict_test_nonoverlap")] or shared[("validation", "strict_test_nonoverlap")]:
        stop("strict test shares exact feature vectors with final train or validation")
    checks.append("strict test shares 0 exact feature vectors with final train and validation")

    removed = {f: stats["official_test_full"]["families"][f] - stats["strict_test_nonoverlap"]["families"][f]
               for f in stats["official_test_full"]["families"]}
    proj = aud["cross_split_overlap"]["primary_evaluation_projection"]
    if sum(removed.values()) != proj["strict_test_excluded_rows"] or any(v < 0 for v in removed.values()):
        stop("official - strict does not reproduce the audited excluded-row count")
    checks.append("official minus strict test reproduces the audited excluded-row count")
    rep = aud["cross_split_overlap"]["representations"][sm["strict_test_policy"]["membership_representation"]]

    families = sorted([f for f in stats["final_train"]["families"] if f != BENIGN],
                      key=lambda f: -stats["final_train"]["families"][f])

    # ---------------- machine-readable data
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "table1_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity"] + [k for k, _ in SPLITS] + ["removed_by_strict_filter"])
        w.writerow(["rows"] + [stats[k]["rows"] for k, _ in SPLITS] + [sum(removed.values())])
        w.writerow(["benign"] + [stats[k]["benign"] for k, _ in SPLITS] + [removed.get(BENIGN, 0)])
        w.writerow(["malicious"] + [stats[k]["malicious"] for k, _ in SPLITS] + [sum(removed.values()) - removed.get(BENIGN, 0)])
        for fam in families:
            w.writerow([f"family:{fam}"] + [stats[k]["families"][fam] for k, _ in SPLITS] + [removed.get(fam, 0)])
        w.writerow(["within_split_duplicate_vectors"] + [stats[k]["dup"] for k, _ in SPLITS] + [""])
        for (x, y), v in shared.items():
            w.writerow([f"shared_vectors:{x}|{y}", "", "", "", "", v])

    # ---------------- LaTeX: Table I
    sp, lic, cit = sm["generator"], dm["license"], dm["citation"]["primary_dataset_reference"]
    ft, va, st, ot = (stats[k] for k, _ in SPLITS)
    rem_mal = sum(removed.values()) - removed.get(BENIGN, 0)
    L = []
    L.append(r"% Generated by analysis/tables/make_table1_dataset.py - do not edit by hand")
    L.append(r"\begin{table}[H]")
    L.append(r"\caption{Dataset characteristics and splits. Source: " + tex(dm["dataset"]["dataset_name"]) + " ("
             + tex(dm["dataset"]["source_organization"]) + r") \cite{moustafa2015unsw}; license: "
             + tex(lic["use_scope"].rstrip(".")) + r". The official supplied training/testing partition is preserved. "
             r"All counts are recomputed from the frozen split files; hashes are listed in Table~S1.}")
    L.append(r"\label{tab:dataset}")
    L.append(r"\begin{adjustwidth}{-\extralength}{0cm}")
    L.append(r"\centering\small")
    L.append(r"\begin{tabular}{lrrrrr}")
    L.append(r"\toprule")
    L.append(r" & \textbf{Final train} & \textbf{Validation} & \textbf{Strict test} & \textbf{Official test} & \textbf{Removed by} \\")
    L.append(r" & & & \textbf{(primary)} & \textbf{(secondary)} & \textbf{strict filter} \\")
    L.append(r"\midrule")
    L.append(f"Rows & {n(ft['rows'])} & {n(va['rows'])} & {n(st['rows'])} & {n(ot['rows'])} & {n(sum(removed.values()))} \\\\")
    L.append(f"Benign ({BENIGN}) & {n(ft['benign'])} & {n(va['benign'])} & {n(st['benign'])} & {n(ot['benign'])} & {n(removed.get(BENIGN, 0))} \\\\")
    L.append(f"Malicious & {n(ft['malicious'])} & {n(va['malicious'])} & \\textbf{{{n(st['malicious'])}}} & {n(ot['malicious'])} & {n(rem_mal)} \\\\")
    L.append(f"Malicious share & {pct(ft['malicious'], ft['rows'])} & {pct(va['malicious'], va['rows'])} & "
             f"{pct(st['malicious'], st['rows'])} & {pct(ot['malicious'], ot['rows'])} & --- \\\\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{6}{l}{\textit{Attack-family counts}} \\")
    for fam in families:
        L.append(f"{tex(fam)} & " + " & ".join(n(stats[k]["families"][fam]) for k, _ in SPLITS)
                 + f" & {n(removed.get(fam, 0))} \\\\")
    L.append(r"\midrule")
    L.append(f"Exact duplicate vectors & {n(ft['dup'])} ({pct(ft['dup'], ft['rows'])}) & {n(va['dup'])} "
             f"({pct(va['dup'], va['rows'])}) & {n(st['dup'])} ({pct(st['dup'], st['rows'])}) & {n(ot['dup'])} "
             f"({pct(ot['dup'], ot['rows'])}) & --- \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append("")
    L.append(r"\vspace{4pt}")
    L.append(r"\begin{tabular}{p{0.25\linewidth}p{0.70\linewidth}}")
    L.append(r"\toprule")
    L.append(r"\multicolumn{2}{l}{\textbf{Split procedure and leakage audit}} \\")
    L.append(r"\midrule")
    L.append(f"Features & {len(feats)} raw model features ({len(pm['policy'].get('numeric_features', []))} numeric, "
             f"{len(pm['policy'].get('categorical_features', []))} categorical; one-hot $\\rightarrow$ "
             f"{pm['feature_space']['transformed_feature_count']} inputs), preprocessor fit on final train only. \\\\")
    L.append(f"Development split & Validation $=$ {100 * sp['validation_fraction']:.0f}\\% of the official training partition, "
             f"{tex(sp['stratification_column'])}-stratified random split (seed {sp['random_seed']}); "
             f"membership overlap with final train: {n(sm['development_split']['membership_overlap_rows'])} rows. \\\\")
    L.append(r"Test partition & Official supplied testing partition preserved unchanged (secondary evaluation). \\")
    L.append(f"Strict test rule & Test rows whose complete {len(feats)}-feature vector occurs exactly in the official training "
             f"partition are excluded ({n(proj['strict_test_excluded_rows'])} rows; "
             f"{n(rep['confirmed_exact_unique_feature_vectors'])} unique shared vectors, of which "
             f"{n(rep['exact_vectors_with_different_label_sets'])} carry different labels across partitions; "
             f"{proj['strict_test_eligible_percentage']:.1f}\\% of the official test retained). Membership is independent of "
             r"labels, attack families and model outputs, and was fixed before model training. \\")
    L.append(f"Train/test contamination & Exact vectors shared with the strict test: "
             f"{shared[('final_train', 'strict_test_nonoverlap')]} (final train), "
             f"{shared[('validation', 'strict_test_nonoverlap')]} (validation). Final train and validation share "
             f"{n(shared[('final_train', 'validation')])} vectors (development partition only). \\\\")
    L.append(r"Duplicates & Exact within-split duplicates are retained, not deduplicated (pre-registered policy). \\")
    tmp = dm["temporal_ordering_assessment"]
    L.append(f"Temporal ordering & "
             + ("The selected files carry no timestamps; no chronological split is claimed."
                if not tmp.get("explicit_timestamp_available_in_selected_split_files") and not tmp.get("chronological_split_claimed")
                else tex(tmp.get("interpretation", ""))) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{adjustwidth}")
    L.append(r"\end{table}")
    (out / "table1_dataset.tex").write_text("\n".join(L) + "\n")

    # ---------------- LaTeX: Table S1 (hashes)
    S = [r"% Generated by analysis/tables/make_table1_dataset.py - do not edit by hand",
         r"\begin{table}[H]",
         r"\caption{SHA-256 identities of the data artifacts underlying Table~\ref{tab:dataset}, recomputed at generation time.}",
         r"\label{tab:S1-hashes}", r"\centering\scriptsize", r"\begin{tabular}{ll}", r"\toprule",
         r"\textbf{Artifact} & \textbf{SHA-256} \\", r"\midrule"]
    for path, h in hashes.items():
        S.append(f"\\texttt{{{tex(path)}}} & \\texttt{{{h}}} \\\\")
    S += [r"\bottomrule", r"\end{tabular}"]
    if hashes[str(DM.relative_to(ROOT))] not in dm_refs.values():
        S.append(r"\par\smallskip\footnotesize Note: the split and preprocessing manifests reference earlier versions of "
                 r"\texttt{dataset\_manifest.json} (" + ", ".join(f"\\texttt{{{v[:16]}}}" for v in dm_refs.values())
                 + r"); the manifest was subsequently updated to record phase completion.")
    S.append(r"\end{table}")
    (out / "tableS1_hashes.tex").write_text("\n".join(S) + "\n")

    prov = {"table": "Table I and Table S1", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": sha(Path(__file__)), "python": platform.python_version(),
            "checks_passed": checks, "inputs_sha256": hashes,
            "dataset_manifest_versions_referenced": dm_refs,
            "methods": {"vector_equality": "exact string equality of the 42 source-feature fields (utf-8-sig, BOM removed)",
                        "duplicates": "occurrences beyond the first of each exact vector within a split",
                        "removed_by_strict_filter": "official_test_full minus strict_test_nonoverlap, per family"},
            "citation": cit,
            "outputs_sha256": {p.name: sha(p) for p in (out / "table1_dataset.tex", out / "tableS1_hashes.tex", out / "table1_data.csv")}}
    (out / "table1_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    for c in checks:
        print("OK:", c)
    print(f"wrote {out}/table1_dataset.tex, tableS1_hashes.tex, table1_data.csv, table1_provenance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
