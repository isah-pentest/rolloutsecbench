#!/usr/bin/env python3
"""
RolloutSecBench - revised Table (Detector revisions), as requested in review:
  * complete confusion counts (TP, FP, TN, FN) for both splits,
  * rows ordered TP, FP, TN, FN, Accuracy, Precision, Recall, F1-score, FPR, AUROC, AUPRC,
  * a difference column (V2 - V1) per split,
  * "F1" renamed "F1-score" and "Recall (TPR)" renamed "Recall",
  * Gate 1: rule written explicitly as V2 - V1 >= -tol, with a V2 - V1 column.

Nothing is recomputed from models or data here and nothing in the repository is modified.
All values are read from table2_data.csv, written by the released, verifying generator
analysis/tables/make_table2_models.py (standard library only). That generator checks the revision
protocol hash, the model files against their manifests, the validation metrics against the
acceptance report, the pairwise strict-test predictions against the freeze hash, and recomputes every
strict-test count and metric from the row-level predictions before it writes anything.

This script then refuses to write unless every value already printed in the submitted Table equals
the generator's value (rates to 4 decimals, counts exactly) and the count identities hold.

Usage, from the repository root (local repository or a clean-room copy with the evidence archive):
    python3 make_table_models_revision.py --out-dir ~/rsb-table-models
    python3 make_table_models_revision.py --table2-dir <existing table2 output dir> --out-dir ~/rsb-table-models
Outputs: table_models.tex (the table), table_models_values.tex (macros for the text),
         table_models_provenance.json (input and output SHA-256).
"""
import argparse, csv, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

# Values printed in the submitted Table (Table 4 of the 3 Nov 2026 PDF). Used only as a guard.
PUBLISHED = {
    "validation": {"n": 35069,
        "v1": {"precision": "0.9610", "recall": "0.9758", "f1": "0.9683", "auroc": "0.9932", "auprc": "0.9968",
               "fpr": "0.0845", "accuracy": "0.9566", "fn": 577, "fp": 946},
        "v2": {"precision": "0.9606", "recall": "0.9753", "f1": "0.9679", "auroc": "0.9927", "auprc": "0.9966",
               "fpr": "0.0853", "accuracy": "0.9560", "fn": 589, "fp": 955}},
    "strict_test": {"n": 73791, "positives": 38263,
        "v1": {"precision": "0.8047", "recall": "0.9829", "f1": "0.8850", "auroc": "0.9824", "auprc": "0.9854",
               "fpr": "0.2569", "accuracy": "0.8675", "fn": 653, "fp": 9126},
        "v2": {"precision": "0.8072", "recall": "0.9820", "f1": "0.8860", "auroc": "0.9818", "auprc": "0.9849",
               "fpr": "0.2527", "accuracy": "0.8690", "fn": 688, "fp": 8977}},
    "gate1": {"f1": {"min": "0.9483", "tol": "0.02"}, "auroc": {"min": "0.9832", "tol": "0.01"}},
}
RATES = ["accuracy", "precision", "recall", "f1", "fpr", "auroc", "auprc"]
COUNTS = ["tp", "fp", "tn", "fn"]
LABEL = {"tp": "TP", "fp": "FP", "tn": "TN", "fn": "FN (missed attacks)", "accuracy": "Accuracy",
         "precision": "Precision", "recall": "Recall", "f1": "F1-score", "fpr": "FPR", "auroc": "AUROC",
         "auprc": "AUPRC"}
HIGHER_IS_BETTER = {"accuracy": True, "precision": True, "recall": True, "f1": True, "fpr": False,
                    "auroc": True, "auprc": True, "tp": True, "tn": True, "fp": False, "fn": False}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stop(msg):
    sys.exit(f"REFUSING TO WRITE: {msg}")


def n(x):
    return f"{x:,}".replace(",", "{,}")


def signed(x, fmt):
    s = format(abs(x), fmt)
    if float(s) == 0:
        return format(0, fmt)
    return ("+" if x > 0 else "\\ensuremath{-}") + s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table2-dir", help="existing output of make_table2_models.py (else it is run now)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    root = Path.cwd()
    out = Path(a.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if a.table2_dir:
        t2 = Path(a.table2_dir).expanduser().resolve()
    else:
        gen = root / "analysis/tables/make_table2_models.py"
        if not gen.is_file():
            stop("run from the repository root (analysis/tables/make_table2_models.py not found)")
        t2 = out / "table2_verified"
        r = subprocess.run([sys.executable, str(gen), "--out-dir", str(t2)], cwd=root)
        if r.returncode != 0:
            stop("make_table2_models.py refused (see its message above); nothing written")
    data_csv = t2 / "table2_data.csv"
    if not data_csv.is_file():
        stop(f"{data_csv} not found")

    v = {"validation": {"v1": {}, "v2": {}}, "strict_test": {"v1": {}, "v2": {}}}
    gmin = {}
    with open(data_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["block"] in v:
                for r_ in ("v1", "v2"):
                    q = row["quantity"]
                    v[row["block"]][r_][q] = int(row[r_]) if q in COUNTS else float(row[r_])
            elif row["block"] == "gate1":
                gmin[row["quantity"].replace("_min_required_v2", "")] = float(row["v2"])

    # guard 1: every value already in the submitted Table is unchanged
    for split in ("validation", "strict_test"):
        for r_ in ("v1", "v2"):
            for k, want in PUBLISHED[split][r_].items():
                got = v[split][r_].get(k)
                if got is None:
                    stop(f"{split}/{r_}/{k} missing from {data_csv.name}")
                if (f"{got:.4f}" if isinstance(want, str) else got) != want:
                    stop(f"{split}/{r_}/{k}: generator gives {got}, the submitted Table prints {want}")
    for k, g in PUBLISHED["gate1"].items():
        if f"{gmin[k]:.4f}" != g["min"]:
            stop(f"Gate 1 {k}: minimum {gmin[k]} differs from the submitted {g['min']}")
        tol = v["validation"]["v1"][k] - gmin[k]
        if abs(tol - float(g["tol"])) > 1e-9:
            stop(f"Gate 1 {k}: tolerance {tol} differs from the submitted {g['tol']}")

    # guard 2: count identities and consistency of the rates with the counts
    for split in ("validation", "strict_test"):
        for r_ in ("v1", "v2"):
            m = v[split][r_]
            tot = m["tp"] + m["fp"] + m["tn"] + m["fn"]
            if tot != PUBLISHED[split]["n"]:
                stop(f"{split}/{r_}: TP+FP+TN+FN = {tot}, not n = {PUBLISHED[split]['n']}")
            chk = {"precision": m["tp"] / (m["tp"] + m["fp"]), "recall": m["tp"] / (m["tp"] + m["fn"]),
                   "fpr": m["fp"] / (m["fp"] + m["tn"]), "accuracy": (m["tp"] + m["tn"]) / tot}
            chk["f1"] = 2 * chk["precision"] * chk["recall"] / (chk["precision"] + chk["recall"])
            for k, x in chk.items():
                if abs(x - m[k]) > 1e-9:
                    stop(f"{split}/{r_}: {k} {m[k]} inconsistent with the counts ({x})")
        pos = {r_: v[split][r_]["tp"] + v[split][r_]["fn"] for r_ in ("v1", "v2")}
        if pos["v1"] != pos["v2"]:
            stop(f"{split}: the two revisions see different numbers of malicious samples")
    if v["strict_test"]["v1"]["tp"] + v["strict_test"]["v1"]["fn"] != PUBLISHED["strict_test"]["positives"]:
        stop("strict test: TP+FN differs from n_m")

    d = {s: {k: v[s]["v2"][k] - v[s]["v1"][k] for k in RATES + COUNTS} for s in v}

    # ---- table
    rows = COUNTS + RATES
    L = [r"% Generated by make_table_models_revision.py from make_table2_models.py output - do not edit by hand",
         r"\begin{table}[H]",
         r"\caption{Detector revisions: training setup, static performance and Gate-1 acceptance.}",
         r"\label{tab:models}",
         r"\centering\small\renewcommand{\arraystretch}{1.15}",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"\multicolumn{7}{l}{\textbf{(a) Training setup}} \\", r"\midrule",
         r" & \multicolumn{1}{c}{\textbf{V1}} & \multicolumn{1}{c}{\textbf{V2}} & & & & \\",
         r"Role & baseline & pre-declared revision & & & & \\"]
    hp = [("n\\_estimators", "300", "450"), ("max\\_depth", "6", "5"), ("learning\\_rate", "0.05", "0.04"),
          ("subsample", "0.8", "0.9"), ("colsample\\_bytree", "0.8", "0.9"), ("min\\_child\\_weight", "1", "2"),
          ("gamma", "0.0", "0.1"), ("reg\\_alpha", "0.0", "0.05"), ("reg\\_lambda", "1.0", "1.5")]
    for k, a1, a2 in hp:   # unchanged from the submitted Table (generator-verified against the protocol)
        L.append(f"\\texttt{{{k}}} & {a1} & {a2} & & & & \\\\")
    L += [r"Decision threshold & 0.5 (fixed) & 0.5 (fixed) & & & & \\",
          r"Shared & \multicolumn{6}{l}{\texttt{binary:logistic}, \texttt{hist}, seed 42; XGBoost 3.4.1; no hyperparameter search} \\",
          r"\midrule", r"\multicolumn{7}{l}{\textbf{(b) Static performance}} \\", r"\midrule",
          r" & \multicolumn{3}{c}{\textbf{Validation} ($n=" + n(PUBLISHED["validation"]["n"]) + r"$)} & "
          r"\multicolumn{3}{c}{\textbf{Strict test} ($n=" + n(PUBLISHED["strict_test"]["n"]) + r"$)} \\",
          r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
          r" & V1 & V2 & V2 $-$ V1 & V1 & V2 & V2 $-$ V1 \\", r"\midrule"]
    for k in rows:
        cells = []
        for s in ("validation", "strict_test"):
            if k in COUNTS:
                cells += [n(v[s]["v1"][k]), n(v[s]["v2"][k]), signed(d[s][k], ",d").replace(",", "{,}")]
            else:
                cells += [f"{v[s]['v1'][k]:.4f}", f"{v[s]['v2'][k]:.4f}", signed(d[s][k], ".4f")]
        L.append(f"{LABEL[k]} & " + " & ".join(cells) + r" \\")
        if k == "fn":
            L.append(r"\midrule")
    L += [r"\midrule", r"\multicolumn{7}{l}{\textbf{(c) Gate 1 (validation only)}} \\", r"\midrule",
          r" & \multicolumn{1}{c}{\textbf{V1}} & \multicolumn{1}{c}{\textbf{V2}} & \multicolumn{1}{c}{\textbf{V2 $-$ V1}} & "
          r"\multicolumn{1}{c}{\textbf{Result}} & & \\",
          r"Core checks$^{a}$ & pass & pass & --- & pass & & \\"]
    for k, lab in (("f1", "F1-score"), ("auroc", "AUROC")):
        margin = v["validation"]["v2"][k] - gmin[k]          # guard only: V2 must not be below its Gate-1 minimum
        if margin < 0:
            stop(f"Gate 1 {k}: V2 below its minimum ({margin}), but the submitted Table reports pass")
        dv = v["validation"]["v2"][k] - v["validation"]["v1"][k]     # V2 - V1, as in part (b)
        L.append(f"{lab} & {v['validation']['v1'][k]:.4f} & "
                 f"{v['validation']['v2'][k]:.4f} & {signed(dv, '.4f')} & pass & & \\\\")
    L += [r"\textbf{Gate 1} & & & & \textbf{PASS} & & \\", r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", "",
          r"\smallskip\raggedright\footnotesize TP, FP, TN, FN: true positives, false positives, true negatives, false negatives "
          r"(malicious is the positive class); V2 $-$ V1 is computed before rounding. "
          r"Gate 1 is one-sided: V2 passes if V2 $-$ V1 is at least $-" + PUBLISHED["gate1"]["f1"]["tol"] + r"$ for the F1-score "
          r"and $-" + PUBLISHED["gate1"]["auroc"]["tol"] + r"$ for AUROC (absolute tolerances in metric units); the protocol sets no upper "
          r"bound for V2; FPR: false-positive "
          r"rate; AUROC, AUPRC: areas under the ROC and precision--recall curves. $^{a}$ Training completes; all "
          r"validation probabilities are finite and in $[0,1]$; all required validation metrics are recorded; the "
          r"serialized model reloads to identical predictions; deterministic reproduction passes. Static metrics on "
          r"the official (secondary) test partition were not computed in the frozen offline evaluation; see "
          r"Section~\ref{sec:limitations}.",
          r"\end{table}"]
    (out / "table_models.tex").write_text("\n".join(L) + "\n")

    # ---- macros for the text
    def better(s, r_):
        return [k for k in RATES if (d[s][k] < 0) == HIGHER_IS_BETTER[k] and d[s][k] != 0] if r_ == "v1" else \
               [k for k in RATES if (d[s][k] > 0) == HIGHER_IS_BETTER[k] and d[s][k] != 0]
    val_v1_better = better("validation", "v1")
    mx = max(RATES, key=lambda k: abs(d["validation"][k]))
    M = [r"% generated by make_table_models_revision.py - do not edit by hand",
         f"\\newcommand{{\\MvTPa}}{{{n(v['validation']['v1']['tp'])}}}",
         f"\\newcommand{{\\MvTPb}}{{{n(v['validation']['v2']['tp'])}}}",
         f"\\newcommand{{\\MvTNa}}{{{n(v['validation']['v1']['tn'])}}}",
         f"\\newcommand{{\\MvTNb}}{{{n(v['validation']['v2']['tn'])}}}",
         f"\\newcommand{{\\MTPa}}{{{n(v['strict_test']['v1']['tp'])}}}",
         f"\\newcommand{{\\MTPb}}{{{n(v['strict_test']['v2']['tp'])}}}",
         f"\\newcommand{{\\MTNa}}{{{n(v['strict_test']['v1']['tn'])}}}",
         f"\\newcommand{{\\MTNb}}{{{n(v['strict_test']['v2']['tn'])}}}",
         f"\\newcommand{{\\MValDeltaMax}}{{{abs(d['validation'][mx]):.4f}}}",
         f"\\newcommand{{\\MValDeltaMaxMetric}}{{{LABEL[mx]}}}",
         f"\\newcommand{{\\MValVoneBetterAll}}{{{'yes' if len(val_v1_better) == len(RATES) else 'no'}}}",
         f"\\newcommand{{\\MValFNdiff}}{{{d['validation']['fn']}}}"]
    (out / "table_models_values.tex").write_text("\n".join(M) + "\n")

    st_v2 = [LABEL[k] for k in better("strict_test", "v2")]
    st_v1 = [LABEL[k] for k in better("strict_test", "v1")]
    prov = {"generated_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__)),
            "inputs_sha256": {str(data_csv): sha(data_csv)},
            "outputs_sha256": {q: sha(out / q) for q in ("table_models.tex", "table_models_values.tex")},
            "validation_v1_better_on": [LABEL[k] for k in val_v1_better],
            "strict_test_v2_better_on": st_v2, "strict_test_v1_better_on": st_v1}
    (out / "table_models_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print("OK: every value of the submitted Table reproduced; count identities hold")
    print(f"validation: V1 better on {', '.join(prov['validation_v1_better_on'])}; "
          f"largest |Delta| {abs(d['validation'][mx]):.4f} ({LABEL[mx]})")
    print(f"strict test: V2 better on {', '.join(st_v2)}; V1 better on {', '.join(st_v1)}")
    for q, h in prov["outputs_sha256"].items():
        print(f"wrote {out / q}  sha256 {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
