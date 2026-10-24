#!/usr/bin/env python3
"""
RolloutSecBench - Gate-4 LaTeX values and per-run table (Section "Gate 4").

READ-ONLY on evidence. Every number printed in the Gate-4 section comes from this script; nothing is typed
by hand. Inputs are verified before anything is written:
  * gate4.json and integrity_report.json match the analysis-of-record ANALYSIS_MANIFEST.sha256
  * integrity_report.json shows PASS with checksums verified
  * per run: the accounting identity  d_r = K12 - K21 - (|X12| - |X21|) * w2_obs  holds exactly
    (served == frozen makes misses = both_miss + K12 + (|X21| - K21)), and the cell mean equals mean d_r

Outputs (--out-dir):
  gate4_values.tex   \\newcommand macros used in the prose  (\\input in the preamble)
  table_gate4.tex    per-run table (tab:gate4)
  gate4_provenance.json

Usage (repo root):
  .venv-blueprint-v2/bin/python analysis/tables/make_gate4_tex.py \\
      --analysis-dir results/derived/phase9_v3/analysis_20261022T200702Z \\
      --out-dir results/derived/phase9_v3/tables/gate4
"""
import argparse
import hashlib
import json
import platform
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
CELLS = {"R1-A0": "Rone", "R2-A0": "Rtwo"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stop(msg):
    print(f"REFUSING TO WRITE GATE-4 TEX: {msg}", file=sys.stderr)
    sys.exit(2)


def num(x, d):
    s = f"{x:.{d}f}"
    return "\\ensuremath{-}" + s[1:] if s.startswith("-") else s   # valid in text and math mode


def pval(p):
    return "$<$0.0001" if p < 1e-4 else f"{p:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--out-dir", default="results/derived/phase9_v3/tables/gate4")
    ap.add_argument("--skip-prose-check", action="store_true", help="tests on synthetic data only")
    a = ap.parse_args()
    adir = (ROOT / a.analysis_dir).resolve()
    out = (ROOT / a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    man = {}
    for line in (adir / "ANALYSIS_MANIFEST.sha256").read_text().splitlines():
        h, name = line.split(None, 1)
        man[name.strip()] = h
    for name in ("gate4.json", "integrity_report.json"):
        if sha(adir / name) != man.get(name):
            stop(f"{name} does not match the analysis manifest")
    integ = json.loads((adir / "integrity_report.json").read_text())
    if not (integ.get("PASS") and integ.get("checksums_verified")):
        stop("analysis of record lacks a full integrity PASS")
    g = json.loads((adir / "gate4.json").read_text())
    pop = integ["population"]
    N, n12, n21 = pop["N"], pop["X12_only_v2_misses"], pop["X21_only_v1_misses"]

    # The Gate-4 prose states these outcomes in words; refuse if the evidence says otherwise.
    PROSE_EXPECTS = {"R1-A0": True, "R2-A0": False}          # reject_additivity_at_0.05
    checks, macros, rows = [], [], []
    for cell, tag in CELLS.items():
        c = g["cells"].get(cell) or stop(f"{cell} missing from gate4.json")
        runs = c["runs"]
        for r in runs:
            ident = r["K12_obs"] - r["K21_obs"] - (n12 - n21) * r["w2_obs"]
            if abs(ident - r["d_samples"]) > 1e-6:
                stop(f"{r['run_id']}: accounting identity fails ({ident} vs {r['d_samples']})")
        if abs(st.fmean(r["d_samples"] for r in runs) - c["mean_d_samples"]) > 1e-9:
            stop(f"{cell}: mean d does not equal the mean of per-run d")
        if "p_holm" not in c:
            stop(f"{cell}: Holm-adjusted p missing (numpy absent at analysis time?)")
        if not a.skip_prose_check and c["reject_additivity_at_0.05"] != PROSE_EXPECTS[cell]:
            stop(f"{cell}: outcome differs from what the Gate-4 prose describes; revise the prose first")
        if not a.skip_prose_check and cell == "R1-A0":
            short12 = n12 * st.fmean(r["w2_obs"] for r in runs) - st.fmean(r["K12_obs"] for r in runs)
            excess21 = st.fmean(r["K21_obs"] for r in runs) - n21 * st.fmean(r["w2_obs"] for r in runs)
            if not (short12 > 0 and excess21 > 0 and excess21 > short12):
                stop("R1-A0: mechanism differs from the prose (X12 under-served, X21 over-served, X21 larger)")
        checks.append(f"{cell}: identity holds for {len(runs)} runs; mean d recomputed")
        w = st.fmean(r["w2_obs"] for r in runs)
        s12 = st.fmean(r["v2_share_in_X12"] for r in runs)
        s21 = st.fmean(r["v2_share_in_X21"] for r in runs)
        k12 = st.fmean(r["K12_obs"] for r in runs)
        k21 = st.fmean(r["K21_obs"] for r in runs)
        vals = {
            "Runs": str(len(runs)),
            "MeanD": num(c["mean_d_samples"], 2),
            "MeanDpp": num(100 * c["mean_d_asr"], 4),
            "NullSD": num(c["null_sd_of_mean_d"], 2),
            "P": pval(c["p_two_sided_mc"]),
            "Holm": pval(c["p_holm"]),
            "Reject": "rejected" if c["reject_additivity_at_0.05"] else "not rejected",
            "MDEexc": str(c["secondary_MDE_exceedances"]),
            "Wobs": num(w, 4),
            "ShareXonetwo": num(s12, 4),
            "ShareXtwoone": num(s21, 4),
            "KonetwoMean": num(k12, 1),
            "KtwooneMean": num(k21, 1),
            "KonetwoExp": num(n12 * w, 1),
            "KtwooneExp": num(n21 * w, 1),
        }
        for k, v in vals.items():
            macros.append(f"\\newcommand{{\\Gfour{tag}{k}}}{{{v}}}")
        for r in sorted(runs, key=lambda x: x["run_id"]):
            rows.append(f"{cell.split('-')[0]} & \\texttt{{{r['run_id']}}} & {r['w2_obs']:.4f} & {r['misses']} & "
                        f"{num(r['d_samples'], 2)} & {r['K12_obs']} & {r['K21_obs']} & "
                        f"{r['v2_share_in_X12']:.3f} & {r['v2_share_in_X21']:.3f} & "
                        f"{'yes' if r['exceeds_superseded_MDE'] else 'no'} \\\\")
        rows.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{cell.split('-')[0]} cell:}} mean $\\bar d={num(c['mean_d_samples'], 2)}$ "
                    f"samples; null SD of $\\bar d$ {num(c['null_sd_of_mean_d'], 2)}; "
                    f"$p_{{\\mathrm{{MC}}}}={pval(c['p_two_sided_mc'])}$; $p_{{\\mathrm{{Holm}}}}={pval(c['p_holm'])}$}} \\\\")
        rows.append("\\midrule")
    rows[-1] = "\\bottomrule"
    macros += [f"\\newcommand{{\\GfourMCdraws}}{{{g['mc_draws']:,}}}".replace(",", "{,}"),
               f"\\newcommand{{\\GfourSeed}}{{{g['seed']}}}",
               f"\\newcommand{{\\GfourN}}{{{N:,}}}".replace(",", "{,}"),
               f"\\newcommand{{\\GfourXonetwo}}{{{n12}}}", f"\\newcommand{{\\GfourXtwoone}}{{{n21}}}"]

    (out / "gate4_values.tex").write_text(
        "% generated by make_gate4_tex.py - do not edit by hand\n" + "\n".join(macros) + "\n")
    table = r"""% generated by make_gate4_tex.py - do not edit by hand
\begin{table}[H]
\caption{Gate-4 additivity test per valid single-attempt run in the mixed rollout states.
$w_{2,\mathrm{obs}}$: realized V2 share; $d_r$: observed misses minus the realized-share mixture
$(1-w_{2,\mathrm{obs}})M_{\mathrm{R0}}+w_{2,\mathrm{obs}}M_{\mathrm{R3}}$, in samples; $K_{12}$, $K_{21}$:
samples of $X_{1\rightarrow2}$ and $X_{2\rightarrow1}$ served by V2; $s_{12}=K_{12}/|X_{1\rightarrow2}|$,
$s_{21}=K_{21}/|X_{2\rightarrow1}|$. MDE: $|d_r|/n_m$ exceeds the superseded MDE
(pre-registered secondary comparison).}
\label{tab:gate4}
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llrrrrrrrc}
\toprule
State & Run & $w_{2,\mathrm{obs}}$ & Misses & $d_r$ & $K_{12}$ & $K_{21}$ & $s_{12}$ & $s_{21}$ & MDE \\
\midrule
""" + "\n".join(rows) + "\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n"
    (out / "table_gate4.tex").write_text(table)

    prov = {"generated_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__)),
            "python": platform.python_version(), "analysis_dir": str(adir),
            "inputs_sha256": {k: man[k] for k in ("gate4.json", "integrity_report.json")},
            "checks": checks,
            "outputs_sha256": {p: sha(out / p) for p in ("gate4_values.tex", "table_gate4.tex")}}
    (out / "gate4_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    for c in checks:
        print("CHECK PASS", c)
    print((out / "gate4_values.tex").read_text())
    print(f"wrote {out / 'gate4_values.tex'}\nwrote {out / 'table_gate4.tex'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
