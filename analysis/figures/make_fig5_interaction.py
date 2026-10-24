#!/usr/bin/env python3
"""
RolloutSecBench - Figure 5: Rollout x Attacker interaction (H5), sample-level evasion outcome (Audit Flag 4).

Outcome per run:  A0 -> C(1)/N (single-attempt ASR);  A1 -> C(K)/N with K = frozen A1 budget
(amendment A1-B5-L16: K = 5, replacing C(20)).

READ-ONLY. Inputs are verified before drawing:
  * freeze manifest against its recorded SHA-256
  * per_run.csv and integrity_report.json against the analysis-of-record ANALYSIS_MANIFEST.sha256,
    the manifest's freeze hash against the freeze, and the live registry against the manifest
  * integrity_report.json must show PASS with checksums verified
Expected values (dashed, open markers) come from the frozen disagreement counts and the exposure law:
  E[C(n)] = both_miss + |X12| (1-(1-w)^n) + |X21| (1-w^n), at the configured w of each state.

Outputs (--out-dir): fig5_interaction.pdf/.png, fig5_data.csv, fig5_provenance.json
Usage (repo root):
  .venv-blueprint-v2/bin/python analysis/make_fig5_interaction.py \
      --analysis-dir results/derived/phase9_v3/analysis_20261022T170703Z \
      --out-dir results/derived/phase9_v3/figures/fig5
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
FREEZE = ROOT / "data/manifests/phase9_v3/phase9_v3_freeze.json"
REGISTRY = ROOT / "results/raw/phase9_v3/registry.csv"
FROZEN_XM = ROOT / "results/derived/malicious_predictions_strict.csv"
STATES = ["R0", "R1", "R2", "R3"]
W = {"R0": 0.0, "R1": 0.1, "R2": 0.5, "R3": 1.0}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stop(msg):
    print(f"REFUSING TO DRAW FIGURE 5: {msg}", file=sys.stderr)
    sys.exit(2)


def tci(xs):
    m = st.fmean(xs)
    if len(xs) < 2:
        return m, 0.0, m, m
    s = st.stdev(xs)
    h = T975[len(xs) - 1] * s / math.sqrt(len(xs))
    return m, s, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--out-dir", default="results/derived/phase9_v3/figures/fig5")
    a = ap.parse_args()
    adir = (ROOT / a.analysis_dir).resolve()
    out = (ROOT / a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- integrity
    rec = (FREEZE.parent / (FREEZE.name + ".sha256")).read_text().split()[0]
    if sha(FREEZE) != rec:
        stop("freeze manifest hash mismatch")
    freeze = json.loads(FREEZE.read_text())
    man = {}
    for line in (adir / "ANALYSIS_MANIFEST.sha256").read_text().splitlines():
        h, name = line.split(None, 1)
        man[name.strip()] = h
    for name in ("per_run.csv", "integrity_report.json"):
        if sha(adir / name) != man.get(name):
            stop(f"{name} does not match the analysis manifest")
    if man.get("phase9_v3_freeze.json") != rec:
        stop("analysis of record was produced from a different freeze")
    if sha(REGISTRY) != man.get("registry.csv"):
        stop("registry changed after the analysis of record")
    integ = json.loads((adir / "integrity_report.json").read_text())
    if not (integ.get("PASS") and integ.get("checksums_verified")):
        stop("analysis of record lacks a full integrity PASS")
    if sha(FROZEN_XM) != freeze["frozen_inputs_sha256"]["results/derived/malicious_predictions_strict.csv"]:
        stop("frozen predictions hash mismatch")

    K = freeze["design"]["attempts"]["A1"]
    pop = integ["population"]
    N, n12, n21, bm = pop["N"], pop["X12_only_v2_misses"], pop["X21_only_v1_misses"], pop["both_miss"]

    # ---------------- data
    rows = list(csv.DictReader(open(adir / "per_run.csv")))
    data = {}
    for r in rows:
        k = 1 if r["attacker"] == "A0" else K
        data.setdefault((r["rollout_state"], r["attacker"]), []).append(
            (r["run_id"], int(r[f"C{k}"]) / N))
    for s in STATES:
        for at in ("A0", "A1"):
            if len(data.get((s, at), [])) != freeze["design"]["required_valid_per_cell"]:
                stop(f"cell {s}-{at} does not have the required number of valid runs")

    def expected(s, n):
        w = W[s]
        return (bm + n12 * (1 - (1 - w) ** n) + n21 * (1 - w ** n)) / N

    summary = []
    for at in ("A0", "A1"):
        for s in STATES:
            vals = [v for _, v in data[(s, at)]]
            m, sd, lo, hi = tci(vals)
            summary.append({"attacker": at, "state": s, "outcome": "C(1)/N" if at == "A0" else f"C({K})/N",
                            "n_runs": len(vals), "mean": m, "sd": sd, "ci95_lo": lo, "ci95_hi": hi,
                            "expected_law": expected(s, 1 if at == "A0" else K)})
    with open(out / "fig5_data.csv", "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["level", "attacker", "state", "run_id", "outcome", "value", "sd", "ci95_lo", "ci95_hi", "expected_law"])
        for at in ("A0", "A1"):
            for s in STATES:
                for rid, v in data[(s, at)]:
                    w_.writerow(["run", at, s, rid, "C(1)/N" if at == "A0" else f"C({K})/N", f"{v:.10f}", "", "", "", ""])
        for x in summary:
            w_.writerow(["cell", x["attacker"], x["state"], "", x["outcome"], f"{x['mean']:.10f}", f"{x['sd']:.10f}",
                         f"{x['ci95_lo']:.10f}", f"{x['ci95_hi']:.10f}", f"{x['expected_law']:.10f}"])

    # ---------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3de"
    COL = {"A0": "#2a78d6", "A1": "#eb6834"}
    MK = {"A0": "o", "A1": "D"}
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    xs = {s: i for i, s in enumerate(STATES)}
    off = {"A0": -0.09, "A1": 0.09}
    for at in ("A0", "A1"):
        cells = [x for x in summary if x["attacker"] == at]
        xm = [xs[x["state"]] + off[at] for x in cells]
        ym = [100 * x["mean"] for x in cells]
        # expected (law) - dashed, open markers
        ax.plot([xs[x["state"]] + off[at] for x in cells], [100 * x["expected_law"] for x in cells],
                color=COL[at], lw=0.8, ls=(0, (3, 2)), marker=MK[at], ms=3.2, mfc="white", mew=0.8, zorder=2)
        # runs
        for x in cells:
            vals = [v for _, v in data[(x["state"], at)]]
            jit = [(-0.035 + 0.0175 * i) for i in range(len(vals))]
            ax.scatter([xs[x["state"]] + off[at] + 0.6 * j for j in jit], [100 * v for v in vals], s=5,
                       color=COL[at], alpha=0.35, linewidths=0, zorder=3)
        # cell means + 95% CI
        ax.plot(xm, ym, color=COL[at], lw=1.2, zorder=4)
        ax.errorbar(xm, ym, yerr=[[100 * (x["mean"] - x["ci95_lo"]) for x in cells],
                                  [100 * (x["ci95_hi"] - x["mean"]) for x in cells]],
                    fmt=MK[at], ms=3.8, color=COL[at], mfc=COL[at], mec="white", mew=0.5, capsize=2, lw=1.0, zorder=5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{s}\n{int(round(100 * (1 - W[s])))}:{int(round(100 * W[s]))}" for s in STATES])
    ax.set_xlim(-0.45, 3.45)
    ax.set_xlabel("Rollout state (configured V1:V2)", color=INK2)
    ax.set_ylabel("Malicious samples evaded (%)", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [
        Line2D([], [], color=COL["A0"], marker="o", ms=3.8, lw=1.2, label="A0: C(1), single attempt"),
        Line2D([], [], color=COL["A1"], marker="D", ms=3.8, lw=1.2, label=f"A1: C({K}), up to {K} attempts"),
        Line2D([], [], color=INK2, marker="o", ms=3.2, mfc="white", lw=0.8, ls=(0, (3, 2)), label="expected (exposure law)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.2, handlelength=2.2)
    pdf, png = out / "fig5_interaction.pdf", out / "fig5_interaction.png"
    fig.savefig(pdf, metadata={"Title": "RolloutSecBench Figure 5: Rollout x Attacker interaction",
                               "Creator": "make_fig5_interaction.py"})
    fig.savefig(png, dpi=600)
    plt.close(fig)

    prov = {"figure": "Figure 5 - Rollout x Attacker interaction (H5)",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": sha(Path(__file__)), "python": platform.python_version(),
            "matplotlib": matplotlib.__version__, "freeze_sha256": rec, "analysis_dir": str(adir.relative_to(ROOT)),
            "inputs_sha256": {"per_run.csv": man["per_run.csv"], "registry.csv": man["registry.csv"]},
            "outcome": {"A0": "C(1)/N", "A1": f"C({K})/N (amendment A1-B5-L16)"},
            "intervals": "95% t-intervals over the valid runs of each cell",
            "outputs_sha256": {p.name: sha(p) for p in (pdf, png, out / "fig5_data.csv")}}
    (out / "fig5_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    for x in summary:
        print(f"{x['attacker']} {x['state']}: mean {100 * x['mean']:.4f}%  CI [{100 * x['ci95_lo']:.4f}, "
              f"{100 * x['ci95_hi']:.4f}]  expected {100 * x['expected_law']:.4f}%")
    print(f"wrote {pdf}\nwrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
