#!/usr/bin/env python3
"""
RolloutSecBench - Figure 3 (revised): cumulative evasion C(n) under repeated probing (A1) in the mixed
rollout states R1 and R2, drawn in ONE panel on a shared axis (reviewer request), instead of the two
panels drawn by workloads/analyze_v3.py.

The analysis of record is not re-run and not modified. This script only reads two of its outputs,
    exposure_law.csv   expected C(n), 95% band of a five-run mean, observed mean and 95% t interval
    per_run.csv        C(1..5) of every valid run
and refuses to draw unless
  * both files hash to the values recorded in the analysis manifest (ANALYSIS_MANIFEST.sha256),
  * the expected curve equals the exposure law recomputed from |X_both|, |X_1->2|, |X_2->1| and w2,
  * the observed means and 95% t intervals recomputed from per_run.csv equal exposure_law.csv,
  * each mixed A1 cell has five valid runs.
Plotted values are therefore exactly those of the analysis of record; only the layout changes.

Usage (repository root, analysis environment with matplotlib):
  .venv-blueprint-v2/bin/python make_fig3_exposure_law.py \
      --analysis-dir results/derived/phase9_v3/analysis_20261022T200702Z --out-dir ~/rsb-fig3
Outputs: fig_exposure_law.pdf, fig_exposure_law.png, fig3_data.csv, fig3_provenance.json
"""
import argparse, csv, hashlib, json, math, platform, statistics as stats, sys
from datetime import datetime, timezone
from pathlib import Path

N_BOTH, N_12, N_21 = 596, 92, 57          # frozen partition of X_m (claims_values.tex: NBoth, NXot, NXto)
W = {"R1": 0.1, "R2": 0.5}
K = 5
T975_DF4 = 2.776                         # t quantile used by analyze_v3.py for five runs (T975[4])
FIGSIZE = (3.19, 2.67)                   # rendered size and font sizes matched to Figure 4 (make_fig5_interaction.py)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stop(msg):
    sys.exit(f"REFUSING TO DRAW FIGURE 3: {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    A = Path(a.analysis_dir).expanduser().resolve()
    out = Path(a.out_dir).expanduser().resolve()

    man = {}
    for line in (A / "ANALYSIS_MANIFEST.sha256").read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            man[parts[1].strip()] = parts[0]
    for f in ("exposure_law.csv", "per_run.csv"):
        if man.get(f) != sha(A / f):
            stop(f"{f} does not match the analysis manifest")

    law = {}
    with open(A / "exposure_law.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            law[(r["state"], int(r["n"]))] = {k: float(v) for k, v in r.items() if k != "state"}
    runs = {st: [] for st in W}
    with open(A / "per_run.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            st = r["cell"].split("-")[0]
            if r["cell"] in ("R1-A1", "R2-A1"):
                runs[st].append({"run_id": r["run_id"], **{n: int(float(r[f"C{n}"])) for n in range(1, K + 1)}})

    for st, w in W.items():
        if len(runs[st]) != 5:
            stop(f"{st}-A1 has {len(runs[st])} valid runs, expected 5")
        for n in range(1, K + 1):
            L = law.get((st, n))
            if L is None:
                stop(f"exposure_law.csv lacks {st}, n={n}")
            q, rr = 1 - (1 - w) ** n, 1 - w ** n
            exp = N_BOTH + N_12 * q + N_21 * rr
            if abs(exp - L["expected_C"]) > 1e-9 or abs(L["w2"] - w) > 1e-12:
                stop(f"{st} n={n}: expected C(n) {L['expected_C']} differs from the exposure law {exp}")
            xs = [r[n] for r in runs[st]]
            m, s = stats.fmean(xs), stats.stdev(xs)
            h = T975_DF4 * s / math.sqrt(len(xs))
            for k, v in (("observed_mean_C", m), ("observed_ci95_lo", m - h), ("observed_ci95_hi", m + h)):
                if abs(v - L[k]) > 1e-9:
                    stop(f"{st} n={n}: recomputed {k} {v} differs from exposure_law.csv {L[k]}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    COL = {"R1": "#0072B2", "R2": "#D55E00"}      # Okabe-Ito blue / vermillion
    MK = {"R1": "s", "R2": "D"}
    ns = list(range(1, K + 1))
    ceiling = N_BOTH + N_12 + N_21

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhline(ceiling, color="#8a8984", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(1.0, ceiling + 0.8, rf"ceiling $|X_{{\mathrm{{both}}}}|+|X_D|$ = {ceiling}", fontsize=6.2,
            color="#52514e", va="bottom", ha="left")
    for st in W:
        c = COL[st]
        lr = [law[(st, n)] for n in ns]
        ax.fill_between(ns, [r["band95_lo"] for r in lr], [r["band95_hi"] for r in lr], color=c, alpha=0.15,
                        lw=0, zorder=2)
        ax.plot(ns, [r["expected_C"] for r in lr], color=c, lw=1.2, zorder=3)
        dx = -0.12 if st == "R1" else 0.12
        for r in runs[st]:
            ax.plot([n + dx for n in ns], [r[n] for n in ns], "o", ms=2.2, color=c, alpha=0.35, mew=0, zorder=4)
        ax.errorbar(ns, [r["observed_mean_C"] for r in lr],
                    yerr=[[r["observed_mean_C"] - r["observed_ci95_lo"] for r in lr],
                          [r["observed_ci95_hi"] - r["observed_mean_C"] for r in lr]],
                    fmt=MK[st], ms=4, color=c, mfc="white", mew=1.0, capsize=2, lw=1.0, zorder=5)
        ax.text(K + 0.22, lr[-1]["expected_C"], rf"{st} ($w_2={W[st]:g}$)", color=c, fontsize=6.5,
                va="center", ha="left", fontweight="bold")
    ax.set_xlim(0.7, K + 0.95)
    # headroom below the data so the legend (lower right) never overlaps a curve, band or interval
    lo = min(min(law[(st, n)]["band95_lo"], law[(st, n)]["observed_ci95_lo"], min(r[n] for r in runs[st]))
             for st in W for n in ns)
    span = ceiling - lo
    ax.set_ylim(lo - 0.32 * span, ceiling + 0.05 * span)
    ax.set_xticks(ns)
    ax.set_xlabel("Attempts per sample, $n$")
    ax.set_ylabel("Cumulative evaded samples, $C(n)$")
    ax.grid(axis="y", color="#e4e3de", lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], color="#333333", lw=1.2, label="Exposure law, expected $C(n)$"),
               Patch(facecolor="#333333", alpha=0.15, label="95% band of a five-run mean"),
               Line2D([], [], color="#333333", marker="s", mfc="white", lw=1.0, ms=4,
                      label="Observed mean, 95% $t$ interval"),
               Line2D([], [], color="#333333", marker="o", ms=2.2, alpha=0.35, lw=0, label="Individual runs")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.2)

    out.mkdir(parents=True, exist_ok=True)
    meta = {"Title": "RolloutSecBench Figure 3: exposure law", "Creator": "make_fig3_exposure_law.py",
            "Subject": f"exposure_law.csv sha256={sha(A / 'exposure_law.csv')}"}
    fig.savefig(out / "fig_exposure_law.pdf", metadata=meta)
    fig.savefig(out / "fig_exposure_law.png", dpi=600)
    plt.close(fig)

    with open(out / "fig3_data.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["state", "w2", "n", "expected_C", "band95_lo", "band95_hi", "observed_mean_C",
                    "observed_ci95_lo", "observed_ci95_hi"] + [r["run_id"] for r in runs["R1"]] + [r["run_id"] for r in runs["R2"]])
        for st in W:
            for n in ns:
                L = law[(st, n)]
                vals = [r[n] for r in runs["R1"]] + [""] * 5 if st == "R1" else [""] * 5 + [r[n] for r in runs["R2"]]
                w.writerow([st, W[st], n] + [repr(L[k]) for k in ("expected_C", "band95_lo", "band95_hi",
                            "observed_mean_C", "observed_ci95_lo", "observed_ci95_hi")] + vals)
    prov = {"figure": "Figure 3 (single panel)", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": sha(Path(__file__)), "python": platform.python_version(),
            "matplotlib": matplotlib.__version__, "analysis_dir": str(A),
            "inputs_sha256": {f: sha(A / f) for f in ("exposure_law.csv", "per_run.csv", "ANALYSIS_MANIFEST.sha256")},
            "checks_passed": ["inputs match ANALYSIS_MANIFEST.sha256", "expected C(n) equals the exposure law",
                              "observed means and 95% t intervals recomputed from per_run.csv",
                              "five valid runs per mixed A1 cell"],
            "outputs_sha256": {p: sha(out / p) for p in ("fig_exposure_law.pdf", "fig_exposure_law.png", "fig3_data.csv")}}
    (out / "fig3_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print("OK: inputs match the analysis manifest; exposure law and observed intervals reproduced")
    for p, h in prov["outputs_sha256"].items():
        print(f"wrote {out / p}  sha256 {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
