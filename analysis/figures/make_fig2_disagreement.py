#!/usr/bin/env python3
"""
RolloutSecBench - Figure 2: Anatomy of malicious revision disagreement (H1, Gate 2).

Reads ONLY frozen Phase-4 artifacts. The script refuses to draw if:
  * any input's SHA-256 differs from environment/checksums.sha256,
  * any stored prediction is not reproducible from its stored probability at the
    frozen decision threshold,
  * any count recomputed from the row-level predictions differs from the frozen
    summaries (malicious_disagreement_strict.json, gate2_malicious_disagreement.json,
    phase4_statistical_summary.json, malicious_disagreement_by_attack_family_strict.csv).

Outputs (in --out-dir):
  fig2_malicious_disagreement.pdf   vector figure for the manuscript (fonts embedded)
  fig2_malicious_disagreement.png   600-dpi preview
  fig2_data.csv                     every plotted number (table view / supplement)
  fig2_provenance.json              input hashes, script hash, environment, checks passed

Dependencies: Python standard library + matplotlib.
Usage:  python make_fig2_disagreement.py --repo /path/to/RolloutSecBench --out-dir results/figures
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

INPUTS = {
    "predictions": "results/derived/malicious_predictions_strict.csv",
    "family_csv": "results/derived/malicious_disagreement_by_attack_family_strict.csv",
    "disagreement_json": "results/derived/malicious_disagreement_strict.json",
    "gate2_json": "results/derived/gate2_malicious_disagreement.json",
    "phase4_stats": "results/derived/phase4_statistical_summary.json",
    "static_metrics": "results/derived/static_model_metrics_strict.json",
}
CELLS = ["both_detect", "v1_detects_v2_misses", "v2_detects_v1_misses", "both_miss"]
N_EXPECTED = 38263  # |X_m| frozen at Phase 4
Z95 = 1.959963984540054


class IntegrityError(Exception):
    pass


def fail(msg):
    raise IntegrityError(msg)


# ----------------------------------------------------------------------------------
# Integrity
# ----------------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_checksums(repo):
    path = repo / "environment" / "checksums.sha256"
    if not path.is_file():
        fail(f"frozen checksum record not found: {path}")
    record = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(None, 1)
        record[name.strip().lstrip("*")] = digest.lower()
    return record


def verify_inputs(repo):
    record = load_frozen_checksums(repo)
    verified = {}
    for rel in INPUTS.values():
        path = repo / rel
        if not path.is_file():
            fail(f"missing input: {rel}")
        if rel not in record:
            fail(f"no frozen checksum recorded for {rel}")
        got = sha256_file(path)
        if got != record[rel]:
            fail(f"SHA-256 mismatch for {rel}: frozen {record[rel]}, found {got}")
        verified[rel] = got
    return verified


# ----------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------
def wilson(k, n, z=Z95):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def classify(v1, v2):
    return {(1, 1): "both_detect", (1, 0): "v1_detects_v2_misses",
            (0, 1): "v2_detects_v1_misses", (0, 0): "both_miss"}[(v1, v2)]


# ----------------------------------------------------------------------------------
# Recompute from row-level predictions and cross-check against frozen summaries
# ----------------------------------------------------------------------------------
def recompute(repo, thresholds):
    total = Counter()
    fam = defaultdict(Counter)
    ids = set()
    thr_mismatch = {">=": 0, ">": 0}
    with open(repo / INPUTS["predictions"], newline="") as fh:
        for row in csv.DictReader(fh):
            if row["label"] != "1":
                fail(f"non-malicious row in X_m: id={row['id']}")
            if row["id"] in ids:
                fail(f"duplicate sample id: {row['id']}")
            ids.add(row["id"])
            v1, v2 = int(row["v1_prediction"]), int(row["v2_prediction"])
            if v1 not in (0, 1) or v2 not in (0, 1):
                fail(f"non-binary prediction: id={row['id']}")
            cell = classify(v1, v2)
            if row["paired_outcome"] != cell:
                fail(f"paired_outcome inconsistent with predictions: id={row['id']}")
            for model, pred in (("v1", v1), ("v2", v2)):
                p, t = float(row[f"{model}_probability"]), thresholds[model]
                thr_mismatch[">="] += int(p >= t) != pred
                thr_mismatch[">"] += int(p > t) != pred
            total[cell] += 1
            fam[row["attack_cat"]][cell] += 1
    n = sum(total.values())
    if n != N_EXPECTED:
        fail(f"|X_m| = {n}, expected {N_EXPECTED}")
    rule = min(thr_mismatch, key=thr_mismatch.get)
    if thr_mismatch[rule] != 0:
        fail(f"stored predictions not reproducible from stored probabilities at frozen "
             f"thresholds {thresholds}: mismatches {thr_mismatch}")
    return n, total, fam, f"probability {rule} threshold"


def cross_check(repo, n, total, fam):
    d = json.loads((repo / INPUTS["disagreement_json"]).read_text())
    g = json.loads((repo / INPUTS["gate2_json"]).read_text())
    s = json.loads((repo / INPUTS["phase4_stats"]).read_text())
    checks = []

    def eq(name, frozen, recomputed):
        if frozen != recomputed:
            fail(f"{name}: frozen {frozen!r} != recomputed {recomputed!r}")
        checks.append(name)

    for c in CELLS:
        eq(f"malicious_disagreement_strict.paired_outcomes.{c}", d["paired_outcomes"][c], total[c])
        eq(f"phase4_statistical_summary.paired_outcome_counts.{c}", s["paired_outcome_counts"][c], total[c])
    eq("malicious_disagreement_strict.partition_sum", d["paired_outcomes"]["partition_sum"], n)
    eq("malicious_disagreement_strict.malicious_population.count", d["malicious_population"]["count"], n)

    n_d = total["v1_detects_v2_misses"] + total["v2_detects_v1_misses"]
    obs = g["observed"]
    eq("gate2.observed.malicious_count", obs["malicious_count"], n)
    eq("gate2.observed.disagreement_count", obs["disagreement_count"], n_d)
    eq("gate2.observed.v1_detects_v2_misses", obs["v1_detects_v2_misses"], total["v1_detects_v2_misses"])
    eq("gate2.observed.v2_detects_v1_misses", obs["v2_detects_v1_misses"], total["v2_detects_v1_misses"])
    if abs(obs["mrd"] - n_d / n) > 1e-15:
        fail(f"gate2 MRD {obs['mrd']} != {n_d}/{n}")
    checks.append("gate2.observed.mrd")

    with open(repo / INPUTS["family_csv"], newline="") as fh:
        frozen_fam = {r["attack_cat"]: r for r in csv.DictReader(fh)}
    if set(frozen_fam) != set(fam):
        fail(f"family sets differ: frozen {sorted(frozen_fam)} vs recomputed {sorted(fam)}")
    for f, r in frozen_fam.items():
        eq(f"family_csv.{f}.malicious_count", int(r["malicious_count"]), sum(fam[f].values()))
        for c in CELLS:
            eq(f"family_csv.{f}.{c}", int(r[c]), fam[f][c])
    return d, g, s, checks


# ----------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------
def draw(n, total, fam, g, out_dir, stamp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3de"
    BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots 1-2
    BLUE_T, ORANGE_T = "#cde2fb", "#fbd9c9"      # tints for heatmap cells
    NEUTRAL_T, MISS = "#f1f0ec", "#52514e"

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0, "xtick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })

    fig = plt.figure(figsize=(7.0, 3.2))  # full page width (MDPI adjustwidth), 7 pt base text
    # explicit panel positions (figure fraction): family labels sit between (a) and (b);
    # the |X_D| column sits between (b) and (c); nothing overlaps
    POS_A = [0.0, 0.16, 0.30, 0.74]
    POS_B = [0.455, 0.20, 0.235, 0.66]
    POS_C = [0.815, 0.20, 0.175, 0.66]

    # -------------------- (a) 2x2 paired outcomes with margins --------------------
    ax = fig.add_axes(POS_A)
    cells = {
        (0, 0): ("both_detect", NEUTRAL_T, MUTED),
        (0, 1): ("v1_detects_v2_misses", BLUE_T, BLUE),
        (1, 0): ("v2_detects_v1_misses", ORANGE_T, ORANGE),
        (1, 1): ("both_miss", "#d6d5cf", MISS),
    }
    sub = {"both_detect": "Both detect", "v1_detects_v2_misses": r"$X_{1\to2}$",
           "v2_detects_v1_misses": r"$X_{2\to1}$", "both_miss": "Both miss"}
    for (r, c), (key, fill, edge) in cells.items():
        y = 1 - r
        ax.add_patch(Rectangle((c + 0.03, y + 0.03), 0.94, 0.94, facecolor=fill,
                               edgecolor=edge, linewidth=1.1 if key != "both_detect" else 0.6))
        k = total[key]
        ax.text(c + 0.5, y + 0.70, sub[key], ha="center", va="center", fontsize=6.8, color=INK2)
        ax.text(c + 0.5, y + 0.47, f"{k:,}", ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=INK)
        ax.text(c + 0.5, y + 0.25, f"{100 * k / n:.2f}%", ha="center", va="center",
                fontsize=6.8, color=INK2)
    # margins = per-revision static totals on X_m
    v1_det = total["both_detect"] + total["v1_detects_v2_misses"]
    v1_miss = total["v2_detects_v1_misses"] + total["both_miss"]
    v2_det = total["both_detect"] + total["v2_detects_v1_misses"]
    v2_miss = total["v1_detects_v2_misses"] + total["both_miss"]
    for y, v in ((1.5, v1_det), (0.5, v1_miss)):
        ax.text(2.08, y, f"{v:,}", ha="left", va="center", fontsize=6.3, color=INK2)
    for x, v in ((0.5, v2_det), (1.5, v2_miss)):
        ax.text(x, -0.08, f"{v:,}", ha="center", va="top", fontsize=6.3, color=INK2)
    ax.text(2.08, -0.08, f"{n:,}", ha="left", va="top", fontsize=6.3, color=INK2,
            fontweight="bold")
    ax.text(2.08, 2.06, "Total", ha="left", va="bottom", fontsize=6.3, color=INK2)
    ax.text(-0.06, -0.08, "Total", ha="right", va="top", fontsize=6.3, color=INK2)
    ax.text(-0.06, 1.5, "V1\ndetects", ha="right", va="center", fontsize=6.8)
    ax.text(-0.06, 0.5, "V1\nmisses", ha="right", va="center", fontsize=6.8)
    ax.text(0.5, 2.06, "V2 detects", ha="center", va="bottom", fontsize=6.8)
    ax.text(1.5, 2.06, "V2 misses", ha="center", va="bottom", fontsize=6.8)
    ax.set_xlim(-0.62, 2.62)
    ax.set_ylim(-0.42, 2.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"(a) Paired V1/V2 outcomes on $X_m$", loc="left", pad=4, fontweight="bold")
    mrd = g["observed"]
    ci = mrd["mrd_ci_95_exact"]
    n_d = mrd["disagreement_count"]
    ax.text(-0.62, -0.30,
            f"MRD = {n_d}/{n:,} = {100 * mrd['mrd']:.3f}%\n"
            f"95% CI {100 * ci['low']:.3f}–{100 * ci['high']:.3f}% (Clopper–Pearson)\n"
            f"Both miss = {100 * total['both_miss'] / n:.2f}%: ASR floor in every rollout state",
            ha="left", va="top", fontsize=6.0, color=INK2, linespacing=1.4)

    # -------------------- family data, ordered by family size --------------------
    order = sorted(fam, key=lambda f: -sum(fam[f].values()))
    rows = []
    for f in order:
        nf = sum(fam[f].values())
        a, b, bm = fam[f]["v1_detects_v2_misses"], fam[f]["v2_detects_v1_misses"], fam[f]["both_miss"]
        rows.append(dict(family=f, n=nf, a=a, b=b, bm=bm,
                         a_ci=wilson(a, nf), b_ci=wilson(b, nf), bm_ci=wilson(bm, nf)))
    ypos = {r["family"]: len(rows) - 1 - i for i, r in enumerate(rows)}
    pa, pb, pbm = total["v1_detects_v2_misses"], total["v2_detects_v1_misses"], total["both_miss"]
    pooled_row = dict(family="All $X_m$", n=n, a=pa, b=pb, bm=pbm,
                      a_ci=wilson(pa, n), b_ci=wilson(pb, n), bm_ci=wilson(pbm, n))
    YP = len(rows) + 0.35            # pooled row sits above a thin separator
    ypos[pooled_row["family"]] = YP
    allrows = [pooled_row] + rows
    n_dis_all = pa + pb

    def style_rate_axis(axr, xmax, ticks):
        axr.set_xlim(0, xmax)
        axr.set_xticks(ticks)
        axr.set_xticklabels([f"{t:g}" for t in ticks])
        axr.set_ylim(-0.6, YP + 0.6)
        axr.axhline(len(rows) - 0.32, color=MUTED, linewidth=0.5, zorder=1)
        axr.grid(axis="x", color=GRID, linewidth=0.5)
        axr.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axr.spines[side].set_visible(False)
        axr.tick_params(axis="y", length=0)

    def dot(axr, x, lo, hi, y, color, marker, xmax):
        clipped = hi > xmax
        axr.plot([lo, min(hi, xmax)], [y, y], color=color, linewidth=1.1, solid_capstyle="butt",
                 zorder=2)
        if clipped:
            axr.annotate("", xy=(xmax, y), xytext=(xmax - 0.06 * xmax, y),
                         arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1,
                                         mutation_scale=6), zorder=3, annotation_clip=False)
        filled = x > 0
        axr.plot([x], [y], marker=marker, markersize=4.2 if marker == "o" else 4.6,
                 markerfacecolor=color if filled else "white", markeredgecolor=color,
                 markeredgewidth=0.9, linestyle="none", zorder=4, clip_on=False)

    # -------------------- (b) directional disagreement by family --------------------
    axb = fig.add_axes(POS_B)
    XB = 2.5
    style_rate_axis(axb, XB, [0, 0.5, 1, 1.5, 2, 2.5])
    off = 0.17
    for r in allrows:
        y = ypos[r["family"]]
        dot(axb, 100 * r["a"] / r["n"], 100 * r["a_ci"][0], 100 * r["a_ci"][1], y + off, BLUE, "o", XB)
        dot(axb, 100 * r["b"] / r["n"], 100 * r["b_ci"][0], 100 * r["b_ci"][1], y - off, ORANGE, "D", XB)
    for r in allrows:  # clipped upper bound: label inside the axis, above the arrows
        hi = max(100 * r["a_ci"][1], 100 * r["b_ci"][1])
        if hi > XB:
            axb.text(XB * 0.97, ypos[r["family"]] + 0.36, f"95% upper bound {hi:.1f}%", ha="right",
                     va="bottom", fontsize=5.6, color=INK2)
    axb.set_yticks([ypos[r["family"]] for r in allrows])
    axb.set_yticklabels([f"{r['family']}  ({r['n']:,})" for r in allrows], fontsize=6.5, color=INK)
    axb.get_yticklabels()[0].set_fontweight("bold")
    axb.tick_params(axis="y", pad=7)
    axb.set_xlabel("Samples in family (%)", color=INK2)
    axb.set_title("(b) Directional disagreement", loc="left", pad=4, fontweight="bold")
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(axb.transAxes, axb.transData)
    axb.text(1.20, YP + 0.62, r"$|X_D|$  (share of all)", ha="center", va="bottom", fontsize=6.0,
             color=INK2, transform=tr, clip_on=False)
    for r in allrows:  # |X_D| per row, in its own column between (b) and (c)
        nd = r["a"] + r["b"]
        txt = f"{nd}" if r is pooled_row else (f"{nd}  ({100 * nd / n_dis_all:.0f}%)" if nd else "0")
        axb.text(1.20, ypos[r["family"]], txt, ha="center", va="center", fontsize=6.2,
                 color=INK if nd else MUTED, fontweight="bold" if r is pooled_row else "normal",
                 transform=tr, clip_on=False)

    # -------------------- (c) both-miss by family --------------------
    axc = fig.add_axes(POS_C, sharey=axb)
    XC = 10
    style_rate_axis(axc, XC, [0, 2.5, 5, 7.5, 10])
    for r in allrows:
        dot(axc, 100 * r["bm"] / r["n"], 100 * r["bm_ci"][0], 100 * r["bm_ci"][1],
            ypos[r["family"]], MISS, "s", XC)
    pooled = 100 * total["both_miss"] / n
    axc.axvline(pooled, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    axc.tick_params(axis="y", labelleft=False)
    axc.set_xlabel("Samples in family (%)", color=INK2)
    axc.set_title("(c) Both miss (ASR floor)", loc="left", pad=4, fontweight="bold")

    # -------------------- legend (identity never by colour alone) --------------------
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=BLUE, marker="o", markersize=4.2, linewidth=1.1,
               label=r"V1 detects, V2 misses ($X_{1\to2}$)"),
        Line2D([], [], color=ORANGE, marker="D", markersize=4.2, linewidth=1.1,
               label=r"V2 detects, V1 misses ($X_{2\to1}$)"),
        Line2D([], [], color=MISS, marker="s", markersize=4.0, linewidth=1.1,
               label="Both miss"),
        Line2D([], [], color=INK2, marker="o", markersize=4.2, markerfacecolor="white",
               linewidth=0, label="Zero events (Wilson upper bound shown)"),
        Line2D([], [], color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)),
               label=f"Both miss, all $X_m$ ({100 * total['both_miss'] / n:.2f}%)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.3,
               handlelength=1.6, columnspacing=1.6, bbox_to_anchor=(0.5, -0.02))

    meta = {"Title": "RolloutSecBench Figure 2: malicious revision disagreement",
            "Subject": stamp, "Creator": "make_fig2_disagreement.py"}
    pdf = out_dir / "fig2_malicious_disagreement.pdf"
    png = out_dir / "fig2_malicious_disagreement.png"
    fig.savefig(pdf, metadata=meta)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    return rows, pdf, png


def write_table(rows, total, n, out_dir):
    path = out_dir / "fig2_data.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "attack_cat", "n", "quantity", "count", "rate",
                    "ci95_low", "ci95_high", "ci_method"])
        for c in CELLS:
            lo, hi = wilson(total[c], n)
            w.writerow(["a", "ALL", n, c, total[c], f"{total[c] / n:.10f}",
                        f"{lo:.10f}", f"{hi:.10f}", "wilson"])
        for r in rows:  # pooled b/c rows equal panel (a) cells and are already written above
            for panel, q, k, ci in (("b", "v1_detects_v2_misses", r["a"], r["a_ci"]),
                                    ("b", "v2_detects_v1_misses", r["b"], r["b_ci"]),
                                    ("c", "both_miss", r["bm"], r["bm_ci"])):
                w.writerow([panel, r["family"], r["n"], q, k, f"{k / r['n']:.10f}",
                            f"{ci[0]:.10f}", f"{ci[1]:.10f}", "wilson"])
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = ap.parse_args()
    repo = args.repo.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        hashes = verify_inputs(repo)
        sm = json.loads((repo / INPUTS["static_metrics"]).read_text())
        thresholds = {m: float(sm["decision_thresholds"][m]) for m in ("v1", "v2")}
        n, total, fam, rule = recompute(repo, thresholds)
        d, g, s, checks = cross_check(repo, n, total, fam)
    except IntegrityError as e:
        print(f"REFUSING TO DRAW FIGURE 2: {e}", file=sys.stderr)
        return 2

    pred_hash = hashes[INPUTS["predictions"]]
    stamp = f"inputs verified; {INPUTS['predictions']} sha256={pred_hash}"
    rows, pdf, png = draw(n, total, fam, g, out_dir, stamp)
    table = write_table(rows, total, n, out_dir)

    import matplotlib
    prov = {
        "artifact": "RolloutSecBench",
        "figure": "Figure 2 - malicious revision disagreement (H1, Gate 2)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": {"path": Path(__file__).name, "sha256": sha256_file(Path(__file__))},
        "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
        "inputs_verified_against": "environment/checksums.sha256",
        "inputs": {rel: {"sha256": h} for rel, h in hashes.items()},
        "thresholds": thresholds,
        "prediction_rule_reproduced": rule,
        "frozen_crosschecks_passed": checks,
        "counts": {"N_m": n, **{c: total[c] for c in CELLS}},
        "intervals": {"panel_a_mrd": "Clopper-Pearson, copied from gate2_malicious_disagreement.json",
                      "panels_b_c": "Wilson score, 95%, computed here"},
        "gate2_status": g["decision"]["status"],
        "outputs": {p.name: sha256_file(p) for p in (pdf, png, table)},
    }
    (out_dir / "fig2_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"OK: {len(checks)} frozen cross-checks passed; prediction rule '{rule}'.")
    for p in (pdf, png, table, out_dir / "fig2_provenance.json"):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
