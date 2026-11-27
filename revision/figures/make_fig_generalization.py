#!/usr/bin/env python3
"""
RolloutSecBench - generalization figure: malicious revision disagreement (row MRD) and expected retry gain for every
revision pair of the generalization experiment, by model family and revision mechanism, on UNSW-NB15 (48 pairs) and
CIC-IDS2017 (24 pairs), with the original V1/V2 pair marked.

Reads only committed results of the generalization experiment and refuses to draw (exit 2, nothing written) unless
  * revision/results/e2/e2_pairs.csv, e2_summary.json and step0_reproduction.json equal their pinned hashes, and
    revision/results/e2_cic/e2_cic_pairs.csv and e2_cic_summary.json equal the hashes in the committed manifest
    revision/manifests/cic/phase5b_e2_cic.sha256 (tracked and unchanged from HEAD);
  * every pair's prediction file equals the preds_sha256 recorded for it, and X12, X21, n, the row MRD and the
    expected retry gain recomputed from that file equal the pair table;
  * the original pair (step 0) has X12 = 92, X21 = 57 on n = 38,263, as in the frozen analysis;
  * the medians by family and by mechanism recomputed here equal the committed summaries.

Expected retry gain: expected Delta(5) = E[C(5)] - E[C(1)] under the exposure law at w2 = 0.1,
  |X12| ((1 - 0.9^5) - 0.1) + |X21| ((1 - 0.1^5) - 0.9).

Outputs (--out-dir, default revision/figures/fig_generalization):
  fig_generalization.pdf / .png, fig_generalization_data.csv (every plotted value), fig_generalization_values.tex
  (macros for the caption and text), fig_generalization_provenance.json.
Usage (repository root):  python revision/figures/make_fig_generalization.py
"""
import argparse, csv, hashlib, json, math, platform, statistics as st, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

UNSW = {"pairs": ("revision/results/e2/e2_pairs.csv",
                  "8dafb37178670d6b17db1ca3eda100431540a5b3c6615b1e793d47518fd0e5e6"),
        "summary": ("revision/results/e2/e2_summary.json",
                    "02c94f55787f84885460399813f6277147e9217e96d175c61a6afe7b9e60b38d"),
        "step0": ("revision/results/e2/step0_reproduction.json",
                  "8b0dbd049075b741246da2c8b399a037c7127144d6c631953beb15d4858c8d0b"),
        "preds_dir": "revision/results/e2/preds", "n": 38263, "pairs_expected": 48}
CIC = {"pairs": "revision/results/e2_cic/e2_cic_pairs.csv", "summary": "revision/results/e2_cic/e2_cic_summary.json",
       "manifest": "revision/manifests/cic/phase5b_e2_cic.sha256",
       "preds_dir": "revision/results/e2_cic/preds", "n": 42053, "pairs_expected": 24}
FAMILIES = [("xgb", "XGBoost"), ("lgbm", "LightGBM"), ("rf", "Random\nforest"), ("mlp", "MLP")]
MECHS = {"M1": "reseed", "M2": "hyperparameter step", "M3": "data refresh", "M4": "feature-set change"}
SEEDS = [101, 202, 303]
W2 = 0.1
TOL = 1e-12


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


def gain(x12, x21, w=W2):
    return x12 * ((1 - (1 - w) ** 5) - w) + x21 * ((1 - w ** 5) - (1 - w))


def load(repo, spec, pairs_rel, dataset):
    rows = []
    with open(repo / pairs_rel, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    need(len(rows) == spec["pairs_expected"], f"{pairs_rel}: {len(rows)} pairs, expected {spec['pairs_expected']}")
    out = []
    for r in rows:
        pf = repo / spec["preds_dir"] / f"{r['pair']}.csv"
        need(pf.is_file(), f"prediction file missing: {pf.relative_to(repo)}")
        need(sha(pf) == r["preds_sha256"], f"prediction file differs from preds_sha256: {r['pair']}")
        x12 = x21 = n = 0
        with open(pf, newline="") as fh:
            for p in csv.DictReader(fh):
                v1, v2 = int(p["v1"]), int(p["v2"])
                n += 1
                x12 += v1 == 1 and v2 == 0
                x21 += v1 == 0 and v2 == 1
        need(n == spec["n"], f"{r['pair']}: {n} malicious test rows, expected {spec['n']}")
        need(x12 == int(r["X12"]) and x21 == int(r["X21"]), f"{r['pair']}: X12/X21 recomputed {x12}/{x21}, "
                                                             f"table {r['X12']}/{r['X21']}")
        mrd = (x12 + x21) / n
        g = gain(x12, x21)
        need(abs(mrd - float(r["MRD_row"])) < TOL, f"{r['pair']}: MRD_row")
        need(abs(g - float(r["expected_gain_w0.1_n5"])) < 1e-9, f"{r['pair']}: expected gain")
        need(r["mechanism"] in MECHS and r["family"] in dict(FAMILIES) and int(r["seed"]) in SEEDS,
             f"{r['pair']}: unexpected family/mechanism/seed")
        out.append({"dataset": dataset, "pair": r["pair"], "family": r["family"], "mechanism": r["mechanism"],
                    "seed": int(r["seed"]), "gate1_pass": r["gate1_pass"].strip().lower() == "true",
                    "X12": x12, "X21": x21, "n": n, "MRD_row": mrd, "expected_gain_w0.1_n5": g,
                    "preds_sha256": r["preds_sha256"]})
    return out


def check_medians(recs, summary, label):
    passing = [r for r in recs if r["gate1_pass"]]
    checked = []
    for grp_key, field in (("by_family", "family"), ("by_mechanism", "mechanism")):
        for name, block in summary[grp_key].items():
            sub = [r for r in passing if r[field] == name]
            for k in ("MRD_row", "expected_gain_w0.1_n5"):
                if not sub:
                    continue
                med = st.median([r[k] for r in sub])
                need(abs(med - block[k]["median"]) < 1e-9, f"{label} {grp_key}.{name}.{k} median {med} vs "
                                                          f"summary {block[k]['median']}")
                checked.append(f"{label}.{grp_key}.{name}.{k}.median")
    return checked


def tex_num(x, d):
    return f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, default=Path("revision/figures/fig_generalization"))
    a = ap.parse_args()
    repo = a.repo.resolve()
    out = a.out_dir if a.out_dir.is_absolute() else repo / a.out_dir
    inputs, checks = {}, []
    try:
        for key in ("pairs", "summary", "step0"):
            rel, want = UNSW[key]
            need(sha(repo / rel) == want, f"{rel} differs from pinned {want}")
            inputs[rel] = want
        need(git_clean(repo, CIC["manifest"]), f"{CIC['manifest']} is not tracked or differs from HEAD")
        for key in ("pairs", "summary"):
            want = manifest_hash(repo, CIC["manifest"], CIC[key])
            need(sha(repo / CIC[key]) == want, f"{CIC[key]} differs from {CIC['manifest']}")
            inputs[CIC[key]] = want
        inputs[CIC["manifest"]] = sha(repo / CIC["manifest"])
        checks.append("pair tables and summaries equal pinned hashes / committed manifest")

        unsw = load(repo, UNSW, UNSW["pairs"][0], "UNSW-NB15")
        cic = load(repo, CIC, CIC["pairs"], "CIC-IDS2017")
        checks.append("every prediction file equals preds_sha256; X12, X21, n, MRD_row and expected gain "
                      "recomputed from it (72 pairs)")

        s0 = json.loads((repo / UNSW["step0"][0]).read_text())
        need(s0.get("predictions_identical") is True, "step 0 did not reproduce the published models")
        op = s0["original_pair"]
        need(op["X12"] == 92 and op["X21"] == 57, f"original pair X12/X21 {op['X12']}/{op['X21']}, expected 92/57")
        ox12, ox21 = int(op["X12"]), int(op["X21"])
        orig = {"MRD_row": (ox12 + ox21) / UNSW["n"], "gain": gain(ox12, ox21)}
        need(abs(op["MRD_row"] - orig["MRD_row"]) < TOL, "original pair MRD_row")
        checks.append("original pair (step 0): predictions identical to the frozen models; X12/X21 = 92/57")

        checks += check_medians(unsw, json.loads((repo / UNSW["summary"][0]).read_text()), "UNSW")
        checks += check_medians(cic, json.loads((repo / CIC["summary"]).read_text()), "CIC")
    except Refuse as e:
        print(f"REFUSING TO DRAW: {e}", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.3, "ytick.labelsize": 6.5, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
        "xtick.color": "#52514e", "ytick.color": "#52514e", "xtick.major.width": 0.6, "ytick.major.width": 0,
        "xtick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    INK, INK2, GRID = "#1f1e1c", "#52514e", "#e4e3de"
    COL = {"M1": "#0072B2", "M2": "#D55E00", "M3": "#009E73", "M4": "#CC79A7"}
    OFF = {101: -0.24, 202: -0.08, 303: 0.08}
    ORIG_OFF = 0.24

    def rows_for(mechs):
        return [(f, m) for f, _ in FAMILIES for m in mechs]

    blocks = [("UNSW-NB15", unsw, rows_for(["M1", "M2", "M3", "M4"])), ("CIC-IDS2017", cic, rows_for(["M1", "M2"]))]
    row_h = 0.150
    gap = 0.62
    left, mid, right = 0.92, 0.22, 0.12
    panel_w = (7.0 - left - mid - right) / 2
    heights = [len(b[2]) * row_h for b in blocks]
    top_pad, bottom_pad, title_h = 0.30, 0.62, 0.0
    fig_h = top_pad + heights[0] + gap + heights[1] + bottom_pad
    fig = plt.figure(figsize=(7.0, fig_h))
    axes = {}
    fam_labels = []
    y_top = fig_h - top_pad
    for bi, (name, recs, rows) in enumerate(blocks):
        h = heights[bi]
        y0 = y_top - h
        for pi, metric in enumerate(("MRD_row", "expected_gain_w0.1_n5")):
            x0 = left + pi * (panel_w + mid)
            ax = fig.add_axes([x0 / 7.0, y0 / fig_h, panel_w / 7.0, h / fig_h])
            axes[(bi, pi)] = ax
            ypos = {rk: len(rows) - 1 - i for i, rk in enumerate(rows)}
            for i, rk in enumerate(rows):
                if i % 2 == 0:
                    ax.axhspan(ypos[rk] - 0.5, ypos[rk] + 0.5, color="#f4f3ef", lw=0, zorder=0)
            fam_ends = [i for i in range(1, len(rows)) if rows[i][0] != rows[i - 1][0]]
            for i in fam_ends:
                ax.axhline(len(rows) - i - 0.5, color="#b9b8b3", lw=0.5, zorder=1)
            vals = []
            for r in recs:
                y = ypos[(r["family"], r["mechanism"])] + OFF[r["seed"]]
                v = 100 * r[metric] if metric == "MRD_row" else r[metric]
                vals.append(v)
                ax.plot(v, y, "o", ms=3.0, color=COL[r["mechanism"]], mec="white", mew=0.4, zorder=4,
                        mfc=COL[r["mechanism"]] if r["gate1_pass"] else "white")
            if name == "UNSW-NB15":
                v = 100 * orig["MRD_row"] if metric == "MRD_row" else orig["gain"]
                vals.append(v)
                ax.axvline(v, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=2)
                ax.plot(v, ypos[("xgb", "M2")] + ORIG_OFF, "*", ms=6.5, color=INK, mec="white", mew=0.4, zorder=5)
            vmax = max(vals)
            ax.set_xlim(-0.03 * vmax, 1.06 * vmax)
            ax.set_ylim(-0.5, len(rows) - 0.5)
            ax.grid(axis="x", color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_yticks([ypos[rk] for rk in rows])
            if pi == 0:
                ax.set_yticklabels([m for _, m in rows], fontsize=6.3, color=INK2)
                for f, flabel in FAMILIES:
                    ys = [ypos[rk] for rk in rows if rk[0] == f]
                    fam_labels.append(ax.annotate(flabel, xy=(0, (min(ys) + max(ys)) / 2),
                                                  xycoords=("axes fraction", "data"),
                                                  xytext=(-(left - 0.06) * 72, 0), textcoords="offset points",
                                                  ha="left", va="center", fontsize=6.6, fontweight="bold",
                                                  color=INK))
            else:
                ax.set_yticklabels([])
            if metric == "MRD_row":
                ax.set_xlabel("Row MRD (% of malicious test samples)", color=INK2)
                title = f"({'a' if bi == 0 else 'c'}) {name}: row MRD"
            else:
                ax.set_xlabel(r"Expected retry gain $\Delta(5)$ at $w_2=0.1$ (samples)", color=INK2)
                title = f"({'b' if bi == 0 else 'd'}) {name}: expected retry gain"
            ax.set_title(title, loc="left", fontsize=7.2, fontweight="bold", color=INK, pad=4)
        y_top = y0 - gap

    handles = [Line2D([], [], color=COL[m], marker="o", ls="", ms=3.6, mec="white", mew=0.4, label=f"{m} {MECHS[m]}")
               for m in MECHS]
    handles.append(Line2D([], [], color=INK, marker="*", ls=(0, (3, 2)), lw=0.6, ms=6.5, mec="white", mew=0.4,
                          label="original V1/V2 pair"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6.3,
                     bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.2)

    # layout guard
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    boxes = []
    for (bi, pi), ax in axes.items():
        boxes.append((f"panel{bi}{pi}-xlabel", ax.xaxis.label.get_window_extent(rend)))
        boxes.append((f"panel{bi}{pi}-title", ax.title.get_window_extent(rend)))
        boxes += [(f"panel{bi}{pi}-yt{i}", t.get_window_extent(rend)) for i, t in enumerate(ax.get_yticklabels())
                  if t.get_text()]
    boxes.append(("legend", leg.get_window_extent(rend)))
    boxes += [(f"family-label-{i}", t.get_window_extent(rend)) for i, t in enumerate(fam_labels)]
    fw = fig.get_figwidth() * fig.dpi
    for i in range(len(boxes)):
        need_ok = boxes[i][1].x0 >= -0.5 and boxes[i][1].x1 <= fw + 0.5
        if not need_ok:
            print(f"REFUSING: layout - {boxes[i][0]} extends beyond the figure", file=sys.stderr)
            return 2
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                print(f"REFUSING: layout - {boxes[i][0]} overlaps {boxes[j][0]}", file=sys.stderr)
                return 2

    out.mkdir(parents=True, exist_ok=True)
    pdf, png = out / "fig_generalization.pdf", out / "fig_generalization.png"
    meta = {"Title": "RolloutSecBench: generalization across revision pairs", "Creator": "make_fig_generalization.py",
            "Subject": f"e2_pairs.csv sha256={UNSW['pairs'][1]}", "CreationDate": None}
    fig.savefig(pdf, metadata=meta)
    fig.savefig(png, dpi=600)
    plt.close(fig)

    # ------------------------------------------------------------------ data, values, provenance
    data = out / "fig_generalization_data.csv"
    with open(data, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "pair", "family", "mechanism", "seed", "gate1_pass", "X12", "X21", "n",
                    "MRD_row", "expected_gain_w0.1_n5", "preds_sha256"])
        for r in unsw + cic:
            w.writerow([r["dataset"], r["pair"], r["family"], r["mechanism"], r["seed"], r["gate1_pass"], r["X12"],
                        r["X21"], r["n"], repr(r["MRD_row"]), repr(r["expected_gain_w0.1_n5"]), r["preds_sha256"]])
        w.writerow(["UNSW-NB15", "original", "xgb", "M2", 42, True, ox12, ox21, UNSW["n"], repr(orig["MRD_row"]),
                    repr(orig["gain"]), ""])

    def summ(recs):
        p = [r for r in recs if r["gate1_pass"]]
        m = [100 * r["MRD_row"] for r in p]
        g = [r["expected_gain_w0.1_n5"] for r in p]
        return {"pairs": len(recs), "pass": len(p), "mrd_min": min(m), "mrd_med": st.median(m), "mrd_max": max(m),
                "gain_med": st.median(g), "gain_max": max(g),
                "zero": sum(1 for r in p if r["X12"] + r["X21"] == 0),
                "x12_gt": sum(1 for r in p if r["X12"] > r["X21"]), "x21_gt": sum(1 for r in p if r["X21"] > r["X12"])}
    su, sc = summ(unsw), summ(cic)
    below = sum(1 for r in unsw if r["gate1_pass"] and r["MRD_row"] < orig["MRD_row"])
    macros = {
        "GenUNSWPairs": str(su["pairs"]), "GenUNSWPass": str(su["pass"]),
        "GenUNSWMRDMin": tex_num(su["mrd_min"], 3), "GenUNSWMRDMed": tex_num(su["mrd_med"], 3),
        "GenUNSWMRDMax": tex_num(su["mrd_max"], 3), "GenUNSWGainMed": tex_num(su["gain_med"], 1),
        "GenUNSWGainMax": tex_num(su["gain_max"], 1), "GenUNSWZero": str(su["zero"]),
        "GenUNSWXotGt": str(su["x12_gt"]), "GenUNSWXtoGt": str(su["x21_gt"]),
        "GenCICPairs": str(sc["pairs"]), "GenCICPass": str(sc["pass"]),
        "GenCICMRDMin": tex_num(sc["mrd_min"], 3), "GenCICMRDMed": tex_num(sc["mrd_med"], 3),
        "GenCICMRDMax": tex_num(sc["mrd_max"], 3), "GenCICGainMed": tex_num(sc["gain_med"], 1),
        "GenCICGainMax": tex_num(sc["gain_max"], 1), "GenCICZero": str(sc["zero"]),
        "GenCICXotGt": str(sc["x12_gt"]), "GenCICXtoGt": str(sc["x21_gt"]),
        "GenOrigMRD": tex_num(100 * orig["MRD_row"], 3), "GenOrigGain": tex_num(orig["gain"], 1),
        "GenOrigBelow": str(below), "GenNmCIC": f"{CIC['n']:,}".replace(",", "{,}"),
    }
    vtex = out / "fig_generalization_values.tex"
    vtex.write_text("% generated by make_fig_generalization.py - do not edit by hand\n"
                    + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
    prov = {
        "artifact": "RolloutSecBench", "figure": "Generalization across revision pairs (UNSW-NB15, CIC-IDS2017)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": {"path": "revision/figures/make_fig_generalization.py", "sha256": sha(Path(__file__))},
        "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
        "inputs": inputs, "checks_passed": checks,
        "expected_retry_gain": "E[C(5)] - E[C(1)] under the exposure law at w2 = 0.1: "
                               "|X12|((1-0.9^5)-0.1) + |X21|((1-0.1^5)-0.9)",
        "original_pair": {"X12": ox12, "X21": ox21, "n": UNSW["n"], **orig},
        "summary": {"UNSW-NB15": su, "CIC-IDS2017": sc, "UNSW_pairs_with_MRD_below_original": below},
        "outputs": {p.name: sha(p) for p in (pdf, png, data, vtex)},
    }
    (out / "fig_generalization_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"OK: {len(checks)} checks passed")
    for p in (pdf, png, data, vtex, out / "fig_generalization_provenance.json"):
        print(f"wrote {p.relative_to(repo) if p.is_relative_to(repo) else p}  {sha(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
