#!/usr/bin/env python3
"""RolloutSecBench revision r1: values and tables of the extension results (Results, Sections 4.1 and 4.7-4.11).

Reads committed outputs only; computes no new outcome. Every input is checked against its revision manifest before
use. Writes, under revision/results/manuscript_values/:
  results_ext_values.tex   \\newcommand macros used in the Results text
  table_robustness.tex     alternative tests of the confirmatory decisions (S3) and H3 intervals (S4)
  table_canary.tex         canary sizing (exposure law) and single-attempt detectability (S2/E1)
  table_policies.tex       routing-policy distributions (E3, simulated) and live verification (E3, live)
  provenance.json          SHA-256 of every input and output
Values that were already printed in committed run logs are cross-checked; any difference stops the script.
Run from the repository root:  python3 revision/analysis/make_results_ext_values.py
"""
import hashlib, json, math, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "revision/results/manuscript_values"
N = 38263


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def manifest(rel):
    m = {}
    for line in (ROOT / rel).read_text().splitlines():
        if line.strip():
            h, p = line.split(None, 1); p = p.strip().lstrip("*"); m[p[2:] if p.startswith("./") else p] = h
    return m


INPUTS = {}


def load(rel):
    """Return the parsed JSON at rel after checking it against every manifest under revision/manifests that lists it."""
    hits = set()
    for mf in sorted((ROOT / "revision/manifests").rglob("*.sha256")):
        m = manifest(mf.relative_to(ROOT))
        for key in (rel, rel.removeprefix("revision/")):
            if key in m: hits.add(m[key])
    if not hits: sys.exit(f"STOP: {rel} not listed in any revision manifest")
    if len(hits) > 1: sys.exit(f"STOP: manifests disagree on {rel}")
    got = sha(ROOT / rel)
    if got not in hits: sys.exit(f"STOP: hash mismatch {rel}")
    INPUTS[rel] = got
    return json.load(open(ROOT / rel))


def need(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d: sys.exit(f"STOP: missing key {'/'.join(map(str, keys))}")
        d = d[k]
    return d


def same(label, a, b, tol=1e-9):
    if (isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) > tol) or \
            (not isinstance(a, (int, float)) and a != b):
        sys.exit(f"STOP: cross-check {label}: output {a!r} != logged {b!r}")


def f(x, d):  # fixed decimals with thousands separators
    s = f"{x:,.{d}f}".replace(",", "{,}")
    return "\\ensuremath{-}" + s[1:] if s.startswith("-") else s


def pv(p):  # p-value in the manuscript style
    if p >= 0.001: return f"{p:.3f}"
    e = math.floor(math.log10(p)); m = p / 10 ** e
    if round(m, 1) >= 10: m, e = m / 10, e + 1  # e.g. 9.99999e-6 prints as 1.0e-5, not 10.0e-6
    return f"{m:.1f}\\times10^{{{e}}}"


S1 = load("revision/results/s1_duplicates/s1_results.json")
S3 = load("revision/results/s3_s4_robustness/s3_s4_results.json")
E1 = load("revision/results/e1_distributions/e1_summary.json")
E3 = load("revision/results/e3_sim/e3_summary.json")
EL = load("revision/results/e3_live/e3_live_results.json")
E4 = load("revision/results/e4/e4_summary.json")
ED = load("revision/results/e4/e4_decomposition.json")
E5 = load("revision/results/e5/e5_summary.json")
FZ_REL = "data/manifests/phase9_v3/phase9_v3_freeze.json"
FZ_SHA = "67d0ac984710b5a3748c44de577eece02bf7652deb9969fc792decf6b75dfcf1"  # FREEZE_SHA in the simulator
if sha(ROOT / FZ_REL) != FZ_SHA: sys.exit("STOP: phase9_v3 freeze hash differs from simulate.py")
INPUTS[FZ_REL] = FZ_SHA
IDENT = {x["run_id"]: x for x in json.load(open(ROOT / FZ_REL))["identities"]}

mac = {}

# ---------------- S1: row MRD cluster bootstrap ----------------
same("S1 rows XD", need(S1, "rows", "XD"), 149); same("S1 rows n", need(S1, "rows", "n"), N)
lo, hi = need(S1, "rows", "cluster_bootstrap_95")
mac["SoneBootLo"], mac["SoneBootHi"] = f(100 * lo, 3), f(100 * hi, 3)

# ---------------- S3/S4: robustness ----------------
if not all(v for v in need(S3, "checks").values() if v is not None): sys.exit("STOP: S3 cross-checks not all true")
tab = need(S3, "robustness_table")
order = ["H2 Kruskal-Wallis", "H4 Wilcoxon exact", "H4 one-sample t", "H5 Stouffer",
         "H5 Mann-Whitney exact + Fisher", "H5 parametric bootstrap + Fisher"]
label = {"H2 Kruskal-Wallis": "Kruskal--Wallis", "H4 Wilcoxon exact": "Wilcoxon signed-rank, exact",
         "H4 one-sample t": "one-sample $t$", "H5 Stouffer": "Stouffer combination of the Welch tests",
         "H5 Mann-Whitney exact + Fisher": "exact Mann--Whitney per state, Fisher combination",
         "H5 parametric bootstrap + Fisher": "parametric bootstrap per state, Fisher combination"}
rows = []
for k in order:
    t = need(tab, k)
    rows.append(f"{t['hypothesis']} & {label[k]} & ${pv(t['original_p'])}$ & ${pv(t['p'])}$ & ${pv(t['p_holm'])}$ & "
                f"{'yes' if t['reject_at_0.05'] else 'no'} \\\\")
mac["RobNreject"] = str(sum(bool(need(tab, k, "reject_at_0.05")) for k in order))
mac["RobNtests"] = str(len(order))
h3 = need(S3, "H3")
for nm, key in (("CRone", "ci95_CR1"), ("CRtwo", "ci95_CR2"), ("Wild", "ci95_wild_cluster_bootstrap_t")):
    mac[f"Hthree{nm}Lo"], mac[f"Hthree{nm}Hi"] = f(h3[key][0], 3), f(h3[key][1], 3)
rob = ["\\begin{table}[!t]", "\\caption{Alternative tests of the confirmatory decisions (S3) and intervals for the H3 slope (S4). "
       "Each alternative $p$-value is Holm-adjusted together with the original $p$-values of the other two confirmatory "
       "hypotheses; the exact Mann--Whitney test with five runs per group has a minimum two-sided $p$ of $2/252$ per state.}",
       "\\label{tab:robustness}", "\\centering\\footnotesize", "\\begin{tabular}{lp{5.4cm}llll}", "\\toprule",
       "\\textbf{Hyp.} & \\textbf{Alternative test} & \\textbf{Original $p$} & \\textbf{$p$} & \\textbf{Holm $p$} & \\textbf{Reject} \\\\",
       "\\midrule", *rows, "\\midrule",
       f"H3 & slope $\\beta={f(h3['beta'], 3)}$, 95\\% interval: CR1 & \\multicolumn{{4}}{{l}}{{{mac['HthreeCRoneLo']}--{mac['HthreeCRoneHi']}}} \\\\",
       f" & CR2 & \\multicolumn{{4}}{{l}}{{{mac['HthreeCRtwoLo']}--{mac['HthreeCRtwoHi']}}} \\\\",
       f" & wild cluster bootstrap-$t$ & \\multicolumn{{4}}{{l}}{{{mac['HthreeWildLo']}--{mac['HthreeWildHi']}}} \\\\",
       "\\bottomrule", "\\end{tabular}", "\\end{table}"]

# ---------------- E1/S2: run variability, Gate 4, canary sizing, detectability, schedule ----------------
cells = need(E1, "cells")
for st, nm in (("R1", "Rone"), ("R2", "Rtwo")):
    d5 = need(cells, f"{st}-A1", "D5"); c1 = need(cells, f"{st}-A0", "C1")
    mac[f"SimDfive{nm}Mean"], mac[f"SimDfive{nm}Lo"], mac[f"SimDfive{nm}Hi"] = f(d5["mean"], 1), f(d5["p2.5"], 0), f(d5["p97.5"], 0)
    mac[f"SimCone{nm}Lo"], mac[f"SimCone{nm}Hi"] = f(100 * c1["p2.5"] / N, 3), f(100 * c1["p97.5"] / N, 3)
    g4 = need(E1, "gate4_vs_simulator", st)
    mac[f"Gsim{nm}P"], mac[f"Gsim{nm}Lo"], mac[f"Gsim{nm}Hi"] = f(g4["p_two_sided"], 4), f(g4["dbar_2.5"], 1), f(g4["dbar_97.5"], 1)
pc = need(E1, "observed_percentiles")
obs = {r: (IDENT[r]["rollout_state"], IDENT[r]["attacker"], v) for r, v in pc.items()}
if len(obs) != 40: sys.exit(f"STOP: expected 40 observed runs, found {len(obs)}")
for st, nm in (("R1", "Rone"), ("R2", "Rtwo")):
    c1 = [v["C1"] for s_, a_, v in obs.values() if s_ == st and a_ == "A0"]
    d5 = [v["D5"] for s_, a_, v in obs.values() if s_ == st and a_ == "A1"]
    if len(c1) != 5 or len(d5) != 5: sys.exit(f"STOP: {st} observed runs per cell")
    mac[f"PctCone{nm}Min"], mac[f"PctCone{nm}Max"] = f(100 * min(c1), 1), f(100 * max(c1), 1)
    mac[f"PctDfive{nm}Min"], mac[f"PctDfive{nm}Max"] = f(100 * min(d5), 1), f(100 * max(d5), 1)
det, cs = need(E1, "detectability"), need(E1, "canary_sizing_expected_gain")
ws = sorted(det, key=float)
can = ["\\begin{table}[!t]", "\\caption{Canary sizing and single-attempt detectability for the V1/V2 pair under P1. "
       "$\\mathbb{E}[\\Delta(n)]$: expected retry gain from Equation~\\eqref{eq:exposure-law}; detectability: share of "
       "10{,}000 simulated single-attempt runs with $C_r(1)\\le M_{\\mathrm{R0}}$.}", "\\label{tab:canary}",
       "\\centering\\footnotesize", "\\begin{tabular}{l" + "r" * len(ws) + "}", "\\toprule",
       "$w_2$ & " + " & ".join(f"{float(w):.2f}" for w in ws) + " \\\\", "\\midrule",
       "$\\mathbb{E}[\\Delta(2)]$ & " + " & ".join(f(cs[w]["2"], 1) for w in ws) + " \\\\",
       "$\\mathbb{E}[\\Delta(5)]$ & " + " & ".join(f(cs[w]["5"], 1) for w in ws) + " \\\\",
       "$\\mathbb{E}[\\Delta(10)]$ & " + " & ".join(f(cs[w]["10"], 1) for w in ws) + " \\\\",
       "$C_r(1)\\le M_{\\mathrm{R0}}$ & " + " & ".join(f"{100 * det[w]['share_runs_C1_le_M_R0']:.1f}\\%" for w in ws) + " \\\\",
       "\\bottomrule", "\\end{tabular}", "\\end{table}"]
for w in ws:
    same(f"detectability law gain {w}", det[w]["expected_retry_gain_n5"], cs[w]["5"], 1e-6)
for w, nm in (("0.01", "One"), ("0.10", "Ten")):
    mac[f"DetW{nm}"] = f(100 * need(det, w, "share_runs_C1_le_M_R0"), 1)
    mac[f"CanaryDfiveW{nm}"], mac[f"CanaryDtenW{nm}"] = f(need(cs, w, "5"), 1), f(need(cs, w, "10"), 1)
sch = need(E1, "schedule")
for sp, nm in (("(5, 0, 0)", "A"), ("(3, 2, 0)", "B"), ("(1, 2, 2)", "C")):
    mac[f"Sched{nm}"] = f(100 * need(sch, sp, "P_X12_reaches_V2"), 1)

# ---------------- E3: routing policies (simulated) and live verification ----------------
ec = need(E3, "cells")
for c in ("P2-R1-A1-one", "P2-R2-A1-one", "P3-R1-A1-one", "P3-R2-A1-one"):
    if need(ec, c, "D5", "max") != 0 or need(ec, c, "D5", "min") != 0: sys.exit(f"STOP: {c} has nonzero Delta(5)")
for st in ("R1", "R2"):
    if need(ec, f"P2-{st}-A1-multi", "D5", "mean") <= 0: sys.exit(f"STOP: P2-{st}-A1-multi mean Delta(5) not positive")
lv = need(EL, "runs")
mac["LiveValid"] = str(len(lv)); mac["LiveInvalid"] = str(len(need(EL, "invalid_runs")))
mac["LiveEqual"] = str(sum(1 for x in lv.values() if x["equals_prediction"]))
if not all(need(EL, "verification", k, "met") for k in ("V-P2-one", "V-P2-multi", "V-P3")): sys.exit("STOP: a verification criterion not met")
if need(EL, "findings_valid_runs_differing_from_prediction"): sys.exit("STOP: live findings present; text assumes none")
same("live valid runs", len(lv), 12); same("live equal", int(mac["LiveEqual"]), 12)
live = {}
for x in lv.values():
    st = x["cell"].split("-")[1]
    live.setdefault((x["policy"], x["attacker"], st), []).append(x["Delta5_live"])
pol_rows = []
spec = [("P1", "A1", "P1, A1", "e1"), ("P2", "A1-one", "P2, A1-one", "e3"), ("P2", "A1-multi", "P2, A1-multi", "e3"),
        ("P3", "A1-one", "P3, A1-one", "e3")]
for p, a, lab, src in spec:
    cellsrc = (lambda st: need(cells, f"{st}-A1")) if src == "e1" else (lambda st, p=p, a=a: need(ec, f"{p}-{st}-{a}"))
    parts = []
    for st in ("R1", "R2"):
        c = cellsrc(st)
        parts.append(f"{f(100 * c['C5']['mean'] / N, 3)} & {f(c['D5']['mean'], 1)} [{f(c['D5']['p2.5'], 0)}, {f(c['D5']['p97.5'], 0)}]")
        lvv = live.get((p, a, st))
        parts.append(", ".join(str(v) for v in sorted(lvv)) if lvv else "---")
    pol_rows.append(f"{lab} & " + " & ".join(parts) + " \\\\")
pol = ["\\begin{table}[!t]", "\\caption{Routing policies at the mixed states. Simulated: mean of $C(5)/n_m$ (\\%) and mean of "
       "$\\Delta(5)$ with its 2.5th and 97.5th percentiles over 10{,}000 identities per cell (P1 from the run-variability "
       "simulation). Live: $\\Delta_r(5)$ of the valid live runs, each equal to its frozen prediction; --- = no live cell.}",
       "\\label{tab:policies}", "\\centering\\footnotesize", "\\begin{tabular}{lrrrrrr}", "\\toprule",
       " & \\multicolumn{3}{c}{\\textbf{R1} ($w_2=0.1$)} & \\multicolumn{3}{c}{\\textbf{R2} ($w_2=0.5$)} \\\\",
       "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
       "\\textbf{Policy, attacker} & $C(5)/n_m$ & $\\Delta(5)$ & live $\\Delta_r(5)$ & $C(5)/n_m$ & $\\Delta(5)$ & live $\\Delta_r(5)$ \\\\",
       "\\midrule", *pol_rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"]

# ---------------- E4: attacker variants ----------------
lg = need(E4, "logs")
if any(need(lg, "gain_k1", st, "mean") != 0 for st in ("R0", "R1", "R2", "R3")): sys.exit("STOP: k = 1 gain not zero")
logged = {"gain_k2": {"R1": 13.2, "R2": 37.6}, "gain_k3": {"R1": 21.2, "R2": 53.2}, "gain_k5": {"R1": 33.8, "R2": 67.6}}
for k, v in logged.items():
    for st, x in v.items(): same(f"E4 {k} {st}", round(need(lg, k, st, "mean"), 2), x, 1e-9)
for k, nm in (("gain_k2", "Two"), ("gain_k3", "Three"), ("gain_k5", "Five")):
    mac[f"Ek{nm}Rone"], mac[f"Ek{nm}Rtwo"] = f(lg[k]["R1"]["mean"], 1), f(lg[k]["R2"]["mean"], 1)
es = [need(lg, "mean_attempts_early_stop", s, "mean") for s in ("R0", "R1", "R2", "R3")]
same("E4 early stop", [round(x, 3) for x in es], [4.932, 4.929, 4.926, 4.928])
mac["EarlyMin"], mac["EarlyMax"] = f(min(es), 3), f(max(es), 3)
nd = need(E4, "near_duplicate")
for k, x in (("unperturbed_samples", 806), ("unperturbed_XD", 5), ("variants", 191315), ("XD", 149),
             ("XD_samples_persisting_all5", 5), ("XD_samples_persisting_ge1", 34), ("new_disagreement_variants", 668),
             ("new_disagreement_samples", 567)):
    same(f"E4 near-duplicate {k}", nd[k], x)
mac["NdUnpert"], mac["NdUnpertXD"] = f(nd["unperturbed_samples"], 0), str(nd["unperturbed_XD"])
mac["NdVariants"] = f(nd["variants"], 0)
mac["NdPersistPct"] = f(100 * nd["XD_variant_persistence_share"], 1)
mac["NdPersistAll"], mac["NdPersistAny"] = str(nd["XD_samples_persisting_all5"]), str(nd["XD_samples_persisting_ge1"])
mac["NdNewVar"], mac["NdNewSamp"] = str(nd["new_disagreement_variants"]), str(nd["new_disagreement_samples"])
mac["NdNewOT"], mac["NdNewTO"] = str(need(nd, "new_disagreements_by_direction", "X12")), str(need(nd, "new_disagreements_by_direction", "X21"))
same("E4 decomposition R0", need(ED, "single_model_fresh_variant", "R0", "Delta5"), 831)
same("E4 decomposition R3", need(ED, "single_model_fresh_variant", "R3", "Delta5"), 867)
same("E4 decomposition R1 mean", round(need(ED, "mixed_states", "R1", "mean_Delta5"), 2), 838.57)
same("E4 decomposition R2 mean", round(need(ED, "mixed_states", "R2", "mean_Delta5"), 2), 861.55)
same("E4 persisting == unperturbed", need(ED, "persisting_all5_equals_unperturbed_XD"), True)
mac["FvRzero"], mac["FvRthree"] = f(ED["single_model_fresh_variant"]["R0"]["Delta5"], 0), f(ED["single_model_fresh_variant"]["R3"]["Delta5"], 0)
for st, nm in (("R1", "Rone"), ("R2", "Rtwo")):
    m = ED["mixed_states"][st]
    mac[f"Fv{nm}"], mac[f"Fv{nm}Lo"], mac[f"Fv{nm}Hi"] = f(m["mean_Delta5"], 1), f(m["p2.5"], 0), f(m["p97.5"], 0)
    mac[f"Fv{nm}IncZero"], mac[f"Fv{nm}IncThree"] = f(m["increment_vs_R0"], 1), f(m["increment_vs_R3"], 1)

cv = need(E4, "campaign_derived")
for att, nm in (("A0", "Az"), ("A1", "Ao")):
    fz = need(cv, f"R1-{att}", "Fuzzers")
    mac[f"CampFuzz{nm}One"], mac[f"CampFuzz{nm}Ten"] = f(100 * need(fz, "P_ge1_missed_m1"), 1), f(100 * need(fz, "P_ge1_missed_m10"), 1)
    mac[f"CampFuzz{nm}Fifty"] = f(100 * need(fz, "P_ge1_missed_m50"), 1)

# ---------------- E5: directional-disagreement gate ----------------
g = need(E5, "E5.1_directional_gate")
logged5 = {"UNSW-NB15": {"0.05%": (45, 48, 11434), "0.10%": (45, 48, 11434), "0.20%": (40, 48, 10620), "0.50%": (16, 48, 6130)},
           "CIC-IDS2017": {"0.05%": (3, 24, 19), "0.10%": (0, 24, 0), "0.20%": (0, 24, 0), "0.50%": (0, 24, 0)}}
for ds, v in logged5.items():
    for t, x in v.items():
        y = need(g, ds, t)
        same(f"E5 {ds} {t}", (y["pairs_blocked"], y["pairs_evaluated"], y["detections_forgone_total_X21"]), x)
for ds, nm in (("UNSW-NB15", "U"), ("CIC-IDS2017", "C")):
    for t, tn in (("0.05%", "A"), ("0.10%", "B"), ("0.20%", "C"), ("0.50%", "D")):
        mac[f"Gate{nm}{tn}Blk"] = str(g[ds][t]["pairs_blocked"])
        mac[f"Gate{nm}{tn}Fgn"] = f(g[ds][t]["detections_forgone_total_X21"], 0)
op = need(g, "original_pair")
same("E5 original pair", [op[f"blocked_at_{t}"] for t in ("0.05%", "0.10%", "0.20%", "0.50%")], [True, True, True, False])
mac["GateOrigShare"] = f(100 * op["X12_share"], 3)

# ---------------- write ----------------
OUT.mkdir(parents=True, exist_ok=True)
files = {
    "results_ext_values.tex": "% generated by revision/analysis/make_results_ext_values.py; do not edit\n" +
                              "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(mac.items())),
    "table_robustness.tex": "\n".join(rob) + "\n", "table_canary.tex": "\n".join(can) + "\n",
    "table_policies.tex": "\n".join(pol) + "\n"}
for name, text in files.items():
    (OUT / name).write_text(text)
prov = {"script_sha256": sha(__file__), "inputs_sha256": INPUTS,
        "outputs_sha256": {n: sha(OUT / n) for n in files}}
(OUT / "provenance.json").write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
for n, h in prov["outputs_sha256"].items(): print(h, n)
print("macros:", len(mac))
