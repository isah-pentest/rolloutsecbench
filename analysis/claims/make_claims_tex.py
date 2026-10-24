#!/usr/bin/env python3
"""
RolloutSecBench - manuscript claims: every number in the prose and figure captions, as LaTeX macros.

READ-ONLY. Each source is verified before use; nothing is typed by hand. The script also checks every
QUALITATIVE statement the prose makes (orderings, "zero", "within the band", "excludes", ...) and refuses
to write if the evidence contradicts one, so the text cannot drift from the data.

Sources and how each is verified
  freeze            data/manifests/phase9_v3/phase9_v3_freeze.json          its .sha256 sidecar
  X_m predictions   results/derived/malicious_predictions_strict.csv        freeze frozen_inputs_sha256
  analysis record   <analysis-dir>/per_run.csv, integrity_report.json       ANALYSIS_MANIFEST.sha256
  confirmatory      <confirmatory-dir>/h2_h5_results.json                   h2_h5_provenance.json outputs
  routing null      <routing-null-dir>/routing_null.json                    provenance.json outputs
  Table I data      <table1-dir>/table1_data.csv                            table1_provenance.json outputs
  split audit       results/derived/dataset_audit.json, split_manifest.json, feature_names.json
                                                                            table1_provenance.json inputs
  Table II data     <table2-dir>/table2_data.csv                            table2_provenance.json outputs

Outputs (--out-dir): claims_values.tex, claims_values.json, claims_provenance.json

Usage (repo root):
  .venv-blueprint-v2/bin/python analysis/claims/make_claims_tex.py \\
      --analysis-dir results/derived/phase9_v3/analysis_20261022T200702Z \\
      --out-dir results/derived/phase9_v3/claims
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
FREEZE = "data/manifests/phase9_v3/phase9_v3_freeze.json"
XM = "results/derived/malicious_predictions_strict.csv"
Z95 = 1.959963984540054
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
W_CONF = {"R0": 0.0, "R1": 0.1, "R2": 0.5, "R3": 1.0}


# ------------------------------------------------------------------ helpers
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stop(msg):
    print(f"REFUSING TO WRITE CLAIMS: {msg}", file=sys.stderr)
    sys.exit(2)


def read_manifest(p):
    out = {}
    for line in Path(p).read_text().splitlines():
        if line.strip():
            h, name = line.split(None, 1)
            out[name.strip()] = h
    return out


def verified(path, expected, label):
    if sha(path) != expected:
        stop(f"{label}: {path} does not match its recorded SHA-256")
    return path


def tci(xs):
    m = st.fmean(xs)
    s = st.stdev(xs)
    h = T975[len(xs) - 1] * s / math.sqrt(len(xs))
    return m, m - h, m + h


def wilson(k, n, z=Z95):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _betacf(a, b, x):
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return h


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def beta_ppf(q, a, b):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta_ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def law_C(n, w, both, x12, x21):
    return both + x12 * (1 - (1 - w) ** n) + x21 * (1 - w ** n)


# formatting
def I(x):
    return f"{int(x):,}".replace(",", "{,}")


def F(x, d):
    s = f"{x:.{d}f}"
    return "\\ensuremath{-}" + s[1:] if s.startswith("-") else s


def P(x, d):
    return F(100 * x, d)


def sig1(p):
    """one significant figure, plain decimal (e.g. 0.004, 0.0003)"""
    if p <= 0:
        return "0"
    e = math.floor(math.log10(p))
    return f"{round(p, -e):.{max(0, -e)}f}"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--confirmatory-dir", default="results/derived/phase9_v3/confirmatory")
    ap.add_argument("--routing-null-dir", default="results/derived/phase9_v3/exploratory/routing_null")
    ap.add_argument("--table1-dir", default="results/derived/phase9_v3/tables/table1")
    ap.add_argument("--table2-dir", default="results/derived/phase9_v3/tables/table2")
    ap.add_argument("--out-dir", default="results/derived/phase9_v3/claims")
    a = ap.parse_args()
    out = (ROOT / a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    M, checks, used = {}, [], {}

    def claim(ok, text):
        if not ok:
            stop(f"the prose states '{text}', but the evidence does not support it")
        checks.append(text)

    # ---------------- freeze
    fz = ROOT / FREEZE
    verified(fz, (fz.parent / (fz.name + ".sha256")).read_text().split()[0], "freeze")
    freeze = json.loads(fz.read_text())
    used[FREEZE] = sha(fz)
    d = freeze["design"]
    claim(d["required_valid_per_cell"] == 5, "five valid runs per cell")
    claim(d["attempts"] == {"A0": 1, "A1": 5}, "A0: one attempt; A1: five attempts per sample")
    claim(d["a1_attempt_policy"].startswith("all attempts always sent"), "all A1 attempts are sent")
    if "identities" in freeze:
        wset = {(i["rollout_state"], i["configured_v2_weight"]) for i in freeze["identities"]}
        claim(wset == set(W_CONF.items()), "configured V2 weights are R0 0, R1 0.1, R2 0.5, R3 1")
    else:
        stop("freeze has no identities list to confirm the configured weights")
    M["QPS"] = F(d["target_qps"], 0)
    M["QPStolPct"] = F(100 * d["qps_tolerance"], 0)
    am = freeze["amendment_A1_B5_L16"]
    m_q = re.match(r"\s*(\d+)\s*->\s*(\d+)", am["changes"]["fixed_load_qps"])
    m_b = re.match(r"\s*(\d+)\s*->\s*(\d+)", am["changes"]["a1_budget"])
    if not (m_q and m_b):
        stop("amendment A1-B5-L16 changes not in the expected 'old -> new' form")
    M["QPSold"], M["BudgetOld"] = m_q.group(1), m_b.group(1)
    claim(int(m_q.group(2)) == d["target_qps"] and int(m_b.group(2)) == d["attempts"]["A1"],
          "the amendment's new load and budget equal the frozen design")
    claim("HTTP-502" in am["rationale"], "the load was lowered after HTTP-502 transport failures")
    claim("H6 is deferred" in am["rationale"], "the amendment records H6 as deferred")
    claim("exceeds 0.999" in am["rationale"], "analytic power above 0.999 for H4/H5 at budget 5")
    claim("p>0.999" in freeze["gate4"]["superseded_rule"]["reason"],
          "under correct routing >=1 of 5 runs exceeds the superseded MDE with probability above 0.999")
    claim("alpha 0.05" in freeze["gate4"]["primary_rule"] and "Holm" in freeze["gate4"]["primary_rule"],
          "Gate-4 primary rule: Holm across R1, R2 at alpha 0.05")
    eps = [re.search(r"\|dp\|<\s*1e-(\d+)", v) for v in freeze["validity_criteria"]]
    eps = [e for e in eps if e]
    if len(eps) != 1:
        stop("serving-equivalence tolerance not found in the freeze validity criteria")
    M["EpsExp"] = eps[0].group(1)
    claim(any(v.startswith("P1") and "pod" in v.lower() for v in freeze["validity_criteria"]),
          "validity: pod identity and restart counts unchanged (P1)")
    claim(any(v.startswith("P2") and "protocol hash" in v and "configmap" in v and "env" in v
              for v in freeze["validity_criteria"]),
          "validity: router protocol hash, source configuration and deployment environment pinned (P2)")

    # ---------------- X_m predictions (H1)
    xm = ROOT / XM
    verified(xm, freeze["frozen_inputs_sha256"][XM], "frozen X_m predictions")
    used[XM] = sha(xm)
    cells, fam = Counter(), defaultdict(Counter)
    with open(xm, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["label"] != "1":
                continue
            v1, v2 = int(r["v1_prediction"]), int(r["v2_prediction"])
            c = {(1, 1): "dd", (1, 0): "x12", (0, 1): "x21", (0, 0): "mm"}[(v1, v2)]
            cells[c] += 1
            fam[r["attack_cat"]][c] += 1
    N, x12, x21, both = sum(cells.values()), cells["x12"], cells["x21"], cells["mm"]
    xd, m0, m3 = x12 + x21, both + x21, both + x12
    fsize = {f: sum(c.values()) for f, c in fam.items()}
    lo, hi = clopper_pearson(xd, N)
    M.update({"NXm": I(N), "NXot": I(x12), "NXto": I(x21), "NXD": I(xd), "NBoth": I(both), "NMzero": I(m0),
              "NMthree": I(m3), "NNet": I(x12 - x21), "NCeilSamples": I(both + xd),
              "PctXotOfXD": P(x12 / xd, 1), "MRDpct": P(xd / N, 3), "MRDlo": P(lo, 3), "MRDhi": P(hi, 3),
              "McNemarP": F(mcnemar_exact(x12, x21), 4), "BothPct": P(both / N, 2),
              "FloorLo": P(m0 / N, 3), "FloorHi": P(m3 / N, 3), "Ceil": P((both + xd) / N, 3)})
    claim(lo > 0, "the MRD interval excludes zero")
    claim(x12 > x21, "more samples only V2 misses than only V1 misses")
    nz = {f for f, c in fam.items() if c["x12"] + c["x21"] > 0}
    claim(nz == {"Fuzzers", "Exploits", "Analysis", "DoS"},
          "disagreement occurs only in Fuzzers, Exploits, Analysis and DoS")
    claim(fam["Analysis"]["x12"] == 0 and fam["Analysis"]["x21"] > 0, "all Analysis disagreements are X_{2->1}")
    claim(max(fsize, key=fsize.get) == "Generic", "Generic is the largest family")
    claim(max(nz, key=lambda f: fam[f]["x12"] + fam[f]["x21"]) == "Fuzzers", "Fuzzers contribute most disagreement")
    claim(max(fam, key=lambda f: fam[f]["mm"]) == "Fuzzers", "Fuzzers contribute most both-miss samples")
    claim("Worms" in fam and fam["Worms"]["x12"] + fam["Worms"]["x21"] == 0, "Worms has zero disagreement")
    fz_ = fam["Fuzzers"]
    M.update({"FuzzShareXm": P(fsize["Fuzzers"] / N, 1), "FuzzXD": I(fz_["x12"] + fz_["x21"]),
              "FuzzXDpct": P((fz_["x12"] + fz_["x21"]) / xd, 1), "FuzzRateOT": P(fz_["x12"] / fsize["Fuzzers"], 2),
              "FuzzRateTO": P(fz_["x21"] / fsize["Fuzzers"], 2),
              "ExplXD": I(fam["Exploits"]["x12"] + fam["Exploits"]["x21"]), "AnaXD": I(fam["Analysis"]["x21"]),
              "DosXD": I(fam["DoS"]["x12"] + fam["DoS"]["x21"]), "GenShareXm": P(fsize["Generic"] / N, 1),
              "WormsN": I(fsize["Worms"]), "WormsUB": P(wilson(0, fsize["Worms"])[1], 1),
              "FuzzBoth": I(fz_["mm"]), "FuzzBothPct": P(fz_["mm"] / both, 1),
              "FuzzBothRate": P(fz_["mm"] / fsize["Fuzzers"], 2)})

    # ---------------- analysis of record (rollout cells)
    adir = (ROOT / a.analysis_dir).resolve()
    man = read_manifest(adir / "ANALYSIS_MANIFEST.sha256")
    for name in ("per_run.csv", "integrity_report.json", "gate4.json"):
        verified(adir / name, man.get(name, ""), "analysis of record")
        used[str((adir / name).relative_to(ROOT))] = man[name]
    integ = json.loads((adir / "integrity_report.json").read_text())
    if not (integ.get("PASS") and integ.get("checksums_verified")):
        stop("analysis of record lacks a full integrity PASS")
    pop = integ["population"]
    claim((pop["N"], pop["X12_only_v2_misses"], pop["X21_only_v1_misses"], pop["both_miss"]) == (N, x12, x21, both),
          "the rollout population equals the frozen X_m counts")
    rows = list(csv.DictReader(open(adir / "per_run.csv")))
    K = d["attempts"]["A1"]
    by = defaultdict(list)
    for r in rows:
        by[(r["rollout_state"], r["attacker"])].append(r)
    M["NValid"] = I(len(rows))
    claim(len(rows) == 40, "40 valid runs")

    def y(r):
        return int(r["C1"] if r["attacker"] == "A0" else r[f"C{K}"])

    for s_, m_ in (("R0", m0), ("R3", m3)):
        for at in ("A0", "A1"):
            claim({y(r) for r in by[(s_, at)]} == {m_}, f"{s_}-{at}: every run equals the deterministic value")
    cm = {}
    for s_ in ("R1", "R2"):
        for at in ("A0", "A1"):
            cm[(s_, at)] = tci([y(r) / N for r in by[(s_, at)]])
    tag = {"R1": "Rone", "R2": "Rtwo"}
    for (s_, at), (m_, l_, h_) in cm.items():
        k = ("Az" if at == "A0" else "Ao") + tag[s_]
        M[k + "Mean"], M[k + "Lo"], M[k + "Hi"] = P(m_, 3), P(l_, 3), P(h_, 3)
    for s_ in ("R1", "R2"):
        g = cm[(s_, "A1")][0] - cm[(s_, "A0")][0]
        M["Gap" + tag[s_]], M["Gap" + tag[s_] + "Samples"] = P(g, 3), I(round(g * N))
    claim(cm[("R2", "A1")][0] - cm[("R2", "A0")][0] > cm[("R1", "A1")][0] - cm[("R1", "A0")][0] > 0,
          "the retry effect is positive at R1 and largest at R2")
    exp = {}
    for s_ in ("R1", "R2"):
        w = W_CONF[s_]
        exp[(s_, "A0")] = law_C(1, w, both, x12, x21)
        exp[(s_, "A1")] = law_C(K, w, both, x12, x21)
    M["ExpAoRtwo"] = P(exp[("R2", "A1")] / N, 3)
    M["ExpAzRone"] = P(exp[("R1", "A0")] / N, 3)
    M["ExpAoRone"] = P(exp[("R1", "A1")] / N, 3)
    az1 = N * cm[("R1", "A0")][0] - exp[("R1", "A0")]
    ao1 = N * cm[("R1", "A1")][0] - exp[("R1", "A1")]
    claim(az1 < 0 < ao1, "the two R1 departures have opposite signs (A0 below, A1 above)")
    M["AzRoneBelow"], M["AoRoneAbove"] = F(-az1, 1), F(ao1, 1)
    M["ExpGapRone"] = P((exp[("R1", "A1")] - exp[("R1", "A0")]) / N, 3)
    ratio_exp = (exp[("R1", "A1")] - exp[("R1", "A0")]) / (exp[("R2", "A1")] - exp[("R2", "A0")])
    ratio_obs = (cm[("R1", "A1")][0] - cm[("R1", "A0")][0]) / (cm[("R2", "A1")][0] - cm[("R2", "A0")][0])
    M["RetryRatioExp"], M["RetryRatioObs"] = F(ratio_exp, 2), F(ratio_obs, 2)
    claim(ratio_obs > ratio_exp, "the observed R1/R2 retry-effect ratio exceeds the law's (raised by the R1 departures)")
    claim(0.4 < ratio_exp < 0.6, "the law's R1/R2 retry-effect ratio is about one half")
    claim(cm[("R1", "A0")][1] <= m0 / N <= cm[("R1", "A0")][2], "the R1 single-attempt interval contains the R0 value")
    claim(cm[("R1", "A1")][1] > m0 / N, "the R1 retrying attacker's success is above the stable baseline")
    # exposure-law curve: R1 offset at every budget; R2 within the band at every budget
    off, inside = [], []
    for s_ in ("R1", "R2"):
        rr, w = by[(s_, "A1")], W_CONF[s_]
        for n in range(1, K + 1):
            q, rq = 1 - (1 - w) ** n, 1 - w ** n
            se = math.sqrt(x12 * q * (1 - q) + x21 * rq * (1 - rq)) / math.sqrt(len(rr))
            dv = st.fmean(int(r[f"C{n}"]) for r in rr) - law_C(n, w, both, x12, x21)
            (off if s_ == "R1" else inside).append(dv if s_ == "R1" else abs(dv) <= Z95 * se)
    claim(all(v > 0 for v in off), "at R1 the retry curve lies above the law at every budget")
    claim(all(inside), "at R2 the observed means lie within the 95% band of the law at every budget")
    M["OffRoneLo"], M["OffRoneHi"] = F(min(off), 1), F(max(off), 1)
    g4 = json.loads((adir / "gate4.json").read_text())
    w_r1 = st.fmean(r["w2_obs"] for r in g4["cells"]["R1-A0"]["runs"])
    cfgdiff = abs((x12 - x21) * (w_r1 - W_CONF["R1"]))
    claim(cfgdiff < 0.02, "at R1 the configured- and realized-share mixtures differ by less than 0.02 samples")
    M["GfourCfgDiffRone"] = F(cfgdiff, 3)
    r2a1 = by[("R2", "A1")]
    d5 = [int(r[f"C{K}"]) - int(r["C1"]) for r in r2a1]
    c1 = st.fmean(int(r["C1"]) for r in r2a1) - law_C(1, 0.5, both, x12, x21)
    M["DfiveRtwoMin"], M["DfiveRtwoMax"] = I(min(d5)), I(max(d5))
    M["AoRtwoOffOne"] = F(c1, 1)
    M["AoRtwoBelowCeil"] = F(both + xd - st.fmean(int(r[f"C{K}"]) for r in r2a1), 1)
    claim(c1 > 0, "the R2 retry runs start above the law at n = 1")
    mixed_a0 = [int(r["C1"]) for r in rows if r["attacker"] == "A0" and r["rollout_state"] in ("R1", "R2")]
    claim(min(mixed_a0) < m0, "some mixed-state single-attempt run falls below the all-V1 value")
    claim(all(both <= v <= both + xd for v in (y(r) for r in rows)), "every run lies between the floor and the ceiling")
    lo_run = min((r for r in rows if r["attacker"] == "A0" and r["rollout_state"] in ("R1", "R2")), key=lambda r: int(r["C1"]))
    M["MixMinPct"], M["MixMinState"] = P(int(lo_run["C1"]) / N, 3), lo_run["rollout_state"]

    # ---------------- registry: run accounting (pre-registered replacement policy)
    reg_p = ROOT / "results/raw/phase9_v3/registry.csv"
    verified(reg_p, man.get("registry.csv", ""), "registry")
    used["results/raw/phase9_v3/registry.csv"] = man["registry.csv"]
    reg = list(csv.DictReader(open(reg_p)))
    valid = [r for r in reg if r["disposition"] == "VALID"]
    invalid = [r for r in reg if r["disposition"] != "VALID"]
    claim({r["run_id"] for r in valid} == {r["run_id"] for r in rows}, "the VALID registry rows are exactly the analysed runs")
    claim(len({r["run_id"] for r in reg}) == len(reg), "each run identity appears once in the registry")
    res_valid = [r for r in valid if r["identity_type"] == "RESERVE"]
    replaced = {}
    for r in reg:
        if r["identity_type"] == "RESERVE":
            claim(r["activated_after"] in {x["run_id"] for x in invalid} and
                  r["activated_after"].rsplit("-", 1)[0] == r["run_id"].rsplit("-", 1)[0],
                  f"reserve {r['run_id']} was activated only after an INVALID run of the same cell")
            replaced[r["activated_after"]] = r["run_id"]
    claim(all(x["run_id"] in replaced for x in invalid), "every INVALID run was replaced by a reserve")
    for cell in {r["cell"] for r in reg}:
        orders = [int(r["reserve_order"]) for r in reg if r["cell"] == cell and r["identity_type"] == "RESERVE"]
        claim(orders == list(range(1, len(orders) + 1)), f"{cell}: reserves were activated in ascending order")
    claim(set(freeze.get("validity_never_uses", [])) >= {"ASR", "C(n)"}, "validity never uses scientific outcomes")
    M.update({"RegExecuted": I(len(reg)), "RegValid": I(len(valid)), "RegInvalid": I(len(invalid)),
              "RegReserves": I(len(res_valid)), "RegReservesNoun": "reserve" if len(res_valid) == 1 else "reserves", "RegPrimaryValid": I(len(valid) - len(res_valid))})

    def esc(t):
        t = t.replace("\\", "\x00")
        for a_, b_ in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"),
                       ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
                       ("$", r"\$")):
            t = t.replace(a_, b_)
        return t.replace("\x00", r"\textbackslash{}")
    acc = [r"% generated by make_claims_tex.py - do not edit by hand",
           r"\begin{table}[H]",
           r"\caption{Run accounting of the confirmatory campaign, from the append-only registry. "
           f"{len(reg)} identities were executed; {len(valid)} were VALID. Each INVALID run was replaced by the "
           r"next ordered reserve of the same cell, as pre-registered; validity never used scientific outcomes.}",
           r"\label{tab:runs}", r"\centering\small", r"\begin{adjustbox}{max width=\linewidth}",
           r"\begin{tabular}{llL{0.46\linewidth}l}", r"\toprule",
           r"\textbf{Run} & \textbf{Cell} & \textbf{Recorded reasons} & \textbf{Replaced by} \\", r"\midrule"]
    if invalid:
        for r in sorted(invalid, key=lambda x: x["run_id"]):
            acc.append(f"\\texttt{{{esc(r['run_id'])}}} & {esc(r['cell'])} & \\texttt{{{esc(r['reasons'] or '(none recorded)').replace(chr(92) + "_", chr(92) + "_" + chr(92) + "allowbreak{}")}}} & "
                       f"\\texttt{{{esc(replaced.get(r['run_id'], '---'))}}} \\\\")
    else:
        acc.append(r"\multicolumn{4}{l}{No INVALID runs.} \\")
    acc += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]

    # ---------------- deviation and execution-note records (pinned)
    REC = {"DEV-001": ("data/manifests/phase9_v3/deviations/DEV-001-transport.json",
                       "0c0e5cb52c17a19115b23b6d9b3a1342b3f1d2b689a8b5b4c01f08e8ec80f1d5"),
           "DEV-002": ("data/manifests/phase9_v3/deviations/DEV-002-runner-termination.json",
                       "d3ce3fcd9ea39473ef4cdb46c5bd5990a33bde7bf7977107263d078fa752804e"),
           "NOTE-001": ("data/manifests/phase9_v3/execution_notes/NOTE-001-host-standby.json",
                        "1e7c0f459cb837d495cd84327e0f8731ad92675ca7782c0d1ad3468f47bab980")}
    rec = {}
    for k, (rel, h) in REC.items():
        verified(ROOT / rel, h, f"{k} record")
        used[rel] = h
        rec[k] = json.loads((ROOT / rel).read_text())
    d1, d2, n1 = rec["DEV-001"], rec["DEV-002"], rec["NOTE-001"]
    claim(d1["freeze_sha256"] == d2["freeze_sha256"] == sha(fz), "both deviation records cite this freeze")
    reg_by = {r["run_id"]: r for r in reg}
    trig = d1["trigger"]["run_id"]
    claim(trig == d2["affected_run"]["run_id"] and reg_by.get(trig, {}).get("disposition") == "INVALID",
          "DEV-001 and DEV-002 concern the same run, recorded INVALID in the registry")
    claim(set(d2["affected_run"]["reasons"]) == {x.strip() for x in reg_by[trig]["reasons"].split(";") if x.strip()},
          "the DEV-002 reasons equal the registry reasons")
    res_name = re.search(r"V3-[A-Z0-9]+-[A-Z0-9]+-R\d+", d2["effect_on_outcomes"])
    claim(bool(res_name) and replaced.get(trig) == res_name.group(0) and res_name.group(0) in d1["trigger"]["handling"],
          "the reserve named in DEV-001 and DEV-002 is the registry replacement")
    claim(d2["corrects"].startswith("DEV-001 cause statement"), "DEV-002 corrects the cause stated in DEV-001")
    claim(d2["prevention"].startswith("runner launched detached from any terminal"),
          "DEV-002 prevention: the runner was relaunched detached from any terminal")
    claim(reg_by.get(n1["run_id"], {}).get("disposition") == "VALID", "the NOTE-001 run is VALID")
    claim("Disposition VALID stands" in n1["assessment"] and "All requests were sent and answered before standby began"
          in n1["assessment"], "NOTE-001: all requests answered before standby; disposition VALID stands")
    v2fail = [x for x in d2["affected_run"]["reasons"] if x.startswith("V2_transportx")]
    claim(len(v2fail) == 1, "DEV-002 records the transport-failure count")
    nreq = re.search(r"completed all ([\d,]+) scheduled requests", d2["corrects"])
    claim(bool(nreq) and int(nreq.group(1).replace(",", "")) == N * K, "the affected client completed its full schedule (n_m x 5)")
    port_old = re.search(r"127\.0\.0\.1:(\d+)", d1["change"]["before"]).group(1)
    port_new = re.search(r"127\.0\.0\.1:(\d+)", d1["change"]["after"]).group(1)
    claim("port-forward" in d1["change"]["before"] and "socat" in d1["change"]["after"] and "NodePort" in d1["change"]["after"],
          "DEV-001 replaces the kubectl port-forward with a socat/NodePort forwarder")
    claim("stratified by transport path" in d1["effect_on_outcomes"], "DEV-001 specifies latency stratified by transport path")
    for u in ("pinned code", "models, preprocessor, routing rule and basis", "population X_m, attempts, 16 QPS schedule"):
        claim(any(x.startswith(u) for x in d1["unchanged"]), f"DEV-001 leaves unchanged: {u}")

    def ts(x):
        try:
            return datetime.fromisoformat(x.replace("Z", "+00:00"))
        except Exception:
            stop(f"registry timestamp not ISO-8601: {x!r}")
    t_d1 = ts(d1["recorded_utc"])
    path_of = {}
    for r in rows:
        mport = re.search(r"127\.0\.0\.1:(\d+)", r["endpoint"])
        claim(bool(mport) and mport.group(1) in (port_old, port_new), f"{r['run_id']}: endpoint is one of the two DEV-001 paths")
        path_of[r["run_id"]] = mport.group(1)
        st_ = ts(reg_by[r["run_id"]]["started_utc"])
        claim((st_ < t_d1) if mport.group(1) == port_old else (st_ > t_d1),
              f"{r['run_id']}: transport path matches its start time relative to DEV-001")
    n_old = sum(v == port_old for v in path_of.values())
    M.update({"DevTrigRun": esc(trig), "DevReserveRun": esc(res_name.group(0)),
              "DevVtwoFails": I(int(v2fail[0].split("x")[-1])), "DevSchedReq": I(N * K),
              "NoteRun": esc(n1["run_id"]), "LatOldRuns": I(n_old), "LatNewRuns": I(len(rows) - n_old),
              "LatOldPort": port_old, "LatNewPort": port_new})
    PATH_LABEL = {port_old: "kubectl port-forward (before DEV-001)", port_new: "socat/NodePort forwarder (DEV-001)"}

    # ---------------- latency by transport path (descriptive)
    lat_p = adir / "latency_by_transport_path.csv"
    verified(lat_p, man.get("latency_by_transport_path.csv", ""), "latency summary")
    used[str(lat_p.relative_to(ROOT))] = man["latency_by_transport_path.csv"]
    lat = list(csv.DictReader(open(lat_p)))
    claim(sum(int(r["runs"]) for r in lat if r["subset"] == "all") == len(rows),
          "every valid run contributes to exactly one transport path")
    paths = sorted({r["endpoint"] for r in lat})
    M["LatPaths"] = I(len(paths))
    claim(len(paths) == 2, "runs used exactly two transport paths")
    lt = [r"% generated by make_claims_tex.py - do not edit by hand", r"\begin{table}[H]",
          r"\caption{Client-measured request latency by transport path. Each value is the median over the valid "
          r"runs on that path of the run's P50, P95 or P99 latency, for all requests and excluding the first 5\,s of "
          r"each run (warm-up sensitivity).}",
          r"\label{tab:latency}", r"\centering\small", r"\begin{adjustbox}{max width=\linewidth}",
          r"\begin{tabular}{llrrrr}", r"\toprule",
          r"\textbf{Transport path} & \textbf{Requests} & \textbf{Runs} & \textbf{P50 (ms)} & \textbf{P95 (ms)} & \textbf{P99 (ms)} \\",
          r"\midrule"]
    for r in sorted(lat, key=lambda x: (x["endpoint"], x["subset"])):
        sub = "all" if r["subset"] == "all" else "after first 5\\,s"
        lport = re.search(r"127\.0\.0\.1:(\d+)", r["endpoint"]).group(1)
        claim(int(r["runs"]) == sum(v == lport for v in path_of.values()),
              f"latency summary run count for :{lport} equals the per-run endpoint count")
        lt.append(f"{PATH_LABEL[lport]}, \\texttt{{:{lport}}} & {sub} & {r['runs']} & {float(r['median_of_run_p50']):.1f} & "
                  f"{float(r['median_of_run_p95']):.1f} & {float(r['median_of_run_p99']):.1f} \\\\")
    lt += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]

    # ---------------- confirmatory results
    cdir = (ROOT / a.confirmatory_dir).resolve()
    cprov = json.loads((cdir / "h2_h5_provenance.json").read_text())
    verified(cdir / "h2_h5_results.json", cprov["outputs_sha256"]["h2_h5_results.json"], "confirmatory results")
    used[str((cdir / "h2_h5_results.json").relative_to(ROOT))] = cprov["outputs_sha256"]["h2_h5_results.json"]
    cres = json.loads((cdir / "h2_h5_results.json").read_text())
    claim(cprov["inputs_sha256"]["per_run.csv"] == man["per_run.csv"],
          "the confirmatory results were computed from this analysis of record")
    hd = cres["H2"]["descriptive"]
    claim(hd["R1"]["mean_minus_mixture"] < 0 and hd["R2"]["mean_minus_mixture"] < 0,
          "both mixed single-attempt means lie below the configured mixture")
    M["HtwoMixRone"], M["HtwoMixRtwo"] = F(-N * hd["R1"]["mean_minus_mixture"], 1), F(-N * hd["R2"]["mean_minus_mixture"], 1)
    claim(all(cres["holm"]["reject"]), "all three confirmatory nulls are rejected after Holm")
    claim(cres["holm"]["alpha"] == 0.05 and cres["holm"]["family"] == ["H2", "H4", "H5"],
          "Holm over {H2, H4, H5} at family-wise alpha 0.05")
    eff = {(e["state"], e["n"]): e for e in csv.DictReader(open(cdir / "h4_effects.csv"))} \
        if (cdir / "h4_effects.csv").is_file() else {}
    if eff:
        verified(cdir / "h4_effects.csv", cprov["outputs_sha256"]["h4_effects.csv"], "H4 effects")
        e5 = eff[("R2", str(K))]
        lawv, lo5, hi5 = float(e5["law_expected_delta"]), float(e5["ci95_lo"]), float(e5["ci95_hi"])
        claim(not (lo5 <= lawv <= hi5) and lawv > hi5, "at R2 the Delta(5) interval excludes the law value, which lies above it")
        M["HfourRtwoGap"] = F(lawv - float(e5["mean_delta_samples"]), 1)
        e1 = eff[("R1", str(K))]
        claim(float(e1["ci95_lo"]) <= float(e1["law_expected_delta"]) <= float(e1["ci95_hi"]),
              "at R1 the Delta(5) interval contains the law value")
    else:
        stop("h4_effects.csv missing from the confirmatory folder")
    h3 = cres["H3_descriptive"]
    claim(h3["beta_ci95_cluster"][0] <= 1 <= h3["beta_ci95_cluster"][1], "the H3 slope interval contains one")

    # ---------------- routing null (exploratory)
    rdir = (ROOT / a.routing_null_dir).resolve()
    rprov = json.loads((rdir / "provenance.json").read_text())
    verified(rdir / "routing_null.json", rprov["outputs_sha256"]["routing_null.json"], "routing null")
    used[str((rdir / "routing_null.json").relative_to(ROOT))] = rprov["outputs_sha256"]["routing_null.json"]
    claim(rprov["per_run_sha256"] == man["per_run.csv"], "the routing null used this analysis of record")
    rn = json.loads((rdir / "routing_null.json").read_text())["summary"]
    rnm = json.loads((rdir / "routing_null.json").read_text())["m_per_cell"]
    M["RnM"] = I(rnm)
    M["RnPone"] = sig1(rn["R1-A0"]["p_two_sided_5run_mean"])
    M["RnPtwo"] = sig1(rn["R2-A1"][f"p_5run_sd_D{K}_as_small"])
    M["RnSDsingle"] = F(rn["R2-A1"][f"single_run_sd_D{K}"], 0)
    claim(abs(rn["R1-A0"]["single_run_sd_d"] - math.sqrt(x12 * .1 * .9 + x21 * .1 * .9)) < 0.15 * math.sqrt(x12 * .09 + x21 * .09),
          "the routing null reproduces the binomial variance at R1")
    claim(abs(rn["R2-A0"]["single_run_sd_d"] - math.sqrt(xd * .25)) < 0.15 * math.sqrt(xd * .25),
          "the routing null reproduces the binomial variance at R2")

    # ---------------- Table I data and split audit
    t1 = (ROOT / a.table1_dir).resolve()
    p1 = json.loads((t1 / "table1_provenance.json").read_text())
    verified(t1 / "table1_data.csv", p1["outputs_sha256"]["table1_data.csv"], "Table I data")
    used[str((t1 / "table1_data.csv").relative_to(ROOT))] = p1["outputs_sha256"]["table1_data.csv"]
    for rel in ("results/derived/dataset_audit.json", "data/manifests/split_manifest.json",
                "data/processed/feature_names.json"):
        verified(ROOT / rel, p1["inputs_sha256"][rel], "Table I input")
        used[rel] = p1["inputs_sha256"][rel]
    t1d = {r["quantity"]: r for r in csv.DictReader(open(t1 / "table1_data.csv"))}
    aud = json.loads((ROOT / "results/derived/dataset_audit.json").read_text())
    sm = json.loads((ROOT / "data/manifests/split_manifest.json").read_text())
    feats = json.loads((ROOT / "data/processed/feature_names.json").read_text())
    nfeat = len(feats["source_feature_order"])
    rep = aud["cross_split_overlap"]["representations"][sm["strict_test_policy"]["membership_representation"]]
    ival = lambda q, col: int(t1d[q][col])
    off_rows, strict_rows = ival("rows", "official_test_full"), ival("rows", "strict_test_nonoverlap")
    removed = ival("rows", "removed_by_strict_filter")
    claim(off_rows - strict_rows == removed, "official minus strict equals the removed rows")
    claim(ival("malicious", "strict_test_nonoverlap") == N, "the strict-test malicious count equals |X_m|")
    rem = {}
    for q in t1d:
        if q.startswith("family:"):
            f = q.split(":", 1)[1]
            rem[f] = ival(q, "removed_by_strict_filter") / ival(q, "official_test_full")
    rem_benign = ival("benign", "removed_by_strict_filter") / ival("benign", "official_test_full")
    order_rem = sorted(rem, key=rem.get, reverse=True)
    claim(order_rem[:2] == ["DoS", "Reconnaissance"], "the filter removes the largest shares of DoS and Reconnaissance")
    claim(rem["Fuzzers"] < 0.05 and rem_benign < 0.05, "Fuzzers and benign traffic are almost fully retained")
    sh = lambda col: ival("malicious", col) / ival("rows", col)
    claim(sh("strict_test_nonoverlap") < sh("final_train"), "the strict test is more balanced than the training data")
    dup = [ival("within_split_duplicate_vectors", c) / ival("rows", c)
           for c in ("final_train", "validation", "strict_test_nonoverlap", "official_test_full")]
    shared_strict = [int(t1d[f"shared_vectors:{x}|strict_test_nonoverlap"]["removed_by_strict_filter"])
                     for x in ("final_train", "validation")]
    claim(shared_strict and all(v == 0 for v in shared_strict), "the strict test shares no vector with train or validation")
    gen = sm.get("generator", {})
    M.update({"DOffRows": I(off_rows), "DRemoved": I(removed), "DRemovedPct": P(removed / off_rows, 1),
              "DShared": I(rep["confirmed_exact_unique_feature_vectors"]),
              "DConflict": I(rep["exact_vectors_with_different_label_sets"]), "DStrictRows": I(strict_rows),
              "DRemDos": P(rem["DoS"], 1), "DRemRecon": P(rem["Reconnaissance"], 1), "DRemFuzz": P(rem["Fuzzers"], 1),
              "DRemBenign": P(rem_benign, 1), "DStrictMal": P(sh("strict_test_nonoverlap"), 1),
              "DTrainMal": P(sh("final_train"), 1), "DDupLo": P(min(dup), 1), "DDupHi": P(max(dup), 1),
              "DFeat": I(nfeat), "DValPct": F(100 * gen["validation_fraction"], 0) if "validation_fraction" in gen else "20"})
    if "validation_fraction" not in gen:
        stop("split_manifest generator.validation_fraction not found")

    # ---------------- Table II data
    t2 = (ROOT / a.table2_dir).resolve()
    p2 = json.loads((t2 / "table2_provenance.json").read_text())
    verified(t2 / "table2_data.csv", p2["outputs_sha256"]["table2_data.csv"], "Table II data")
    used[str((t2 / "table2_data.csv").relative_to(ROOT))] = p2["outputs_sha256"]["table2_data.csv"]
    t2d = {(r["block"], r["quantity"]): r for r in csv.DictReader(open(t2 / "table2_data.csv"))}
    proto_rel = "data/manifests/model_revision_protocol_v1.0.json"
    verified(ROOT / proto_rel, p2["inputs_sha256"][proto_rel], "model revision protocol")
    used[proto_rel] = p2["inputs_sha256"][proto_rel]
    proto = json.loads((ROOT / proto_rel).read_text())
    thr = {r: proto[f"revision_{r}"]["threshold_policy"]["threshold"] for r in ("v1", "v2")}
    claim(thr["v1"] == thr["v2"], "both revisions use the same fixed decision threshold")
    M["MThreshold"] = f"{thr['v1']:g}"
    g = lambda blk, q, v: float(t2d[(blk, q)][v])
    metrics = ["precision", "recall", "f1", "auroc", "auprc", "fpr", "accuracy"]
    tdiff = {k: abs(g("strict_test", k, "v1") - g("strict_test", k, "v2")) for k in metrics}
    kmax = max(tdiff, key=tdiff.get)
    claim(int(g("strict_test", "fn", "v1")) == m0 and int(g("strict_test", "fn", "v2")) == m3,
          "strict-test missed attacks equal the frozen V1/V2 misses")
    claim(g("strict_test", "fp", "v2") < g("strict_test", "fp", "v1"), "V2 raises fewer false alarms than V1")
    claim(int(g("strict_test", "fp", "v1") - g("strict_test", "fp", "v2")) == xd,
          "the false-alarm difference coincides numerically with |X_D|")
    claim(min(g("strict_test", "recall", "v1"), g("strict_test", "recall", "v2")) > 0.98, "strict-test recall above 0.98")
    if ("gate1", "f1_min_required_v2") not in t2d or ("gate1", "auroc_min_required_v2") not in t2d:
        stop("table2_data.csv lacks the Gate-1 F1/AUROC minimum rows")
    fone_min, auc_min = g("gate1", "f1_min_required_v2", "v2"), g("gate1", "auroc_min_required_v2", "v2")
    claim(g("validation", "f1", "v2") >= fone_min and g("validation", "auroc", "v2") >= auc_min, "V2 passes Gate 1")
    M.update({"MvFa": F(g("validation", "f1", "v1"), 4), "MvFb": F(g("validation", "f1", "v2"), 4),
              "MvAUa": F(g("validation", "auroc", "v1"), 4), "MvAUb": F(g("validation", "auroc", "v2"), 4),
              "MFdiff": F(g("validation", "f1", "v1") - g("validation", "f1", "v2"), 4),
              "MFallow": F(g("validation", "f1", "v1") - fone_min, 2),
              "MAUdiff": F(g("validation", "auroc", "v1") - g("validation", "auroc", "v2"), 4),
              "MAUallow": F(g("validation", "auroc", "v1") - auc_min, 2),
              "MTestMaxDiff": F(tdiff[kmax], 4), "MTestMaxDiffMetric": {"fpr": "FPR", "f1": "F1", "auroc": "AUROC",
                                                                         "auprc": "AUPRC", "recall": "recall",
                                                                         "precision": "precision",
                                                                         "accuracy": "accuracy"}[kmax],
              "MFPa": I(g("strict_test", "fp", "v1")), "MFPb": I(g("strict_test", "fp", "v2")),
              "MFPdiff": I(g("strict_test", "fp", "v1") - g("strict_test", "fp", "v2")),
              "MTestFPR": F(st.fmean([g("strict_test", "fpr", v) for v in ("v1", "v2")]), 2),
              "MValFPR": F(st.fmean([g("validation", "fpr", v) for v in ("v1", "v2")]), 3),
              "MTestRecallMin": F(min(g("strict_test", "recall", "v1"), g("strict_test", "recall", "v2")), 4)})

    # ---------------- write
    names = sorted(M)
    for k in names:
        if not re.fullmatch(r"[A-Za-z]+", k):
            stop(f"macro name {k} is not letters-only")
    tex = ["% generated by make_claims_tex.py - do not edit by hand"] + \
          [f"\\newcommand{{\\{k}}}{{{M[k]}}}" for k in names]
    (out / "claims_values.tex").write_text("\n".join(tex) + "\n")
    (out / "run_accounting.tex").write_text("\n".join(acc) + "\n")
    (out / "latency_table.tex").write_text("\n".join(lt) + "\n")
    (out / "claims_values.json").write_text(json.dumps({"macros": M, "qualitative_checks": checks}, indent=2) + "\n")
    prov = {"generated_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__)),
            "python": platform.python_version(), "inputs_sha256": used, "qualitative_checks_passed": len(checks),
            "outputs_sha256": {p: sha(out / p) for p in ("claims_values.tex", "claims_values.json",
                                                          "run_accounting.tex", "latency_table.tex")}}
    (out / "claims_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    for c in checks:
        print("CHECK PASS", c)
    print(f"{len(names)} macros, {len(checks)} qualitative checks\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
