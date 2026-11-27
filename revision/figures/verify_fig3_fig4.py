#!/usr/bin/env python3
"""
RolloutSecBench - verification record for Figures 3 and 4 (retry curve; rollout x attacker interaction).

The figures are not redrawn. This script checks that the published figure outputs are exactly the values of the
analysis of record, establishes the facts stated in their captions, and preserves the published outputs in the
repository. It refuses (exit 2) on any mismatch and writes nothing in that case.

Checks
  1. analysis inputs  exposure_law.csv and per_run.csv equal ANALYSIS_MANIFEST.sha256; 5 valid runs in every cell.
  2. scripts          analysis/figures/make_fig3_exposure_law.py and make_fig5_interaction.py are tracked, unchanged
                      from HEAD and equal the pinned hashes; each figure's provenance names its script hash.
  3. outputs          every output listed in each provenance file is present with the recorded hash.
  4. Figure 3 band    sd_single_run = sqrt(|X12| q(1-q) + |X21| r(1-r)), q = 1-(1-w2)^n, r = 1-w2^n (independent
                      binomial routing of the disagreement samples); band = expected +/- 1.96 sd/sqrt(5).
  5. Figure 3 data    fig3_data.csv equals exposure_law.csv and per_run.csv; means and 95% t intervals
                      (t_0.975,4 = 2.776) recomputed from the runs.
  6. Figure 4 data    fig5_data.csv cell means and 95% t intervals recomputed from per_run.csv
                      (A0: C(1)/n_m, A1: C(5)/n_m); expected values from the exposure law.

Usage (repository root):
  python revision/figures/verify_fig3_fig4.py --fig3-dir ~/rsb-fig3 --fig4-dir ~/rsb-fig4
Writes revision/figures/fig3/ and revision/figures/fig4/ (copies of the published outputs) and
revision/figures/fig3_fig4_verification.json.
"""
import argparse, csv, hashlib, json, math, platform, shutil, statistics as st, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ANALYSIS = "results/derived/phase9_v3/analysis_20261022T200702Z"
SCRIPTS = {"fig3": ("analysis/figures/make_fig3_exposure_law.py",
                    "674be61601e9e66dfca6e81e2e87e6f9a7b53eabbab08a577510d401a341a8a1"),
           "fig4": ("analysis/figures/make_fig5_interaction.py",
                    "a79a0f53ce7f1d4b15e0f24f08cd261d044bce149f77ca2f21ba66bf564ed4dd")}
N_BOTH, N_12, N_21, N_M = 596, 92, 57, 38263
W = {"R0": 0.0, "R1": 0.1, "R2": 0.5, "R3": 1.0}
T975_DF4 = 2.776          # value used by both figure scripts for five runs
Z = 1.96                  # value used by analyze_v3.py for the band
TOL = 1e-9


class Refuse(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Refuse(msg)


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


def strings_in(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from strings_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings_in(v)


def tci(xs):
    m = st.fmean(xs)
    s = st.stdev(xs) if len(xs) > 1 else 0.0
    h = T975_DF4 * s / math.sqrt(len(xs))
    return m, m - h, m + h


def law_C(w, n):
    return N_BOTH + N_12 * (1 - (1 - w) ** n) + N_21 * (1 - w ** n)


def provenance_outputs(prov):
    """Map output file name -> sha256 from a provenance JSON (keys 'outputs' or 'outputs_sha256')."""
    for key in ("outputs_sha256", "outputs"):
        if isinstance(prov.get(key), dict):
            out = {}
            for k, v in prov[key].items():
                out[k] = v["sha256"] if isinstance(v, dict) else v
            return out
    raise Refuse("provenance has no 'outputs' or 'outputs_sha256' map")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--fig3-dir", type=Path, required=True)
    ap.add_argument("--fig4-dir", type=Path, required=True)
    a = ap.parse_args()
    repo = a.repo.resolve()
    dirs = {"fig3": a.fig3_dir.expanduser().resolve(), "fig4": a.fig4_dir.expanduser().resolve()}
    A = repo / ANALYSIS
    checks, hashes = [], {}
    try:
        # 1. analysis inputs
        man = {}
        for line in (A / "ANALYSIS_MANIFEST.sha256").read_text().splitlines():
            p = line.split(None, 1)
            if len(p) == 2:
                man[p[1].strip()] = p[0]
        for f in ("exposure_law.csv", "per_run.csv"):
            need(man.get(f) == sha(A / f), f"{f} differs from ANALYSIS_MANIFEST.sha256")
            hashes[f"{ANALYSIS}/{f}"] = man[f]
        checks.append("exposure_law.csv and per_run.csv equal ANALYSIS_MANIFEST.sha256")
        runs = {}
        with open(A / "per_run.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                runs.setdefault(r["cell"], []).append(r)
        cells = sorted(runs)
        need(cells == sorted(f"{s}-{t}" for s in W for t in ("A0", "A1")), f"unexpected cells {cells}")
        for c in cells:
            need(len(runs[c]) == 5, f"{c} has {len(runs[c])} valid runs, expected 5")
        for rs in runs.values():
            for r in rs:
                need(abs(int(float(r["C1"])) / float(r["ASR_C1"]) - N_M) < 1e-6 if float(r["ASR_C1"]) else True,
                     f"n_m implied by {r['run_id']} is not {N_M}")
        checks.append("5 valid runs in each of the 8 cells; n_m = 38263")

        # 2. scripts
        for fig, (rel, want) in SCRIPTS.items():
            need(git_clean(repo, rel), f"{rel} is not tracked or differs from HEAD")
            need(sha(repo / rel) == want, f"{rel} sha256 differs from pinned {want}")
            hashes[rel] = want
        checks.append("figure scripts tracked, unchanged from HEAD, equal pinned hashes")

        # 3. published outputs
        provs = {}
        for fig, d in dirs.items():
            pf = d / ("fig3_provenance.json" if fig == "fig3" else "fig5_provenance.json")
            need(pf.is_file(), f"{pf} not found")
            prov = json.loads(pf.read_text())
            need(SCRIPTS[fig][1] in set(strings_in(prov)), f"{pf.name} does not name script sha256 {SCRIPTS[fig][1]}")
            outs = provenance_outputs(prov)
            for name, h in outs.items():
                need((d / name).is_file(), f"{fig}: output {name} listed in provenance is missing from {d}")
                need(sha(d / name) == h, f"{fig}: {name} differs from its provenance hash")
            provs[fig] = (pf, outs)
        checks.append("every published output equals its provenance hash; provenance names the pinned script")

        # 4. Figure 3 band
        law = {}
        with open(A / "exposure_law.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                law[(r["state"], int(r["n"]))] = {k: float(v) for k, v in r.items() if k != "state"}
        for (s, n), L in law.items():
            w = W[s]
            q, rr = 1 - (1 - w) ** n, 1 - w ** n
            sd = math.sqrt(N_12 * q * (1 - q) + N_21 * rr * (1 - rr))
            need(abs(L["expected_C"] - law_C(w, n)) < TOL, f"expected C {s} n={n}")
            need(abs(L["sd_single_run"] - sd) < TOL, f"sd_single_run {s} n={n}: {L['sd_single_run']} vs binomial {sd}")
            need(abs(L["se_mean_of_runs"] - sd / math.sqrt(5)) < TOL, f"se {s} n={n}")
            need(abs(L["band95_lo"] - (L["expected_C"] - Z * sd / math.sqrt(5))) < TOL, f"band lo {s} n={n}")
            need(abs(L["band95_hi"] - (L["expected_C"] + Z * sd / math.sqrt(5))) < TOL, f"band hi {s} n={n}")
        checks.append("Figure 3 band = expected C(n) +/- 1.96*sd/sqrt(5), sd from independent binomial routing "
                      "of X12 and X21 (all 20 rows)")

        # 5. Figure 3 data
        with open(dirs["fig3"] / "fig3_data.csv", newline="") as fh:
            f3 = list(csv.DictReader(fh))
        need(len(f3) == 10, f"fig3_data.csv has {len(f3)} rows, expected 10")
        for row in f3:
            s, n = row["state"], int(row["n"])
            L = law[(s, n)]
            for k in ("expected_C", "band95_lo", "band95_hi", "observed_mean_C", "observed_ci95_lo", "observed_ci95_hi"):
                need(abs(float(row[k]) - L[k]) < TOL, f"fig3_data {s} n={n} {k}")
            m, lo, hi = tci([float(r[f"C{n}"]) for r in runs[f"{s}-A1"]])
            need(abs(m - L["observed_mean_C"]) < TOL and abs(lo - L["observed_ci95_lo"]) < TOL
                 and abs(hi - L["observed_ci95_hi"]) < TOL, f"t interval {s} n={n}")
            run_vals = sorted(int(float(row[r["run_id"]])) for r in runs[f"{s}-A1"])
            need(run_vals == sorted(int(float(r[f"C{n}"])) for r in runs[f"{s}-A1"]), f"fig3 run values {s} n={n}")
        checks.append("fig3_data.csv equals exposure_law.csv and per_run.csv; 95% t intervals reproduced (t=2.776)")

        # 6. Figure 4 data
        with open(dirs["fig4"] / "fig5_data.csv", newline="") as fh:
            f4 = list(csv.DictReader(fh))
        cell_rows = [r for r in f4 if r.get("ci95_lo", "") not in ("", None) and not r.get("run_id")]
        levels = sorted({r.get("level", "") for r in f4})
        need(len(cell_rows) == 8, f"fig5_data.csv: found {len(cell_rows)} cell rows (levels {levels}), expected 8")
        f4_summary = {}
        for row in cell_rows:
            at, s = row["attacker"], row["state"]
            n_att = 1 if at == "A0" else 5
            vals = [float(r[f"ASR_C{n_att}"]) for r in runs[f"{s}-{at}"]]
            m, lo, hi = tci(vals)
            exp = law_C(W[s], n_att) / N_M
            for k, v in (("value", m), ("ci95_lo", lo), ("ci95_hi", hi), ("expected_law", exp)):
                need(abs(float(row[k]) - v) < TOL, f"fig5_data {at} {s} {k}: {row[k]} vs recomputed {v}")
            f4_summary[f"{s}-{at}"] = {"outcome": f"C({n_att})/n_m", "n_runs": len(vals), "mean_pct": 100 * m,
                                       "ci95_pct": [100 * lo, 100 * hi], "expected_pct": 100 * exp}
        checks.append("fig5_data.csv cell means, 95% t intervals and exposure-law values reproduced from per_run.csv")
    except Refuse as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return 2

    # preserve the published outputs in the repository
    copied = {}
    for fig, (pf, outs) in provs.items():
        dest = repo / "revision/figures" / fig
        dest.mkdir(parents=True, exist_ok=True)
        for name in list(outs) + [pf.name]:
            shutil.copy2(dirs[fig] / name, dest / name)
            copied[f"revision/figures/{fig}/{name}"] = sha(dest / name)
    rec = {
        "artifact": "RolloutSecBench",
        "record": "Figures 3 and 4 - verification of published outputs and caption facts (figures not redrawn)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": {"path": "revision/figures/verify_fig3_fig4.py", "sha256": sha(Path(__file__))},
        "python": platform.python_version(),
        "analysis_of_record": ANALYSIS,
        "inputs": hashes,
        "checks_passed": checks,
        "figure3": {"script": SCRIPTS["fig3"][0], "runs_per_state": 5, "observed_interval": "95% t, t_0.975,4 = 2.776",
                    "band": "expected C(n) +/- 1.96*sd/sqrt(5); sd^2 = |X12| q(1-q) + |X21| r(1-r), "
                            "q = 1-(1-w2)^n, r = 1-w2^n (normal approximation, independent binomial routing)",
                    "ceiling": N_BOTH + N_12 + N_21},
        "figure4": {"script": SCRIPTS["fig4"][0], "cells": f4_summary,
                    "interval": "95% t over 5 runs per cell, t_0.975,4 = 2.776"},
        "preserved_outputs": copied,
    }
    out = repo / "revision/figures/fig3_fig4_verification.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print("OK: " + str(len(checks)) + " checks passed")
    for c in checks:
        print("  -", c)
    for p, h in copied.items():
        print(f"preserved {p}  {h}")
    print(f"wrote {out}  {sha(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
