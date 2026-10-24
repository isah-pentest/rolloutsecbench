#!/usr/bin/env python3
"""
RolloutSecBench - Table II: V1/V2 training setup, static performance, and Gate-1 acceptance.

Nothing is hard-coded. Standard library only (no scikit-learn / xgboost needed).

Reads (read-only)
  data/manifests/model_revision_protocol_v1.0.json (+ .sha256 sidecar)   hyperparameters, thresholds, acceptance rules
  models/v1/model_manifest.json, models/v2/model_manifest.json           model hashes, environment, validation metrics
  models/v1/model.json, models/v2/model.json                             re-hashed against the manifests
  results/derived/model_revision_acceptance.json                         Gate-1 decision (validation only)
  results/derived/static_model_metrics_strict.json                       frozen strict-test metrics
  results/derived/pairwise_predictions_strict.csv                        row-level strict-test predictions (re-scored here)
  data/manifests/phase9_v3/phase9_v3_freeze.json                         frozen hash of the pairwise predictions

Checks before writing (any failure -> exit 2)
  * protocol hash = sidecar = hash referenced by both model manifests and the acceptance report
  * model files hash to the manifest values; hyperparameters and thresholds in the manifests equal the protocol
  * validation metrics in the manifests equal the acceptance report
  * pairwise predictions hash to the freeze value; every stored prediction is reproduced from its probability at the
    frozen threshold; confusion counts, precision, recall, F1, FPR, accuracy, AUROC and AUPRC recomputed from the
    rows equal static_model_metrics_strict.json (counts exactly, rates within 1e-9)
  * Gate-1 recomputed from the protocol rules and validation metrics equals the acceptance report
Reported, not blocking: model-manifest hashes that differ from those recorded in the acceptance report.

Outputs (--out-dir): table2_models.tex, table2_data.csv, table2_provenance.json
Usage (repo root):
  .venv-blueprint-v2/bin/python analysis/tables/make_table2_models.py --out-dir results/derived/phase9_v3/tables/table2
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
PROTO = ROOT / "data/manifests/model_revision_protocol_v1.0.json"
SIDECAR = ROOT / "data/manifests/model_revision_protocol_v1.0.sha256"
MAN = {"v1": ROOT / "models/v1/model_manifest.json", "v2": ROOT / "models/v2/model_manifest.json"}
ACC = ROOT / "results/derived/model_revision_acceptance.json"
STATIC = ROOT / "results/derived/static_model_metrics_strict.json"
PAIR = ROOT / "results/derived/pairwise_predictions_strict.csv"
FREEZE = ROOT / "data/manifests/phase9_v3/phase9_v3_freeze.json"
METRICS = ["precision", "recall", "f1", "auroc", "auprc", "fpr", "accuracy"]
LABEL = {"precision": "Precision", "recall": "Recall (TPR)", "f1": "F1", "auroc": "AUROC", "auprc": "AUPRC",
         "fpr": "FPR", "accuracy": "Accuracy"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stop(msg):
    print(f"REFUSING TO WRITE TABLE II: {msg}", file=sys.stderr)
    sys.exit(2)


def tex(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def n(x):
    return f"{x:,}".replace(",", "{,}")


def auroc(y, s):
    """Mann-Whitney AUROC with ties counted as 1/2 (identical to the trapezoidal ROC area)."""
    pairs = sorted(zip(s, y))
    pos = sum(y)
    neg = len(y) - pos
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        rank_sum += avg_rank * sum(1 for k in range(i, j + 1) if pairs[k][1] == 1)
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def auprc(y, s):
    """Average precision (step-wise, ties grouped): sum over thresholds of (R_k - R_{k-1}) * P_k."""
    pairs = sorted(zip(s, y), key=lambda t: -t[0])
    pos = sum(y)
    tp = fp = 0
    ap, prev_r, i = 0.0, 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            tp += pairs[j][1]
            fp += 1 - pairs[j][1]
            j += 1
        r = tp / pos
        ap += (r - prev_r) * (tp / (tp + fp))
        prev_r, i = r, j
    return ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/derived/phase9_v3/tables/table2")
    a = ap.parse_args()
    out = (ROOT / a.out_dir).resolve()
    checks, notes = [], []

    proto = json.loads(PROTO.read_text())
    proto_sha = sha(PROTO)
    if SIDECAR.read_text().split()[0] != proto_sha:
        stop("revision protocol hash differs from its sidecar")
    man = {r: json.loads(p.read_text()) for r, p in MAN.items()}
    acc = json.loads(ACC.read_text())
    static = json.loads(STATIC.read_text())
    if man["v1"]["protocol"]["sha256"] != proto_sha or man["v2"]["protocol"]["sha256"] != proto_sha \
            or acc["frozen_protocol"]["sha256"] != proto_sha:
        stop("model manifests / acceptance report reference a different revision protocol")
    checks.append("revision protocol hash = sidecar = model manifests = acceptance report")

    # model files and hyperparameters
    hp = {}
    for r in ("v1", "v2"):
        m = man[r]
        art = m["artifacts"]["model"]
        if sha(ROOT / art["path"]) != art["sha256"]:
            stop(f"{art['path']} hash differs from its manifest")
        h_m = (m.get("training") or {}).get("hyperparameters")
        h_p = proto[f"revision_{r}"]["hyperparameters"]
        if h_m != h_p:
            stop(f"{r}: manifest hyperparameters differ from the frozen protocol")
        hp[r] = h_p
        thr_p = proto[f"revision_{r}"]["threshold_policy"]
        thr_m = (m.get("training") or {}).get("threshold_policy") or (m.get("decision_policy") or {}).get("threshold_policy")
        if thr_m != thr_p:
            stop(f"{r}: manifest threshold policy differs from the frozen protocol")
        rec = acc["input_manifests"][r]["sha256"]
        if sha(MAN[r]) != rec:
            notes.append(f"{MAN[r].relative_to(ROOT)} hash {sha(MAN[r])[:16]} differs from the version recorded in the "
                         f"acceptance report ({rec[:16]}); the model file and its metrics are unchanged.")
    checks.append("model files match manifests; hyperparameters and threshold policies equal the frozen protocol")

    # validation metrics consistency
    val = {}
    for r in ("v1", "v2"):
        vm = (man[r].get("validation") or {}).get("metrics") or man[r].get("validation_metrics")
        if vm != acc["validation_metrics"][r]:
            stop(f"{r}: validation metrics in the manifest differ from the acceptance report")
        val[r] = vm
    checks.append("validation metrics in the model manifests equal the acceptance report")

    # Gate 1 recomputed
    pol = proto["model_acceptance_policy"]["v2"]["relative_noninferiority_requirements"]
    g1 = {}
    for k, rule in pol.items():
        need = val["v1"][k] - rule["absolute_tolerance"]
        g1[k] = {"v1": val["v1"][k], "v2": val["v2"][k], "tol": rule["absolute_tolerance"], "min": need,
                 "passed": val["v2"][k] >= need}
        rp = acc["prospective_acceptance_policy"][k]
        if rp["passed"] != g1[k]["passed"] or abs(rp["minimum_required_v2"] - need) > 1e-12:
            stop(f"Gate-1 {k} recomputation disagrees with the acceptance report")
    gr = acc["gate_result"]
    if gr["status"] != ("PASS" if all(x["passed"] for x in g1.values()) and gr["v1_core_checks_passed"]
                        and gr["v2_core_checks_passed"] else "FAIL"):
        stop("Gate-1 status disagrees with its components")
    checks.append(f"Gate 1 recomputed from protocol rules and validation metrics: {gr['status']}")

    # strict test: re-score row-level predictions
    freeze = json.loads(FREEZE.read_text())
    if sha(PAIR) != freeze["frozen_inputs_sha256"]["results/derived/pairwise_predictions_strict.csv"]:
        stop("pairwise_predictions_strict.csv differs from the frozen hash")
    if static["evaluation_set"]["sha256"] != freeze["frozen_inputs_sha256"]["data/splits/strict_test_nonoverlap.csv"]:
        stop("static metrics refer to a different strict test file")
    y, s, p = [], {"v1": [], "v2": []}, {"v1": [], "v2": []}
    with open(PAIR, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            y.append(int(row["label"]))
            for r in ("v1", "v2"):
                s[r].append(float(row[f"{r}_probability"]))
                p[r].append(int(row[f"{r}_prediction"]))
    if len(y) != static["evaluation_set"]["rows"]:
        stop("row count differs from static metrics")
    test, rule_used = {}, {}
    for r in ("v1", "v2"):
        thr = static["decision_thresholds"][r]
        if thr != proto[f"revision_{r}"]["threshold_policy"]["threshold"]:
            stop(f"{r}: static-evaluation threshold differs from the protocol")
        ge = all((sc >= thr) == bool(pr) for sc, pr in zip(s[r], p[r]))
        gt = all((sc > thr) == bool(pr) for sc, pr in zip(s[r], p[r]))
        if not (ge or gt):
            stop(f"{r}: stored predictions are not reproducible from probabilities at threshold {thr}")
        rule_used[r] = ">=" if ge else ">"
        tp = sum(1 for yi, pi in zip(y, p[r]) if yi == 1 and pi == 1)
        fp = sum(1 for yi, pi in zip(y, p[r]) if yi == 0 and pi == 1)
        tn = sum(1 for yi, pi in zip(y, p[r]) if yi == 0 and pi == 0)
        fn = sum(1 for yi, pi in zip(y, p[r]) if yi == 1 and pi == 0)
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        mt = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": prec, "recall": rec,
              "f1": 2 * prec * rec / (prec + rec), "fpr": fp / (fp + tn), "accuracy": (tp + tn) / len(y),
              "auroc": auroc(y, s[r]), "auprc": auprc(y, s[r])}
        fz = static["metrics"][r]
        for k in ("tp", "fp", "tn", "fn"):
            if mt[k] != fz[k]:
                stop(f"{r}: recomputed {k} {mt[k]} != frozen {fz[k]}")
        for k in METRICS:
            if abs(mt[k] - fz[k]) > 1e-9:
                stop(f"{r}: recomputed {k} {mt[k]:.12f} != frozen {fz[k]:.12f}")
        test[r] = mt
    checks.append("strict-test confusion counts and all seven metrics recomputed from row-level predictions "
                  "equal static_model_metrics_strict.json")
    official = ROOT / "results/derived/static_model_metrics_official.json"
    if not official.exists():
        notes.append("Official-test (secondary) static metrics are not present in the frozen records "
                     "(no static_model_metrics_official*.json); that column is reported as not computed.")

    # ---------------- outputs
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "table2_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "quantity", "v1", "v2"])
        for k in hp["v1"]:
            w.writerow(["hyperparameter", k, hp["v1"][k], hp["v2"][k]])
        for k in METRICS + ["tp", "fp", "tn", "fn"]:
            w.writerow(["validation", k, val["v1"][k], val["v2"][k]])
        for k in METRICS + ["tp", "fp", "tn", "fn"]:
            w.writerow(["strict_test", k, test["v1"][k], test["v2"][k]])
        for k, g in g1.items():
            w.writerow(["gate1", f"{k}_min_required_v2", "", g["min"]])

    env = man["v1"]["environment"]
    show_hp = [k for k in hp["v1"] if k not in ("objective", "eval_metric", "tree_method", "random_state", "n_jobs")]
    L = [r"% Generated by analysis/tables/make_table2_models.py - do not edit by hand",
         r"\begin{table}[H]",
         r"\caption{Detector revisions: training setup, static performance and Gate-1 acceptance. Both revisions are "
         + tex(proto["scientific_objective"]["model_family"]) + r"s trained on the final training split with the frozen "
         r"preprocessor; V2 is a pre-declared change of complexity, learning rate, ensemble size, sampling and "
         r"regularization. Acceptance was decided on the validation split only; test metrics did not influence it.}",
         r"\label{tab:models}", r"\centering\small",
         r"\begin{tabular}{lcc|cc}", r"\toprule",
         r"\multicolumn{5}{l}{\textbf{(a) Training setup}} \\", r"\midrule",
         r" & \textbf{V1} & \textbf{V2} & & \\",
         r"Role & baseline & pre-declared revision & & \\"]
    for k in show_hp:
        L.append(f"\\texttt{{{tex(k)}}} & {hp['v1'][k]} & {hp['v2'][k]} & & \\\\")
    sp = proto["shared_training_policy"]
    L.append(f"Shared & \\multicolumn{{4}}{{l}}{{\\texttt{{{tex(sp['objective'])}}}, \\texttt{{{tex(sp['tree_method'])}}}, "
             f"seed {sp['random_seed']}; XGBoost {tex(env.get('xgboost', '?'))}; no hyperparameter search}} \\\\")
    L.append(f"Decision threshold & {proto['revision_v1']['threshold_policy']['threshold']} (fixed) & "
             f"{proto['revision_v2']['threshold_policy']['threshold']} (fixed) & & \\\\")
    L += [r"\midrule",
          r"\multicolumn{5}{l}{\textbf{(b) Static performance}} \\", r"\midrule",
          r" & \multicolumn{2}{c|}{\textbf{Validation} (" + n(val["v1"]["tp"] + val["v1"]["fp"] + val["v1"]["tn"] + val["v1"]["fn"])
          + r")} & \multicolumn{2}{c}{\textbf{Strict test} (" + n(len(y)) + r")} \\",
          r" & V1 & V2 & V1 & V2 \\"]
    for k in METRICS:
        L.append(f"{LABEL[k]} & {val['v1'][k]:.4f} & {val['v2'][k]:.4f} & {test['v1'][k]:.4f} & {test['v2'][k]:.4f} \\\\")
    L.append(f"FN (missed attacks) & {n(val['v1']['fn'])} & {n(val['v2']['fn'])} & {n(test['v1']['fn'])} & {n(test['v2']['fn'])} \\\\")
    L.append(f"FP & {n(val['v1']['fp'])} & {n(val['v2']['fp'])} & {n(test['v1']['fp'])} & {n(test['v2']['fp'])} \\\\")
    L += [r"\midrule", r"\multicolumn{5}{l}{\textbf{(c) Gate 1 (validation only)}} \\", r"\midrule",
          r"Core checks$^{a}$ & "
          + ("pass" if gr["v1_core_checks_passed"] else "fail") + " & " + ("pass" if gr["v2_core_checks_passed"] else "fail")
          + r" & & \\"]
    for k, g in g1.items():
        L.append(f"{k.upper() if k != 'f1' else 'F1'} ($\\geq$ V1 $-$ {g['tol']}) & "
                 f"{g['v1']:.4f} & {g['v2']:.4f} $\\geq$ {g['min']:.4f} & \\multicolumn{{2}}{{l}}{{"
                 f"{'pass' if g['passed'] else 'fail'}}} \\\\")
    L.append(f"\\textbf{{Gate 1}} & \\multicolumn{{4}}{{l}}{{\\textbf{{{tex(gr['status'])}}}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    L.append(r"\par\smallskip\footnotesize $^{a}$ " + tex("; ".join(x.rstrip(".") for x in proto["model_acceptance_policy"]["v1"]["requirements"]) + "."))
    if notes:
        L.append(r"\par\footnotesize " + " ".join(tex(x) for x in notes))
    L.append(r"\end{table}")
    (out / "table2_models.tex").write_text("\n".join(L) + "\n")

    prov = {"table": "Table II", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": sha(Path(__file__)), "python": platform.python_version(),
            "checks_passed": checks, "notes": notes, "prediction_rule": rule_used,
            "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                              [PROTO, MAN["v1"], MAN["v2"], ACC, STATIC, PAIR, ROOT / man["v1"]["artifacts"]["model"]["path"],
                               ROOT / man["v2"]["artifacts"]["model"]["path"]]},
            "metric_definitions": {"auroc": "Mann-Whitney U with ties = 1/2", "auprc": "average precision, ties grouped"},
            "outputs_sha256": {q.name: sha(q) for q in (out / "table2_models.tex", out / "table2_data.csv")}}
    (out / "table2_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    for c in checks:
        print("OK:", c)
    for x in notes:
        print("NOTE:", x)
    print(f"wrote {out}/table2_models.tex, table2_data.csv, table2_provenance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
