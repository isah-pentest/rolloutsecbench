# Reproducing the extension analyses (release v1.1.0)

The extension analyses read the frozen artifacts of the confirmatory campaign (evidence archive version 1) and write their outputs under `revision/results/`. Their outputs and the raw evidence of the live routing-policy runs are in the extension evidence archive (`rolloutsecbench-evidence-ext-v1.1.tar.gz`). Every script checks its inputs against the hashes in `revision/manifests/` or in the frozen manifests and stops on a mismatch.

## Level 2: regenerate the reported values from the archived outputs

From the repository root, with both archives unpacked into it:

```
sha256sum -c --quiet <(grep -v '^#' ARCHIVE_MANIFEST.sha256)
sha256sum -c --quiet <(grep -v '^#' EXTENSION_MANIFEST.sha256)
python3 revision/analysis/make_results_ext_values.py
sha256sum -c revision/manifests/phase9_results_values.sha256
```

`make_results_ext_values.py` uses the Python standard library only. It writes `revision/results/manuscript_values/` (`results_ext_values.tex`, `table_robustness.tex`, `table_canary.tex`, `table_policies.tex`, `provenance.json`); the last command confirms that they are byte-identical to the files the paper was compiled with. `release/cleanroom_test.sh --archive-ext` runs these steps from a fresh copy.

## Level 3: re-run an analysis

Re-running writes new outputs; compare them with the archived files and their manifests. The revision environment is `revision/manifests/env_rsb_revision.lock` (training stack plus LightGBM 4.7.0). The E2 and S1 analyses also need the UNSW-NB15 splits, regenerated with `training/generate_splits.py`; the CIC-IDS2017 pairs need the corrected release from its providers, prepared with `revision/analysis/cic_prepare.py`.

| Analysis | Script | Outputs (`revision/results/`) | Manifest (`revision/manifests/`) |
|---|---|---|---|
| Simulator validation (40 runs) | `revision/simulator/validate_simulator.py` | `e1_validation/` | hash of `validation_report.json` pinned in `s2_e1_distributions.py` |
| S1 | `revision/analysis/s1_duplicates.py` | `s1_duplicates/` | `phase4_s1.sha256` |
| S2 / E1 | `revision/analysis/s2_e1_distributions.py` | `e1_distributions/` | `phase4_s2.sha256` |
| S3 / S4 | `revision/analysis/s3_s4_robustness.py` | `s3_s4_robustness/` | `phase4_s3_s4.sha256` |
| E2, UNSW-NB15 | `revision/analysis/e2_generalization.py`, `e2_report.py` | `e2/` | `phase5_e2.sha256` |
| CIC-IDS2017 preparation | `revision/analysis/cic_prepare.py` | `cic/` | `cic/phase5b_prepare.sha256` |
| E2, CIC-IDS2017 | `revision/analysis/e2_cic.py` | `e2_cic/` | `cic/phase5b_e2_cic.sha256` |
| E3 identities | `revision/analysis/e3_freeze.py` | `revision/prereg/E3_FREEZE.json`, `E3_SIM_IDENTITIES.txt` | hash pinned in `e3_simulate.py` and `make_ext1_live_freeze.py` |
| E3, simulated | `revision/analysis/e3_simulate.py` | `e3_sim/` | `phase6_e3_sim.sha256` |
| EXT1 live runs | `workloads/ext1_campaign.py` (with `ext1_client.py`, `validate_ext1_run.py`, `make_ext1_live_freeze.py`, `make_ext1_k8s.py`, `ext1_serving_gate.py`) | `results/raw/ext1/` (archive) | `revision/results/e3_live/raw_evidence.sha256` |
| E3, live comparison | `revision/analysis/e3_live_analysis.py` | `e3_live/` | `phase6_e3_live.sha256` |
| E4 | `revision/analysis/e4_attacker.py` | `e4/` | `phase7_e4.sha256` |
| E4 decomposition | `revision/analysis/e4_decompose.py` | `e4/e4_decomposition.json` | `phase7_e4_decomposition.sha256` |
| E5 | `revision/analysis/e5_mitigations.py` | `e5/` | `phase8_e5.sha256` |
| Manuscript values | `revision/analysis/make_results_ext_values.py` | `manuscript_values/` | `phase9_results_values.sha256` |
| Figures and tables | `revision/figures/*.py`, `revision/tables/*.py` | `revision/figures/`, `revision/tables/` (archive) | `fig2_revision.sha256`, `fig3_fig4_revision.sha256`, `fig_generalization.sha256`, `table_dataset_cic.sha256`, `table_s4.sha256` |

Several scripts refuse to run when their output already exists; run them in a fresh copy.
