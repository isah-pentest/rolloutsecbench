# Reproduction map: paper element → generator → inputs

Section numbers refer to the paper.

| Paper element | Generator | Main verified inputs |
|---|---|---|
| Table 1, Table S1 (Section 3.3) | `analysis/tables/make_table1_dataset.py` | raw and split files, dataset/split/preprocessing manifests, `dataset_audit.json`, preprocessor |
| Table 2, Gate 1 (Section 3.4) | `analysis/tables/make_table2_models.py` | model revision protocol (+ sidecar), model manifests and files, acceptance report, static metrics |
| Table 3 (Section 3.5) | — (the frozen analysis plan, stated in the paper) | — |
| Table 4, run accounting (Section 3.6) | `analysis/claims/make_claims_tex.py` → `run_accounting.tex` | `registry.csv`, freeze manifest |
| Table 5, deviations (Section 3.6) | — (protocol amendments and deviation records, stated in the paper) | — |
| Figure 1 (Section 3.1) | `paper/figures/fig1_architecture.tex` | identities shown on its provenance rail (`tools/verify_release.py`, check `prefixes`) |
| Figure 2, H1 (Section 4.1) | `analysis/figures/make_fig2_disagreement.py` | frozen offline predictions and disagreement summaries, `environment/checksums.sha256` |
| H2, H3 (Section 4.2) | `analysis/confirmatory/h2_h5_tests.py` → `h2_h5_values.tex` | `per_run.csv`, `integrity_report.json`, `gate4.json` of the analysis directory |
| Figure 3, H4 (Section 4.3) | `workloads/analyze_v3.py` → `fig_exposure_law.pdf`; `h2_h5_tests.py` for the H4 test | VALID runs, freeze manifest, frozen predictions |
| Figure 4, H5 (Section 4.4) | `analysis/figures/make_fig5_interaction.py` (file name `fig5_interaction.pdf`); `h2_h5_tests.py` for the H5 test | `per_run.csv`, `integrity_report.json` |
| Gate 4, Table 6 (Section 4.5) | `workloads/analyze_v3.py` (statistics); `analysis/tables/make_gate4_tex.py` → `gate4_values.tex`, `table_gate4.tex` | `gate4.json`, `per_run.csv`, analysis manifest |
| Exploratory routing-rule check (Sections 4.5 and 6) | `analysis/exploratory/routing_null_check.py` | frozen routing rule (`workloads/make_v3_freeze.py`), frozen inputs, `per_run.csv` |
| Table 7, latency (Section 4.6) | `workloads/analyze_v3.py` → `latency_by_transport_path.csv`; `make_claims_tex.py` → `latency_table.tex` | per-run client request logs |
| Every other number in the prose | `analysis/claims/make_claims_tex.py` → `claims_values.tex` | all of the above, each verified by hash; qualitative statements are checked against the evidence |

The paper source loads `gate4_values.tex`, `h2_h5_values.tex` and `claims_values.tex`, and inputs `run_accounting.tex`, `table_gate4.tex` and `latency_table.tex`. If any of these files is absent, the affected values print as `[C?]`, `[G4?]`, `[H?]` or `[T?]` rather than as stale numbers.
