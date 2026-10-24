# Reproducing RolloutSecBench

Run every command from the repository root, with Python 3.11 or later.

```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## 1. Integrity

```
python3 tools/verify_release.py
```

| Check | What it verifies |
|---|---|
| `release` | Every file is identical to `RELEASE_MANIFEST.sha256` |
| `published` | Every artifact present matches the SHA-256 printed in the paper (Table S1, Figure 1, frozen predictions) |
| `prefixes` | The models and the freeze manifest match the abbreviated identities in Figure 1 |
| `pinned` | The campaign code is byte-identical to the code pinned in the freeze manifest |
| `routing` | The analysis and the validator implement the routing rule of the frozen router protocol |

Artifacts that are not yet in place are reported as SKIP. Add `--strict` once the dataset and the evidence archive are installed: SKIP then counts as failure.

## 2. Data

UNSW-NB15 is distributed by its authors (Moustafa and Slay, MilCIS 2015). It is free for academic research; commercial use requires agreement with the authors.

1. Obtain `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` from the dataset authors.
2. Place them in `data/raw/`.
3. Run `python3 tools/verify_release.py` and confirm that both files report PASS against the hashes in `environment/published_hashes.sha256`. The study used exactly these byte sequences; a file with any other hash is a different dataset version.

## 3. Evidence archive

The evidence archive contains the frozen offline artifacts and the request-level evidence of the confirmatory campaign. Unpack it at the repository root; it creates the paths below. Its contents are listed in `ARCHIVE_MANIFEST.sha256` inside the archive.

**Evidence archive:** [https://doi.org/10.5281/zenodo.23621512](https://doi.org/10.5281/zenodo.23621512) (`rolloutsecbench-evidence-v1.tar.gz`, SHA-256 `17b61615728b68b27c689b48258a1b78a3d095f14f5fc11a0d8eca8e34dd12ca`)

| Path | Role |
|---|---|
| `data/processed/feature_names.json`, `data/manifests/{dataset,split,preprocessing}_manifest.json` | Preprocessing and split records (Table 1, Table S1). The split files themselves are rows of UNSW-NB15 and are not redistributed: `training/generate_splits.py` recreates them from the two official files, and `tools/verify_release.py --strict` checks them against the hashes in Table S1 |
| `models/preprocessing/preprocessor.joblib`, `models/v1/`, `models/v2/`, `data/manifests/model_revision_protocol_v1.0.json` (+ `.sha256`) | Frozen preprocessor, revisions and acceptance rules (Table 2) |
| `results/derived/` (frozen offline evaluation: `malicious_predictions_strict.csv`, `pairwise_predictions_strict.csv`, `static_model_metrics_*.json`, `model_revision_acceptance.json`, `dataset_audit.json`, disagreement summaries) | Reference outcome of every malicious sample under each revision (H1, Figure 2) |
| `environment/checksums.sha256` | Hashes of the offline artifacts, checked by the Figure 2 script |
| `data/manifests/phase9_v3/phase9_v3_freeze.json` (+ `.sha256`) | Confirmatory freeze: design, run identities, reserves, predicted outcomes, pinned code and runtime |
| `results/raw/phase9_v3/registry.csv` and `results/raw/phase9_v3/<run_id>/` | Append-only validity registry and, per run, the client request log, router evidence, validation record and checksums (41 runs, about 8.5 GB) |
| `data/manifests/phase9_v3/deviations/`, `data/manifests/phase9_v3/execution_notes/` | Deviation records DEV-001 and DEV-002 and execution note NOTE-001 (Table 5) |
| `paper_inputs/` | The generated LaTeX values and tables the paper was compiled with (`claims_values.tex`, `gate4_values.tex`, `h2_h5_values.tex`, `table_gate4.tex`, `run_accounting.tex`, `latency_table.tex`) |
| `results/derived/phase9_v3/analysis_20261022T200702Z/` and its reference outputs (`claims/`, `confirmatory/`, `exploratory/`, `tables/`, `figures/`, `full_analysis_run.log`) | The analysis of record and the values printed in the paper, for comparison |

Before step 4, recreate the splits in the training environment (`environment/requirements.lock`). `training/generate_splits.py` refuses to overwrite a frozen artifact, so first move the archived `data/manifests/split_manifest.json` aside. Then run `python training/generate_splits.py`, unpack `data/manifests/` from the archive again so that the archived manifests are back in place, and run `python3 tools/verify_release.py --strict`. The split files must match the hashes in Table S1.

`release/cleanroom_test.sh` runs Sections 2–4 end to end in an empty directory, starting from this repository, the archive and the two official files.

Some provenance records in the archive contain absolute paths of the original workstation, for example the analysis directory. They are kept byte for byte, because rewriting them would break the recorded hashes.

## 4. Regenerate the paper's numbers, tables and figures

```
bash tools/reproduce_claims.sh --reference-dir paper_inputs
```

The script runs these steps in order; each step refuses to write if its inputs differ from their recorded hashes.

| Step | Script | Produces |
|---|---|---|
| 1 | `workloads/analyze_v3.py` | Per-run outcomes of the VALID runs, Gate-4 statistics, latency summaries, Figure 3 |
| 2 | `analysis/confirmatory/h2_h5_tests.py` | H2, H4 and H5 tests with Holm adjustment; H3 regression |
| 3 | `analysis/exploratory/routing_null_check.py --m 1000` | Exploratory routing-rule check |
| 4 | `analysis/tables/make_gate4_tex.py` | Gate-4 values and Table 6 |
| 5 | `analysis/tables/make_table1_dataset.py`, `make_table2_models.py` | Tables 1, S1 and 2 |
| 6 | `analysis/figures/make_fig2_disagreement.py` | Figure 2 |
| 7 | `analysis/figures/make_fig5_interaction.py` | Figure 4 |
| 8 | `analysis/claims/make_claims_tex.py` | Every number in the prose (`claims_values.tex`) and Tables 4 and 7 |
| 9 | — | Collects the paper inputs under `results/reproduced/<UTC>/paper/` with `GENERATED.sha256` |

**Comparing with the reference values.** `tools/compare_generated.py` compares the regenerated LaTeX values with the reference set macro by macro, so it ignores timestamps.

**Monte-Carlo quantities.** The Gate-4 *p*-values (10^6 draws) and the routing-rule check (resampling) use the frozen seeds in the scripts, so they should reproduce exactly with the same Python and NumPy versions as the original analysis (recorded in the provenance files where the scripts write them). Other versions can change the pseudo-random stream; the values then agree only within Monte-Carlo error.

## 5. Re-executing the apparatus

Re-execution produces new evidence under the same frozen protocol. It does not reproduce the published runs byte for byte.

Environments: `environment/requirements.lock` is the training and serving environment (Python 3.14.4, XGBoost 3.4.1, pandas 3.0.5, NumPy 2.5.3, scikit-learn 1.9.1, as recorded in the model manifests). `environment/analysis_environment.lock` is the environment of the analysis of record (Python 3.14.4, NumPy 2.5.4, Matplotlib 3.11.2). `environment/campaign_environment.txt` records the workstation, WSL, Docker, kind and Kubernetes versions and the packages installed in the running campaign containers. It was produced read-only by `release/collect_environment.sh` after the campaign and is summarized in Table S2 of the paper.

1. **Offline artifacts.** With the two UNSW-NB15 files in `data/raw/`, run the scripts in `training/` in phase order (each script's header states its phase, inputs and outputs):
   - `generate_splits.py`
   - `audit_dataset.py`
   - `fit_preprocessor.py`
   - `train_v1.py`, `train_v2.py`
   - `evaluate_static_test.py`
   - `evaluate_model_revision_acceptance.py`
   - `evaluate_malicious_disagreement.py`

   Then run `analysis/close_phase4_gate2.py`. Compare every output with `environment/published_hashes.sha256` and with the archive. Reproduction of the model files byte for byte depends on the library versions and platform; the archived models remain the reference.
2. **Containers.** Build both images from the repository root (Docker Desktop with WSL 2 in the original run):
   - **Model serving:** `docker build -f serving/docker/Dockerfile -t rolloutsecbench-serving:phase5-v3 .`. It copies the serving application, the serving contract `data/manifests/phase5_serving_protocol_v1.0.json`, and, from the evidence archive, the frozen models, preprocessor and feature names.
   - **Router:** `docker build -f serving/router/execution003/Dockerfile -t rolloutsecbench-router:gate3-v2.1-execution003-layoutfix1 .`. In the campaign the Deployment overrides the image's `CMD` and runs `serving/router/phase7/router_phase7_v1.py`, mounted from a ConfigMap. That router imports only the standard library, `httpx`, `fastapi` and `pydantic`. The other files the image copies (`serving/router/gate3_v2_candidates/*_candidate003.py`, `data/manifests/phase6_router_validation_protocol_v2.1.json`) were present in the container but are not executed. They are released so that the image can be rebuilt with the same contents.
   - **Loading:** load both images with `kind load docker-image <tag> --name rolloutsecbench-qualification`. The Deployments use `imagePullPolicy: Never`.

   `kubernetes/campaign/CONTAINER_FILES.json` lists every file under `/app` in the three running containers, with its SHA-256. Every file equals the released file at the same path, or the archive file at the same path; the router source is mounted. The base image `python:3.14-slim` is not pinned by digest in either Dockerfile; the containers that ran used Python 3.14.7. A rebuild therefore reproduces the application layer but not necessarily the base layer. The image IDs that ran are recorded in `runtime_at_freeze` and `EXPORT_VERIFICATION.json`.
3. **Cluster.** `kubernetes/campaign/` holds the Kubernetes objects that ran the campaign, exported from the live cluster as JSON manifests:
   - the model and router Deployments;
   - the Services, including the DEV-001 NodePort service `router-direct`;
   - the ConfigMaps `phase7-router-source` and `phase7-router-protocol`, which mount the released router source and protocol byte for byte.

   `EXPORT_VERIFICATION.json` records that the router image, command and environment, both ConfigMaps and all three pod images match `runtime_at_freeze` in the freeze manifest. To recreate the cluster:
   - create it with `kind create cluster --config kubernetes/campaign/kind-cluster.yaml`;
   - load the two images built in step 2;
   - apply the namespace `rolloutsecbench`;
   - apply `kubectl apply -f kubernetes/campaign/`.
4. **Serving equivalence.** Run `serving/run_phase5_equivalence.py` and `workloads/xm_equivalence_gate.py`. Every revision must reproduce its frozen decisions exactly and its probabilities within 1e-4.
5. **Freeze and execute.** Generate a new freeze with `workloads/make_v3_freeze.py`, using new run identifiers and never reusing published ones. Then execute with `workloads/v3_campaign.py`, which runs `workloads/v3_client.py` and `workloads/validate_v3_run.py` for each identity.
6. **Analyse.** Analyse as in Section 4.

Validity is decided only by the pre-specified apparatus criteria (completeness, transport, route reconstruction, identity, served = frozen, offered load, client–router join, pod stability). It never depends on outcomes.

## Running the reproduction inside the private repository

Before the release is built, there is no `RELEASE_MANIFEST.sha256`, so run `bash tools/reproduce_claims.sh --allow-skips` there. In the built public tree, with the dataset and the archive in place, run it without that flag.
