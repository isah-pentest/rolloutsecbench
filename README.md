# RolloutSecBench

Code and reproduction tooling for the paper **"Model Rollout Evasion: Quantifying and Predicting Evasion Exposure During Progressive Deployment of Machine Learning Detectors."**

RolloutSecBench measures what happens to a deployed ML intrusion detector while a progressive rollout serves two accepted revisions at the same time. Two frozen XGBoost revisions, V1 and V2, are trained on UNSW-NB15 and served behind a weighted router on a single-node Kubernetes cluster. Every request records the revision that served it.

The confirmatory campaign crosses two factors:
- four rollout states, with 0, 10, 50 or 100% of traffic configured to V2;
- two attacker strategies: each malicious test sample submitted once, or five times unchanged.

Each of the eight cells has five valid runs. Release v1.1.0 adds the extension analyses of the revised paper (see below).

```
git clone https://github.com/isah-pentest/rolloutsecbench.git
cd rolloutsecbench
python3 tools/verify_release.py
```

## Repository layout

| Path | Contents | Paper |
|---|---|---|
| `configs/`, `training/` | Splits, strict non-overlap filter, dataset audit, preprocessing, V1/V2 training, static evaluation, acceptance (Gate 1), offline malicious disagreement | Sections 3.3, 3.4, 4.1 |
| `analysis/close_phase4_gate2.py` | Closure of the offline disagreement analysis (H1) | Section 4.1 |
| `serving/api/`, `serving/docker/` | Model-serving application and its container | Section 3.1 |
| `serving/run_phase5_equivalence.py` | Offline-versus-container serving equivalence | Section 3.1 |
| `serving/router/phase7/`, `data/manifests/phase7/router/` | Weighted router of the confirmatory campaign and its frozen protocol | Section 3.1 |
| `kubernetes/campaign/`, `data/manifests/phase9_v3/deviations/` | Kubernetes objects exported from the campaign cluster and verified against the pre-registration; kind cluster definition; NodePort service of deviation DEV-001 | Sections 3.1, 3.9 |
| `workloads/` | Request loader, attack client, campaign runner, run validator, freeze generator, confirmatory analysis, serving-equivalence gate | Sections 3.1, 3.2, 3.9, 4 |
| `analysis/` | H2–H5 tests, exploratory routing-rule check, table, figure and claims generators | Section 4 |
| `paper/figures/fig1_architecture.tex` | Figure 1 | Section 3.1 |
| `revision/` | Extension analyses (release v1.1.0); see [docs/REPRODUCTION_R1.md](docs/REPRODUCTION_R1.md) | Sections 3.5–3.8, 4 |
| `serving/router/ext1/`, `data/manifests/ext1/`, `kubernetes/ext1/`, `workloads/*ext1*` | Router with a configurable routing policy (P1–P3), its protocol, Kubernetes objects, client, validator, campaign runner and live-run freeze of the routing-policy experiment | Sections 3.1, 3.7, 4.9 |
| `tools/`, `release/` | Release verification, claims reproduction, comparison of regenerated values, release and archive builders, clean-room test | — |
| `serving/router/execution003/`, `serving/router/gate3_v2_candidates/` | Build of the campaign router image (see `docs/RELEASE_DECISIONS.md`) | Section 3.1 |
| `environment/published_hashes.sha256` | SHA-256 identities of the data and frozen artifacts reported in the paper | Table S1, Figure 1 |
| `environment/*.lock`, `environment/campaign_environment.txt`, `revision/manifests/env_rsb_revision.lock` | Training, serving, analysis and revision environments; workstation and cluster versions | Section 3.1 |
| `docs/` | Reproduction maps, identifiers used in the code, release decisions | — |

**Not in this repository:**
- The UNSW-NB15 files, which come from the dataset authors, and the corrected CIC-IDS2017 release, which comes from its providers.
- The frozen models, the request-level evidence, the analysis of record and the outputs of the extension analyses, which are distributed as two evidence archives on Zenodo (see [REPRODUCE.md](REPRODUCE.md) and [docs/REPRODUCTION_R1.md](docs/REPRODUCTION_R1.md)).

## Routing rule

Request *i* is served by V2 if and only if

```
u_i = int.from_bytes(SHA256("<basis>|<run_id>|<key_i>")[:8], "big") / 2**64  <  w2
```

The basis is the fixed value recorded in the router protocol. In the confirmatory campaign (policy P1) the key is the request's sequence number, so the rule depends on the request's position, never on its content. The routing-policy experiment also uses P2 (key = session identifier sent by the client) and P3 (key = SHA-256 of the canonical feature vector). Neither the client nor the router retries failed requests, and a valid run requires every request to return HTTP 200. `python3 tools/verify_release.py` checks that the router protocol, the analysis and the validator all implement the P1 rule.

## Three levels of reproduction

1. **Integrity (minutes, no data):**
   ```
   python3 tools/verify_release.py
   ```
2. **Claims (offline, from the frozen evidence):**
   ```
   bash tools/reproduce_claims.sh --reference-dir paper_inputs
   python3 revision/analysis/make_results_ext_values.py
   ```
   The first command regenerates every number, table and figure of the confirmatory campaign and compares each regenerated file with the one the paper was compiled with. The second regenerates the values and tables of the extension analyses from their archived outputs, after checking every input against `revision/manifests/`. In the authors' clean-room test, all regenerated files were byte-identical.
3. **Operation (re-execution):** rebuild the offline artifacts, the containers and the cluster, then execute new runs under the frozen protocol. New runs are new evidence, not a replacement for the published runs.

See [REPRODUCE.md](REPRODUCE.md), [docs/REPRODUCTION_MAP.md](docs/REPRODUCTION_MAP.md) and [docs/REPRODUCTION_R1.md](docs/REPRODUCTION_R1.md).

## Extension analyses (release v1.1.0)

The revised paper adds analyses that do not change the 40 confirmatory runs or their decisions: duplicate-aware disagreement, alternative tests of the confirmatory decisions, a routing simulator validated on all 40 runs, 72 further revision pairs on UNSW-NB15 and CIC-IDS2017, live runs under session-affinity and content-keyed routing, offline attacker variants and a directional-disagreement gate. They were specified in a time-stamped plan before any output was generated; the plan itself is not released, and its SHA-256 and timestamp proof are listed in Table S4 of the paper. Identifiers used in this code are mapped to the paper in [docs/IDENTIFIERS_R1.md](docs/IDENTIFIERS_R1.md).

## Scope

The live measurements cover one dataset, one model family, one revision pair, a single-node deployment at 16 queries per second and a non-adaptive attacker; session-affinity and content-keyed routing were run live at the 10% and 50% states only. The paper's Limitations section gives the full scope.

## Licence and citation

The code is released under the MIT License (see [LICENSE](LICENSE)); copyright 2026 Isah Mohammed, Hyuk Cho and Narasimha Shashidhar. To cite the software, use [CITATION.cff](CITATION.cff) (GitHub shows it under "Cite this repository").

The datasets are subject to their providers' terms. UNSW-NB15 is free for academic research, while commercial use requires agreement with its authors.
