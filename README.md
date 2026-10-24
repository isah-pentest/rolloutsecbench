# RolloutSecBench

Code and reproduction tooling for the paper **"Model Rollout Evasion: Security Consistency of ML Threat Detection Under Continuous Deployment."**

RolloutSecBench measures what happens to a deployed ML intrusion detector while a progressive rollout serves two accepted revisions at the same time. Two frozen XGBoost revisions, V1 and V2, are trained on UNSW-NB15 and served behind a weighted router on a single-node Kubernetes cluster. Every request records the revision that served it.

The experiment crosses two factors:
- four rollout states, with 0, 10, 50 or 100% of traffic configured to V2;
- two attacker strategies: each malicious test sample submitted once, or five times unchanged.

Each of the eight cells has five valid runs.

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
| `serving/router/phase7/`, `data/manifests/phase7/router/` | Weighted router and its frozen protocol | Section 3.1 |
| `kubernetes/campaign/`, `data/manifests/phase9_v3/deviations/` | Kubernetes objects exported from the campaign cluster and verified against the pre-registration; kind cluster definition; NodePort service of deviation DEV-001 | Sections 3.1, 3.6 |
| `workloads/` | Request loader, attack client, campaign runner, run validator, freeze generator, confirmatory analysis, serving-equivalence gate | Sections 3.1, 3.6, 4 |
| `analysis/` | H2–H5 tests, exploratory routing-rule check, Tables 1/S1/2/6, Figures 2 and 4, claims generator | Section 4 |
| `paper/figures/fig1_architecture.tex` | Figure 1 | Section 3.1 |
| `tools/`, `release/` | Release verification, claims reproduction, comparison of regenerated values, release builder | — |
| `serving/router/execution003/`, `serving/router/gate3_v2_candidates/` | Build of the campaign router image (see `docs/RELEASE_DECISIONS.md`) | Section 3.1 |
| `environment/published_hashes.sha256` | SHA-256 identities of the data and frozen artifacts reported in the paper | Table S1, Figure 1 |
| `environment/*.lock`, `environment/campaign_environment.txt` | Training, serving and analysis environments; workstation and cluster versions | Table S2 |
| `docs/` | Reproduction map, identifiers used in the code, release decisions | — |

**Not in this repository:**
- The UNSW-NB15 files, which come from the dataset authors.
- The frozen models, the request-level evidence and the analysis of record, which are distributed as a separate archive (see [REPRODUCE.md](REPRODUCE.md)).

## Routing rule

Request *i* is served by V2 if and only if

```
u_i = int.from_bytes(SHA256("<basis>|<run_id>|<request_sequence>")[:8], "big") / 2**64  <  w2
```

The basis is the fixed value recorded in the router protocol. The rule depends on the request's position, never on its content. Neither the client nor the router retries failed requests, and a valid run requires every request to return HTTP 200. `python3 tools/verify_release.py` checks that the router protocol, the analysis and the validator all implement this same rule.

## Three levels of reproduction

1. **Integrity (minutes, no data):**
   ```
   python3 tools/verify_release.py
   ```
2. **Claims (offline, from the frozen evidence):**
   ```
   bash tools/reproduce_claims.sh --reference-dir paper_inputs
   ```
   This regenerates every number, table and figure the paper inputs, verifies every input by hash, and compares each regenerated file with the one the paper was compiled with. In the authors' clean-room test all six regenerated files were byte-identical.
3. **Operation (re-execution):** rebuild the offline artifacts, the containers and the cluster, then execute new runs under the frozen protocol. New runs are new evidence, not a replacement for the published runs.

See [REPRODUCE.md](REPRODUCE.md) and [docs/REPRODUCTION_MAP.md](docs/REPRODUCTION_MAP.md).

## Scope

The results cover one dataset, one model family, one revision pair, one routing policy, a single-node deployment at 16 queries per second, and a non-adaptive attacker. The paper's Limitations section gives the full scope.

## Licence and citation

The code is released under the MIT License (see [LICENSE](LICENSE)); copyright 2026 Isah Mohammed, Hyuk Cho and Narasimha Shashidhar. To cite the software, use [CITATION.cff](CITATION.cff) (GitHub shows it under "Cite this repository").

The dataset is subject to the UNSW-NB15 terms of use: free for academic research, while commercial use requires agreement with its authors.
