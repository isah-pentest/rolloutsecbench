# Identifiers of the extension analyses (revision r1)

Code, manifests and commit messages of release v1.1.0 use the identifiers below. Section and table numbers refer to the revised paper.

| Identifier in code | Meaning | Paper |
|---|---|---|
| S1 | Duplicate-aware disagreement: row MRD, cluster bootstrap with the distinct feature vector as cluster, distinct-vector MRD, exact McNemar | Sections 3.6, 4.1 |
| S2 / E1 | Simulated routing distributions under P1 (10,000 identities per cell), percentiles of the observed runs, Gate 4 against the simulator, canary sizing, detectability, rollout schedules | Sections 3.7, 4.7, 4.8 |
| S3 / S4 | Alternative tests of the H2, H4 and H5 decisions; CR2 and wild cluster bootstrap intervals for the H3 slope | Section 4.7 |
| E2 | Generalization experiment: 48 revision pairs on UNSW-NB15 and 24 on CIC-IDS2017 (mechanisms M1–M4) | Sections 3.5, 4.2 |
| Phase 5b, CNS2022 | Preparation of the corrected CIC-IDS2017 release and its E2 pairs | Section 3.3 |
| E3 | Routing-policy comparison: simulated cells under P1–P3 and live-run predictions | Sections 3.7, 4.9 |
| EXT1 | Live routing-policy runs (P2 with A1-one and A1-multi, P3 with A1-one, at R1 and R2), their router, client, validator, qualification and freeze | Sections 3.1, 3.7, 4.9 |
| E4 | Attacker variants: detection cost, early stopping, near-duplicate replay, campaign view | Sections 3.6, 4.10 |
| E5 | Mitigation analysis: directional-disagreement gate; sticky routing and shortened mixed windows taken from E3 and E1 | Sections 3.8, 4.11 |
| Phase 9 manuscript values | `make_results_ext_values.py`: macros and tables of Sections 4.1 and 4.7–4.11, generated from the committed outputs | Section 4 |
| DEV-R1-001 … DEV-R1-008 | Recorded deviations of the extension analyses | Table 10 |
| P1, P2, P3 | Per-request, session-affinity and content-keyed routing | Section 3.1 |
| A1-one, A1-multi | Retrying attacker under P2 with one session per sample, or a new session per attempt | Section 3.2 |
| "pre-registration", "prereg", "§n", "Section n" in comments | Sections of the time-stamped extension plan, which is not released; its SHA-256 and timestamp proof are in Table S4 | Table S4 |
| `revision/prereg/E3_FREEZE.json`, `E3_SIM_IDENTITIES.txt` | Identity freezes of E3, written before any E3 outcome and read by `e3_simulate.py` and `make_ext1_live_freeze.py` | Section 3.7 |
