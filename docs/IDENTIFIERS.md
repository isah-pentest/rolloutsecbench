# Identifiers used in the code

The code keeps the identifiers of the study protocol unchanged, because several files are pinned by hash in the freeze manifest and editing them, even their comments, would break that verification. This table maps the identifiers to the paper.

| Identifier in code | Meaning | Paper |
|---|---|---|
| Phase 0 | Protocol freeze | Section 3.5 |
| Phase 1 | Dataset audit and splits | Section 3.3 |
| Phase 2, Phase 3 | Training of V1 and V2; static evaluation | Section 3.4 |
| Phase 4 | Offline malicious-disagreement analysis (frozen predictions on $X_m$) | Sections 3.1, 4.1 |
| Phase 5 | Serving validation (serving-equivalence gate, \|Δp\| < 1e-4) | Section 3.1 |
| Phase 6, Phase 7 | Router validation; pilot (pilot evidence is not confirmatory) | Sections 3.1, 3.6 |
| Phase 8 | Final protocol freeze before the main experiment | Sections 3.5, 3.6 |
| Phase 9 (v3) | Main confirmatory campaign reported in the paper | Sections 3.6, 4 |
| Phase 10 | Analysis | Section 4 |
| Gate 1 | Model acceptance (non-inferiority on validation) | Section 3.4 |
| Gate 2 | Malicious revision disagreement (H1) | Section 4.1 |
| Gate 4 | Mixture-additivity test of single-attempt exposure | Section 4.5 |
| Amendment A1-B5-L16 | A1 budget 20 → 5 attempts, load 64 → 16 queries per second | Table 5 |
| DEV-001, DEV-002, NOTE-001 | Recorded deviations during execution | Table 5 |
| V1–V7, P1–P2 | Run validity criteria (completeness, transport, route reconstruction, identity, served = frozen, offered load, client–router join; pod stability, pinned router configuration) | Section 3.1 |
| ASR | Attack success rate, the share of malicious samples evaded, $C(n)/n_m$ | Section 3.2 |
| MDE | Minimum detectable effect of the superseded Gate-4 rule, estimated from pilot data | Section 4.5, Table 5 |
| "Section K", "Section AH", "Section AJ", "audit Flag 1–5" | Clauses of internal planning documents that are not distributed. Respectively: the serving-equivalence tolerance, the offered-load tolerance, the hash chain, and the items of the pre-execution statistical analysis plan | Sections 3.1, 3.5 |
| `.venv-blueprint-v2` | Name of the original local Python environment, which appears in usage examples; any Python ≥ 3.11 environment with `requirements.txt` works | — |
