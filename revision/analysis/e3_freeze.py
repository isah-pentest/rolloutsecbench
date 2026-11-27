"""Revision r1, Phase 6, step 1: E3 identity freeze (pre-registration Sections 2 and 5).

Generates every E3 identity by rule and the live execution order, and computes NO outcome. Deterministic (no
timestamps), so the file hash is fixed by the rules alone. Output: revision/prereg/E3_FREEZE.json, which is
committed and timestamped before e3_simulate.py computes anything.

Simulated cells (P1 cells already exist from S2/E1 and are not repeated):
  P2 x {R0, R1, R2, R3} x {A0, A1-one, A1-multi}   (A1-multi is P2-only, Section 5)
  P3 x {R0, R1, R2, R3} x {A0, A1-one}
  identities SIM-{policy}-{state}-{attacker}-{index:05d}, index 00001-10000
Live cells: {R1, R2} x {P2 + A1-one, P2 + A1-multi, P3 + A1-one}
  identities EXT1-{policy}-{state}-{attacker}-{P|R}{index:02d}, 2 primary + 2 reserve per cell
  execution order: random.Random(11242026).shuffle(all primaries)
"""
import hashlib, json, pathlib, random

ROOT = pathlib.Path(__file__).resolve().parents[2]
STATES = ["R0", "R1", "R2", "R3"]
SIM_CELLS = [("P2", s, a) for s in STATES for a in ("A0", "A1-one", "A1-multi")] + [("P3", s, a) for s in STATES for a in ("A0", "A1-one")]
LIVE_CELLS = [(p, s, a) for s in ("R1", "R2") for (p, a) in (("P2", "A1-one"), ("P2", "A1-multi"), ("P3", "A1-one"))]
NSIM = 10_000

sim = {f"{p}-{s}-{a}": [f"SIM-{p}-{s}-{a}-{i:05d}" for i in range(1, NSIM + 1)] for p, s, a in SIM_CELLS}
live = {f"{p}-{s}-{a}": {"primary": [f"EXT1-{p}-{s}-{a}-P{i:02d}" for i in (1, 2)],
                         "reserve": [f"EXT1-{p}-{s}-{a}-R{i:02d}" for i in (1, 2)]} for p, s, a in LIVE_CELLS}
order = [x for c in live.values() for x in c["primary"]]
random.Random(11242026).shuffle(order)
sim_text = "\n".join(x for c in sim.values() for x in c) + "\n"
freeze = {
    "status": "FROZEN_BEFORE_ANY_E3_OUTCOME",
    "rules": "pre-registration v1.0 Sections 2 and 5",
    "weights": {"R0": 0.0, "R1": 0.1, "R2": 0.5, "R3": 1.0},
    "sim_cells": [list(c) for c in SIM_CELLS], "sim_per_cell": NSIM,
    "sim_identities_sha256": hashlib.sha256(sim_text.encode()).hexdigest(),
    "sim_identities_first_last": {k: [v[0], v[-1]] for k, v in sim.items()},
    "live_cells": [list(c) for c in LIVE_CELLS], "live_identities": live,
    "live_execution_order": order, "live_order_rule": "random.Random(11242026).shuffle(primaries in cell order)",
    "reserve_rule": "used only after an objective INVALID in the same cell, ascending; no re-run of a consumed identity",
}
out = ROOT / "revision/prereg/E3_FREEZE.json"
out.write_text(json.dumps(freeze, indent=1, sort_keys=True) + "\n")
(ROOT / "revision/prereg/E3_SIM_IDENTITIES.txt").write_text(sim_text)
print("sim identities:", sum(map(len, sim.values())), freeze["sim_identities_sha256"])
print("live execution order:", order)
print("wrote", out, hashlib.sha256(out.read_bytes()).hexdigest())
