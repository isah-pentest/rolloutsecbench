"""RolloutSecBench revision r1, E1: exact routing simulator.

Implemented from the frozen router protocol (phase9_v3 freeze), not by importing router code:
    V2 iff int.from_bytes(SHA256(material)[:8], "big") / 2**64 < w2
    P1 material = basis|run_id|request_sequence
    P2 material = basis|run_id|session_id
    P3 material = basis|run_id|SHA256(canonical feature vector)
Scoring follows Eq. (1), (3), (4) of the manuscript, using the frozen offline predictions (1 = malicious, 0 = benign).
Standard library only. Every input is checked against the freeze before use.
"""
import csv, hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
FREEZE_PATH = ROOT / "data/manifests/phase9_v3/phase9_v3_freeze.json"
FREEZE_SHA = "67d0ac984710b5a3748c44de577eece02bf7652deb9969fc792decf6b75dfcf1"
WEIGHTS = {"R0": 0.0, "R1": 0.1, "R2": 0.5, "R3": 1.0}
ATTEMPTS = {"A0": 1, "A1": 5, "A1-one": 5, "A1-multi": 5}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def uniform(material: str) -> float:
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") / 2**64


class Simulator:
    def __init__(self):
        if sha256_file(FREEZE_PATH) != FREEZE_SHA:
            sys.exit("freeze manifest hash mismatch")
        self.freeze = json.load(open(FREEZE_PATH))
        for rel, want in {**self.freeze["frozen_inputs_sha256"], **self.freeze["pinned_code_sha256"]}.items():
            if sha256_file(ROOT / rel) != want:
                sys.exit(f"input hash mismatch: {rel}")
        # The freeze field "basis_sha256" holds the 64-hex-digit routing basis b itself (Section 3.1).
        self.basis = self.freeze["routing"]["basis_sha256"]
        sys.path.insert(0, str(ROOT / "workloads"))
        import xm_loader  # pinned (hash checked above)
        samples, _ = xm_loader.load_xm(ROOT)
        self.order = [str(s["sample_id"]) for s in samples]
        self.pred = {}
        with open(ROOT / "results/derived/malicious_predictions_strict.csv", newline="") as f:
            for r in csv.DictReader(f):
                self.pred[r["id"]] = {"v1": int(r["v1_prediction"]), "v2": int(r["v2_prediction"])}
        names = json.load(open(ROOT / "data/processed/feature_names.json"))
        if not isinstance(names, list):
            names = next(v for v in names.values() if isinstance(v, list) and len(v) == 42)
        self.vec_hash = {}
        with open(ROOT / "data/splits/strict_test_nonoverlap.csv", newline="") as f:
            for r in csv.DictReader(f):
                if r["id"] in self.pred:
                    self.vec_hash[r["id"]] = hashlib.sha256("|".join(r[n] for n in names).encode()).hexdigest()
        self.sets = {k: {s for s, p in self.pred.items() if (p["v1"], p["v2"]) == v}
                     for k, v in {"X12": (1, 0), "X21": (0, 1), "both": (0, 0)}.items()}

    def schedule(self, attacker):
        seq = 0
        for sid in self.order:
            for a in range(1, ATTEMPTS[attacker] + 1):
                seq += 1
                yield seq, sid, a

    def material(self, policy, run_id, seq, sid, a, attacker):
        if policy == "P1":
            key = str(seq)
        elif policy == "P2":
            key = f"{run_id}-S{sid}-A{a}" if attacker == "A1-multi" else f"{run_id}-S{sid}"
        elif policy == "P3":
            key = self.vec_hash[sid]
        else:
            raise ValueError(policy)
        return f"{self.basis}|{run_id}|{key}"

    def run(self, run_id, state, attacker, policy="P1", keep_requests=False):
        w2 = WEIGHTS[state]
        first_evade = {}
        reqs = []
        v2 = 0
        k12 = set(); k21 = set()
        n = 0
        for seq, sid, a in self.schedule(attacker):
            u = uniform(self.material(policy, run_id, seq, sid, a, attacker))
            rev = "v2" if u < w2 else "v1"
            v2 += rev == "v2"; n += 1
            decision = self.pred[sid][rev]
            if a == 1 and rev == "v2":
                if sid in self.sets["X12"]: k12.add(sid)
                if sid in self.sets["X21"]: k21.add(sid)
            if decision == 0 and sid not in first_evade:
                first_evade[sid] = a
            if keep_requests:
                reqs.append((seq, sid, a, u, rev, decision))
        A = ATTEMPTS[attacker]
        C = {m: sum(1 for x in first_evade.values() if x <= m) for m in range(1, A + 1)}
        out = {"run_id": run_id, "state": state, "attacker": attacker, "policy": policy, "requests": n,
               "realized_v2_share": v2 / n, "C": C, "K12": len(k12), "K21": len(k21)}
        if attacker == "A0":
            m0 = len(self.sets["both"]) + len(self.sets["X21"])  # misses at R0
            m3 = len(self.sets["both"]) + len(self.sets["X12"])  # misses at R3
            w = out["realized_v2_share"]
            out["d"] = C[1] - ((1 - w) * m0 + w * m3)
        if keep_requests:
            out["requests_detail"] = reqs
        return out
