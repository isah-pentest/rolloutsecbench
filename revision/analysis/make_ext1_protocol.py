"""Revision r1, Phase 6 live: build the EXT1 router protocol from the deployed phase7 protocol (pre-registration Section 5).
Copies rollout_states and routing from the phase7 protocol verbatim (hash-checked). Adds the three policies and the
P3 feature order (from data/processed/feature_names.json source_feature_order, verified identical to the simulator's
order by ext1_p3_canon_check.py). Deterministic. Output: data/manifests/ext1/ext1_router_protocol_v1.0.json
"""
import hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
P7 = ROOT / "data/manifests/phase7/router/phase7_router_protocol_v1.0.json"
P7_SHA = "4226b83926cb83f65472e0f7cb35c5ff01a893866cd32b58e73904499f8ac781"
if hashlib.sha256(P7.read_bytes()).hexdigest() != P7_SHA: sys.exit("phase7 protocol hash mismatch")
p7 = json.loads(P7.read_text())
fn = json.loads((ROOT / "data/processed/feature_names.json").read_text())
order = fn["source_feature_order"]
if len(order) != 42: sys.exit("expected 42 features")
proto = {
    "protocol_name": "RolloutSecBench EXT1 Router Protocol (revision r1, E3 routing policies)",
    "protocol_version": "1.0",
    "protocol_status": "PROSPECTIVE_PREEXECUTION",
    "parent_router_protocol": {"path": str(P7.relative_to(ROOT)), "sha256": P7_SHA},
    "rollout_states": p7["rollout_states"],
    "routing": {k: p7["routing"][k] for k in ("algorithm", "route_seed_basis_sha256", "digest_bytes", "uniform_denominator", "selection_rule")},
    "policies": {
        "P1": {"material": "route_seed_basis_sha256|run_id|request_sequence"},
        "P2": {"material": "route_seed_basis_sha256|run_id|session_id", "session_id_required": True},
        "P3": {"material": "route_seed_basis_sha256|run_id|SHA256_hex(canonical feature vector)",
               "canonical_vector": "values in p3_feature_order joined with '|'; str values as is, numbers as Python repr "
                                   "of the JSON-decoded value (equals the strict-test CSV text for all X_m samples)"},
    },
    "default_policy": "P1",
    "policy_selection": "deployment environment variable ROUTING_POLICY (one router build; configuration value)",
    "p3_feature_order": order,
    "evidence": {"router_evidence_path": "/evidence/ext1_router_requests.jsonl", "append_only": True},
}
out = ROOT / "data/manifests/ext1/ext1_router_protocol_v1.0.json"; out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(proto, indent=2, sort_keys=True) + "\n")
print("wrote", out, hashlib.sha256(out.read_bytes()).hexdigest())
