from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PROTOCOL = (
    ROOT / "data/manifests/"
    "phase6_router_validation_protocol_v2.1.json"
)

EXPECTED_PROTOCOL_SHA256 = (
    "dec25e9d7a303c5d104595ca907cf307"
    "17eaadde9a9934c3e037b7da330e1d05"
)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_protocol(path=DEFAULT_PROTOCOL):
    observed = sha256_file(path)

    if observed != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            f"Frozen protocol SHA mismatch: {observed}"
        )

    p = json.loads(path.read_text(encoding="utf-8"))

    if p.get("protocol_status") != "FROZEN_PREEXECUTION":
        raise RuntimeError("Protocol is not FROZEN_PREEXECUTION.")

    return p


def oracle_route(state, replicate_id, sequence, protocol=None):
    if protocol is None:
        protocol = load_protocol()

    if state not in protocol["rollout_states"]:
        raise ValueError("Unknown rollout_state.")

    if replicate_id not in protocol["replication"]["replicate_ids"]:
        raise ValueError("Unknown replicate_id.")

    if not isinstance(sequence, int) or sequence < 0:
        raise ValueError("Invalid zero-based request sequence.")

    seed = (
        protocol["precommitted_seed_registry"]
        [state][replicate_id]["sha256"]
    )

    basis = protocol["protocol_identity"]["identity_basis_sha256"]

    reconstructed = hashlib.sha256(
        f"{basis}|{state}|{replicate_id}".encode("utf-8")
    ).hexdigest()

    if seed != reconstructed:
        raise RuntimeError(
            "Precommitted seed registry mismatch."
        )

    digest = hashlib.sha256(
        f"{seed}|{sequence}".encode("utf-8")
    ).digest()

    integer = int.from_bytes(
        digest[:8],
        "big",
        signed=False,
    )

    u = integer / (2 ** 64)

    weight = float(
        protocol["rollout_states"][state]["v2_weight"]
    )

    revision = "v2" if u < weight else "v1"

    return {
        "revision": revision,
        "seed_digest": seed,
        "routing_uniform_value": u,
        "configured_v2_weight": weight,
    }


if __name__ == "__main__":
    p = load_protocol()
    print({
        "status": "PASS",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "protocol_status": p["protocol_status"],
    })
