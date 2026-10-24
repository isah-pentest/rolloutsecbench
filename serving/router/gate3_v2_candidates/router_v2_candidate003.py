from pathlib import Path
import hashlib
import json
import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


ROOT = Path(__file__).resolve().parents[3]

PROTOCOL_PATH = Path(
    os.environ.get(
        "PHASE6_PROTOCOL_PATH",
        str(
            ROOT / "data/manifests/"
            "phase6_router_validation_protocol_v2.1.json"
        ),
    )
)

EXPECTED_PROTOCOL_SHA256 = (
    "dec25e9d7a303c5d104595ca907cf307"
    "17eaadde9a9934c3e037b7da330e1d05"
)

V1_BACKEND_URL = os.environ.get(
    "V1_BACKEND_URL",
    "http://model-v1:8000",
)

V2_BACKEND_URL = os.environ.get(
    "V2_BACKEND_URL",
    "http://model-v2:8000",
)

EVIDENCE_PATH = os.environ.get(
    "ROUTER_EVIDENCE_PATH",
    "",
)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_protocol():
    observed = sha256_file(PROTOCOL_PATH)

    if observed != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "Frozen Gate-3 v2.1 protocol SHA mismatch."
        )

    p = json.loads(
        PROTOCOL_PATH.read_text(encoding="utf-8")
    )

    if p.get("protocol_status") != "FROZEN_PREEXECUTION":
        raise RuntimeError(
            "Protocol is not frozen pre-execution."
        )

    return p


PROTOCOL = load_protocol()


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    run_id: str
    request_id: str
    request_sequence: int
    replicate_id: str
    sample_id: str | int
    ground_truth: int
    attack_family: str | None = None
    rollout_state: str
    connection_mode: str
    features: dict


app = FastAPI()


def expected_backend_identity(selected):
    identities = (
        PROTOCOL
        .get("provenance", {})
        .get("backend_identities", {})
    )

    obj = identities.get(selected)

    if obj:
        return obj

    # The v2.1 contract requires runtime verification.
    # If model identities are not represented in this protocol
    # location, deployment must supply them explicitly.
    prefix = selected.upper()

    model_sha = os.environ.get(
        f"{prefix}_MODEL_SHA256"
    )

    if not model_sha:
        raise HTTPException(
            500,
            f"{prefix}_MODEL_SHA256_NOT_CONFIGURED",
        )

    return {
        "model_sha256": model_sha,
    }


def route(req):
    state = req.rollout_state
    replicate = req.replicate_id
    seq = req.request_sequence

    if state not in PROTOCOL["rollout_states"]:
        raise HTTPException(400, "Unknown rollout_state")

    if replicate not in PROTOCOL["replication"]["replicate_ids"]:
        raise HTTPException(400, "Unknown replicate_id")

    if req.connection_mode not in PROTOCOL["connection_modes"]:
        raise HTTPException(400, "Unknown connection_mode")

    if seq < 0 or seq >= 11200:
        raise HTTPException(
            400,
            "request_sequence outside frozen condition range",
        )

    seed = (
        PROTOCOL["precommitted_seed_registry"]
        [state][replicate]["sha256"]
    )

    basis = (
        PROTOCOL["protocol_identity"]
        ["identity_basis_sha256"]
    )

    reconstructed_seed = hashlib.sha256(
        f"{basis}|{state}|{replicate}".encode("utf-8")
    ).hexdigest()

    if seed != reconstructed_seed:
        raise HTTPException(
            500,
            "Precommitted seed registry mismatch",
        )

    digest = hashlib.sha256(
        f"{seed}|{seq}".encode("utf-8")
    ).digest()

    u = int.from_bytes(
        digest[:8],
        "big",
        signed=False,
    ) / (2 ** 64)

    weight = float(
        PROTOCOL["rollout_states"][state]["v2_weight"]
    )

    selected = "v2" if u < weight else "v1"

    return selected, seed, u, weight


@app.get("/health")
def health():
    return {
        "status": "ok",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
    }


@app.post("/predict")
def predict(req: PredictRequest):

    if int(req.ground_truth) != 0:
        raise HTTPException(
            400,
            "Gate-3 v2.1 accepts benign validation traffic only.",
        )

    selected, seed, u, weight = route(req)

    backend_url = (
        V2_BACKEND_URL
        if selected == "v2"
        else V1_BACKEND_URL
    )

    # Exact frozen backend request contract.
    backend_payload = {
        "request_id": req.request_id,
        "features": req.features,
    }

    t0 = time.perf_counter()

    try:
        response = httpx.post(
            backend_url.rstrip("/") + "/predict",
            json=backend_payload,
            timeout=30.0,
        )
    except Exception as exc:
        raise HTTPException(
            502,
            "Backend request failed:"
            + type(exc).__name__,
        )

    router_backend_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    if response.status_code != 200:
        raise HTTPException(
            502,
            f"Backend HTTP {response.status_code}",
        )

    try:
        backend = response.json()
    except Exception as exc:
        raise HTTPException(
            502,
            "Backend JSON decode failure:"
            + type(exc).__name__,
        )

    required = {
        "request_id",
        "model_version",
        "prediction",
        "probability",
        "processing_ms",
        "model_sha256",
        "preprocessor_sha256",
        "raw_feature_count",
        "transformed_feature_count",
    }

    missing = sorted(required - set(backend))

    if missing:
        raise HTTPException(
            500,
            "Backend response missing fields:"
            + ",".join(missing),
        )

    if backend["request_id"] != req.request_id:
        raise HTTPException(
            500,
            "Backend request_id mismatch",
        )

    backend_revision = str(
        backend["model_version"]
    ).lower()

    if backend_revision != selected:
        raise HTTPException(
            500,
            "Selected/backend model_version mismatch",
        )

    dependency = PROTOCOL[
        "frozen_execution_dependencies"
    ]

    expected_preprocessor = (
        dependency["preprocessor"]["sha256"]
    )

    if (
        backend["preprocessor_sha256"]
        != expected_preprocessor
    ):
        raise HTTPException(
            500,
            "Backend preprocessor SHA mismatch",
        )

    expected_raw = int(
        dependency["feature_names"]["raw_feature_count"]
    )

    expected_transformed = int(
        dependency["feature_names"]
        ["transformed_feature_count"]
    )

    if int(backend["raw_feature_count"]) != expected_raw:
        raise HTTPException(
            500,
            "Backend raw_feature_count mismatch",
        )

    if (
        int(backend["transformed_feature_count"])
        != expected_transformed
    ):
        raise HTTPException(
            500,
            "Backend transformed_feature_count mismatch",
        )

    identity = expected_backend_identity(selected)

    expected_model_sha = identity.get(
        "model_sha256"
    )

    if (
        not expected_model_sha
        or backend["model_sha256"] != expected_model_sha
    ):
        raise HTTPException(
            500,
            "Backend model SHA mismatch",
        )

    internal_record = {
        "experiment_id": req.experiment_id,
        "run_id": req.run_id,
        "request_id": req.request_id,
        "request_sequence": req.request_sequence,
        "replicate_id": req.replicate_id,
        "sample_id": req.sample_id,
        "ground_truth": req.ground_truth,
        "attack_family": req.attack_family,
        "rollout_state": req.rollout_state,
        "connection_mode": req.connection_mode,
        "configured_v2_weight": weight,
        "route_seed_digest": seed,
        "routing_uniform_value": u,
        "router_selected_revision": selected,
        "backend_request_id": backend["request_id"],
        "backend_revision": backend_revision,
        "backend_model_sha256": backend["model_sha256"],
        "backend_preprocessor_sha256":
            backend["preprocessor_sha256"],
        "backend_raw_feature_count":
            backend["raw_feature_count"],
        "backend_transformed_feature_count":
            backend["transformed_feature_count"],
        "prediction": backend["prediction"],
        "probability": backend["probability"],
        "backend_processing_ms":
            backend["processing_ms"],
        "router_backend_ms": router_backend_ms,
        "route_match": True,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
    }

    if not EVIDENCE_PATH:
        raise HTTPException(
            500,
            "ROUTER_EVIDENCE_PATH_NOT_CONFIGURED",
        )

    serialized = json.dumps(
        internal_record,
        sort_keys=True,
        separators=(",", ":"),
    )

    try:
        with open(
            EVIDENCE_PATH,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(serialized + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as exc:
        raise HTTPException(
            500,
            "ROUTER_EVIDENCE_PERSISTENCE_FAILURE:"
            + type(exc).__name__,
        )

    # Oracle/revision remains hidden from the traffic client.
    return {
        "request_id": req.request_id,
        "prediction": backend["prediction"],
        "probability": backend["probability"],
        "processing_ms": backend["processing_ms"],
    }
