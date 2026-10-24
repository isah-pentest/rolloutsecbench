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
        "PHASE7_PROTOCOL_PATH",
        str(
            ROOT / "data/manifests/phase7/router/"
            "phase7_router_protocol_v1.0.json"
        ),
    )
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

V1_MODEL_SHA256 = os.environ.get(
    "V1_MODEL_SHA256",
    "",
)

V2_MODEL_SHA256 = os.environ.get(
    "V2_MODEL_SHA256",
    "",
)

EXPECTED_PREPROCESSOR_SHA256 = os.environ.get(
    "EXPECTED_PREPROCESSOR_SHA256",
    "",
)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def load_protocol():
    return json.loads(
        PROTOCOL_PATH.read_text(encoding="utf-8")
    )


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
    attempt_number: int = 1


app = FastAPI()


def route(req):
    state = req.rollout_state

    if state not in PROTOCOL["rollout_states"]:
        raise HTTPException(
            400,
            "Unknown rollout_state",
        )

    if req.request_sequence < 0:
        raise HTTPException(
            400,
            "Negative request_sequence",
        )

    if req.attempt_number < 1:
        raise HTTPException(
            400,
            "Invalid attempt_number",
        )

    weight = float(
        PROTOCOL["rollout_states"][state]
        ["v2_weight"]
    )

    basis = (
        PROTOCOL["routing"]
        ["route_seed_basis_sha256"]
    )

    # request_sequence must be globally unique within a run.
    material = (
        f"{basis}|{req.run_id}|"
        f"{req.request_sequence}"
    )

    digest = hashlib.sha256(
        material.encode("utf-8")
    ).digest()

    u = (
        int.from_bytes(
            digest[:8],
            "big",
            signed=False,
        )
        / (2 ** 64)
    )

    selected = "v2" if u < weight else "v1"

    return (
        selected,
        material,
        u,
        weight,
    )


def append_evidence(record):
    if not EVIDENCE_PATH:
        raise HTTPException(
            500,
            "ROUTER_EVIDENCE_PATH_NOT_CONFIGURED",
        )

    p = Path(EVIDENCE_PATH)
    p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with p.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                record,
                sort_keys=True,
            )
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


@app.get("/health")
def health():
    return {
        "status": "ok",
        "phase": 7,
        "protocol_sha256":
            sha256_file(PROTOCOL_PATH),
    }


@app.post("/predict")
def predict(req: PredictRequest):

    selected, material, u, weight = route(req)

    backend_url = (
        V2_BACKEND_URL
        if selected == "v2"
        else V1_BACKEND_URL
    )

    backend_payload = {
        "request_id": req.request_id,
        "features": req.features,
    }

    t0 = time.perf_counter()

    try:
        response = httpx.post(
            backend_url.rstrip("/")
            + "/predict",
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

    missing = sorted(
        required - set(backend)
    )

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

    expected_model = (
        V2_MODEL_SHA256
        if selected == "v2"
        else V1_MODEL_SHA256
    )

    if (
        expected_model
        and backend["model_sha256"]
        != expected_model
    ):
        raise HTTPException(
            500,
            "Backend model SHA mismatch",
        )

    if (
        EXPECTED_PREPROCESSOR_SHA256
        and backend["preprocessor_sha256"]
        != EXPECTED_PREPROCESSOR_SHA256
    ):
        raise HTTPException(
            500,
            "Backend preprocessor SHA mismatch",
        )

    evidence = {
        "experiment_id":
            req.experiment_id,
        "run_id":
            req.run_id,
        "request_id":
            req.request_id,
        "request_sequence":
            req.request_sequence,
        "replicate_id":
            req.replicate_id,
        "sample_id":
            req.sample_id,
        "ground_truth":
            req.ground_truth,
        "attack_family":
            req.attack_family,
        "attacker_attempt_number":
            req.attempt_number,
        "rollout_state":
            req.rollout_state,
        "connection_mode":
            req.connection_mode,
        "selected_revision":
            selected,
        "routing_material":
            material,
        "routing_uniform_value":
            u,
        "configured_v2_weight":
            weight,
        "backend_request_id":
            backend["request_id"],
        "backend_model_version":
            backend["model_version"],
        "prediction":
            backend["prediction"],
        "probability":
            backend["probability"],
        "backend_processing_ms":
            backend["processing_ms"],
        "router_backend_ms":
            router_backend_ms,
        "model_sha256":
            backend["model_sha256"],
        "preprocessor_sha256":
            backend["preprocessor_sha256"],
        "raw_feature_count":
            backend["raw_feature_count"],
        "transformed_feature_count":
            backend["transformed_feature_count"],
        "protocol_sha256":
            sha256_file(PROTOCOL_PATH),
    }

    append_evidence(evidence)

    return {
        "request_id":
            req.request_id,
        "selected_revision":
            selected,
        "prediction":
            backend["prediction"],
        "probability":
            backend["probability"],
        "model_sha256":
            backend["model_sha256"],
        "preprocessor_sha256":
            backend["preprocessor_sha256"],
        "routing_uniform_value":
            u,
        "configured_v2_weight":
            weight,
        "processing_ms":
            backend["processing_ms"],
    }
