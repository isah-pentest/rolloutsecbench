#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


ROOT = Path("/app")

MODEL_VERSION = os.environ["MODEL_VERSION"].strip().lower()
MODEL_RELATIVE_PATH = os.environ["MODEL_RELATIVE_PATH"]

if MODEL_VERSION not in {"v1", "v2"}:
    raise RuntimeError(
        f"MODEL_VERSION must be v1 or v2, got {MODEL_VERSION!r}"
    )

MODEL_PATH = ROOT / MODEL_RELATIVE_PATH
PREPROCESSOR_PATH = ROOT / "models/preprocessing/preprocessor.joblib"
FEATURE_NAMES_PATH = ROOT / "data/processed/feature_names.json"
PROTOCOL_PATH = ROOT / "data/manifests/phase5_serving_protocol_v1.0.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


protocol = json.loads(PROTOCOL_PATH.read_text())

expected_model = protocol["models"][MODEL_VERSION]

MODEL_SHA256 = sha256_file(MODEL_PATH)
PREPROCESSOR_SHA256 = sha256_file(PREPROCESSOR_PATH)
FEATURE_NAMES_SHA256 = sha256_file(FEATURE_NAMES_PATH)

if MODEL_SHA256 != expected_model["model_sha256"]:
    raise RuntimeError(
        f"{MODEL_VERSION} model hash mismatch: "
        f"{MODEL_SHA256} != {expected_model['model_sha256']}"
    )

if PREPROCESSOR_SHA256 != protocol["preprocessing"]["sha256"]:
    raise RuntimeError(
        "Preprocessor hash mismatch."
    )

if FEATURE_NAMES_SHA256 != protocol["preprocessing"]["feature_names_sha256"]:
    raise RuntimeError(
        "Feature-names hash mismatch."
    )

THRESHOLD = float(expected_model["decision_threshold"])

RAW_FEATURE_COUNT = int(
    protocol["preprocessing"]["raw_feature_count"]
)

TRANSFORMED_FEATURE_COUNT = int(
    protocol["preprocessing"]["transformed_feature_count"]
)

EXCLUDED = set(
    protocol["input_contract"]["excluded_from_model_input"]
)

CATEGORICAL_FEATURES = set(
    protocol["input_contract"]["categorical_features"]
)

preprocessor = joblib.load(PREPROCESSOR_PATH)

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

feature_names_payload = json.loads(FEATURE_NAMES_PATH.read_text())

if not isinstance(feature_names_payload, dict):
    raise RuntimeError(
        "feature_names.json must be a structured JSON object."
    )

required_feature_keys = {
    "raw_feature_count",
    "source_feature_order",
    "transformed_feature_count",
    "transformed_feature_order",
}

missing_feature_keys = sorted(
    required_feature_keys.difference(feature_names_payload)
)

if missing_feature_keys:
    raise RuntimeError(
        "feature_names.json missing required key(s): "
        + ", ".join(missing_feature_keys)
    )

source_feature_order = feature_names_payload["source_feature_order"]
feature_names = feature_names_payload["transformed_feature_order"]

if feature_names_payload["raw_feature_count"] != RAW_FEATURE_COUNT:
    raise RuntimeError(
        "feature_names.json raw feature count does not match protocol."
    )

if len(source_feature_order) != RAW_FEATURE_COUNT:
    raise RuntimeError(
        "Source feature-order length does not match protocol."
    )

if feature_names_payload["transformed_feature_count"] != TRANSFORMED_FEATURE_COUNT:
    raise RuntimeError(
        "feature_names.json transformed feature count does not match protocol."
    )

if len(feature_names) != TRANSFORMED_FEATURE_COUNT:
    raise RuntimeError(
        "Transformed feature-order length does not match protocol."
    )

preprocessor_feature_names = preprocessor.get_feature_names_out().tolist()

if len(preprocessor_feature_names) != TRANSFORMED_FEATURE_COUNT:
    raise RuntimeError(
        "Frozen preprocessor transformed feature count does not match protocol."
    )

if feature_names != preprocessor_feature_names:
    raise RuntimeError(
        "Frozen transformed feature order does not match preprocessor output order."
    )


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    features: dict[str, Any]


class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    prediction: int
    probability: float
    processing_ms: float
    model_sha256: str
    preprocessor_sha256: str
    raw_feature_count: int
    transformed_feature_count: int


app = FastAPI(
    title="RolloutSecBench Inference Service",
    version="1.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "model_sha256": MODEL_SHA256,
        "preprocessor_sha256": PREPROCESSOR_SHA256,
        "threshold": THRESHOLD,
        "raw_feature_count": RAW_FEATURE_COUNT,
        "transformed_feature_count": TRANSFORMED_FEATURE_COUNT,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()

    features = req.features

    forbidden_present = sorted(
        key for key in EXCLUDED if key in features
    )

    if forbidden_present:
        raise HTTPException(
            status_code=422,
            detail=(
                "Non-model field(s) supplied in features: "
                + ", ".join(forbidden_present)
            ),
        )

    expected_feature_set = set(source_feature_order)
    supplied_feature_set = set(features)

    missing_features = sorted(
        expected_feature_set.difference(supplied_feature_set)
    )
    unexpected_features = sorted(
        supplied_feature_set.difference(expected_feature_set)
    )

    if missing_features or unexpected_features:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "RAW_FEATURE_SCHEMA_MISMATCH",
                "missing_features": missing_features,
                "unexpected_features": unexpected_features,
                "expected_feature_count": len(source_feature_order),
                "received_feature_count": len(features),
            },
        )

    if len(features) != RAW_FEATURE_COUNT:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected exactly {RAW_FEATURE_COUNT} raw model "
                f"features; received {len(features)}."
            ),
        )

    try:
        row = pd.DataFrame(
            [[features[name] for name in source_feature_order]],
            columns=source_feature_order,
        )

        transformed = preprocessor.transform(row)

        if transformed.shape != (1, TRANSFORMED_FEATURE_COUNT):
            raise RuntimeError(
                f"Unexpected transformed shape: {transformed.shape}"
            )

        probability = float(
            model.predict_proba(transformed)[0, 1]
        )

        prediction = int(probability >= THRESHOLD)

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Inference failed: {type(exc).__name__}: {exc}",
        ) from exc

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return PredictResponse(
        request_id=req.request_id,
        model_version=MODEL_VERSION,
        prediction=prediction,
        probability=probability,
        processing_ms=elapsed_ms,
        model_sha256=MODEL_SHA256,
        preprocessor_sha256=PREPROCESSOR_SHA256,
        raw_feature_count=RAW_FEATURE_COUNT,
        transformed_feature_count=TRANSFORMED_FEATURE_COUNT,
    )
