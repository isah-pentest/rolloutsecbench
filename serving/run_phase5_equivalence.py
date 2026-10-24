#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_PATH = (
    ROOT / "data/manifests/phase5_serving_protocol_v1.0.json"
)

RUNTIME_PATH = (
    ROOT
    / "results/derived/phase5_preflight/phase5_runtime_manifest.json"
)

PRE_STATE_PATH = (
    ROOT
    / "results/derived/phase5_preflight/pre_equivalence_runtime_state.json"
)

FEATURE_NAMES_PATH = ROOT / "data/processed/feature_names.json"

DERIVED_ROOT = ROOT / "results/derived/phase5_equivalence"
RAW_ROOT = ROOT / "results/raw/phase5_serving/equivalence"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def native(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def require_file(path: Path):
    if not path.is_file():
        raise SystemExit(
            f"Required file missing: {path.relative_to(ROOT)}"
        )


def docker_inspect(name: str) -> dict:
    return json.loads(
        subprocess.check_output(
            ["docker", "inspect", name],
            text=True,
        )
    )[0]


def percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]

    pos = (len(s) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)

    if lower == upper:
        return s[lower]

    weight = pos - lower
    return (
        s[lower] * (1.0 - weight)
        + s[upper] * weight
    )


def numeric_summary(values):
    values = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]

    if not values:
        return {
            "n": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    return {
        "n": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


for required in (
    PROTOCOL_PATH,
    RUNTIME_PATH,
    PRE_STATE_PATH,
    FEATURE_NAMES_PATH,
):
    require_file(required)


protocol = json.loads(PROTOCOL_PATH.read_text())
runtime = json.loads(RUNTIME_PATH.read_text())
pre_state = json.loads(PRE_STATE_PATH.read_text())
feature_manifest = json.loads(FEATURE_NAMES_PATH.read_text())


# ------------------------------------------------------------------
# Scientific constraints
# ------------------------------------------------------------------

constraints = protocol["scientific_constraints"]

for key in (
    "training_allowed",
    "model_refit_allowed",
    "preprocessor_refit_allowed",
    "threshold_tuning_allowed",
    "hyperparameter_tuning_allowed",
    "malicious_disagreement_recomputation_allowed",
    "router_analysis_allowed",
    "rollout_analysis_allowed",
):
    if constraints[key] is not False:
        raise SystemExit(
            f"Scientific constraint violation: {key}"
        )

if constraints["test_data_access_required"] is not False:
    raise SystemExit(
        "Phase-5 protocol unexpectedly requires test-data access."
    )


# ------------------------------------------------------------------
# Frozen contract
# ------------------------------------------------------------------

population = protocol["equivalence_population"]
preprocessing = protocol["preprocessing"]
api_contract = protocol["api_contract"]
rules = protocol["equivalence_rules"]

validation_path = ROOT / population["path"]

offline_paths = {
    revision:
        ROOT / protocol["models"][revision]["offline_reference_path"]
    for revision in protocol["models"]
}

require_file(validation_path)

for p in offline_paths.values():
    require_file(p)


# ------------------------------------------------------------------
# Hash verification
# ------------------------------------------------------------------

if sha256_file(validation_path) != population["sha256"]:
    raise SystemExit("Validation split hash mismatch.")

if (
    sha256_file(FEATURE_NAMES_PATH)
    != preprocessing["feature_names_sha256"]
):
    raise SystemExit("Feature-name artifact hash mismatch.")

for revision in protocol["models"]:
    expected = (
        protocol["models"][revision]["offline_reference_sha256"]
    )
    observed = sha256_file(offline_paths[revision])

    if observed != expected:
        raise SystemExit(
            f"{revision}: offline-reference hash mismatch."
        )


# ------------------------------------------------------------------
# Frozen population/schema
# ------------------------------------------------------------------

validation = pd.read_csv(validation_path)

if len(validation) != int(population["rows"]):
    raise SystemExit("Validation population row-count mismatch.")

if len(validation.columns) != int(population["columns"]):
    raise SystemExit("Validation population column-count mismatch.")

join_key = population["join_key"]
label_col = population["ground_truth_column"]
attack_col = population["reporting_only_column"]

if (
    population["join_key_unique"]
    and validation[join_key].duplicated().any()
):
    raise SystemExit("Validation join key is not unique.")

raw_feature_order = feature_manifest["source_feature_order"]

if (
    len(raw_feature_order)
    != preprocessing["raw_feature_count"]
):
    raise SystemExit("Raw feature count mismatch.")

excluded = protocol["input_contract"]["excluded_from_model_input"]

expected_columns = (
    [join_key]
    + raw_feature_order
    + [attack_col, label_col]
)

if list(validation.columns) != expected_columns:
    raise SystemExit(
        "Validation schema/order differs from frozen feature contract."
    )

if set(excluded).intersection(raw_feature_order):
    raise SystemExit(
        "Excluded columns unexpectedly appear in model feature order."
    )


# ------------------------------------------------------------------
# Offline references
# ------------------------------------------------------------------

offline = {}

for revision in protocol["models"]:

    df = pd.read_csv(offline_paths[revision])

    expected_cols = [
        join_key,
        attack_col,
        label_col,
        "probability",
        "prediction",
    ]

    if list(df.columns) != expected_cols:
        raise SystemExit(
            f"{revision}: offline-reference schema mismatch."
        )

    if df[join_key].duplicated().any():
        raise SystemExit(
            f"{revision}: offline join key is not unique."
        )

    if len(df) != len(validation):
        raise SystemExit(
            f"{revision}: offline population length mismatch."
        )

    if set(df[join_key]) != set(validation[join_key]):
        raise SystemExit(
            f"{revision}: offline population identity mismatch."
        )

    offline[revision] = df.set_index(join_key, drop=False)


# ------------------------------------------------------------------
# Equivalence semantics
# ------------------------------------------------------------------

if rules["probability_metric"] != "ABSOLUTE_DIFFERENCE":
    raise SystemExit("Unsupported probability metric.")

if rules["probability_operator"] != "STRICTLY_LESS_THAN":
    raise SystemExit("Unsupported probability operator.")

if rules["discrete_prediction_rule"] != "EXACT_MATCH":
    raise SystemExit("Unsupported discrete-prediction rule.")

epsilon = float(rules["probability_epsilon"])


# ------------------------------------------------------------------
# Live runtime identity check
# ------------------------------------------------------------------

for revision in protocol["models"]:

    expected = runtime["runtime"][revision]

    current = docker_inspect(expected["container_name"])

    if not current["State"]["Running"]:
        raise SystemExit(
            f"{revision}: container is not running."
        )

    if current["State"]["OOMKilled"]:
        raise SystemExit(
            f"{revision}: container is OOM-killed."
        )

    if current["RestartCount"] != 0:
        raise SystemExit(
            f"{revision}: nonzero restart count."
        )

    if current["Id"] != expected["container_id"]:
        raise SystemExit(
            f"{revision}: container identity drift."
        )

    if current["Image"] != expected["image_id"]:
        raise SystemExit(
            f"{revision}: image identity drift."
        )


# ------------------------------------------------------------------
# Unique run directories
# ------------------------------------------------------------------

run_id = f"P5-EQ-{run_timestamp()}"

raw_dir = RAW_ROOT / run_id
derived_dir = DERIVED_ROOT / run_id

raw_dir.mkdir(parents=True, exist_ok=False)
derived_dir.mkdir(parents=True, exist_ok=False)

raw_jsonl = raw_dir / "requests.jsonl"
stats_jsonl = raw_dir / "docker_stats.jsonl"
pre_container_state = raw_dir / "pre_container_state.json"
post_container_state = raw_dir / "post_container_state.json"
summary_path = derived_dir / "summary.json"
csv_path = derived_dir / "serving_equivalence.csv"
manifest_path = derived_dir / "manifest.json"


# ------------------------------------------------------------------
# Capture pre-container state
# ------------------------------------------------------------------

pre_snapshot = {
    revision: docker_inspect(
        runtime["runtime"][revision]["container_name"]
    )
    for revision in protocol["models"]
}

pre_container_state.write_text(
    json.dumps(pre_snapshot, indent=2) + "\n"
)


# ------------------------------------------------------------------
# Background raw Docker statistics
# Diagnostic only: not a Phase-5 PASS criterion.
# ------------------------------------------------------------------

stop_stats = threading.Event()


def stats_worker():
    with stats_jsonl.open(
        "x",
        encoding="utf-8",
        buffering=1,
    ) as f:

        while not stop_stats.is_set():

            for revision in protocol["models"]:

                name = runtime["runtime"][revision]["container_name"]

                try:
                    output = subprocess.check_output(
                        [
                            "docker",
                            "stats",
                            "--no-stream",
                            "--format",
                            "{{json .}}",
                            name,
                        ],
                        text=True,
                        stderr=subprocess.STDOUT,
                    ).strip()

                    payload = (
                        json.loads(output)
                        if output
                        else None
                    )

                    record = {
                        "timestamp_utc": utc_now(),
                        "revision": revision,
                        "container_name": name,
                        "container_id":
                            runtime["runtime"][revision]["container_id"],
                        "image_id":
                            runtime["runtime"][revision]["image_id"],
                        "stats": payload,
                        "error": None,
                    }

                except Exception as exc:
                    record = {
                        "timestamp_utc": utc_now(),
                        "revision": revision,
                        "container_name": name,
                        "container_id":
                            runtime["runtime"][revision]["container_id"],
                        "image_id":
                            runtime["runtime"][revision]["image_id"],
                        "stats": None,
                        "error":
                            f"{type(exc).__name__}: {exc}",
                    }

                f.write(
                    json.dumps(
                        record,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                f.flush()

            stop_stats.wait(5.0)


stats_thread = threading.Thread(
    target=stats_worker,
    daemon=True,
)

stats_thread.start()


# ------------------------------------------------------------------
# Execute full frozen equivalence population
# ------------------------------------------------------------------

records = []

run_started_utc = utc_now()
run_started_perf = time.perf_counter()

sequence = 0

try:

    with raw_jsonl.open(
        "x",
        encoding="utf-8",
        buffering=1,
    ) as raw_out:

        for revision in protocol["models"]:

            model_contract = protocol["models"][revision]

            endpoint = (
                f"http://127.0.0.1:"
                f"{runtime['runtime'][revision]['host_port']}"
                f"{api_contract['endpoint']}"
            )

            reference_df = offline[revision]

            for row_number, (_, row) in enumerate(
                validation.iterrows(),
                start=1,
            ):

                sequence += 1

                sample_id = native(row[join_key])

                reference = reference_df.loc[sample_id]

                request_id = (
                    f"{run_id}-{revision.upper()}-{row_number:06d}"
                )

                features = {
                    name: native(row[name])
                    for name in raw_feature_order
                }

                payload = {
                    "request_id": request_id,
                    "features": features,
                }

                request_bytes = json.dumps(
                    payload,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")

                request = urllib.request.Request(
                    endpoint,
                    data=request_bytes,
                    headers={
                        "Content-Type": "application/json"
                    },
                    method=api_contract["method"],
                )

                start_utc = utc_now()
                t0 = time.perf_counter_ns()

                http_status = None
                response_body = b""
                error_type = None
                error_message = None

                try:
                    with urllib.request.urlopen(
                        request,
                        timeout=30,
                    ) as response:

                        http_status = response.status
                        response_body = response.read()

                except urllib.error.HTTPError as exc:

                    http_status = exc.code
                    response_body = exc.read()
                    error_type = type(exc).__name__
                    error_message = str(exc)

                except Exception as exc:

                    error_type = type(exc).__name__
                    error_message = str(exc)

                t1 = time.perf_counter_ns()
                end_utc = utc_now()

                client_latency_ms = (
                    t1 - t0
                ) / 1_000_000.0

                parsed = None
                parse_error = None

                if response_body:
                    try:
                        parsed = json.loads(response_body)
                    except Exception as exc:
                        parse_error = (
                            f"{type(exc).__name__}: {exc}"
                        )

                offline_probability = float(
                    reference["probability"]
                )
                offline_prediction = int(
                    reference["prediction"]
                )

                container_probability = None
                container_prediction = None
                absolute_difference = None

                if isinstance(parsed, dict):

                    p = parsed.get("probability")

                    if isinstance(p, (int, float)):
                        container_probability = float(p)

                    pred = parsed.get("prediction")

                    if pred in (0, 1):
                        container_prediction = int(pred)

                if (
                    container_probability is not None
                    and math.isfinite(container_probability)
                ):
                    absolute_difference = abs(
                        offline_probability
                        - container_probability
                    )

                probability_within_epsilon = (
                    absolute_difference is not None
                    and absolute_difference < epsilon
                )

                prediction_exact_match = (
                    container_prediction is not None
                    and container_prediction
                    == offline_prediction
                )

                request_id_match = (
                    isinstance(parsed, dict)
                    and parsed.get("request_id")
                    == request_id
                )

                model_version_match = (
                    isinstance(parsed, dict)
                    and parsed.get("model_version")
                    == revision
                )

                model_hash_match = (
                    isinstance(parsed, dict)
                    and parsed.get("model_sha256")
                    == model_contract["model_sha256"]
                )

                preprocessor_hash_match = (
                    isinstance(parsed, dict)
                    and parsed.get("preprocessor_sha256")
                    == preprocessing["sha256"]
                )

                raw_feature_count_match = (
                    isinstance(parsed, dict)
                    and parsed.get("raw_feature_count")
                    == preprocessing["raw_feature_count"]
                )

                transformed_feature_count_match = (
                    isinstance(parsed, dict)
                    and parsed.get("transformed_feature_count")
                    == preprocessing["transformed_feature_count"]
                )

                response_fields_complete = (
                    isinstance(parsed, dict)
                    and set(
                        api_contract["response_required_fields"]
                    ).issubset(parsed)
                )

                processing_ms = (
                    parsed.get("processing_ms")
                    if isinstance(parsed, dict)
                    else None
                )

                processing_ms_valid = (
                    isinstance(processing_ms, (int, float))
                    and math.isfinite(float(processing_ms))
                    and float(processing_ms) >= 0
                )

                row_pass = all([
                    http_status == 200,
                    isinstance(parsed, dict),
                    response_fields_complete,
                    request_id_match,
                    model_version_match,
                    model_hash_match,
                    preprocessor_hash_match,
                    raw_feature_count_match,
                    transformed_feature_count_match,
                    probability_within_epsilon,
                    prediction_exact_match,
                    processing_ms_valid,
                ])

                record = {
                    "equivalence_run_id": run_id,
                    "request_sequence": sequence,
                    "revision_row_number": row_number,
                    "request_id": request_id,
                    "sample_id": sample_id,
                    "ground_truth":
                        native(row[label_col]),
                    "attack_family":
                        str(row[attack_col]),
                    "expected_revision":
                        revision,
                    "reported_revision":
                        parsed.get("model_version")
                        if isinstance(parsed, dict)
                        else None,
                    "expected_model_sha256":
                        model_contract["model_sha256"],
                    "reported_model_sha256":
                        parsed.get("model_sha256")
                        if isinstance(parsed, dict)
                        else None,
                    "expected_preprocessor_sha256":
                        preprocessing["sha256"],
                    "reported_preprocessor_sha256":
                        parsed.get("preprocessor_sha256")
                        if isinstance(parsed, dict)
                        else None,
                    "offline_probability":
                        offline_probability,
                    "container_probability":
                        container_probability,
                    "absolute_probability_difference":
                        absolute_difference,
                    "probability_within_epsilon":
                        probability_within_epsilon,
                    "offline_prediction":
                        offline_prediction,
                    "container_prediction":
                        container_prediction,
                    "prediction_exact_match":
                        prediction_exact_match,
                    "request_id_match":
                        request_id_match,
                    "model_version_match":
                        model_version_match,
                    "model_hash_match":
                        model_hash_match,
                    "preprocessor_hash_match":
                        preprocessor_hash_match,
                    "raw_feature_count_match":
                        raw_feature_count_match,
                    "transformed_feature_count_match":
                        transformed_feature_count_match,
                    "response_required_fields_complete":
                        response_fields_complete,
                    "request_start_utc":
                        start_utc,
                    "request_end_utc":
                        end_utc,
                    "client_latency_ms":
                        client_latency_ms,
                    "server_processing_ms":
                        float(processing_ms)
                        if processing_ms_valid
                        else None,
                    "request_payload_bytes":
                        len(request_bytes),
                    "response_bytes":
                        len(response_body),
                    "http_status":
                        http_status,
                    "response_parse_error":
                        parse_error,
                    "error_type":
                        error_type,
                    "error_message":
                        error_message,
                    "row_pass":
                        row_pass,
                }

                raw_out.write(
                    json.dumps(
                        record,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                raw_out.flush()

                records.append(record)

                if (
                    row_number == 1
                    or row_number % 1000 == 0
                    or row_number == len(validation)
                ):
                    print(
                        f"{revision}: "
                        f"{row_number}/{len(validation)} "
                        f"requests completed"
                    )

finally:
    stop_stats.set()
    stats_thread.join(timeout=10)


run_end_perf = time.perf_counter()
run_ended_utc = utc_now()

wall_seconds = run_end_perf - run_started_perf


# ------------------------------------------------------------------
# Post-container state
# ------------------------------------------------------------------

post_snapshot = {
    revision: docker_inspect(
        runtime["runtime"][revision]["container_name"]
    )
    for revision in protocol["models"]
}

post_container_state.write_text(
    json.dumps(post_snapshot, indent=2) + "\n"
)


# ------------------------------------------------------------------
# Derived request-level CSV from immutable raw records
# ------------------------------------------------------------------

csv_fields = [
    "equivalence_run_id",
    "request_sequence",
    "revision_row_number",
    "request_id",
    "sample_id",
    "ground_truth",
    "attack_family",
    "expected_revision",
    "reported_revision",
    "expected_model_sha256",
    "reported_model_sha256",
    "expected_preprocessor_sha256",
    "reported_preprocessor_sha256",
    "offline_probability",
    "container_probability",
    "absolute_probability_difference",
    "probability_within_epsilon",
    "offline_prediction",
    "container_prediction",
    "prediction_exact_match",
    "request_id_match",
    "model_version_match",
    "model_hash_match",
    "preprocessor_hash_match",
    "raw_feature_count_match",
    "transformed_feature_count_match",
    "response_required_fields_complete",
    "request_start_utc",
    "request_end_utc",
    "client_latency_ms",
    "server_processing_ms",
    "request_payload_bytes",
    "response_bytes",
    "http_status",
    "response_parse_error",
    "error_type",
    "error_message",
    "row_pass",
]

with csv_path.open(
    "x",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_fields,
    )

    writer.writeheader()

    for record in records:
        writer.writerow(
            {
                field: record.get(field)
                for field in csv_fields
            }
        )


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

revisions = list(protocol["models"])

expected_requests = (
    int(population["rows"])
    * len(revisions)
)

attempted_requests = len(records)

http_failures = sum(
    r["http_status"] != 200
    for r in records
)

prediction_mismatches = sum(
    not r["prediction_exact_match"]
    for r in records
)

probability_violations = sum(
    not r["probability_within_epsilon"]
    for r in records
)

model_version_mismatches = sum(
    not r["model_version_match"]
    for r in records
)

model_hash_mismatches = sum(
    not r["model_hash_match"]
    for r in records
)

preprocessor_hash_mismatches = sum(
    not r["preprocessor_hash_match"]
    for r in records
)

request_id_mismatches = sum(
    not r["request_id_match"]
    for r in records
)

schema_mismatches = sum(
    not (
        r["raw_feature_count_match"]
        and r["transformed_feature_count_match"]
        and r["response_required_fields_complete"]
    )
    for r in records
)

failed_rows = sum(
    not r["row_pass"]
    for r in records
)

differences = [
    r["absolute_probability_difference"]
    for r in records
    if r["absolute_probability_difference"] is not None
]

client_latencies = [
    r["client_latency_ms"]
    for r in records
]

server_times = [
    r["server_processing_ms"]
    for r in records
    if r["server_processing_ms"] is not None
]


per_revision = {}

for revision in revisions:

    subset = [
        r
        for r in records
        if r["expected_revision"] == revision
    ]

    per_revision[revision] = {
        "expected_requests":
            int(population["rows"]),
        "attempted_requests":
            len(subset),
        "failed_rows":
            sum(not r["row_pass"] for r in subset),
        "http_failures":
            sum(r["http_status"] != 200 for r in subset),
        "prediction_mismatches":
            sum(
                not r["prediction_exact_match"]
                for r in subset
            ),
        "probability_violations":
            sum(
                not r["probability_within_epsilon"]
                for r in subset
            ),
        "model_version_mismatches":
            sum(
                not r["model_version_match"]
                for r in subset
            ),
        "maximum_absolute_probability_difference":
            max(
                [
                    r["absolute_probability_difference"]
                    for r in subset
                    if r["absolute_probability_difference"]
                    is not None
                ],
                default=None,
            ),
    }


post_runtime_pass = True

for revision in revisions:

    before = runtime["runtime"][revision]
    after = post_snapshot[revision]

    state_pass = all([
        after["State"]["Running"],
        not after["State"]["OOMKilled"],
        after["RestartCount"] == 0,
        after["Id"] == before["container_id"],
        after["Image"] == before["image_id"],
    ])

    post_runtime_pass = (
        post_runtime_pass and state_pass
    )


phase5_pass = all([
    attempted_requests == expected_requests,
    http_failures
        <= int(rules["maximum_http_failures"]),
    prediction_mismatches
        <= int(
            rules["maximum_discrete_prediction_mismatches"]
        ),
    model_version_mismatches
        <= int(
            rules["maximum_model_version_mismatches"]
        ),
    probability_violations == 0,
    model_hash_mismatches == 0,
    preprocessor_hash_mismatches == 0,
    request_id_mismatches == 0,
    schema_mismatches == 0,
    failed_rows == 0,
    post_runtime_pass,
])


summary = {
    "artifact": protocol["artifact"],
    "phase": protocol["phase"],
    "record_type": "FULL_SERVING_EQUIVALENCE_SUMMARY",
    "equivalence_run_id": run_id,
    "run_started_utc": run_started_utc,
    "run_ended_utc": run_ended_utc,
    "wall_clock_seconds": wall_seconds,
    "population": {
        "name": population["name"],
        "path": population["path"],
        "rows_per_revision": population["rows"],
        "revisions": revisions,
        "expected_requests": expected_requests,
    },
    "equivalence_rule": {
        "probability_metric":
            rules["probability_metric"],
        "probability_operator":
            rules["probability_operator"],
        "probability_epsilon":
            rules["probability_epsilon"],
        "discrete_prediction_rule":
            rules["discrete_prediction_rule"],
    },
    "counts": {
        "attempted_requests":
            attempted_requests,
        "http_failures":
            http_failures,
        "prediction_mismatches":
            prediction_mismatches,
        "probability_violations":
            probability_violations,
        "model_version_mismatches":
            model_version_mismatches,
        "model_hash_mismatches":
            model_hash_mismatches,
        "preprocessor_hash_mismatches":
            preprocessor_hash_mismatches,
        "request_id_mismatches":
            request_id_mismatches,
        "schema_mismatches":
            schema_mismatches,
        "failed_rows":
            failed_rows,
    },
    "probability_difference": numeric_summary(
        differences
    ),
    "client_latency_ms": numeric_summary(
        client_latencies
    ),
    "server_processing_ms": numeric_summary(
        server_times
    ),
    "throughput_requests_per_second": (
        attempted_requests / wall_seconds
        if wall_seconds > 0
        else None
    ),
    "per_revision": per_revision,
    "post_runtime_pass": post_runtime_pass,
    "phase5_pass": phase5_pass,
    "phase6_authorized": (
        phase5_pass
        and protocol["continuation_rule"][
            "phase6_router_validation_authorized_only_after_phase5_pass"
        ]
    ),
}

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
        allow_nan=False,
    )
    + "\n"
)


# ------------------------------------------------------------------
# Manifest
# ------------------------------------------------------------------

script_path = Path(__file__).resolve()

manifest = {
    "artifact": protocol["artifact"],
    "phase": protocol["phase"],
    "record_type": "FULL_SERVING_EQUIVALENCE_MANIFEST",
    "equivalence_run_id": run_id,
    "created_at_utc": utc_now(),
    "inputs": {
        "runner": {
            "path": str(script_path.relative_to(ROOT)),
            "sha256": sha256_file(script_path),
        },
        "protocol": {
            "path": str(PROTOCOL_PATH.relative_to(ROOT)),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "runtime_manifest": {
            "path": str(RUNTIME_PATH.relative_to(ROOT)),
            "sha256": sha256_file(RUNTIME_PATH),
        },
        "pre_equivalence_runtime_state": {
            "path": str(PRE_STATE_PATH.relative_to(ROOT)),
            "sha256": sha256_file(PRE_STATE_PATH),
        },
        "feature_names": {
            "path": str(FEATURE_NAMES_PATH.relative_to(ROOT)),
            "sha256": sha256_file(FEATURE_NAMES_PATH),
        },
        "validation_population": {
            "path": str(validation_path.relative_to(ROOT)),
            "sha256": sha256_file(validation_path),
        },
        "offline_references": {
            revision: {
                "path": str(
                    offline_paths[revision].relative_to(ROOT)
                ),
                "sha256": sha256_file(
                    offline_paths[revision]
                ),
            }
            for revision in revisions
        },
    },
    "runtime": {
        revision: {
            "container_name":
                runtime["runtime"][revision]["container_name"],
            "container_id":
                runtime["runtime"][revision]["container_id"],
            "image_id":
                runtime["runtime"][revision]["image_id"],
            "host_port":
                runtime["runtime"][revision]["host_port"],
        }
        for revision in revisions
    },
    "outputs": {
        "raw_requests": {
            "path": str(raw_jsonl.relative_to(ROOT)),
            "sha256": sha256_file(raw_jsonl),
        },
        "docker_stats": {
            "path": str(stats_jsonl.relative_to(ROOT)),
            "sha256": sha256_file(stats_jsonl),
        },
        "pre_container_state": {
            "path": str(
                pre_container_state.relative_to(ROOT)
            ),
            "sha256": sha256_file(
                pre_container_state
            ),
        },
        "post_container_state": {
            "path": str(
                post_container_state.relative_to(ROOT)
            ),
            "sha256": sha256_file(
                post_container_state
            ),
        },
        "derived_csv": {
            "path": str(csv_path.relative_to(ROOT)),
            "sha256": sha256_file(csv_path),
        },
        "summary": {
            "path": str(summary_path.relative_to(ROOT)),
            "sha256": sha256_file(summary_path),
        },
    },
    "decision": {
        "phase5_pass":
            phase5_pass,
        "phase6_authorized":
            summary["phase6_authorized"],
    },
}

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
        allow_nan=False,
    )
    + "\n"
)


print()
print("=" * 70)
print("PHASE 5 FULL SERVING EQUIVALENCE")
print("=" * 70)
print("Run ID:", run_id)
print("Expected requests:", expected_requests)
print("Attempted requests:", attempted_requests)
print("HTTP failures:", http_failures)
print("Prediction mismatches:", prediction_mismatches)
print("Probability violations:", probability_violations)
print("Model-version mismatches:", model_version_mismatches)
print("Model-hash mismatches:", model_hash_mismatches)
print(
    "Preprocessor-hash mismatches:",
    preprocessor_hash_mismatches,
)
print("Request-ID mismatches:", request_id_mismatches)
print("Schema mismatches:", schema_mismatches)
print("Failed rows:", failed_rows)
print(
    "Max probability difference:",
    summary["probability_difference"]["max"],
)
print("Post-runtime state:", "PASS" if post_runtime_pass else "FAIL")
print("PHASE 5:", "PASS" if phase5_pass else "FAIL")
print(
    "PHASE 6 AUTHORIZED:",
    "YES" if summary["phase6_authorized"] else "NO",
)
print()
print("Raw requests SHA256 :", sha256_file(raw_jsonl))
print("Derived CSV SHA256  :", sha256_file(csv_path))
print("Summary SHA256      :", sha256_file(summary_path))
print("Manifest SHA256     :", sha256_file(manifest_path))

if not phase5_pass:
    raise SystemExit(1)
