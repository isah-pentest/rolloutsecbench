FROM python:3.14-slim

WORKDIR /app

COPY serving/router/requirements.txt \
     /app/serving/router/requirements.txt

RUN pip install --no-cache-dir \
    -r /app/serving/router/requirements.txt

COPY serving/router/gate3_v2_candidates/route_oracle_v2_candidate003.py \
     /app/serving/router/gate3_v2_candidates/route_oracle_v2_candidate003.py

COPY serving/router/gate3_v2_candidates/router_v2_candidate003.py \
     /app/serving/router/gate3_v2_candidates/router_v2_candidate003.py

COPY data/manifests/phase6_router_validation_protocol_v2.1.json \
     /app/data/manifests/phase6_router_validation_protocol_v2.1.json

CMD ["uvicorn", "serving.router.gate3_v2_candidates.router_v2_candidate003:app", "--host", "0.0.0.0", "--port", "8000"]
