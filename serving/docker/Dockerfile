FROM python@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0

WORKDIR /app

COPY serving/api/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY serving/api/app.py /app/serving/api/app.py

COPY models/preprocessing/preprocessor.joblib \
     /app/models/preprocessing/preprocessor.joblib

COPY data/processed/feature_names.json \
     /app/data/processed/feature_names.json

COPY data/manifests/phase5_serving_protocol_v1.0.json \
     /app/data/manifests/phase5_serving_protocol_v1.0.json

COPY models/v1/model.json /app/models/v1/model.json
COPY models/v2/model.json /app/models/v2/model.json

EXPOSE 8000

CMD ["uvicorn", "serving.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
