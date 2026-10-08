# AI-Powered Behavioural Anomaly Detection

Streaming pipeline: synthetic access-event generator → Kafka → detection +
classification + explainability → SQLite → Streamlit SOC dashboard.

**Detection:** an LSTM autoencoder is the primary detector — it scores a
sliding window of an entity's recent behaviour by reconstruction error,
so it can catch anomalies that only show up as a change in *pattern*
across several events, not just a single outlier event. An
IsolationForest is kept as a fallback for entities that don't have
enough history yet for a sequence window (cold start), or if the LSTM
runtime isn't available in a given deployment.

**Explainability:** every alert is explained with SHAP feature
attribution (`shap.TreeExplainer` for the IsolationForest fallback and
the RandomForest classifier, `shap.KernelExplainer` for the LSTM),
showing exactly which features pushed that specific event toward
"anomalous" and by how much — not just a generic rule. The original
rule-based explainer is kept as an automatic fallback if SHAP itself
fails to load or errors on a given alert.

## Project structure

```
behavioral-anomaly-detection/
├── docker-compose.yml          # wires kafka + producer + detector + dashboard
├── producer/                   # generates & streams synthetic access events
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── config.py           # simulation rates, Kafka topic/broker
│       ├── entity_profiles.py  # per-entity habitual baseline (Faker)
│       ├── event_generator.py  # normal event sampling
│       ├── attack_injectors.py # one function per attack pattern
│       ├── kafka_producer.py   # Kafka send wrapper
│       └── main.py             # entrypoint loop
├── detector/                   # consumes events, scores, classifies, explains
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── models/                 # trained .joblib artifacts go here
│   ├── train/
│   │   └── train_models.py     # OFFLINE: run before starting the stack
│   └── src/
│       ├── config.py
│       ├── kafka_consumer.py   # Kafka consume/publish wrapper
│       ├── baseline_profile.py # rolling per-entity state + cold-start
│       ├── feature_engineering.py
│       ├── detection_model.py  # HybridDetectionModel: routes to LSTM or IsolationForest
│       ├── lstm_detector.py    # LSTM autoencoder — PRIMARY detector
│       ├── classification_model.py  # RandomForest attack-type prediction
│       ├── shap_explainability.py   # SHAP per-alert explanation (primary)
│       ├── explainability.py   # rule-based per-alert explanation (fallback)
│       ├── storage.py          # SQLite read/write
│       └── main.py             # entrypoint stream loop
├── dashboard/                  # Streamlit SOC analyst UI
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── config.py
│       ├── data_access.py      # SQLite queries
│       ├── components.py       # reusable UI render functions
│       └── app.py              # entrypoint page
└── data/                       # local scratch space (CSV exports, notes)
```

## 1. Train the models (once, before first run)

```bash
python -m venv venv && source venv/bin/activate
pip install -r producer/requirements.txt -r detector/requirements.txt scikit-learn
python -m detector.train.train_models
```

This writes the LSTM autoencoder (`lstm_autoencoder.keras` +
`lstm_scaler.joblib` + `lstm_threshold.joblib`), the IsolationForest
fallback (`isolation_forest.joblib` + `scaler.joblib`), the classifier
(`attack_classifier.joblib`), and a small SHAP background sample
(`shap_background.joblib`) into `detector/models/` — required before
the detector container can start. Training the LSTM needs TensorFlow
and SHAP, both already listed in `detector/requirements.txt`.

## 2. Run the full stack locally

```bash
docker compose up --build
```

- Kafka broker: `localhost:9092`
- Dashboard: http://localhost:8501

The producer starts streaming synthetic events immediately; the
dashboard will populate once enough events have passed through the
rolling alert-threshold window (a few hundred events).

## 3. Deploying the dashboard live (free tier)

Streamlit Community Cloud / Hugging Face Spaces cannot run Kafka or
docker-compose — only a single Python app. For a live clickable demo,
run `detector/train/train_models.py` locally, pre-generate a batch of
scored alerts into a static `alerts.db`, commit that DB, and deploy
`dashboard/` alone (point `DB_PATH` at the committed file) to:
- share.streamlit.io, or
- huggingface.co/spaces (Streamlit SDK)

The full Kafka + Docker + streaming architecture remains demonstrated
via `docker compose up` locally (record a short screen capture for the
submission) — see the report for the scalability discussion.

## Notes

- All services are free/open-source: Kafka (bitnami image), scikit-learn,
  Streamlit, Faker — no paid APIs used anywhere.
- Ground-truth `label` field is generated but only used for offline
  training/evaluation — the detector never reads it at inference time.
