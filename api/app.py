"""Flask REST API для real-time кредитного скорингу.

Endpoints:
  GET  /health         — перевірка, що сервіс живий і модель завантажена
  GET  /model/info     — метрики, ваги ансамблю, поріг
  GET  /schema         — список полів і допустимих значень
  POST /predict        — скоринг одного клієнта (JSON-об'єкт)
  POST /predict/batch  — скоринг списку клієнтів (JSON-масив)
"""
import os
import sys
import time
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # щоб joblib знайшов клас src.model.CreditScorer

from src.data import CODE_MAPS  # noqa: E402
from src.features import CATEGORICAL, NUMERIC_RAW  # noqa: E402

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "credit_scorer.joblib"))

app = Flask(__name__)
scorer = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

REQUIRED = CATEGORICAL + NUMERIC_RAW
ALLOWED = {c: sorted(CODE_MAPS[c].values()) for c in CATEGORICAL}


def validate(client: dict) -> list:
    """Повертає список помилок валідації (порожній = все гаразд)."""
    errors = []
    if not isinstance(client, dict):
        return ["очікується JSON-об'єкт"]
    for f in REQUIRED:
        if f not in client:
            errors.append(f"відсутнє поле '{f}'")
    for f in CATEGORICAL:
        if f in client and client[f] not in ALLOWED[f]:
            errors.append(f"'{f}' має бути одним із {ALLOWED[f]}")
    for f in NUMERIC_RAW:
        if f in client:
            try:
                if float(client[f]) < 0:
                    errors.append(f"'{f}' не може бути від'ємним")
            except (TypeError, ValueError):
                errors.append(f"'{f}' має бути числом")
    if not errors and float(client["duration"]) == 0:
        errors.append("'duration' має бути > 0")
    return errors


def model_required():
    if scorer is None:
        return jsonify({"error": f"модель не знайдена: {MODEL_PATH}. Запустіть train.py"}), 503
    return None


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": scorer is not None})


@app.get("/model/info")
def model_info():
    if (err := model_required()):
        return err
    return jsonify(scorer.metrics)


@app.get("/schema")
def schema():
    return jsonify({"categorical": ALLOWED, "numeric": NUMERIC_RAW})


@app.post("/predict")
def predict():
    if (err := model_required()):
        return err
    client = request.get_json(silent=True)
    if (errors := validate(client)):
        return jsonify({"error": "validation_failed", "details": errors}), 400
    t0 = time.perf_counter()
    result = scorer.score_client({k: client[k] for k in REQUIRED})
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return jsonify(result)


@app.post("/predict/batch")
def predict_batch():
    if (err := model_required()):
        return err
    clients = request.get_json(silent=True)
    if not isinstance(clients, list) or not clients:
        return jsonify({"error": "очікується непорожній JSON-масив клієнтів"}), 400
    results = []
    for i, client in enumerate(clients):
        if (errors := validate(client)):
            results.append({"index": i, "error": errors})
        else:
            r = scorer.score_client({k: client[k] for k in REQUIRED})
            r.pop("explanation")  # у пакетному режимі пояснення не повертаємо
            results.append({"index": i, **r})
    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
