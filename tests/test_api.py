"""Тести API. Запуск: pytest -q  (спершу треба python train.py)"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.app import app, scorer  # noqa: E402

CLIENT = {
    "checking_status": "lt_0", "duration": 36, "credit_history": "existing_paid",
    "purpose": "car_new", "credit_amount": 6000, "savings": "lt_100",
    "employment": "1_4", "installment_rate": 4, "other_debtors": "none",
    "residence_since": 2, "property": "car_other", "age": 28,
    "other_installment_plans": "none", "housing": "rent", "existing_credits": 1,
    "job": "skilled", "num_dependents": 1, "telephone": "no",
}

pytestmark = pytest.mark.skipif(scorer is None, reason="спершу запустіть train.py")


@pytest.fixture
def client():
    return app.test_client()


def test_health(client):
    assert client.get("/health").json["model_loaded"] is True


def test_predict(client):
    r = client.post("/predict", json=CLIENT)
    assert r.status_code == 200
    body = r.json
    assert 0 <= body["default_probability"] <= 1
    assert 300 <= body["credit_score"] <= 850
    assert body["decision"] in {"approve", "reject"}
    assert len(body["explanation"]["factors"]) > 0


def test_validation_error(client):
    bad = {**CLIENT, "savings": "a_lot"}
    bad.pop("age")
    r = client.post("/predict", json=bad)
    assert r.status_code == 400
    assert len(r.json["details"]) == 2


def test_riskier_client_has_higher_probability(client):
    safe = {**CLIENT, "checking_status": "no_account", "savings": "ge_1000",
            "duration": 12, "credit_amount": 1500, "credit_history": "critical"}
    p_risky = client.post("/predict", json=CLIENT).json["default_probability"]
    p_safe = client.post("/predict", json=safe).json["default_probability"]
    assert p_safe < p_risky


def test_batch(client):
    r = client.post("/predict/batch", json=[CLIENT, {**CLIENT, "age": "x"}])
    assert r.status_code == 200
    assert "credit_score" in r.json[0] and "error" in r.json[1]
