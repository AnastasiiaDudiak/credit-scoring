"""Ансамбль моделей на деревах, метрики скорингу та пояснення рішень."""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from src.features import FEATURES, add_features, build_preprocessor

try:
    from lightgbm import LGBMClassifier
except ImportError:  # дозволяє запуск і без lightgbm (буде попередження)
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# Матриця вартості помилок з опису German Credit:
# видати кредит поганому клієнту (FN) у 5 разів дорожче, ніж відмовити доброму (FP)
COST_FN, COST_FP = 5, 1


def make_models(random_state: int = 42) -> dict:
    """Три моделі на деревах з помірними параметрами (датасет малий — 1000 рядків,
    тому неглибокі дерева та невеликий learning rate проти перенавчання)."""
    models = {
        "rf": RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=random_state),
    }
    if LGBMClassifier is not None:
        models["lgb"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=5,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, class_weight="balanced",
            random_state=random_state, verbose=-1)
    else:
        warnings.warn("lightgbm не встановлено — модель 'lgb' пропущено")
    if XGBClassifier is not None:
        models["xgb"] = XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=7 / 3,
            eval_metric="auc", random_state=random_state)
    else:
        warnings.warn("xgboost не встановлено — модель 'xgb' пропущено")
    return models


def clone_model(model):
    from sklearn.base import clone
    return clone(model)


# ----------------------------- метрики ---------------------------------------
def ks_statistic(y_true, y_prob) -> float:
    """Колмогоров–Смірнов: максимальна відстань між розподілами скорів
    поганих і добрих клієнтів = max(TPR - FPR)."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def best_cost_threshold(y_true, y_prob) -> float:
    """Поріг, що мінімізує сумарну вартість помилок COST_FN*FN + COST_FP*FP."""
    y_true = np.asarray(y_true)
    grid = np.linspace(0.05, 0.95, 181)
    costs = []
    for t in grid:
        pred = (y_prob >= t).astype(int)
        fn = np.sum((pred == 0) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        costs.append(COST_FN * fn + COST_FP * fp)
    return float(grid[int(np.argmin(costs))])


def prob_to_score(p_default, base_score=600, base_odds=1.0, pdo=50):
    """Переводить імовірність дефолту у скоровий бал (як у бюро кредитних історій).
    score = offset + factor * ln(odds_good), де factor = PDO / ln 2.
    PDO (points to double the odds): +50 балів => шанси «добрий:поганий» удвічі вищі."""
    p = np.clip(np.asarray(p_default, dtype=float), 1e-6, 1 - 1e-6)
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    score = offset + factor * np.log((1 - p) / p)
    return np.clip(np.round(score), 300, 850).astype(int)


# ----------------------------- ансамбль --------------------------------------
class CreditScoringEnsemble:
    """Зважений ансамбль: вага кожної моделі пропорційна її ROC-AUC на крос-валідації."""

    def __init__(self, random_state: int = 42, n_splits: int = 5):
        self.random_state = random_state
        self.n_splits = n_splits
        self.models = make_models(random_state)
        self.weights = {}
        self.cv_auc = {}
        self.oof_proba = None

    def fit(self, X, y):
        y = np.asarray(y)
        skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        oof = {name: np.zeros(len(y)) for name in self.models}
        # 1) Out-of-fold прогнози кожної моделі -> чесна оцінка якості
        for train_idx, val_idx in skf.split(X, y):
            for name, model in self.models.items():
                m = clone_model(model)
                m.fit(X[train_idx], y[train_idx])
                oof[name][val_idx] = m.predict_proba(X[val_idx])[:, 1]
        # 2) Ваги = AUC, нормовані до суми 1
        self.cv_auc = {n: float(roc_auc_score(y, p)) for n, p in oof.items()}
        total = sum(self.cv_auc.values())
        self.weights = {n: a / total for n, a in self.cv_auc.items()}
        self.oof_proba = sum(self.weights[n] * oof[n] for n in self.models)
        # 3) Фінальне навчання кожної моделі на всіх тренувальних даних
        for model in self.models.values():
            model.fit(X, y)
        return self

    def predict_proba_each(self, X) -> dict:
        return {n: m.predict_proba(X)[:, 1] for n, m in self.models.items()}

    def predict_proba(self, X) -> np.ndarray:
        """Зважене середнє: sum(w_i * p_i), де sum(w_i) = 1."""
        each = self.predict_proba_each(X)
        return sum(self.weights[n] * p for n, p in each.items())


# ----------------------------- скорер (обгортка для API) ----------------------
class CreditScorer:
    """Усе, що потрібно для продакшн-прогнозу: препроцесинг + ансамбль + поріг."""

    def __init__(self, random_state: int = 42):
        self.preprocessor = build_preprocessor()
        self.ensemble = CreditScoringEnsemble(random_state)
        self.threshold = 0.5
        self.feature_names = FEATURES
        self.baseline = None   # «типовий клієнт» для пояснень без SHAP
        self.metrics = {}

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(add_features(df)[FEATURES]).astype(float)

    def fit(self, df: pd.DataFrame, y):
        X = self.preprocessor.fit_transform(add_features(df)[FEATURES]).astype(float)
        self.ensemble.fit(X, y)
        # Поріг підбираємо на OOF-прогнозах train, а не на тесті (щоб не було витоку)
        self.threshold = best_cost_threshold(y, self.ensemble.oof_proba)
        self.baseline = np.median(X, axis=0)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.ensemble.predict_proba(self._transform(df))

    def explain(self, df: pd.DataFrame, top_k: int = 6) -> dict:
        """Пояснення рішення для ОДНОГО клієнта.

        Якщо є LightGBM — точні SHAP-значення (TreeSHAP, вбудований у lightgbm:
        predict(pred_contrib=True)). Додатне значення = ознака ПІДВИЩУЄ ризик дефолту.
        Без LightGBM — пертурбаційний метод: замінюємо ознаку «типовим» значенням
        і дивимось, як змінюється ймовірність ансамблю.
        """
        X = self._transform(df.iloc[[0]])
        if "lgb" in self.ensemble.models:
            contrib = self.ensemble.models["lgb"].predict(X, pred_contrib=True)[0][:-1]
            method = "shap_lightgbm (log-odds)"
        else:
            p0 = self.ensemble.predict_proba(X)[0]
            contrib = np.zeros(X.shape[1])
            for j in range(X.shape[1]):
                Xp = X.copy()
                Xp[0, j] = self.baseline[j]
                contrib[j] = p0 - self.ensemble.predict_proba(Xp)[0]
            method = "perturbation (probability)"
        order = np.argsort(-np.abs(contrib))[:top_k]
        raw = add_features(df.iloc[[0]]).iloc[0]
        factors = [
            {"feature": self.feature_names[j],
             "value": raw[self.feature_names[j]].item()
             if hasattr(raw[self.feature_names[j]], "item") else raw[self.feature_names[j]],
             "contribution": round(float(contrib[j]), 4),
             "effect": "increases_risk" if contrib[j] > 0 else "decreases_risk"}
            for j in order
        ]
        return {"method": method, "factors": factors}

    def score_client(self, client: dict) -> dict:
        """Повна відповідь для API: ймовірність, бал, рішення, пояснення."""
        df = pd.DataFrame([client])
        p = float(self.predict_proba(df)[0])
        t = self.threshold
        if p < t * 0.6:
            risk = "low"
        elif p < t:
            risk = "medium"
        else:
            risk = "high"
        return {
            "default_probability": round(p, 4),
            "credit_score": int(prob_to_score(p)),
            "threshold": round(t, 3),
            "decision": "approve" if p < t else "reject",
            "risk_level": risk,
            "explanation": self.explain(df),
        }
