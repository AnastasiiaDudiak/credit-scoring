"""Навчання моделі кредитного скорингу.

Запуск:  python train.py
Результат: models/credit_scorer.joblib, reports/metrics.json, reports/*.png
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

from src.data import load_data
from src.features import FEATURES
from src.model import COST_FN, COST_FP, CreditScorer, ks_statistic, prob_to_score

ROOT = Path(__file__).resolve().parent
MODELS_DIR, REPORTS_DIR = ROOT / "models", ROOT / "reports"


def eda(df: pd.DataFrame):
    """Кілька базових графіків розвідувального аналізу (EDA)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    df["target"].map({0: "good", 1: "bad"}).value_counts().plot.bar(
        ax=axes[0], color=["#4c9f70", "#d1495b"])
    axes[0].set_title("Баланс класів")
    sns.boxplot(data=df, x="target", y="credit_amount", ax=axes[1])
    axes[1].set_title("Сума кредиту vs дефолт (1 = bad)")
    (df.groupby("checking_status")["target"].mean().sort_values()
       .plot.barh(ax=axes[2], color="#30638e"))
    axes[2].set_title("Частка дефолтів за станом рахунку")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda.png", dpi=120)
    plt.close(fig)


def plots(y_test, p_test, scorer: CreditScorer):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    # ROC
    fpr, tpr, _ = roc_curve(y_test, p_test)
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, p_test):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(title="ROC-крива", xlabel="FPR", ylabel="TPR")
    axes[0].legend()
    # KS: кумулятивні розподіли скорів поганих і добрих
    order = np.sort(p_test)
    cdf_bad = [np.mean(p_test[y_test == 1] <= s) for s in order]
    cdf_good = [np.mean(p_test[y_test == 0] <= s) for s in order]
    axes[1].plot(order, cdf_bad, label="bad")
    axes[1].plot(order, cdf_good, label="good")
    axes[1].set(title=f"KS = {ks_statistic(y_test, p_test):.3f}",
                xlabel="P(default)", ylabel="CDF")
    axes[1].legend()
    # Розподіл скорових балів
    scores = prob_to_score(p_test)
    for cls, name in [(0, "good"), (1, "bad")]:
        axes[2].hist(scores[y_test == cls], bins=25, alpha=0.6, label=name)
    axes[2].set(title="Розподіл кредитних балів", xlabel="score")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "evaluation.png", dpi=120)
    plt.close(fig)

    # Важливість ознак (з LightGBM, якщо є, інакше з Random Forest)
    name = "lgb" if "lgb" in scorer.ensemble.models else "rf"
    model = scorer.ensemble.models[name]
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    imp.plot.barh(ax=ax, color="#30638e")
    ax.set_title(f"Feature importance ({name})")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)
    return imp.sort_values(ascending=False)


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_data()
    print(f"Дані: {df.shape[0]} клієнтів, частка дефолтів = {df['target'].mean():.1%}")
    eda(df)

    X, y = df.drop(columns="target"), df["target"].values
    # stratify: зберігаємо однакову частку дефолтів у train і test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    scorer = CreditScorer().fit(X_train, y_train)
    ens = scorer.ensemble
    print("CV ROC-AUC моделей:", {k: round(v, 4) for k, v in ens.cv_auc.items()})
    print("Ваги ансамблю:     ", {k: round(v, 3) for k, v in ens.weights.items()})
    print(f"Поріг (мін. вартість помилок {COST_FN}:{COST_FP}): {scorer.threshold:.3f}")

    # --- Оцінка на відкладеній тестовій вибірці ---
    Xt = scorer._transform(X_test)
    p_each = ens.predict_proba_each(Xt)
    p_test = ens.predict_proba(Xt)
    pred = (p_test >= scorer.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    auc = roc_auc_score(y_test, p_test)
    metrics = {
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "default_rate": round(float(y.mean()), 4),
        "models": list(ens.models),
        "cv_auc": {k: round(v, 4) for k, v in ens.cv_auc.items()},
        "weights": {k: round(v, 4) for k, v in ens.weights.items()},
        "test_auc_per_model": {k: round(float(roc_auc_score(y_test, p)), 4)
                               for k, p in p_each.items()},
        "threshold": round(scorer.threshold, 3),
        "test": {
            "roc_auc": round(float(auc), 4),
            "gini": round(float(2 * auc - 1), 4),
            "ks": round(ks_statistic(y_test, p_test), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred)), 4),
            "recall": round(float(recall_score(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred)), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "cost": int(COST_FN * fn + COST_FP * fp),
        },
    }
    # Дані для інтерактивних графіків у Streamlit
    fpr, tpr, _ = roc_curve(y_test, p_test)
    metrics["curves"] = {
        "roc": {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]},
        "scores": {"score": prob_to_score(p_test).tolist(), "bad": y_test.astype(int).tolist()},
    }
    imp = plots(y_test, p_test, scorer)
    metrics["top_features"] = {k: float(v) for k, v in imp.head(10).items()}
    scorer.metrics = metrics

    joblib.dump(scorer, MODELS_DIR / "credit_scorer.joblib")
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Тест:", json.dumps(metrics["test"], ensure_ascii=False))
    print(f"Модель збережено: {MODELS_DIR / 'credit_scorer.joblib'}")


if __name__ == "__main__":
    main()
