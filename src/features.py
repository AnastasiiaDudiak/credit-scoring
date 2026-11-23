"""Інженерія ознак і препроцесинг.

Однакова функція add_features() використовується і при навчанні, і в API —
це гарантує, що модель бачить ознаки в однаковому вигляді (немає train/serve skew).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

# Чутливі ознаки (стать/сімейний стан, іноземний працівник) НЕ використовуємо:
# це питання справедливості (fairness) та антидискримінаційних вимог до скорингу.
EXCLUDED = ["personal_status_sex", "foreign_worker"]

CATEGORICAL = [
    "checking_status", "credit_history", "purpose", "savings", "employment",
    "other_debtors", "property", "other_installment_plans", "housing", "job",
    "telephone",
]
NUMERIC_RAW = [
    "duration", "credit_amount", "installment_rate", "residence_since", "age",
    "existing_credits", "num_dependents",
]
ENGINEERED = ["monthly_payment", "credit_to_age", "log_credit_amount"]
NUMERIC = NUMERIC_RAW + ENGINEERED
FEATURES = CATEGORICAL + NUMERIC


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Додає нові ознаки, що мають бізнес-зміст."""
    df = df.copy()
    # Середній платіж на місяць: велике навантаження -> вищий ризик
    df["monthly_payment"] = df["credit_amount"] / df["duration"].clip(lower=1)
    # Сума кредиту відносно віку (молодий клієнт + великий кредит = ризик)
    df["credit_to_age"] = df["credit_amount"] / df["age"].clip(lower=18)
    # Логарифм суми: зменшує вплив дуже великих значень (правий хвіст розподілу)
    df["log_credit_amount"] = np.log1p(df["credit_amount"])
    return df


def build_preprocessor() -> ColumnTransformer:
    """Категорії -> цілі числа (OrdinalEncoder). Деревам масштабування не потрібне,
    а one-hot не обов'язковий — дерево саме знаходить потрібні розбиття."""
    return ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ],
        verbose_feature_names_out=False,
    )
