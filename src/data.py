"""Завантаження та декодування датасету German Credit (UCI Statlog).

Оригінальний файл german.data містить 1000 клієнтів, 20 ознак і цільову змінну
(1 = good / добрий позичальник, 2 = bad / поганий). Категоріальні ознаки
закодовані як A11, A12, ... — тут ми перетворюємо їх на зрозумілі значення.
"""
from pathlib import Path
import urllib.request

import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "german.data"

COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age",
    "other_installment_plans", "housing", "existing_credits", "job",
    "num_dependents", "telephone", "foreign_worker", "target",
]

# Розшифровка кодів UCI -> читабельні значення (саме їх приймає API)
CODE_MAPS = {
    "checking_status": {"A11": "lt_0", "A12": "0_200", "A13": "ge_200", "A14": "no_account"},
    "credit_history": {"A30": "no_credits", "A31": "all_paid_this_bank", "A32": "existing_paid",
                       "A33": "delay_in_past", "A34": "critical"},
    "purpose": {"A40": "car_new", "A41": "car_used", "A42": "furniture", "A43": "radio_tv",
                "A44": "appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
                "A48": "retraining", "A49": "business", "A410": "other"},
    "savings": {"A61": "lt_100", "A62": "100_500", "A63": "500_1000", "A64": "ge_1000",
                "A65": "unknown"},
    "employment": {"A71": "unemployed", "A72": "lt_1", "A73": "1_4", "A74": "4_7", "A75": "ge_7"},
    "personal_status_sex": {"A91": "male_div", "A92": "female_div_mar", "A93": "male_single",
                            "A94": "male_mar_wid", "A95": "female_single"},
    "other_debtors": {"A101": "none", "A102": "co_applicant", "A103": "guarantor"},
    "property": {"A121": "real_estate", "A122": "savings_insurance", "A123": "car_other",
                 "A124": "none"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "free"},
    "job": {"A171": "unskilled_nonres", "A172": "unskilled_res", "A173": "skilled",
            "A174": "highly_qualified"},
    "telephone": {"A191": "no", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}


def download_data(path: Path = DATA_PATH) -> Path:
    """Завантажує german.data з UCI, якщо файлу ще немає."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Завантажую датасет з {DATA_URL} ...")
    try:
        urllib.request.urlretrieve(DATA_URL, path)
    except Exception as exc:  # немає інтернету / сайт недоступний
        raise SystemExit(
            f"Не вдалося завантажити датасет ({exc}).\n"
            f"Скачайте вручну {DATA_URL}\nі покладіть файл у {path}"
        )
    return path


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Читає german.data і повертає DataFrame з декодованими ознаками.

    target: 1 = дефолт (поганий позичальник), 0 = добрий.
    """
    download_data(path)
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    for col, mapping in CODE_MAPS.items():
        df[col] = df[col].map(mapping)
    # В UCI: 1 = good, 2 = bad. Ми моделюємо ймовірність дефолту => bad = 1
    df["target"] = (df["target"] == 2).astype(int)
    return df
