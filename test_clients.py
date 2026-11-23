"""Перевірка системи на реальних клієнтах з відомою правильною відповіддю.

Бере клієнтів з ТЕСТОВОЇ вибірки (той самий спліт, що й у train.py, тобто
модель їх не бачила), надсилає в API і порівнює рішення з фактом.

Запуск (API має працювати):  python test_clients.py            # 5 добрих + 5 поганих
                             python test_clients.py --n 20      # 20 + 20
Також зберігає data/test_clients.csv: його можна завантажити у вкладку «Пакетний скоринг».
"""
import argparse
import sys

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from src.data import load_data
from src.features import CATEGORICAL, NUMERIC_RAW

API_URL = "http://localhost:5000"
FIELDS = CATEGORICAL + NUMERIC_RAW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="скільки добрих і скільки поганих клієнтів")
    n = ap.parse_args().n

    df = load_data()
    X, y = df.drop(columns="target"), df["target"]
    # Точно такий самий поділ, як у train.py => це клієнти, яких модель не бачила
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    test = X_test[FIELDS].assign(true_label=y_test.map({0: "good", 1: "bad"}))
    sample = pd.concat([test[test.true_label == "good"].head(n),
                        test[test.true_label == "bad"].head(n)])

    try:
        resp = requests.post(f"{API_URL}/predict/batch",
                             json=sample[FIELDS].to_dict("records"), timeout=30)
    except requests.RequestException:
        sys.exit("API не відповідає. Спершу запустіть: python api/app.py")
    res = pd.DataFrame(resp.json()).set_index("index")
    sample = sample.reset_index(names="row_in_dataset").join(res)

    # «Правильне» рішення: good -> approve, bad -> reject
    sample["expected"] = sample["true_label"].map({"good": "approve", "bad": "reject"})
    sample["correct"] = sample["decision"] == sample["expected"]

    cols = ["row_in_dataset", "true_label", "expected", "decision",
            "default_probability", "credit_score", "correct"]
    print(sample[cols].to_string(index=False))
    for label in ["good", "bad"]:
        part = sample[sample.true_label == label]
        print(f"{label:>4}: правильно {part.correct.sum()} з {len(part)}")
    print(f"Разом: {sample.correct.mean():.0%} правильних рішень")

    sample[FIELDS + ["true_label"]].to_csv("data/test_clients.csv", index=False)
    print("Збережено data/test_clients.csv: завантаж у вкладку «Пакетний скоринг»")


if __name__ == "__main__":
    main()
