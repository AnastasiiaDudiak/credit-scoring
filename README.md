# 💳 Credit Scoring System

![CI](https://github.com/<your-login>/credit-scoring/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-XGBoost-2dd4bf)
![Flask](https://img.shields.io/badge/Flask-REST_API-000000?logo=flask)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

End-to-end ML service that scores loan applications in real time, converts default probability
into a 300–850 credit score and **explains every decision** with SHAP.

| | |
|---|---|
| **Model** | Random Forest + LightGBM + XGBoost ensemble (weights = 5-fold CV ROC-AUC) |
| **Test quality** | ROC-AUC 0.776 · Gini 0.551 · KS 0.436 · Recall (bad) 0.867 |
| **Business logic** | Decision threshold minimises 5:1 cost of lending to a defaulter vs. rejecting a good client |
| **Fairness** | Sex / marital status and foreign-worker flag excluded from the model |
| **Serving** | Flask REST API (gunicorn) + Streamlit UI (UA/EN), Docker Compose, pytest, GitHub Actions |

![Scoring](docs/screenshots/scoring.png)
![Model performance](docs/screenshots/model.png)

---

## Проєкт 3: Кредитний скоринг (Credit Scoring)

**POC-сценарій:** система автоматичної оцінки кредитоспроможності з real-time скорингом
та поясненням рішень.

Датасет: **German Credit (UCI Statlog)**: 1000 клієнтів, 20 ознак, 30 % «поганих» позичальників.
Скрипт сам завантажує `german.data` з UCI. Якщо інтернет недоступний, скачайте
<https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data>
і покладіть у `data/german.data`.

---

## 1. Структура проєкту

```
credit-scoring/
├── src/
│   ├── data.py        # завантаження та декодування датасету (A11 → "lt_0")
│   ├── features.py    # інженерія ознак + препроцесинг (ColumnTransformer)
│   └── model.py       # ансамбль RF+LightGBM+XGBoost, метрики KS/Gini, поріг, пояснення
├── train.py           # навчання, оцінка, графіки, збереження моделі
├── api/app.py         # Flask REST API
├── ui/streamlit_app.py# Streamlit-інтерфейс (UA/EN)
├── ui/i18n.py         # тексти інтерфейсу двома мовами
├── .streamlit/config.toml  # тема інтерфейсу
├── .github/workflows/ci.yml # CI: навчання + тести на кожен push
├── docs/screenshots/  # скріншоти для README
├── tests/test_api.py  # pytest-тести API
├── models/            # credit_scorer.joblib (з'являється після train.py)
├── reports/           # metrics.json, eda.png, evaluation.png, feature_importance.png
├── api/Dockerfile, ui/Dockerfile, docker-compose.yml
└── requirements.txt
```

## 2. Запуск

### Локально
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

python train.py                      # 1) навчити модель
pytest -q                            # 2) тести (5 шт.)
python api/app.py                    # 3) API на http://localhost:5000
streamlit run ui/streamlit_app.py    # 4) в іншому терміналі: UI на http://localhost:8501
```

### Docker
```bash
python train.py              # модель має існувати ДО збирання образу
docker compose up --build    # API :5000, UI :8501
```

### Приклад запиту до API
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{
 "checking_status":"lt_0","duration":36,"credit_history":"existing_paid","purpose":"car_new",
 "credit_amount":6000,"savings":"lt_100","employment":"1_4","installment_rate":4,
 "other_debtors":"none","residence_since":2,"property":"car_other","age":28,
 "other_installment_plans":"none","housing":"rent","existing_credits":1,"job":"skilled",
 "num_dependents":1,"telephone":"no"}'
```
Відповідь:
```json
{
  "default_probability": 0.61, "credit_score": 568, "threshold": 0.32,
  "decision": "reject", "risk_level": "high",
  "explanation": {"method": "shap_lightgbm (log-odds)", "factors": [
     {"feature": "checking_status", "value": "lt_0", "contribution": 0.42, "effect": "increases_risk"}, ...]},
  "latency_ms": 12.3
}
```

| Endpoint | Метод | Опис |
|---|---|---|
| `/health` | GET | чи живий сервіс, чи завантажена модель |
| `/model/info` | GET | метрики, ваги ансамблю, поріг, топ-ознаки |
| `/schema` | GET | допустимі значення категоріальних полів |
| `/predict` | POST | скоринг одного клієнта + пояснення |
| `/predict/batch` | POST | скоринг масиву клієнтів |

---

## 3. Конспект теорії

### 3.1 Що таке кредитний скоринг
**Кредитний скоринг** (credit scoring, *оцінювання кредитоспроможності*): статистична модель,
яка за даними про позичальника оцінює **ймовірність дефолту** (probability of default, PD), тобто ймовірність того,
що клієнт не поверне кредит. Це задача **бінарної класифікації** (binary classification):
`1 = bad` (дефолт), `0 = good`.

Особливості задачі:
- **Незбалансовані класи** (imbalanced classes): «поганих» клієнтів менше (тут 30 %, у реальних банках 2–10 %).
  Тому accuracy (*точність*) не підходить: модель «всім схвалити» дасть 70 % accuracy і нульову користь.
- **Асиметрична вартість помилок** (cost-sensitive learning): видати кредит шахраю/неплатнику (FN)
  набагато дорожче, ніж відмовити доброму клієнту (FP). Для German Credit задано співвідношення **5 : 1**.
- **Пояснюваність** (explainability): банк зобов'язаний пояснити клієнту причину відмови.
- **Справедливість** (fairness): модель не повинна дискримінувати за статтю, національністю тощо.

### 3.2 Дані та ознаки (features)
20 вихідних ознак: стан поточного рахунку, строк, кредитна історія, мета, сума, заощадження,
стаж, вік, житло, майно тощо.

**Виключені ознаки:** `personal_status_sex` (стать + сімейний стан) і `foreign_worker`.
Це **захищені атрибути** (protected attributes); їх використання = пряма дискримінація.

**Інженерія ознак** (feature engineering, *створення нових ознак*):
| Ознака | Формула | Бізнес-зміст |
|---|---|---|
| `monthly_payment` | сума / строк | місячне боргове навантаження |
| `credit_to_age` | сума / вік | великий кредит у молодому віці = ризик |
| `log_credit_amount` | ln(1 + сума) | стискає «довгий хвіст» великих сум |

**Кодування категорій:** `OrdinalEncoder` (кожній категорії ціле число). Деревам не потрібне
масштабування (scaling) і one-hot: дерево розбиває за порогом і саме знаходить корисні групи.
`handle_unknown="use_encoded_value"` означає, що невідоме значення в API не «зламає» модель.

**Train/serve skew** (*розбіжність між навчанням і продом*): та сама функція `add_features()`
викликається і в `train.py`, і в API, тому ознаки рахуються однаково.

### 3.3 Розбиття даних
- `train_test_split(test_size=0.2, stratify=y)`: 80 % навчання, 20 % тест;
  **стратифікація** (stratification) зберігає однакову частку дефолтів в обох частинах.
- **Крос-валідація** (cross-validation, CV) `StratifiedKFold(5)`: train ділиться на 5 частин (folds),
  модель 5 разів навчається на 4 і прогнозує 5-ту. Отримуємо **out-of-fold (OOF) прогнози**,
  тобто чесні прогнози для кожного рядка train, яких модель «не бачила».

### 3.4 Моделі на деревах
**Дерево рішень** (decision tree): послідовність питань «ознака ≤ поріг?», що ділить клієнтів на групи
з максимально однорідним класом (критерій Gini / entropy). Одне дерево легко перенавчається (overfitting).

**Random Forest** (*випадковий ліс*), метод **bagging** (bootstrap aggregating):
- кожне дерево вчиться на bootstrap-вибірці (випадкова вибірка з поверненням);
- у кожному вузлі розглядається випадкова підмножина ознак;
- прогноз = середнє по деревах → менша дисперсія (variance).
- Параметри: `n_estimators=500` (кількість дерев), `max_depth=8`, `min_samples_leaf=5`
  (мінімум об'єктів у листі; проти перенавчання), `class_weight="balanced"` (вага класу ∝ 1/частота).

**Gradient Boosting** (*градієнтний бустинг*): дерева будуються **послідовно**; кожне нове
дерево виправляє помилки (градієнт функції втрат, log-loss) попередніх.
`learning_rate` (*крок навчання*) зменшує внесок кожного дерева: менше значення означає більше дерев і кращу узагальненість.

**XGBoost:** бустинг з регуляризацією L1/L2 (`reg_lambda`), обробкою пропусків, рівневим
(level-wise) ростом дерев. `scale_pos_weight = n_good/n_bad ≈ 7/3` компенсує дисбаланс.

**LightGBM:** бустинг з ростом дерев **по листках** (leaf-wise: розщеплює лист з найбільшим
приростом), гістограмним пошуком порогів → дуже швидкий. Головний параметр `num_leaves=15`;
`subsample` / `colsample_bytree` = частка рядків / ознак для кожного дерева (стохастичність).

Датасет малий (1000 рядків), тому параметри **консервативні**: неглибокі дерева, малий learning rate.
У програмі курсу `n_estimators=1000` без обмежень, що на 1000 рядках гарантовано дає перенавчання.

### 3.5 Ансамбль (ensemble)
`CreditScoringEnsemble` = **зважене усереднення** (weighted averaging, soft voting):

  P(default) = w_rf·p_rf + w_lgb·p_lgb + w_xgb·p_xgb,  де  w_i = AUC_i / Σ AUC

Ваги рахуються за **OOF ROC-AUC** на крос-валідації, тобто краща модель має більший вплив.
(У лістингу з програми `np.mean(pred * weight)` ділить ще й на кількість моделей; тут виправлено на суму з нормованими вагами.)
Ансамбль різних алгоритмів стабільніший: помилки моделей частково компенсуються.

### 3.6 Метрики якості
- **Confusion matrix** (*матриця помилок*): TP, FP, TN, FN.
  FN = поганий клієнт отримав кредит (найдорожча помилка), FP = доброму відмовили.
- **Precision** (*точність*) = TP / (TP + FP): частка справді поганих серед відхилених.
- **Recall** (*повнота*) = TP / (TP + FN): яку частку поганих ми «зловили».
- **ROC-крива**: TPR (= recall) проти FPR = FP/(FP+TN) при всіх можливих порогах.
- **ROC-AUC** (area under curve): ймовірність, що випадковий поганий клієнт отримає вищий
  ризик, ніж випадковий добрий. 0.5 = випадкове вгадування, 1 = ідеал. Для скорингу 0.75–0.85 = добре.
- **Gini** = 2·AUC − 1. Стандартна банківська метрика (Gini 0.5 ⇔ AUC 0.75).
- **KS** (Kolmogorov–Smirnov) = max(TPR − FPR): максимальна відстань між кумулятивними
  розподілами скорів добрих і поганих клієнтів (графік `reports/evaluation.png`).
  KS > 0.4 вважається добрим розділенням.

### 3.7 Поріг рішення (decision threshold)
Модель видає ймовірність; рішення «схвалити/відмовити» = порівняння з порогом.
Поріг 0.5 не оптимальний за асиметричних втрат. Ми мінімізуємо **вартість**:

  Cost(t) = 5 · FN(t) + 1 · FP(t)

Поріг шукається на **OOF-прогнозах train** (не на тесті!), інакше буде **витік даних**
(data leakage) і завищена оцінка. Через ціну 5:1 поріг виходить низьким (~0.3), тобто модель
обережна: краще відмовити, ніж видати поганому.

Рівні ризику: `low` (p < 0.6·t), `medium` (до t), `high` (≥ t).

### 3.8 Скоровий бал (credit score)
Ймовірність переводиться в бал, як у кредитних бюро:

  score = offset + factor · ln(odds),  odds = (1−p)/p,  factor = PDO / ln 2

**PDO** (points to double the odds) = 50: +50 балів означає, що шанси «добрий:поганий» удвічі вищі.
Тут 600 балів ↔ шанси 1:1 (p = 0.5); бал обмежено діапазоном 300–850.

### 3.9 Пояснення рішень (explainability)
**SHAP** (SHapley Additive exPlanations) з теорії ігор: прогноз розкладається на суму
внесків ознак: `f(x) = базове значення + Σ φ_j`. φ_j > 0 означає, що ознака **підвищує** ризик дефолту.
LightGBM рахує точні TreeSHAP-значення вбудовано: `model.predict(X, pred_contrib=True)`
(у log-odds). API повертає топ-6 факторів (саме вони показуються клієнту як причини відмови).
Якщо lightgbm не встановлено, використовується **пертурбаційний метод**: ознака замінюється на медіанне значення,
і дивимось, як змінилась ймовірність.

**Feature importance** (*важливість ознак*), глобальна: наскільки сумарно кожна ознака зменшує
помилку в усіх деревах (`reports/feature_importance.png`). На German Credit зазвичай
лідирують стан рахунку, строк, сума, кредитна історія, вік.

### 3.10 Архітектура сервісу
```
[Streamlit UI :8501] --HTTP/JSON--> [Flask API :5000] --> credit_scorer.joblib
                                         (gunicorn)          (препроцесор + ансамбль + поріг)
```
- **Серіалізація** (serialization): `joblib.dump` зберігає весь об'єкт `CreditScorer`
  (препроцесор + 3 моделі + поріг + метрики) в один файл → один артефакт для деплою.
- **REST API**: валідація вхідних даних (400 Bad Request зі списком помилок), 503 якщо модель
  не завантажена, `/health` для health-check у Docker.
- **gunicorn**: production WSGI-сервер (вбудований сервер Flask лише для розробки).
- **Docker Compose**: 2 контейнери; UI стартує лише коли API «healthy» (`depends_on: condition`).
  Усередині мережі compose UI звертається до API за іменем сервісу `http://api:5000`.

### 3.11 Можливі питання на захисті
1. *Чому не accuracy?* Через дисбаланс класів і різну ціну помилок.
2. *Чим bagging відрізняється від boosting?* Паралельні незалежні дерева (зменшують дисперсію) vs послідовні дерева, що виправляють помилки (зменшують зміщення/bias).
3. *Як обрано поріг?* Мінімізація вартості 5·FN + FP на OOF-прогнозах.
4. *Що таке KS і Gini?* Див. 3.6.
5. *Чому виключено стать і громадянство?* Fairness / антидискримінаційне регулювання.
6. *Як пояснюється відмова?* SHAP-внески ознак для конкретного клієнта.
7. *Як уникнули витоку даних?* Стратифікований відкладений тест, поріг і ваги лише з train-CV.

### 3.12 Що можна покращити
Підбір гіперпараметрів (Optuna / GridSearchCV), калібрування ймовірностей (`CalibratedClassifierCV`),
WoE/IV-аналіз та логістична регресія як банківський бейзлайн, більший датасет
(Give Me Some Credit / Home Credit), моніторинг дрейфу даних (PSI), логування запитів у БД.
