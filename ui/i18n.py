"""Тексти інтерфейсу українською та англійською (перемикач мови в сайдбарі)."""

TEXT = {
    "UA": {
        "title": "Система кредитного скорингу",
        "subtitle": "Real-time оцінка кредитоспроможності з поясненням кожного рішення",
        "tab_score": "🎯 Оцінка клієнта", "tab_batch": "📦 Пакетний скоринг",
        "tab_model": "📊 Якість моделі", "tab_about": "ℹ️ Про проєкт",
        "api_down": "API недоступне або модель не завантажена ({url}). "
                    "Запустіть `python train.py` і `python api/app.py`.",
        "api_ok": "API онлайн", "api_bad": "API офлайн",
        "presets": "Швидкі приклади:", "p_reliable": "🟢 Надійний клієнт",
        "p_average": "🟡 Типовий клієнт", "p_risky": "🔴 Ризиковий клієнт",
        "sec_credit": "Кредит", "sec_finance": "Фінанси", "sec_client": "Клієнт",
        "credit_amount": "Сума кредиту (DM)", "duration": "Строк, місяців",
        "installment_rate": "Платіж, % від доходу (1–4)", "existing_credits": "Кредитів у цьому банку",
        "age": "Вік", "residence_since": "Років за поточною адресою",
        "num_dependents": "Кількість утриманців",
        "btn_score": "Оцінити клієнта",
        "approve": "Кредит схвалено", "reject": "У кредиті відмовлено",
        "prob": "Ймовірність дефолту", "score": "Кредитний бал", "risk": "Рівень ризику",
        "risk_low": "Низький", "risk_medium": "Середній", "risk_high": "Високий",
        "threshold_note": "Поріг відмови: {t:.1%} (підібрано за мінімальною вартістю помилок 5:1)",
        "why": "Чому таке рішення?",
        "why_hint": "Внесок кожної ознаки (SHAP): червоне підвищує ризик, зелене знижує",
        "inc": "підвищує ризик", "dec": "знижує ризик", "contribution": "Внесок",
        "key_factors": "Головні фактори",
        "empty_result": "Заповніть анкету або оберіть приклад і натисніть «Оцінити клієнта».",
        "latency": "Метод пояснення: {m} · відповідь API за {ms} мс",
        "kpi_auc": "ROC-AUC", "kpi_gini": "Gini", "kpi_ks": "KS",
        "kpi_recall": "Recall (bad)", "kpi_precision": "Precision (bad)",
        "kpi_auc_h": "якість ранжування", "kpi_gini_h": "банківський стандарт",
        "kpi_ks_h": "розділення класів", "kpi_recall_h": "знайдено поганих",
        "kpi_precision_h": "точність відмов",
        "roc": "ROC-крива", "dist": "Розподіл кредитних балів (тест)",
        "cm": "Матриця помилок", "weights": "Ансамбль: ваги моделей",
        "fi": "Важливість ознак", "actual": "Факт", "pred": "Прогноз",
        "count": "Кількість", "model": "Модель", "weight": "Вага",
        "retrain": "Перенавчіть модель (`python train.py`), щоб побачити ROC і розподіл балів.",
        "batch_intro": "Завантажте CSV з анкетами клієнтів: API оцінить усіх одразу.",
        "batch_template": "⬇️ Завантажити шаблон CSV", "batch_upload": "CSV-файл з клієнтами",
        "batch_run": "Оцінити всіх", "batch_download": "⬇️ Завантажити результати",
        "batch_total": "Клієнтів", "batch_approved": "Схвалено", "batch_rejected": "Відмовлено",
        "batch_avg": "Середній бал", "batch_errors": "Рядків з помилками",
        "decision": "Рішення",
        "lang": "Мова", "sidebar_model": "Модель", "stack": "Стек",
        "author": "Автор",
        "about_md": """
### Задача
Банк отримує заявку на кредит і має за секунди вирішити: **схвалити чи відмовити**,
та пояснити клієнту причину. Помилка «видали кредит неплатнику» у **5 разів дорожча**,
ніж «відмовили доброму клієнту».

### Рішення
- **Дані:** German Credit (UCI), 1000 заявок, 30 % дефолтів
- **Модель:** ансамбль Random Forest + LightGBM + XGBoost, ваги за ROC-AUC на 5-fold CV
- **Поріг рішення:** мінімізація бізнес-вартості помилок, а не стандартні 0.5
- **Скоровий бал 300–850:** шкала як у кредитних бюро (PDO = 50)
- **Пояснення:** SHAP-внески ознак для кожного клієнта
- **Етика:** стать і громадянство виключені з моделі (fairness)

### Архітектура
`Streamlit UI` → `Flask REST API (gunicorn)` → `ансамбль моделей (joblib)`,
все в **Docker Compose**; тести на pytest.
""",
    },
    "EN": {
        "title": "Credit Scoring System",
        "subtitle": "Real-time creditworthiness assessment with an explanation for every decision",
        "tab_score": "🎯 Score a client", "tab_batch": "📦 Batch scoring",
        "tab_model": "📊 Model performance", "tab_about": "ℹ️ About",
        "api_down": "API is unavailable or the model is not loaded ({url}). "
                    "Run `python train.py` and `python api/app.py`.",
        "api_ok": "API online", "api_bad": "API offline",
        "presets": "Quick examples:", "p_reliable": "🟢 Reliable client",
        "p_average": "🟡 Typical client", "p_risky": "🔴 Risky client",
        "sec_credit": "Loan", "sec_finance": "Finances", "sec_client": "Applicant",
        "credit_amount": "Loan amount (DM)", "duration": "Duration, months",
        "installment_rate": "Installment, % of income (1–4)", "existing_credits": "Credits at this bank",
        "age": "Age", "residence_since": "Years at current address",
        "num_dependents": "Number of dependents",
        "btn_score": "Score client",
        "approve": "Loan approved", "reject": "Loan rejected",
        "prob": "Default probability", "score": "Credit score", "risk": "Risk level",
        "risk_low": "Low", "risk_medium": "Medium", "risk_high": "High",
        "threshold_note": "Rejection threshold: {t:.1%} (optimised for 5:1 error cost)",
        "why": "Why this decision?",
        "why_hint": "Contribution of each feature (SHAP): red raises risk, green lowers it",
        "inc": "raises risk", "dec": "lowers risk", "contribution": "Contribution",
        "key_factors": "Key factors",
        "empty_result": "Fill in the application or pick an example and press “Score client”.",
        "latency": "Explanation method: {m} · API responded in {ms} ms",
        "kpi_auc": "ROC-AUC", "kpi_gini": "Gini", "kpi_ks": "KS",
        "kpi_recall": "Recall (bad)", "kpi_precision": "Precision (bad)",
        "kpi_auc_h": "ranking quality", "kpi_gini_h": "banking standard",
        "kpi_ks_h": "class separation", "kpi_recall_h": "bad loans caught",
        "kpi_precision_h": "rejection precision",
        "roc": "ROC curve", "dist": "Credit score distribution (test)",
        "cm": "Confusion matrix", "weights": "Ensemble: model weights",
        "fi": "Feature importance", "actual": "Actual", "pred": "Predicted",
        "count": "Count", "model": "Model", "weight": "Weight",
        "retrain": "Retrain the model (`python train.py`) to see ROC and score distribution.",
        "batch_intro": "Upload a CSV of applications — the API will score them all at once.",
        "batch_template": "⬇️ Download CSV template", "batch_upload": "CSV file with clients",
        "batch_run": "Score all", "batch_download": "⬇️ Download results",
        "batch_total": "Clients", "batch_approved": "Approved", "batch_rejected": "Rejected",
        "batch_avg": "Average score", "batch_errors": "Rows with errors",
        "decision": "Decision",
        "lang": "Language", "sidebar_model": "Model", "stack": "Stack",
        "author": "Author",
        "about_md": """
### Problem
A bank receives a loan application and must decide within seconds whether to **approve or reject**
it, and explain the reason to the client. Lending to a defaulter costs **5× more**
than rejecting a good client.

### Solution
- **Data:** German Credit (UCI), 1,000 applications, 30% defaults
- **Model:** Random Forest + LightGBM + XGBoost ensemble, weighted by 5-fold CV ROC-AUC
- **Decision threshold:** minimises business cost of errors instead of the default 0.5
- **Credit score 300–850:** bureau-style scale (PDO = 50)
- **Explainability:** per-client SHAP feature contributions
- **Ethics:** sex and citizenship excluded from the model (fairness)

### Architecture
`Streamlit UI` → `Flask REST API (gunicorn)` → `model ensemble (joblib)`,
packaged with **Docker Compose**; tested with pytest.
""",
    },
}

LABELS = {
    "UA": {
        "checking_status": ("Поточний рахунок (DM)", {
            "lt_0": "< 0", "0_200": "0–200", "ge_200": "≥ 200", "no_account": "немає рахунку"}),
        "credit_history": ("Кредитна історія", {
            "no_credits": "кредитів не було", "all_paid_this_bank": "все сплачено в цьому банку",
            "existing_paid": "поточні сплачуються вчасно", "delay_in_past": "були затримки",
            "critical": "критичний рахунок / кредити деінде"}),
        "purpose": ("Мета кредиту", {
            "car_new": "нове авто", "car_used": "вживане авто", "furniture": "меблі",
            "radio_tv": "радіо / ТВ", "appliances": "побутова техніка", "repairs": "ремонт",
            "education": "освіта", "vacation": "відпустка", "retraining": "перекваліфікація",
            "business": "бізнес", "other": "інше"}),
        "savings": ("Заощадження (DM)", {
            "lt_100": "< 100", "100_500": "100–500", "500_1000": "500–1000", "ge_1000": "≥ 1000",
            "unknown": "невідомо / немає"}),
        "employment": ("Стаж на поточній роботі", {
            "unemployed": "безробітний", "lt_1": "< 1 року", "1_4": "1–4 роки",
            "4_7": "4–7 років", "ge_7": "≥ 7 років"}),
        "other_debtors": ("Поручителі", {
            "none": "немає", "co_applicant": "співпозичальник", "guarantor": "поручитель"}),
        "property": ("Майно", {
            "real_estate": "нерухомість", "savings_insurance": "заощадження / страхування",
            "car_other": "авто чи інше", "none": "немає / невідомо"}),
        "other_installment_plans": ("Інші розстрочки", {
            "bank": "у банку", "stores": "у магазинах", "none": "немає"}),
        "housing": ("Житло", {"rent": "оренда", "own": "власне", "free": "безкоштовне"}),
        "job": ("Робота", {
            "unskilled_nonres": "некваліфікована (нерезидент)", "unskilled_res": "некваліфікована",
            "skilled": "кваліфікований працівник", "highly_qualified": "керівник / висока кваліфікація"}),
        "telephone": ("Телефон", {"no": "немає", "yes": "є"}),
    },
    "EN": {
        "checking_status": ("Checking account (DM)", {
            "lt_0": "< 0", "0_200": "0–200", "ge_200": "≥ 200", "no_account": "no account"}),
        "credit_history": ("Credit history", {
            "no_credits": "no credits taken", "all_paid_this_bank": "all paid at this bank",
            "existing_paid": "existing paid on time", "delay_in_past": "past delays",
            "critical": "critical / credits elsewhere"}),
        "purpose": ("Purpose", {
            "car_new": "new car", "car_used": "used car", "furniture": "furniture",
            "radio_tv": "radio / TV", "appliances": "appliances", "repairs": "repairs",
            "education": "education", "vacation": "vacation", "retraining": "retraining",
            "business": "business", "other": "other"}),
        "savings": ("Savings (DM)", {
            "lt_100": "< 100", "100_500": "100–500", "500_1000": "500–1000", "ge_1000": "≥ 1000",
            "unknown": "unknown / none"}),
        "employment": ("Employment length", {
            "unemployed": "unemployed", "lt_1": "< 1 year", "1_4": "1–4 years",
            "4_7": "4–7 years", "ge_7": "≥ 7 years"}),
        "other_debtors": ("Guarantors", {
            "none": "none", "co_applicant": "co-applicant", "guarantor": "guarantor"}),
        "property": ("Property", {
            "real_estate": "real estate", "savings_insurance": "savings / life insurance",
            "car_other": "car or other", "none": "none / unknown"}),
        "other_installment_plans": ("Other installment plans", {
            "bank": "bank", "stores": "stores", "none": "none"}),
        "housing": ("Housing", {"rent": "rent", "own": "own", "free": "free"}),
        "job": ("Job", {
            "unskilled_nonres": "unskilled (non-resident)", "unskilled_res": "unskilled",
            "skilled": "skilled employee", "highly_qualified": "management / highly qualified"}),
        "telephone": ("Telephone", {"no": "no", "yes": "yes"}),
    },
}

FEATURES = {
    "UA": {"duration": "Строк", "credit_amount": "Сума кредиту", "installment_rate": "Платіж, % доходу",
           "residence_since": "Років за адресою", "age": "Вік", "existing_credits": "Кількість кредитів",
           "num_dependents": "Утриманці", "monthly_payment": "Платіж на місяць",
           "credit_to_age": "Сума / вік", "log_credit_amount": "log(сума)"},
    "EN": {"duration": "Duration", "credit_amount": "Loan amount", "installment_rate": "Installment %",
           "residence_since": "Years at address", "age": "Age", "existing_credits": "Existing credits",
           "num_dependents": "Dependents", "monthly_payment": "Monthly payment",
           "credit_to_age": "Amount / age", "log_credit_amount": "log(amount)"},
}


def feature_name(lang: str, f: str) -> str:
    if f in LABELS[lang]:
        return LABELS[lang][f][0]
    return FEATURES[lang].get(f, f)


def value_label(lang: str, f: str, v) -> str:
    if f in LABELS[lang]:
        return LABELS[lang][f][1].get(v, str(v))
    if isinstance(v, float):
        return f"{v:,.1f}" if abs(v) < 100 else f"{v:,.0f}"
    return str(v)
