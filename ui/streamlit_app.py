"""Streamlit-інтерфейс кредитного скорингу (портфоліо-версія).

Запуск з кореня проєкту:  streamlit run ui/streamlit_app.py
Тема інтерфейсу лежить у .streamlit/config.toml
"""
import json
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

from i18n import LABELS, TEXT, feature_name, value_label

API_URL = os.getenv("API_URL", "http://localhost:5000")
AUTHOR = "Анастасія Дудяк"
GITHUB_URL = ""   # встав посилання на репозиторій, напр. "https://github.com/<login>/credit-scoring"

RED, GREEN, AMBER, TEAL, MUTED = "#f87171", "#34d399", "#fbbf24", "#2dd4bf", "#94a3b8"

st.set_page_config(page_title="Credit Scoring", page_icon="💳", layout="wide")

# ----------------------------------------------------------------------------- стилі
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1350px;}
.hero {background: linear-gradient(120deg, #0f766e 0%, #1e3a8a 55%, #312e81 100%);
       border-radius: 20px; padding: 28px 34px; margin-bottom: 18px;
       box-shadow: 0 10px 30px rgba(0,0,0,.35);}
.hero h1 {color: #fff; font-size: 2.2rem; margin: 0 0 4px 0; padding: 0;}
.hero p {color: #cbd5e1; font-size: 1.05rem; margin: 0 0 14px 0;}
.badge {display: inline-block; background: rgba(255,255,255,.12); color: #e2e8f0;
        border: 1px solid rgba(255,255,255,.18); border-radius: 999px;
        padding: 3px 12px; margin: 0 6px 6px 0; font-size: .8rem;}
.card {background: #131c31; border: 1px solid #1e293b; border-radius: 16px;
       padding: 18px 20px; height: 100%;}
.kpi-label {color: #94a3b8; font-size: .85rem; margin-bottom: 4px;}
.kpi-value {color: #f1f5f9; font-size: 1.9rem; font-weight: 700; line-height: 1.1;}
.kpi-hint {color: #64748b; font-size: .78rem; margin-top: 4px;}
.decision {border-radius: 16px; padding: 18px 22px; font-size: 1.4rem; font-weight: 700;
           display: flex; align-items: center; gap: 12px;}
.decision.ok {background: rgba(52,211,153,.12); border: 1px solid rgba(52,211,153,.5); color: #34d399;}
.decision.no {background: rgba(248,113,113,.12); border: 1px solid rgba(248,113,113,.5); color: #f87171;}
.factor {display: flex; justify-content: space-between; padding: 8px 0;
         border-bottom: 1px solid #1e293b; font-size: .92rem;}
.factor:last-child {border-bottom: none;}
.factor .v {color: #94a3b8;}
.dot {height: 9px; width: 9px; border-radius: 50%; display: inline-block; margin-right: 6px;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {background: #131c31; border-radius: 10px 10px 0 0; padding: 8px 18px;}
div[data-testid="stForm"] {background: #0f172a; border: 1px solid #1e293b; border-radius: 16px;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- API
@st.cache_data(ttl=30, show_spinner=False)
def api_get(path):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=5)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def api_post(path, payload):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    return r.ok, r.json()


# ----------------------------------------------------------------------------- приклади клієнтів
PRESETS = {
    "p_reliable": dict(checking_status="no_account", duration=12, credit_history="critical",
                       purpose="car_used", credit_amount=2000, savings="ge_1000", employment="ge_7",
                       installment_rate=2, other_debtors="none", residence_since=4,
                       property="real_estate", age=45, other_installment_plans="none",
                       housing="own", existing_credits=2, job="skilled", num_dependents=1,
                       telephone="yes"),
    "p_average": dict(checking_status="0_200", duration=24, credit_history="existing_paid",
                      purpose="radio_tv", credit_amount=3000, savings="lt_100", employment="1_4",
                      installment_rate=3, other_debtors="none", residence_since=2,
                      property="car_other", age=33, other_installment_plans="none",
                      housing="own", existing_credits=1, job="skilled", num_dependents=1,
                      telephone="no"),
    "p_risky": dict(checking_status="lt_0", duration=48, credit_history="delay_in_past",
                    purpose="car_new", credit_amount=9000, savings="lt_100", employment="lt_1",
                    installment_rate=4, other_debtors="none", residence_since=1, property="none",
                    age=23, other_installment_plans="bank", housing="rent", existing_credits=1,
                    job="unskilled_res", num_dependents=1, telephone="no"),
}
FIELDS = list(PRESETS["p_average"])


def load_preset(name):
    for f, v in PRESETS[name].items():
        st.session_state[f"in_{f}"] = v
    st.session_state.pop("result", None)


if "in_age" not in st.session_state:       # перший запуск: типовий клієнт
    load_preset("p_average")


# ----------------------------------------------------------------------------- допоміжні елементи
def kpi(label, value, hint="", color="#f1f5f9"):
    st.markdown(f"""<div class="card"><div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-hint">{hint}</div></div>""", unsafe_allow_html=True)


def gauge(score: int, label: str) -> str:
    """SVG-спідометр для балу 300–850."""
    import math
    f = (min(max(score, 300), 850) - 300) / 550
    ang = math.pi * (1 - f)
    x, y = 150 + 92 * math.cos(ang), 140 - 92 * math.sin(ang)
    color = RED if score < 560 else AMBER if score < 650 else GREEN
    return f"""
<div class="card" style="text-align:center">
<svg viewBox="0 0 300 175" width="100%" style="max-width:340px">
  <defs><linearGradient id="g" x1="0" x2="1">
    <stop offset="0%" stop-color="{RED}"/><stop offset="50%" stop-color="{AMBER}"/>
    <stop offset="100%" stop-color="{GREEN}"/></linearGradient></defs>
  <path d="M 40 140 A 110 110 0 0 1 260 140" fill="none" stroke="#1e293b" stroke-width="22"
        stroke-linecap="round"/>
  <path d="M 40 140 A 110 110 0 0 1 260 140" fill="none" stroke="url(#g)" stroke-width="16"
        stroke-linecap="round" opacity=".9"/>
  <line x1="150" y1="140" x2="{x:.1f}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="4"
        stroke-linecap="round"/>
  <circle cx="150" cy="140" r="8" fill="#e2e8f0"/>
  <text x="40" y="168" fill="{MUTED}" font-size="12" text-anchor="middle">300</text>
  <text x="260" y="168" fill="{MUTED}" font-size="12" text-anchor="middle">850</text>
  <text x="150" y="112" fill="{color}" font-size="34" font-weight="700" text-anchor="middle">{score}</text>
</svg>
<div class="kpi-label">{label}</div></div>"""


def chart(c, height=300):
    st.altair_chart(c.properties(height=height).configure_view(strokeWidth=0)
                    .configure_axis(gridColor="#1e293b", domainColor="#334155",
                                    labelColor=MUTED, titleColor=MUTED),
                    width="stretch")


# ----------------------------------------------------------------------------- сайдбар
with st.sidebar:
    lang = st.segmented_control("🌐", ["UA", "EN"], default="UA", key="lang") or "UA"
    T = TEXT[lang]
    health = api_get("/health")
    online = bool(health and health.get("model_loaded"))
    st.markdown(f'<span class="dot" style="background:{GREEN if online else RED}"></span>'
                f'{T["api_ok"] if online else T["api_bad"]} · <code>{API_URL}</code>',
                unsafe_allow_html=True)
    info = api_get("/model/info") if online else None
    if info:
        st.markdown(f"#### {T['sidebar_model']}")
        st.markdown(f"Ensemble: **{' + '.join(m.upper() for m in info['models'])}**  \n"
                    f"ROC-AUC **{info['test']['roc_auc']:.3f}** · Gini **{info['test']['gini']:.3f}**  \n"
                    f"Train / test: {info['n_train']} / {info['n_test']}")
    st.markdown(f"#### {T['stack']}")
    st.markdown("".join(f'<span class="badge">{s}</span>' for s in
                        ["Python", "scikit-learn", "LightGBM", "XGBoost", "SHAP", "Flask",
                         "Streamlit", "Docker", "pytest"]), unsafe_allow_html=True)
    st.markdown(f"#### {T['author']}")
    st.markdown(AUTHOR + (f"  \n[GitHub]({GITHUB_URL})" if GITHUB_URL else ""))

# ----------------------------------------------------------------------------- шапка
st.markdown(f"""<div class="hero"><h1>💳 {T['title']}</h1><p>{T['subtitle']}</p>
<span class="badge">Random Forest + LightGBM + XGBoost</span>
<span class="badge">SHAP explainability</span><span class="badge">REST API</span>
<span class="badge">Cost-sensitive threshold</span></div>""", unsafe_allow_html=True)

if not online:
    st.error(T["api_down"].format(url=API_URL))
    st.stop()

tab_score, tab_batch, tab_model, tab_about = st.tabs(
    [T["tab_score"], T["tab_batch"], T["tab_model"], T["tab_about"]])

# ============================================================================= 1. Оцінка клієнта
with tab_score:
    pc = st.columns([1.1, 1, 1, 1, 2])
    pc[0].markdown(f"<div style='padding-top:8px;color:{MUTED}'>{T['presets']}</div>",
                   unsafe_allow_html=True)
    for col, name in zip(pc[1:4], PRESETS):
        col.button(T[name], on_click=load_preset, args=(name,), width="stretch")

    L = LABELS[lang]

    def select(field):
        title, opts = L[field]
        st.selectbox(title, list(opts), format_func=opts.get, key=f"in_{field}")

    with st.form("client"):
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown(f"##### 💰 {T['sec_credit']}")
            st.number_input(T["credit_amount"], 250, 20000, step=250, key="in_credit_amount")
            st.slider(T["duration"], 4, 72, key="in_duration")
            st.slider(T["installment_rate"], 1, 4, key="in_installment_rate")
            for f in ["purpose", "credit_history", "other_installment_plans"]:
                select(f)
        with c2:
            st.markdown(f"##### 🏦 {T['sec_finance']}")
            for f in ["checking_status", "savings", "property", "other_debtors"]:
                select(f)
            st.slider(T["existing_credits"], 1, 4, key="in_existing_credits")
        with c3:
            st.markdown(f"##### 👤 {T['sec_client']}")
            st.slider(T["age"], 19, 75, key="in_age")
            for f in ["employment", "job", "housing", "telephone"]:
                select(f)
            st.slider(T["residence_since"], 1, 4, key="in_residence_since")
            st.slider(T["num_dependents"], 1, 2, key="in_num_dependents")
        submitted = st.form_submit_button(T["btn_score"], type="primary", width="stretch")

    if submitted:
        client = {f: st.session_state[f"in_{f}"] for f in FIELDS}
        ok, res = api_post("/predict", client)
        if ok:
            st.session_state["result"] = res
        else:
            st.error(res)

    res = st.session_state.get("result")
    if not res:
        st.info(T["empty_result"])
    else:
        approved = res["decision"] == "approve"
        st.markdown(f"""<div class="decision {'ok' if approved else 'no'}">
            {'✅' if approved else '⛔'} {T['approve'] if approved else T['reject']}</div>""",
                    unsafe_allow_html=True)
        st.write("")
        g, k1, k2 = st.columns([1.3, 1, 1], gap="medium")
        with g:
            st.markdown(gauge(res["credit_score"], T["score"]), unsafe_allow_html=True)
        p = res["default_probability"]
        risk_color = {"low": GREEN, "medium": AMBER, "high": RED}[res["risk_level"]]
        with k1:
            kpi(T["prob"], f"{p:.1%}", T["threshold_note"].format(t=res["threshold"]),
                color=RED if p >= res["threshold"] else GREEN)
        with k2:
            kpi(T["risk"], T[f"risk_{res['risk_level']}"], f"P(default) = {p:.3f}",
                color=risk_color)

        # --- пояснення
        st.markdown(f"### {T['why']}")
        st.caption(T["why_hint"])
        fac = pd.DataFrame(res["explanation"]["factors"])
        fac["name"] = fac["feature"].map(lambda f: feature_name(lang, f))
        fac["val"] = [value_label(lang, f, v) for f, v in zip(fac["feature"], fac["value"])]
        fac["label"] = fac["name"] + ": " + fac["val"]
        fac["effect_t"] = fac["effect"].map({"increases_risk": T["inc"], "decreases_risk": T["dec"]})
        fac["abs"] = fac["contribution"].abs()

        e1, e2 = st.columns([1.6, 1], gap="large")
        with e1:
            bars = alt.Chart(fac).mark_bar(cornerRadius=4).encode(
                y=alt.Y("label:N", sort=alt.EncodingSortField("abs", order="descending"), title=None,
                        axis=alt.Axis(labelLimit=260)),
                x=alt.X("contribution:Q", title=T["contribution"]),
                color=alt.Color("effect_t:N", title=None,
                                scale=alt.Scale(domain=[T["inc"], T["dec"]], range=[RED, GREEN]),
                                legend=alt.Legend(orient="bottom")),
                tooltip=[alt.Tooltip("label:N", title=""),
                         alt.Tooltip("contribution:Q", format="+.3f", title=T["contribution"])])
            rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#475569").encode(x="x:Q")
            chart(bars + rule, height=280)
        with e2:
            rows = "".join(
                f'<div class="factor"><span>{"🔺" if r.effect == "increases_risk" else "🔻"} '
                f'{r.name}</span><span class="v">{r.val}</span></div>'
                for r in fac.itertuples())
            st.markdown(f'<div class="card"><div class="kpi-label">{T["key_factors"]}</div>{rows}</div>',
                        unsafe_allow_html=True)
        st.caption(T["latency"].format(m=res["explanation"]["method"], ms=res["latency_ms"]))

# ============================================================================= 2. Пакетний скоринг
with tab_batch:
    st.write(T["batch_intro"])
    template = pd.DataFrame(PRESETS.values())
    st.download_button(T["batch_template"], template.to_csv(index=False).encode("utf-8"),
                       "clients_template.csv", "text/csv")
    up = st.file_uploader(T["batch_upload"], type="csv")
    if up is not None:
        df = pd.read_csv(up)
        st.dataframe(df.head(20), width="stretch", hide_index=True)
        if st.button(T["batch_run"], type="primary"):
            ok, out = api_post("/predict/batch", json.loads(df.to_json(orient="records")))
            if not ok:
                st.error(out)
            else:
                res_df = pd.DataFrame(out).set_index("index")
                full = df.join(res_df)
                n_err = int(full["error"].notna().sum()) if "error" in full else 0
                good = full[full.get("decision").notna()] if "decision" in full else full.iloc[0:0]
                b = st.columns(5)
                with b[0]:
                    kpi(T["batch_total"], len(full))
                with b[1]:
                    kpi(T["batch_approved"], int((good["decision"] == "approve").sum()), color=GREEN)
                with b[2]:
                    kpi(T["batch_rejected"], int((good["decision"] == "reject").sum()), color=RED)
                with b[3]:
                    kpi(T["batch_avg"], f"{good['credit_score'].mean():.0f}" if len(good) else "—")
                with b[4]:
                    kpi(T["batch_errors"], n_err, color=AMBER if n_err else "#f1f5f9")
                st.write("")
                show = full.copy()
                if "decision" in show:
                    show[T["decision"]] = show["decision"].map(
                        {"approve": "✅ " + T["approve"], "reject": "⛔ " + T["reject"]})
                    show = show.drop(columns="decision")
                # Результати — на початок таблиці, анкета — після них
                first = [c for c in ["true_label", T["decision"], "default_probability",
                                     "credit_score", "risk_level", "error"] if c in show]
                show = show[first + [c for c in show.columns if c not in first]]
                st.dataframe(
                    show, width="stretch", hide_index=True,
                    column_config={
                        "default_probability": st.column_config.ProgressColumn(
                            T["prob"], min_value=0, max_value=1, format="percent"),
                        "credit_score": st.column_config.NumberColumn(T["score"])})
                st.download_button(T["batch_download"], full.to_csv(index=False).encode("utf-8"),
                                   "scoring_results.csv", "text/csv", type="primary")

# ============================================================================= 3. Якість моделі
with tab_model:
    t = info["test"]
    cols = st.columns(5)
    for col, (key, val) in zip(cols, [("auc", t["roc_auc"]), ("gini", t["gini"]), ("ks", t["ks"]),
                                      ("recall", t["recall"]), ("precision", t["precision"])]):
        with col:
            kpi(T[f"kpi_{key}"], f"{val:.3f}", T[f"kpi_{key}_h"])
    st.write("")

    curves = info.get("curves")
    r1, r2 = st.columns(2, gap="large")
    with r1:
        st.markdown(f"##### {T['roc']}")
        if curves:
            roc = pd.DataFrame(curves["roc"])
            base = alt.Chart(roc).encode(
                x=alt.X("fpr:Q", title="False Positive Rate", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("tpr:Q", title="True Positive Rate", stack=None,
                        scale=alt.Scale(domain=[0, 1])))
            area = base.mark_area(interpolate="step-after", color=TEAL, opacity=.18)
            line = base.mark_line(interpolate="step-after", color=TEAL, strokeWidth=2.5)
            diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
                strokeDash=[5, 5], color="#475569").encode(x="x:Q", y="y:Q")
            line = area + line
            chart(line + diag)
        else:
            st.info(T["retrain"])
    with r2:
        st.markdown(f"##### {T['dist']}")
        if curves:
            sd = pd.DataFrame(curves["scores"])
            sd["class"] = sd["bad"].map({0: "good", 1: "bad"})
            hist = alt.Chart(sd).mark_bar(opacity=.65, binSpacing=1).encode(
                x=alt.X("score:Q", bin=alt.Bin(maxbins=25), title=T["score"]),
                y=alt.Y("count():Q", stack=None, title=T["count"]),
                color=alt.Color("class:N", scale=alt.Scale(domain=["good", "bad"], range=[GREEN, RED]),
                                legend=alt.Legend(orient="top-left", title=None)))
            chart(hist)
        else:
            st.info(T["retrain"])

    r3, r4, r5 = st.columns([1, 1, 1.3], gap="large")
    with r3:
        st.markdown(f"##### {T['cm']}")
        cm = t["confusion_matrix"]
        cmd = pd.DataFrame([
            {"a": "good", "p": "good", "n": cm["tn"]}, {"a": "good", "p": "bad", "n": cm["fp"]},
            {"a": "bad", "p": "good", "n": cm["fn"]}, {"a": "bad", "p": "bad", "n": cm["tp"]}])
        base = alt.Chart(cmd).encode(
            x=alt.X("p:N", title=T["pred"], sort=["good", "bad"], axis=alt.Axis(labelAngle=0)),
            y=alt.Y("a:N", title=T["actual"], sort=["good", "bad"]))
        heat = base.mark_rect(cornerRadius=6).encode(
            color=alt.Color("n:Q", scale=alt.Scale(range=["#132036", "#0f766e"]), legend=None))
        txt = base.mark_text(fontSize=22, fontWeight="bold", color="#f1f5f9").encode(text="n:Q")
        chart(heat + txt, height=260)
        st.caption(f"threshold = {info['threshold']:.3f} · cost = {t['cost']}")
    with r4:
        st.markdown(f"##### {T['weights']}")
        w = pd.DataFrame({"model": [m.upper() for m in info["weights"]],
                          "weight": list(info["weights"].values()),
                          "cv_auc": [info["cv_auc"][m] for m in info["weights"]],
                          "test_auc": [info["test_auc_per_model"][m] for m in info["weights"]]})
        wc = alt.Chart(w).mark_arc(innerRadius=55, cornerRadius=4).encode(
            theta="weight:Q",
            color=alt.Color("model:N", scale=alt.Scale(range=[TEAL, "#818cf8", AMBER]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["model", alt.Tooltip("weight:Q", format=".3f"),
                     alt.Tooltip("cv_auc:Q", format=".4f"), alt.Tooltip("test_auc:Q", format=".4f")])
        chart(wc, height=260)
        st.dataframe(w.rename(columns={"model": T["model"], "weight": T["weight"],
                                       "cv_auc": "CV AUC", "test_auc": "Test AUC"}),
                     hide_index=True, width="stretch")
    with r5:
        st.markdown(f"##### {T['fi']}")
        fi = pd.DataFrame({"f": list(info["top_features"]), "v": list(info["top_features"].values())})
        fi["name"] = fi["f"].map(lambda f: feature_name(lang, f))
        fic = alt.Chart(fi).mark_bar(cornerRadius=4, color=TEAL).encode(
            y=alt.Y("name:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
            x=alt.X("v:Q", title=None),
            tooltip=[alt.Tooltip("name:N", title=""), alt.Tooltip("v:Q", format=".0f")])
        chart(fic, height=330)

ARCH = """\
┌──────────────┐  HTTP/JSON  ┌────────────────┐
│ Streamlit UI │ ──────────▶ │ Flask REST API │
│    :8501     │ ◀────────── │ gunicorn :5000 │
└──────────────┘             └───────┬────────┘
                                     │ joblib
                          ┌──────────▼─────────┐
                          │ Preprocessor       │
                          │ RF · LGBM · XGB    │
                          │ threshold · SHAP   │
                          └────────────────────┘
   $ docker compose up --build"""

# ============================================================================= 4. Про проєкт
with tab_about:
    a1, a2 = st.columns([1.4, 1], gap="large")
    with a1:
        st.markdown(T["about_md"])
    with a2:
        st.markdown("##### Architecture")
        st.code(ARCH, language=None)
