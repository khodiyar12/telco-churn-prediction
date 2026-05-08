import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import MODEL_DIR, CATEGORICAL_COLS, NUMERIC_COLS
from feature_engineering import add_features

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📡", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #fafaf8; }
.risk-high   { color: #d85a30; font-size: 2rem; font-weight: 700; }
.risk-medium { color: #ba7517; font-size: 2rem; font-weight: 700; }
.risk-low    { color: #0f6e56; font-size: 2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    for name in ["best_model.joblib", "lgbm_best.joblib", "logreg_baseline.joblib"]:
        path = MODEL_DIR / name
        if path.exists():
            return joblib.load(path), name
    return None, None

model, model_name = load_model()


def predict_customer(customer_dict):
    df = pd.DataFrame([customer_dict])
    df = add_features(df)
    feature_cols = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c in df.columns]
    X    = df[feature_cols]
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)
    tier = "High" if prob >= 0.60 else ("Medium" if prob >= 0.30 else "Low")
    return prob, pred, tier


def draw_gauge(prob):
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#fafaf8")
    ax.set_facecolor("#fafaf8")
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(theta, [1]*200, color="#e8e6e0", linewidth=12)
    color = "#d85a30" if prob >= 0.60 else ("#ba7517" if prob >= 0.30 else "#0f6e56")
    theta_val = np.linspace(np.pi, np.pi - prob * np.pi, 200)
    ax.plot(theta_val, [1]*200, color=color, linewidth=12)
    angle = np.pi - prob * np.pi
    ax.annotate("", xy=(angle, 0.85), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#0f0e0d", lw=2))
    ax.text(0, 0, f"{prob:.0%}", ha="center", va="center",
            fontsize=18, fontweight="bold", color=color)
    ax.set_ylim(0, 1.2)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.axis("off")
    return fig


st.title("📡 Telco Customer Churn Predictor")
st.caption(f"Model: `{model_name}`  |  IBM Telco dataset")

st.sidebar.title("Customer Profile")

gender         = st.sidebar.selectbox("Gender",            ["Male", "Female"])
senior         = st.sidebar.selectbox("Senior Citizen",    ["No", "Yes"])
partner        = st.sidebar.selectbox("Partner",           ["Yes", "No"])
dependents     = st.sidebar.selectbox("Dependents",        ["No", "Yes"])
tenure         = st.sidebar.slider("Tenure (months)",      0, 72, 12)
phone_service  = st.sidebar.selectbox("Phone Service",     ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines",    ["No", "Yes", "No phone service"])
internet       = st.sidebar.selectbox("Internet Service",  ["Fiber optic", "DSL", "No"])
online_sec     = st.sidebar.selectbox("Online Security",   ["No", "Yes", "No internet service"])
online_bkp     = st.sidebar.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
device_prot    = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support   = st.sidebar.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
streaming_tv   = st.sidebar.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
streaming_mov  = st.sidebar.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])
contract       = st.sidebar.selectbox("Contract",          ["Month-to-month", "One year", "Two year"])
paperless      = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment        = st.sidebar.selectbox("Payment Method",    [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_chg    = st.sidebar.slider("Monthly Charges ($)",  18.0, 120.0, 70.0, 0.5)
total_chg      = st.sidebar.number_input("Total Charges ($)", min_value=0.0,
                                          value=float(monthly_chg * max(tenure, 1)), step=10.0)

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    if model is None:
        st.error("No trained model found. Run `python src/train.py` first.")
    else:
        customer = {
            "Gender": gender, "Senior Citizen": senior,
            "Partner": partner, "Dependents": dependents,
            "Tenure Months": tenure, "Phone Service": phone_service,
            "Multiple Lines": multiple_lines, "Internet Service": internet,
            "Online Security": online_sec, "Online Backup": online_bkp,
            "Device Protection": device_prot, "Tech Support": tech_support,
            "Streaming TV": streaming_tv, "Streaming Movies": streaming_mov,
            "Contract": contract, "Paperless Billing": paperless,
            "Payment Method": payment,
            "Monthly Charges": monthly_chg, "Total Charges": total_chg,
        }

        prob, pred, tier = predict_customer(customer)

        col1, col2, col3 = st.columns([1.2, 1, 1])

        with col1:
            st.markdown("**Churn Probability**")
            fig = draw_gauge(prob)
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Risk Assessment**")
            css = f"risk-{tier.lower()}"
            st.markdown(f'<div class="{css}">{tier} Risk</div>', unsafe_allow_html=True)
            verdict = "⚠️ Likely to Churn" if pred == 1 else "✅ Likely to Stay"
            st.markdown(f"**Verdict:** {verdict}")
            st.metric("Churn Probability", f"{prob:.1%}")
            st.metric("Contract", contract)
            st.metric("Tenure", f"{tenure} months")

        with col3:
            st.markdown("**Key Risk Factors**")
            factors = []
            if contract == "Month-to-month":
                factors.append(("🔴 Month-to-month contract", "Highest churn driver"))
            if tenure < 6:
                factors.append(("🔴 New customer (<6 mo)", "High early-churn risk"))
            if payment == "Electronic check":
                factors.append(("🟡 Electronic check", "3x higher churn rate"))
            if online_sec == "No":
                factors.append(("🟡 No online security", "Adds churn risk"))
            if tech_support == "No":
                factors.append(("🟡 No tech support", "Adds churn risk"))
            if contract == "Two year":
                factors.append(("🟢 Two-year contract", "Strong retention signal"))
            if tenure >= 48:
                factors.append(("🟢 Long-term customer", "Low churn tendency"))
            if not factors:
                factors.append(("ℹ️ No strong signals", "Average risk profile"))
            for label, desc in factors[:6]:
                st.markdown(f"**{label}**  \n_{desc}_")

        st.divider()
        st.markdown("##### Full customer profile")
        # Convert all values to strings to avoid Arrow serialization issues
        profile_df = pd.DataFrame({
            "Field": list(customer.keys()),
            "Value": [str(v) for v in customer.values()]
        })
        st.dataframe(profile_df, hide_index=True)

with tab2:
    st.markdown("Upload a CSV with customer data to score all rows at once.")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded and model is not None:
        df_up = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df_up):,} customers")
        try:
            df_feat = add_features(df_up)
            feature_cols = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c in df_feat.columns]
            probs = model.predict_proba(df_feat[feature_cols])[:, 1]
            df_up["churn_probability"] = probs.round(4)
            df_up["risk_tier"] = pd.cut(probs, bins=[0, 0.30, 0.60, 1.0],
                                         labels=["Low", "Medium", "High"])
            c1, c2, c3 = st.columns(3)
            c1.metric("High Risk",   int((df_up["risk_tier"] == "High").sum()))
            c2.metric("Medium Risk", int((df_up["risk_tier"] == "Medium").sum()))
            c3.metric("Low Risk",    int((df_up["risk_tier"] == "Low").sum()))
            st.dataframe(df_up.sort_values("churn_probability", ascending=False).head(50).astype(str))
            st.download_button("Download predictions CSV",
                               df_up.to_csv(index=False).encode(),
                               "churn_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")
