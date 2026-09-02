import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATH CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = [
    os.path.join(BASE_DIR, "Model", "saved_models", "fraud_detection_all_models.pkl"),
    os.path.join(BASE_DIR, "model", "saved_models", "fraud_detection_all_models.pkl"),
    os.path.join(BASE_DIR, "saved_models", "fraud_detection_all_models.pkl"),
    os.path.join(BASE_DIR, "fraud_detection_all_models.pkl"),
    r"..\Model\saved_models\fraud_detection_all_models.pkl"
]

DATASET_PATHS = [
    os.path.join(BASE_DIR, "Dataset", "creditcard.csv"),
    os.path.join(BASE_DIR, "..", "Dataset", "creditcard.csv"),
    os.path.join(BASE_DIR, "Model", "..", "Dataset", "creditcard.csv"),
    r"C:\Users\bhave\OneDrive\Desktop\Creadit_card_project\Dataset\creditcard.csv"
]


# ============================================================
# LOAD MODEL & DATASET
# ============================================================
@st.cache_resource
def load_saved_data():
    for path in MODEL_PATHS:
        path = os.path.normpath(path)
        if os.path.isfile(path):
            try:
                with open(path, "rb") as file:
                    package = pickle.load(file)
                return package, None
            except Exception as e:
                return None, f"Error loading model:\n{e}"
    return None, "❌ Model file not found."


@st.cache_data
def load_dataset():
    for path in DATASET_PATHS:
        path = os.path.normpath(path)
        if os.path.isfile(path):
            try:
                df = pd.read_csv(path)
                return df, None
            except Exception as e:
                return None, f"Error loading dataset:\n{e}"
    return None, "❌ Could not find Dataset/creditcard.csv"


data_pkg, model_error = load_saved_data()
if data_pkg is None:
    st.error(model_error)
    st.stop()

df, dataset_error = load_dataset()
if df is None:
    st.error(dataset_error)
    st.stop()

scaler = data_pkg["scaler"]
lr_model = data_pkg["logistic_model"]
rf_model = data_pkg["random_forest_model"]

best_model_name = data_pkg.get("best_model_name", "Random Forest")
feature_names = data_pkg.get("feature_names", ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================
st.sidebar.markdown("## ⚙️ Model Configuration")

model_choice = st.sidebar.radio(
    "Select ML Algorithm:",
    ["Random Forest", "Logistic Regression"],
    index=0
)

selected_model = rf_model if model_choice == "Random Forest" else lr_model

st.sidebar.markdown("---")
st.sidebar.info(
    f"""
    **Best Model**
    {best_model_name}

    **Features Used**
    Time + 4 Major V-Groups + Amount
    """
)

# ============================================================
# MOBILE RESPONSIVE CUSTOM CSS
# ============================================================
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f5f7ff 0%, #eef4ff 50%, #f8f5ff 100%); color: #1e293b; }

    .block-container { 
        max-width: 1450px; 
        padding-left: 2rem; 
        padding-right: 2rem; 
    }

    .dashboard-title { font-size: 32px; font-weight: 800; margin-bottom: 5px; }
    .title-text { background: linear-gradient(90deg, #2563eb, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .section-title { color: #1e3a8a; font-size: 20px; font-weight: 750; margin-top: 25px; margin-bottom: 15px; padding-left: 10px; border-left: 5px solid #6366f1; }

    .result-card { text-align: center; padding: 25px; border-radius: 18px; margin-top: 15px; color: white; }
    .result-card h1 { font-size: 26px; margin-bottom: 10px; color: white; }
    .risk-number { font-size: 40px; font-weight: 800; color: white; }
    .legitimate { background: linear-gradient(135deg, #0f5132, #198754); border: 2px solid #20c997; }
    .fraud { background: linear-gradient(135deg, #842029, #dc3545); border: 2px solid #ff6b6b; }

    /* MOBILE RESPONSIVE MEDIA QUERIES */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
            padding-top: 1rem !important;
        }
        .dashboard-title {
            font-size: 24px !important;
            text-align: center;
        }
        .section-title {
            font-size: 18px !important;
            margin-top: 18px !important;
        }
        .result-card {
            padding: 15px !important;
        }
        .result-card h1 {
            font-size: 20px !important;
        }
        .risk-number {
            font-size: 32px !important;
        }
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
            margin-bottom: 10px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="dashboard-title">💳 <span class="title-text">Credit Card Fraud Detection Dashboard</span></div>',
    unsafe_allow_html=True)

# ============================================================
# SECTION 1: TOP KPIs
# ============================================================
st.markdown('<div class="section-title">1. Top KPIs</div>', unsafe_allow_html=True)

total_transactions = len(df)
fraud_cases = int((df["Class"] == 1).sum())
legitimate_cases = int((df["Class"] == 0).sum())
fraud_rate = (fraud_cases / total_transactions * 100) if total_transactions > 0 else 0

selected_metrics = data_pkg["random_forest_metrics"] if model_choice == "Random Forest" else data_pkg[
    "logistic_metrics"]
model_f1 = selected_metrics["f1"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Transactions", f"{total_transactions:,}")
k2.metric("Fraud Cases", f"{fraud_cases:,}")
k3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
k4.metric("Model F1-Score", f"{model_f1:.4f}")

# ============================================================
# SECTION 2: OVERVIEW
# ============================================================
st.markdown('<div class="section-title">2. Overview</div>', unsafe_allow_html=True)

ov_col1, ov_col2 = st.columns(2)
with ov_col1:
    st.subheader("Fraud vs Legitimate Transactions")
    chart_data = pd.DataFrame({"Type": ["Legitimate", "Fraud"], "Count": [legitimate_cases, fraud_cases]})
    st.bar_chart(chart_data.set_index("Type"))

with ov_col2:
    st.subheader("Distribution Breakdown")
    st.dataframe(chart_data, use_container_width=True, hide_index=True)
    st.info("Notice the high imbalance: Precision & Recall are critical evaluation metrics for this dataset.")

# ============================================================
# SECTION 3: TRANSACTION ANALYSIS
# ============================================================
st.markdown('<div class="section-title">3. Transaction Analysis</div>', unsafe_allow_html=True)

an_col1, an_col2 = st.columns(2)
with an_col1:
    st.subheader("💰 Amount Distribution")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df["Amount"], bins=40, color="#6366f1")
    ax.set_xlabel("Amount (₹)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig, clear_figure=True)

with an_col2:
    st.subheader("📈 Fraud Trend Over Time")
    fraud_df = df[df["Class"] == 1].copy()
    if len(fraud_df) > 0:
        fraud_df["Time Bin"] = pd.cut(fraud_df["Time"], bins=15)
        trend = fraud_df.groupby("Time Bin", observed=False).size()
        trend_df = pd.DataFrame({"Time Bins": [str(x) for x in trend.index], "Fraud Cases": trend.values})
        st.line_chart(trend_df.set_index("Time Bins"))

# ============================================================
# SECTION 4: PREDICTION (4 INPUTS & ACTION BUTTONS)
# ============================================================
st.markdown('<div class="section-title">4. Simple Transaction Input (Divided in 4 Groups)</div>',
            unsafe_allow_html=True)

# Action Buttons
btn_c1, btn_c2, btn_c3 = st.columns(3)

if btn_c1.button("⚡ Fill Legitimate Case", use_container_width=True):
    st.session_state['time_val'] = 1000.0
    st.session_state['amount_val'] = 150.0
    st.session_state['g1'] = 0.0
    st.session_state['g2'] = 0.0
    st.session_state['g3'] = 0.0
    st.session_state['g4'] = 0.0

if btn_c2.button("🚨 Fill Fraudulent Case", use_container_width=True):
    st.session_state['time_val'] = 406.0
    st.session_state['amount_val'] = 1150.0
    st.session_state['g1'] = -10.0
    st.session_state['g2'] = -15.0
    st.session_state['g3'] = -12.0
    st.session_state['g4'] = -5.0

if btn_c3.button("🔄 Reset Inputs", use_container_width=True):
    st.session_state['time_val'] = 50000.0
    st.session_state['amount_val'] = 100.0
    st.session_state['g1'] = 0.0
    st.session_state['g2'] = 0.0
    st.session_state['g3'] = 0.0
    st.session_state['g4'] = 0.0
    if "prediction_result" in st.session_state:
        del st.session_state["prediction_result"]

p_col1, p_col2 = st.columns(2)
with p_col1:
    time_input = st.number_input("⏱️ Transaction Time", min_value=0.0, value=st.session_state.get('time_val', 50000.0))
with p_col2:
    amount_input = st.number_input("💰 Transaction Amount (₹)", min_value=0.0,
                                   value=st.session_state.get('amount_val', 100.0), format="%.2f")

st.subheader("⚙️ Main Anomaly Risk Indicators (Divided in 4 Inputs)")

v_col1, v_col2, v_col3, v_col4 = st.columns(4)

with v_col1:
    group1 = st.number_input("🔹 Group 1 (V1-V7 Risk)", value=st.session_state.get('g1', 0.0), format="%.2f")
with v_col2:
    group2 = st.number_input("⚠️ Group 2 (V8-V14 Critical Risk)", value=st.session_state.get('g2', 0.0), format="%.2f")
with v_col3:
    group3 = st.number_input("🔹 Group 3 (V15-V21 Risk)", value=st.session_state.get('g3', 0.0), format="%.2f")
with v_col4:
    group4 = st.number_input("🔹 Group 4 (V22-V28 Risk)", value=st.session_state.get('g4', 0.0), format="%.2f")

predict_click = st.button("🔍 Predict Transaction", type="primary", use_container_width=True)

# Prediction Logic
if predict_click:
    input_dict = {"Time": time_input}

    for i in range(1, 8): input_dict[f"V{i}"] = group1
    for i in range(8, 15): input_dict[f"V{i}"] = group2
    for i in range(15, 22): input_dict[f"V{i}"] = group3
    for i in range(22, 29): input_dict[f"V{i}"] = group4

    input_dict["Amount"] = amount_input

    input_df = pd.DataFrame([input_dict])[feature_names]

    if model_choice == "Logistic Regression":
        input_scaled = scaler.transform(input_df)
        prediction = selected_model.predict(input_scaled)[0]
        prob = selected_model.predict_proba(input_scaled)[0][1] * 100
    else:
        prediction = selected_model.predict(input_df)[0]
        prob = selected_model.predict_proba(input_df)[0][1] * 100

    risk = "HIGH" if prob >= 70 else ("MEDIUM" if prob >= 30 else "LOW")
    st.session_state["prediction_result"] = {
        "prediction": int(prediction),
        "prob": prob,
        "risk": risk,
        "amount": amount_input,
        "model": model_choice
    }

# ============================================================
# SECTION 5: RESULT
# ============================================================
st.markdown('<div class="section-title">5. Result</div>', unsafe_allow_html=True)

res = st.session_state.get("prediction_result")
if res:
    if res["prediction"] == 1 or res["prob"] >= 50:
        st.markdown(f'''
            <div class="result-card fraud">
                <h1>🚨 FRAUDULENT TRANSACTION</h1>
                <div class="risk-number">{res["prob"]:.2f}%</div>
                <p>Fraud Risk Probability</p>
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
            <div class="result-card legitimate">
                <h1>✅ LEGITIMATE TRANSACTION</h1>
                <div class="risk-number">{res["prob"]:.2f}%</div>
                <p>Fraud Risk Probability</p>
            </div>
        ''', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Transaction Amount", f"₹{res['amount']:,.2f}")
    r2.metric("Status", "FRAUD" if res["prediction"] == 1 or res["prob"] >= 50 else "LEGITIMATE")
    r3.metric("Risk Level", res["risk"])
    r4.metric("Selected Model", res["model"])

# ============================================================
# SECTION 6: MODEL PERFORMANCE
# ============================================================
st.markdown('<div class="section-title">6. Model Performance</div>', unsafe_allow_html=True)

m_col1, m_col2 = st.columns(2)
with m_col1:
    st.subheader("Confusion Matrix")
    cm = np.array(selected_metrics["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=["Actual Legitimate", "Actual Fraud"], columns=["Pred Legitimate", "Pred Fraud"])
    st.dataframe(cm_df, use_container_width=True)

with m_col2:
    st.subheader(f"{model_choice} Evaluation Metrics")
    st.metric("Precision", f"{selected_metrics['precision']:.4f}")
    st.metric("Recall", f"{selected_metrics['recall']:.4f}")
    st.metric("F1-Score", f"{selected_metrics['f1']:.4f}")
    st.metric("ROC-AUC", f"{selected_metrics['roc_auc']:.4f}")

# ============================================================
# SECTION 7: DATA TABLE
# ============================================================
st.markdown('<div class="section-title">7. Data Table (Sample Transactions)</div>', unsafe_allow_html=True)

recent_df = df[["Time", "Amount", "Class"]].tail(50).copy()
recent_df["Prediction"] = recent_df["Class"].map({0: "Legitimate", 1: "Fraud"})
st.dataframe(recent_df, use_container_width=True, hide_index=True)
