import os
import pickle
import pandas as pd
import streamlit as st

# ---------------- LOAD MODEL & SCALER (.pkl) ----------------
@st.cache_resource
def load_saved_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    possible_paths = [
        r"C:\Users\bhave\OneDrive\Desktop\Creadit_card_project\Model\saved_models\fraud_detection_all_models.pkl",
        os.path.join(BASE_DIR, 'Model', 'saved_models', 'fraud_detection_all_models.pkl'),
        os.path.join(BASE_DIR, 'saved_models', 'fraud_detection_all_models.pkl'),
        os.path.join(BASE_DIR, 'fraud_detection_all_models.pkl')
    ]

    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        st.error("Error: Could not find 'fraud_detection_all_models.pkl' in any expected directory.")
        return None

    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading pickle file: {e}")
        return None

data_pkg = load_saved_data()

if data_pkg:
    scaler = data_pkg['scaler']
    lr_model = data_pkg['logistic_model']
    rf_model = data_pkg['random_forest_model']
else:
    st.stop()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# ---------------- SIDEBAR: MODEL SELECTION ----------------
st.sidebar.header("⚙️ Model Configuration")
model_choice = st.sidebar.radio(
    "Select ML Algorithm:",
    ["Random Forest", "Logistic Regression"],
    index=0
)


selected_model = rf_model if model_choice == "Random Forest" else lr_model
f1_score_val = "0.85" if model_choice == "Random Forest" else "0.78"


if "last_selected_model" not in st.session_state:
    st.session_state["last_selected_model"] = model_choice

if st.session_state["last_selected_model"] != model_choice:
    st.session_state["show_result"] = False
    st.session_state["last_selected_model"] = model_choice

selected_model = rf_model if model_choice == "Random Forest" else lr_model
f1_score_val = "0.85" if model_choice == "Random Forest" else "0.78"

# ---------------- CSS STYLING ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7ff 0%, #eef4ff 50%, #f8f5ff 100%);
        color: #1e293b;
    }
    .block-container {
        max-width: 1400px;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .dashboard-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .dashboard-title .title-text {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-title {
        color: #1e3a8a;
        font-size: 23px;
        font-weight: 750;
        margin-top: 28px;
        margin-bottom: 15px;
        padding-left: 12px;
        border-left: 5px solid #6366f1;
    }
    .kpi-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.10);
        border: 1px solid rgba(226, 232, 240, 0.9);
    }
    .kpi-title { color: #64748b; font-size: 14px; font-weight: 600; }
    .kpi-value {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 30px; font-weight: 800;
    }
    .result-box-safe {
        padding: 25px; border-radius: 16px; border: 1px solid #bbf7d0;
        background: linear-gradient(135deg, #f0fdf4, #dcfce7); text-align: center;
    }
    .result-box-fraud {
        padding: 25px; border-radius: 16px; border: 1px solid #fecaca;
        background: linear-gradient(135deg, #fef2f2, #fee2e2); text-align: center;
    }
    .result-value-safe { font-size: 28px; font-weight: 800; color: #16a34a; }
    .result-value-fraud { font-size: 28px; font-weight: 800; color: #dc2626; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="dashboard-title">
    <span>💳</span>
    <span class="title-text">Credit Card Fraud Detection Dashboard</span>
</div>
""", unsafe_allow_html=True)

# ---------------- SECTION 1: TOP KPIs ----------------
st.markdown('<div class="section-title">1. Top KPIs</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="kpi-card"><div class="kpi-title">Total Transactions</div><div class="kpi-value">2,84,807</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="kpi-card"><div class="kpi-title">Fraud Cases</div><div class="kpi-value">492</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="kpi-card"><div class="kpi-title">Fraud Rate</div><div class="kpi-value">0.17%</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Model F1-Score ({model_choice})</div><div class="kpi-value">{f1_score_val}</div></div>', unsafe_allow_html=True)

# ---------------- SECTION 2: OVERVIEW ----------------
st.markdown('<div class="section-title">2. Overview</div>', unsafe_allow_html=True)
st.subheader("Fraud vs. Legitimate Transaction Chart")
chart_data = pd.DataFrame({
    "Type": ["Legitimate", "Fraud"],
    "Count": [284315, 492]
})
st.bar_chart(chart_data, x="Type", y="Count")

# ---------------- SECTION 3: TRANSACTION ANALYSIS ----------------
st.markdown('<div class="section-title">3. Transaction Analysis</div>', unsafe_allow_html=True)
analysis_col1, analysis_col2 = st.columns(2)

with analysis_col1:
    st.subheader("Amount Distribution Chart")
    amount_df = pd.DataFrame({
        "Amount Range ($)": ["0-100", "100-500", "500-1000", "1000+"],
        "Transactions": [3500, 4200, 1700, 600]
    })
    st.bar_chart(amount_df, x="Amount Range ($)", y="Transactions")

with analysis_col2:
    st.subheader("Fraud Trend Chart")
    trend_df = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "Fraud Cases": [30, 42, 35, 51, 39, 48]
    })
    st.line_chart(trend_df, x="Month", y="Fraud Cases")

# ---------------- SECTION 4: PREDICTION ----------------
st.markdown('<div class="section-title">4. Prediction</div>', unsafe_allow_html=True)
st.write("Transaction Input Fields:")

transaction_amount = st.number_input(
    "Transaction Amount ($)",
    min_value=0.0,
    value=150.0,
    step=10.0
)

if st.button("Predict Fraud Status"):
    input_dict = {'Time': 0.0}
    for i in range(1, 29):
        input_dict[f'V{i}'] = 0.0
    input_dict['Amount'] = transaction_amount

    input_df = pd.DataFrame([input_dict])
    scaled_values = scaler.transform(input_df[['Time', 'Amount']])
    input_df['Time'] = scaled_values[:, 0]
    input_df['Amount'] = scaled_values[:, 1]

    prediction = selected_model.predict(input_df)[0]
    probabilities = selected_model.predict_proba(input_df)[0]
    risk_score = probabilities[1] * 100

    st.session_state["show_result"] = True
    st.session_state["prediction"] = prediction
    st.session_state["risk_score"] = risk_score

# ---------------- SECTION 5: RESULT ----------------
st.markdown('<div class="section-title">5. Result</div>', unsafe_allow_html=True)

if st.session_state.get("show_result", False):
    is_fraud = st.session_state["prediction"] == 1
    risk = st.session_state["risk_score"]

    if is_fraud:
        st.markdown(f"""
        <div class="result-box-fraud">
            <div class="result-title">Prediction Result ({model_choice})</div>
            <div class="result-value-fraud">🚨 Potential Fraud</div>
            <p style="font-size:18px; color:#dc2626; margin-top:8px;"><b>Risk Probability: {risk:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box-safe">
            <div class="result-title">Prediction Result ({model_choice})</div>
            <div class="result-value-safe">✅ Legitimate Transaction</div>
            <p style="font-size:18px; color:#16a34a; margin-top:8px;"><b>Risk Probability: {risk:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Click 'Predict Fraud Status' above to display prediction results.")

# ---------------- SECTION 6: MODEL PERFORMANCE ----------------
st.markdown('<div class="section-title">6. Model Performance</div>', unsafe_allow_html=True)
perf_col1, perf_col2 = st.columns(2)

with perf_col1:
    st.subheader("Confusion Matrix")
    matrix_df = pd.DataFrame({
        "Actual Status": ["Actual Legitimate", "Actual Fraud"],
        "Pred Legitimate": [56850, 14],
        "Pred Fraud": [14, 84]
    })
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)

with perf_col2:
    st.subheader("Performance Metrics (Precision | Recall | F1 | ROC-AUC)")
    metrics_df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1 Score", "ROC-AUC"],
        "Score": [0.86, 0.85, 0.85, 0.96]
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# ---------------- SECTION 7: DATA TABLE ----------------
st.markdown('<div class="section-title">7. Data Table</div>', unsafe_allow_html=True)
st.subheader("Recent / Sample Transactions with Prediction")

sample_data = pd.DataFrame({
    "Transaction ID": ["TXN1001", "TXN1002", "TXN1003", "TXN1004", "TXN1005"],
    "Amount ($)": [250.00, 1250.50, 80.00, 5400.00, 320.75],
    "Prediction": ["Legitimate", "Fraud", "Legitimate", "Fraud", "Legitimate"],
    "Risk Probability": ["0.12%", "98.45%", "0.04%", "99.12%", "0.85%"]
})

st.dataframe(sample_data, use_container_width=True, hide_index=True)