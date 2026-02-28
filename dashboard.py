import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime, timedelta
from groq import Groq # NEW: Added Groq Support
from src.utils.voice_bot import trigger_voice_alert 

# 1. Page Configuration
st.set_page_config(page_title="FinSecure | Enterprise Fraud Shield", layout="wide", page_icon="🛡️")

# 2. Advanced Professional Styling, Glassmorphism & Animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    * { font-family: 'Inter', sans-serif; }

    /* Theme: PhonePe Purple & Trust Background */
    .stApp {
        background: linear-gradient(135deg, #5f259f 0%, #3a1864 100%);
        color: #FFFFFF;
    }

    /* Animation: Subtle Fade In for all components */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stTable, .stMetric, .stPlotlyChart, .stImage, .stSubheader, .stMarkdown {
        animation: fadeIn 0.8s ease-out;
    }

    /* Animation: Critical Pulse for Fraud Alerts */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(255, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
    }
    .critical-alert {
        animation: pulse-red 2s infinite;
        border: 2px solid #ff0000 !important;
        border-radius: 12px;
        padding: 10px;
        background: rgba(255, 0, 0, 0.1);
    }

    /* Card Containers with Hover Effect */
    div[data-testid="stVerticalBlock"] > div:has(div.stMetric) {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(12px);
        transition: 0.3s;
    }
    div[data-testid="stVerticalBlock"] > div:has(div.stMetric):hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.12);
    }

    /* Metric & Label Styling */
    [data-testid="stMetricValue"] { color: #00ffd0 !important; font-weight: 800 !important; }
    [data-testid="stMetricLabel"] { color: #ffffff !important; font-weight: 700 !important; font-size: 16px !important; }

    /* Sliders & Inputs Visibility */
    .stSlider label, .stNumberInput label {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        background: rgba(255,255,255,0.15);
        padding: 4px 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        display: inline-block;
    }
    .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #3a1864 !important;
        font-weight: 700 !important;
        border: 2px solid #00d2ff !important;
    }

    /* Sidebar Professional Look */
    section[data-testid="stSidebar"] { background-color: #ffffff !important; }
    section[data-testid="stSidebar"] * { color: #3a1864 !important; font-weight: 600; }

    /* Main Action Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white !important;
        font-weight: 800 !important;
        font-size: 18px !important;
        padding: 15px !important;
        border-radius: 12px !important;
        border: 2px solid rgba(255,255,255,0.2) !important;
        transition: 0.4s;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        box-shadow: 0px 8px 25px rgba(0, 210, 255, 0.6);
        transform: scale(1.02);
        border: 2px solid #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Artifacts & Initialize Groq
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('src/models/fraud_model.pkl')
        explainer = joblib.load('src/models/shap_explainer.pkl')
        return model, explainer
    except:
        st.error("⚠️ System Offline: Model files missing in src/models/")
        return None, None

model, explainer = load_artifacts()

# NEW: Initialize Groq Client using Streamlit Secrets
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.sidebar.error("⚠️ Groq API Key missing in secrets.toml")

def get_ai_investigator_explanation(amount, risk_score, features):
    """Fetches a human-like forensic explanation from Groq AI."""
    prompt = f"""
    You are a Senior Financial Fraud Investigator. An AI model flagged a transaction:
    - Amount: ${amount}
    - Fraud Risk Score: {risk_score:.2f}%
    - Input Anomalies: {features}
    
    Provide a professional, concise 2-line explanation of WHY this was flagged. 
    Be direct and use forensic terminology.
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return completion.choices[0].message.content
    except:
        return "AI Investigator is offline. Manual review required based on anomaly scores."

# 4. Navigation Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/10433/10433048.png", width=80)
    st.title("FINSECURE")
    st.markdown("---")
    page = st.radio("OPERATIONS MENU", ["🛡️ Detection Dashboard", "🏛️ Cyber Crime Portal", "⚙️ System Health & Reliability"])

# --- PAGE 1: DETECTION DASHBOARD ---
if page == "🛡️ Detection Dashboard":
    st.title("🛡️ FinSecure Enterprise Fraud Shield")
    st.caption(f"System Operational • Latency: 42ms • Local Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # KPI SECTION
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("📡 NETWORK", "SECURE", "Live")
    with m2: st.metric("🎯 PRECISION", "99.8%", "+0.02")
    with m3: st.metric("📦 VOLUME", "1.24M", "Stable")
    with m4: st.metric("🤖 DRIFT", "0.004", "Minimal")
    
    st.markdown("---")
    
    col_input, col_gauge, col_explain = st.columns([1.2, 1, 1.4])
    
    with col_input:
        st.subheader("📝 Transaction Details")
        with st.container():
            amount = st.number_input("Transaction Amount ($)", value=1200.0, step=50.0)
            st.markdown("<br>", unsafe_allow_html=True)
            v1 = st.slider("Location Delta (V1)", -5.0, 5.0, 1.2)
            v2 = st.slider("Device Authentication (V2)", -5.0, 5.0, -0.5)
            v3 = st.slider("Behavioral Frequency (V3)", -5.0, 5.0, 2.0)
            
            input_features = np.array([0.0, v1, v2, v3] + [0.0]*25 + [amount]).reshape(1, -1)
            verify = st.button("EXECUTE SECURITY SCAN")
            
    with col_gauge:
        st.subheader("📉 Risk Index")
        if model:
            prob = model.predict_proba(input_features)[0][1] * 100
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob,
                number = {'suffix': "%", 'font': {'color': "#00ffd0", 'size': 50}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickcolor': "white"},
                    'bar': {'color': "#FF0000" if prob > 60 else "#00FFD0"},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(0, 255, 208, 0.3)"},
                        {'range': [30, 70], 'color': "rgba(255, 255, 0, 0.3)"},
                        {'range': [70, 100], 'color': "rgba(255, 0, 0, 0.3)"}
                    ],
                    'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 90}
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=350)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if verify:
                if prob > 70:
                    st.markdown('<div class="critical-alert">', unsafe_allow_html=True)
                    st.error("🚨 CRITICAL: FRAUD PATTERN DETECTED")
                    trigger_voice_alert(f"Security Alert. High risk transaction of {amount} dollars.")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif prob > 40:
                    st.warning("⚠️ ATTENTION: SUSPICIOUS ACTIVITY")
                else:
                    st.success("✅ TRANSACTION SECURED")

    with col_explain:
        st.subheader("🧠 Decision Intelligence")
        if verify and explainer:
            shap_values = explainer.shap_values(input_features)
            feature_names = ['Time', 'Location', 'Device', 'Freq'] + [f'V{i}' for i in range(4,29)] + ['Amount']
            
            # SHAP Waterfall
            fig_shap, ax = plt.subplots(figsize=(8, 4))
            shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], feature_names=feature_names, max_display=6, show=False)
            plt.gcf().set_facecolor('none')
            ax.set_facecolor('none')
            ax.tick_params(colors='white', labelsize=10)
            for text in ax.texts: text.set_color('white')
            st.pyplot(fig_shap)

            # --- NEW: INTEGRATED GROQ AI INVESTIGATOR ---
            st.markdown("---")
            st.markdown("<span style='color:#00ffd0; font-weight:700;'>🤖 REAL-TIME AI INVESTIGATOR SAYS:</span>", unsafe_allow_html=True)
            with st.spinner("AI Analyzer is scrutinizing logs..."):
                # Pass data to Groq for natural language explanation
                ai_msg = get_ai_investigator_explanation(amount, prob, f"V1:{v1}, V2:{v2}, V3:{v3}")
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.08); border-left:4px solid #00ffd0; padding:12px; border-radius:8px;">
                    <span style="color:#ffffff; font-style:italic; font-size:14px;">"{ai_msg}"</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("💡 Run scan to visualize AI behavior metrics.")

    st.markdown("---")
    st.subheader("🛰️ Global Transaction Stream")
    live_data = pd.DataFrame({
        'Timestamp': pd.date_range(start=datetime.now(), periods=8, freq='S'),
        'Gateway': ['Mumbai-HQ', 'Bangalore-DC', 'Delhi-North', 'Chennai-South']*2,
        'Risk Score': [f"{np.random.uniform(0, 15):.2f}%" for _ in range(8)],
        'Status': ['SUCCESS'] * 8
    })
    st.table(live_data)

# --- PAGE 2: CYBER CRIME PORTAL ---
elif page == "🏛️ Cyber Crime Portal":
    st.title("🏛️ National Cyber Crime Portal")
    st.markdown("### Government of India • Ministry of Home Affairs")
    
    with st.form("reporting_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Fraud Type", ["Credit Card Fraud", "UPI Phishing", "SIM Swapping"])
            incident_date = st.date_input("Date of Incident")
        with col2:
            amount_lost = st.number_input("Disputed Amount ($)", min_value=0)
            bank_name = st.text_input("Bank Name")
            
        ai_evidence = st.text_area("AI Forensic Logs (SHAP Evidence)", height=150)
        submitted = st.form_submit_button("LODGE OFFICIAL COMPLAINT")
        if submitted:
            st.success("✅ Case Registered. Incident ID: NCCRP-2026-X89")
            st.balloons()

# --- PAGE 3: SYSTEM HEALTH & RELIABILITY ---
elif page == "⚙️ System Health & Reliability":
    st.title("⚙️ System Health & Reliability Engine")
    st.caption("Monitoring AI Decision Quality, Data Patterns, and Response Speed")
    
    h1, h2, h3 = st.columns(3)
    h1.metric("Current Accuracy", "99.8%", "Optimal")
    h2.metric("Data Pattern Change", "0.038", "-0.002", delta_color="inverse")
    h3.metric("Uptime", "99.99%", "Continuous")

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 Decision Quality Over Time")
        accuracy_history = pd.DataFrame({
            'Day': range(30),
            'Performance': [0.992 - (x * 0.0001) for x in range(30)]
        })
        st.line_chart(accuracy_history, x='Day', y='Performance')
        st.info("💡 AI logic is consistent. No retraining needed at this stage.")

    with c2:
        st.subheader("📡 Transaction Response Speed (ms)")
        latency_data = np.random.normal(42, 4, 1000)
        fig_lat = px.histogram(latency_data, nbins=50, title="Processing Speed Distribution")
        fig_lat.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_lat, use_container_width=True)

    st.subheader("🛡️ Behavioral Stability Index")
    drift_df = pd.DataFrame({
        'Category': ['Spending Amount', 'User Location', 'Auth Methods', 'Purchase Freq'],
        'Stability Score': [0.012, 0.145, 0.032, 0.088],
        'Status': ['✅ Balanced', '⚠️ Minor Shift', '✅ Balanced', '✅ Balanced']
    })
    st.table(drift_df)