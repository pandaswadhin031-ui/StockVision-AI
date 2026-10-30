import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import glob

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="StockVision AI",
    page_icon="📈",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f8ff;
    color: #172b4d;
}

header {
    background: transparent !important;
}

.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}

h1, h2, h3 {
    color: #153b75;
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5eaf3;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5eaf3;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(20,50,100,0.04);
}

[data-testid="stMetricLabel"] {
    color: #64748b;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #173f7a;
    font-weight: 800;
}

.stButton button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 9px;
    height: 48px;
    font-weight: 700;
}

.stButton button:hover {
    background: #1748b5;
    color: white;
}

.hero {
    background: linear-gradient(110deg, #173f7a, #2563eb);
    color: white;
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 32px;
}

.hero p {
    color: #dbeafe;
    font-size: 15px;
}

.info-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e5eaf3;
    margin-bottom: 15px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>📈 StockVision AI</h1>
    <p>Intelligent Stock Market Prediction Powered by Machine Learning</p>
    <p>Analyze historical trends and explore AI-based price forecasts.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
data_files = glob.glob("stock_data/*.csv")
model_files = glob.glob("*.pkl")

model_companies = {
    os.path.splitext(os.path.basename(f))[0]
    for f in model_files
}

company_files = {
    os.path.splitext(os.path.basename(f))[0]: f
    for f in data_files
}

companies = sorted(set(company_files) & model_companies)


if not companies:
    st.error("No trained company models found.")
    st.stop()


# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🏦 StockVision AI")
st.sidebar.caption("Machine Learning Dashboard")
st.sidebar.markdown("---")

company = st.sidebar.selectbox(
    "Select Company",
    companies
)

st.sidebar.markdown("---")

st.sidebar.markdown("### ⚙️ Project Information")
st.sidebar.write("• 50 trained company models")
st.sidebar.write("• Linear Regression")
st.sidebar.write("• Random Forest")
st.sidebar.write("• Historical stock dataset")

st.sidebar.markdown("---")
st.sidebar.caption("Academic ML Project")


# ---------------- DASHBOARD ----------------
st.subheader("Market Analysis Dashboard")

st.write(
    f"Selected company: **{company}**"
)

predict_button = st.button(
    "🚀 Generate Stock Prediction",
    use_container_width=True
)


if predict_button:

    try:

        # Load dataset
        df = pd.read_csv(company_files[company])

        if "Series" in df.columns:
            df = df[df["Series"] == "EQ"]

        features = ["Open", "High", "Low", "Close", "Volume"]

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        for col in features:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(subset=features + ["Date"])
        df = df.sort_values("Date")

        if df.empty:
            st.error("No valid stock data found.")
            st.stop()

        # Load trained model
        model_path = f"{company}.pkl"

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Prepare latest record
        latest = df.iloc[-1]

        input_data = pd.DataFrame(
            [latest[features].values],
            columns=features
        )

        # Predict return and price
        predicted_return = float(
            model.predict(input_data)[0]
        )

        latest_price = float(latest["Close"])

        predicted_price = latest_price * (
            1 + predicted_return
        )

        percentage_change = predicted_return * 100

        st.success(
            f"Prediction generated successfully for {company}"
        )

        # ---------------- METRICS ----------------
        st.markdown("### 📊 Prediction Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Latest Closing Price",
            f"₹{latest_price:,.2f}"
        )

        col2.metric(
            "Predicted Next-Day Price",
            f"₹{predicted_price:,.2f}",
            f"{percentage_change:+.2f}%"
        )

        col3.metric(
            "Last Available Data",
            latest["Date"].strftime("%d %b %Y")
        )

        st.markdown("---")

        # ---------------- CHART ----------------
        st.markdown("### 📈 Historical Price Analysis")

        chart_data = (
            df.tail(100)
            .set_index("Date")[["Close"]]
        )

        st.line_chart(
            chart_data,
            use_container_width=True
        )

        # ---------------- INSIGHTS ----------------
        st.markdown("### 🔍 AI Prediction Insights")

        if percentage_change > 0:
            st.success(
                f"The model estimates a potential increase of {percentage_change:.2f}%."
            )

        elif percentage_change < 0:
            st.warning(
                f"The model estimates a potential decrease of {abs(percentage_change):.2f}%."
            )

        else:
            st.info("The model estimates almost no price change.")

        st.markdown("---")

        st.markdown("### 🧠 Machine Learning Information")

        st.write(
            "The selected company's trained model analyzes "
            "Open, High, Low, Close and Volume data to estimate "
            "the next trading day's percentage return."
        )

        st.caption(
            "Experimental academic prediction only. "
            "The dataset is historical, not live market data. "
            "This is not financial advice."
        )

    except Exception as e:

        st.error(f"Prediction error: {e}")


else:

    st.markdown("### ✨ Welcome to StockVision AI")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 AI-Powered Analysis</h3>
            <p>Machine learning models analyze historical stock data
            to estimate future price movements.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Data Visualization</h3>
            <p>Explore historical closing prices and understand
            stock price trends through charts.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Select a company from the sidebar and click "
        "'Generate Stock Prediction' to begin."
    )


# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown(
    "<center>StockVision AI | Machine Learning Project | 2026</center>",
    unsafe_allow_html=True
)
