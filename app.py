import streamlit as st
from src.data import load_dataset

st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 14px 16px;
        border-radius: 12px;
    }
    div[data-testid="stForm"] {
        border: 1px solid rgba(128,128,128,.2);
        border-radius: 12px;
        padding: 18px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📦 Supply Chain Intelligence")
st.subheader("Late Delivery Risk Prediction & Decision Support")
st.write(
    "An interactive application for predicting late-delivery risk, "
    "exploring supply-chain performance, and identifying operational actions."
)

try:
    df = load_dataset()
except FileNotFoundError:
    st.error(
        "Dataset not found. Place `DataCoSupplyChainDataset.csv` in the project "
        "root folder (the same folder as `app.py`) and restart the app."
    )
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Cleaned records", f"{len(df):,}")
c2.metric("Late-delivery risk rate", f"{df['Late_delivery_risk'].mean() * 100:.1f}%")
c3.metric("Regions covered", f"{df['Order Region'].nunique():,}")

st.markdown("---")
st.markdown("### Explore the application")
left, middle, right = st.columns(3)

with left:
    st.markdown("#### 1. Risk Prediction")
    st.write("Enter order-time information and estimate the model's late-delivery risk probability.")
    st.page_link("pages/1_Risk_Prediction.py", label="Open Risk Prediction", icon="🔮")

with middle:
    st.markdown("#### 2. Analytics Dashboard")
    st.write("Filter orders by payment type, region, customer segment, shipping mode, and time.")
    st.page_link("pages/2_Analytics_Dashboard.py", label="Open Analytics", icon="📊")

with right:
    st.markdown("#### 3. Recommendations & Report")
    st.write("Review high-delay drivers and download a concise operational action report.")
    st.page_link("pages/3_Recommendations_Report.py", label="Open Recommendations", icon="🧭")

st.info(
    "Model output is a statistical estimate based on historical DataCo records, "
    "not a guarantee that an individual order will be delayed."
)
