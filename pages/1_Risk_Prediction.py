import streamlit as st
import pandas as pd

from src.data import load_dataset, DATA_PATH
from src.model import FEATURES, train_model, predict_one, get_feature_importance

st.set_page_config(page_title="Late Delivery Risk Prediction", page_icon="🔮", layout="wide")
st.title("🔮 Late Delivery Risk Prediction")
st.caption("Enter the order details to estimate the probability of late-delivery risk.")

try:
    df = load_dataset()
    model, metrics = train_model(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load data or train the model: {exc}")
    st.stop()

def choices(column):
    return sorted(df[column].dropna().astype(str).unique().tolist())

st.markdown("### Order information")
with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    with c1:
        payment_type = st.selectbox("Payment Type (Type)", choices("Type"))
        scheduled_days = st.selectbox(
            "Days for shipment (scheduled)",
            sorted(df["Days for shipment (scheduled)"].dropna().astype(int).unique().tolist())
        )
        category = st.selectbox("Category Name", choices("Category Name"))
        segment = st.selectbox("Customer Segment", choices("Customer Segment"))
        department = st.selectbox("Department Name", choices("Department Name"))
    with c2:
        region = st.selectbox("Order Region", choices("Order Region"))
        shipping = st.selectbox("Shipping Mode", choices("Shipping Mode"))
        month = st.selectbox("Order Month", list(range(1, 13)), index=5)
        day = st.selectbox(
            "Order Day",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        )
        hour = st.slider("Order Hour", min_value=0, max_value=23, value=12)

    submitted = st.form_submit_button("Predict Late Delivery Risk", type="primary", use_container_width=True)

if submitted:
    input_values = {
        "Type": payment_type,
        "Days for shipment (scheduled)": scheduled_days,
        "Category Name": category,
        "Customer Segment": segment,
        "Department Name": department,
        "Order Region": region,
        "Shipping Mode": shipping,
        "order_month": month,
        "order_day": day,
        "order_hour": hour,
    }

    prediction, probability = predict_one(model, input_values)
    st.markdown("---")
    st.subheader("Prediction result")

    a, b, c = st.columns(3)
    a.metric("Estimated late-risk probability", f"{probability * 100:.1f}%")
    b.metric("Predicted class", "Late delivery risk" if prediction == 1 else "No late-delivery risk")
    b.caption("Class threshold: 50%")
    if probability < 0.40:
        risk_label, explanation = "Low", "The model estimates a comparatively lower late-delivery risk."
    elif probability < 0.70:
        risk_label, explanation = "Medium", "The model indicates a moderate late-delivery risk. Review the order conditions."
    else:
        risk_label, explanation = "High", "The model estimates a higher late-delivery risk. Consider proactive monitoring."
    c.metric("Risk band", risk_label)

    st.progress(probability)
    st.write(explanation)
    st.info(
        f"The model assigns {probability * 100:.1f}% probability to class 1. "
        "This is a model probability, not a guarantee or a calibrated measure of certainty."
    )

    if prediction == 1:
        st.warning(
            "Suggested next step: review carrier capacity, the selected shipping mode, "
            "regional fulfilment constraints, and dispatch readiness before promising the delivery date."
        )
    else:
        st.success(
            "Suggested next step: continue standard fulfilment monitoring and keep shipment status updated."
        )

    st.markdown("#### Input summary")
    st.dataframe(pd.DataFrame([input_values]), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### Model validation snapshot")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
m2.metric("Precision", f"{metrics['precision']:.1%}")
m3.metric("Recall", f"{metrics['recall']:.1%}")
m4.metric("F1 score", f"{metrics['f1']:.1%}")
st.caption(
    f"Hold-out test rows: {metrics['test_rows']:,}. The model is re-trained from the supplied dataset "
    "with a train-fitted frequency encoder and SMOTE applied only within the training pipeline."
)

with st.expander("View model feature importance"):
    importance = get_feature_importance(model)
    st.bar_chart(importance.set_index("Feature")["Importance"])
    st.dataframe(importance, use_container_width=True, hide_index=True)
