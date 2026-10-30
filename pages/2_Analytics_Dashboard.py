import streamlit as st
import pandas as pd
import plotly.express as px

from src.data import load_dataset
from src.analytics import apply_filters, FILTER_COLUMNS, delay_rate_by

st.set_page_config(page_title="Supply Chain Analytics", page_icon="📊", layout="wide")
st.title("📊 Supply Chain Analytics Dashboard")
st.caption("Explore historical delivery performance using interactive filters.")

try:
    df = load_dataset()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

with st.sidebar:
    st.header("Dashboard filters")
    selections = {}
    for label, column in FILTER_COLUMNS.items():
        if column in df.columns:
            options = sorted(df[column].dropna().astype(str).unique().tolist())
            selections[label] = st.multiselect(label, options, default=options)
    selections["Order Month"] = st.multiselect("Order Month", list(range(1, 13)), default=list(range(1, 13)))
    selections["Order Hour"] = st.multiselect("Order Hour", list(range(24)), default=list(range(24)))
    scheduled_options = sorted(df["Days for shipment (scheduled)"].dropna().astype(int).unique().tolist())
    selections["Scheduled Days"] = st.multiselect("Scheduled Shipment Days", scheduled_options, default=scheduled_options)
    if st.button("Reset filters"):
        st.rerun()

filtered = apply_filters(df, selections)
if filtered.empty:
    st.warning("No orders match the selected filters. Widen your filters.")
    st.stop()

total = len(filtered)
late = int(filtered["Is_Delayed"].sum())
late_pct = late / total * 100
on_time_pct = 100 - late_pct
avg_delay = filtered.loc[filtered["Is_Delayed"] == 1, "Delay"].mean()
profit = filtered["Order Profit Per Order"].sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Orders", f"{total:,}")
k2.metric("Late Deliveries", f"{late:,}")
k3.metric("Late Delivery Rate", f"{late_pct:.1f}%")
k4.metric("On-time / Not-late Rate", f"{on_time_pct:.1f}%")
k5.metric("Average Delay", f"{avg_delay:.2f} days" if pd.notna(avg_delay) else "N/A")

st.caption("Delivery KPIs above use actual shipping dates and the notebook's Delay > 0 definition. The model target is `Late_delivery_risk`.")
st.markdown("---")

left, right = st.columns(2)
with left:
    delay_dist = filtered["Delay"].value_counts().sort_index().rename_axis("Delay (days)").reset_index(name="Orders")
    fig = px.bar(delay_dist, x="Delay (days)", y="Orders", title="Order distribution by delay (days)")
    st.plotly_chart(fig, use_container_width=True)

with right:
    profitability = filtered["Profitability Flag"].value_counts().rename_axis("Profitability").reset_index(name="Orders")
    fig = px.pie(profitability, names="Profitability", values="Orders", hole=0.45,
                 title="Profitability distribution")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Where are delays concentrated?")
c1, c2 = st.columns(2)
with c1:
    region = delay_rate_by(filtered, "Order Region")
    fig = px.bar(region, x="Order Region", y="delay_pct", color="delay_pct",
                 title="Actual late-delivery rate by region", text_auto=".1f")
    fig.update_layout(xaxis_tickangle=-35, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    shipping = delay_rate_by(filtered, "Shipping Mode")
    fig = px.bar(shipping, x="Shipping Mode", y="delay_pct", color="delay_pct",
                 title="Actual late-delivery rate by shipping mode", text_auto=".1f")
    fig.update_layout(xaxis_tickangle=-25, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Time-based patterns")
month_data = filtered.groupby("order_month")["Is_Delayed"].mean().mul(100).reset_index(name="Delay %")
day_data = filtered.groupby("order_day")["Is_Delayed"].mean().mul(100).reset_index(name="Delay %")
hour_data = filtered.groupby("order_hour")["Is_Delayed"].mean().mul(100).reset_index(name="Delay %")

t1, t2, t3 = st.columns(3)
with t1:
    st.plotly_chart(px.line(month_data, x="order_month", y="Delay %", markers=True, title="Delay rate by month"), use_container_width=True)
with t2:
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_data["order_day"] = pd.Categorical(day_data["order_day"], categories=day_order, ordered=True)
    day_data = day_data.sort_values("order_day")
    st.plotly_chart(px.bar(day_data, x="order_day", y="Delay %", title="Delay rate by weekday"), use_container_width=True)
with t3:
    st.plotly_chart(px.line(hour_data, x="order_hour", y="Delay %", markers=True, title="Delay rate by order hour"), use_container_width=True)

st.markdown("### Filtered data")
st.dataframe(
    filtered[[
        "Type", "Order Region", "Customer Segment", "Shipping Mode",
        "Department Name", "Category Name", "Days for shipment (scheduled)",
        "Delay", "Is_Delayed", "Late_delivery_risk"
    ]].head(500),
    use_container_width=True,
    hide_index=True,
)
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered orders (CSV)", csv, "filtered_supply_chain_orders.csv", "text/csv")
