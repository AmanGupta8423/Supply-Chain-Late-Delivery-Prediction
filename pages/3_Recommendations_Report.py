import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from src.data import load_dataset
from src.analytics import delay_rate_by

st.set_page_config(page_title="Recommendations & Report", page_icon="🧭", layout="wide")
st.title("🧭 Recommendations & Executive Report")
st.caption("Evidence-led operational recommendations based on observed historical delay patterns.")

try:
    df = load_dataset()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

regions = ["All regions"] + sorted(df["Order Region"].dropna().unique().tolist())
selected_region = st.selectbox("Select an order region", regions)
scope = df if selected_region == "All regions" else df[df["Order Region"] == selected_region].copy()

if scope.empty:
    st.warning("No records found for this selection.")
    st.stop()

overall_rate = scope["Is_Delayed"].mean() * 100
late_count = int(scope["Is_Delayed"].sum())
avg_late_delay = scope.loc[scope["Is_Delayed"] == 1, "Delay"].mean()

a, b, c = st.columns(3)
a.metric("Orders in scope", f"{len(scope):,}")
b.metric("Actual late-delivery rate", f"{overall_rate:.1f}%")
c.metric("Average delay among late orders", f"{avg_late_delay:.2f} days" if pd.notna(avg_late_delay) else "N/A")

st.markdown("---")
st.markdown("### Highest observed delay drivers")
driver_columns = ["Shipping Mode", "Customer Segment", "Department Name", "Type", "Order Status"]
driver_frames = []
for col in driver_columns:
    if col in scope.columns:
        temp = delay_rate_by(scope, col, min_orders=30)
        if not temp.empty:
            temp["Driver"] = col
            temp["Factor Level"] = col + " : " + temp[col].astype(str)
            driver_frames.append(temp[["Driver", "Factor Level", "total_orders", "late_orders", "delay_pct", "avg_delay"]])

if driver_frames:
    drivers = pd.concat(driver_frames, ignore_index=True)
    top_drivers = drivers.sort_values("delay_pct", ascending=False).head(10)
    fig = px.bar(
        top_drivers.sort_values("delay_pct"),
        x="delay_pct", y="Factor Level", color="delay_pct",
        orientation="h", text_auto=".1f",
        title="Top 10 factor levels by observed late-delivery rate",
        labels={"delay_pct": "Late delivery rate (%)", "Factor Level": "Driver factor"},
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(top_drivers, use_container_width=True, hide_index=True)
else:
    drivers = pd.DataFrame()
    st.info("Not enough records to calculate driver-level rates.")

st.markdown("### Recommended mitigation actions")
recommendations = []

if overall_rate >= 50:
    recommendations.append({
        "Priority": "High",
        "Observation": f"Late-delivery rate is {overall_rate:.1f}% in the selected scope.",
        "Recommended action": "Run a daily exception review for orders at risk; assign an owner and escalation deadline to each exception.",
        "Measure": "Late-delivery rate and number of unresolved exceptions",
    })
else:
    recommendations.append({
        "Priority": "Medium",
        "Observation": f"Late-delivery rate is {overall_rate:.1f}% in the selected scope.",
        "Recommended action": "Maintain routine monitoring and investigate segments that perform materially worse than the overall baseline.",
        "Measure": "Weekly late-delivery rate by region and shipping mode",
    })

if not drivers.empty:
    for _, row in top_drivers.head(4).iterrows():
        driver = row["Driver"]
        level = str(row["Factor Level"]).split(" : ", 1)[-1]
        rate = row["delay_pct"]
        if driver == "Shipping Mode":
            action = "Review carrier capacity, pickup cut-off times, route coverage, and service-level commitments for this shipping mode."
        elif driver == "Type":
            action = "Check whether payment authorization or payment-release timing is delaying order release; coordinate payment and fulfilment teams."
        elif driver == "Customer Segment":
            action = "Review service promises and order-priority rules for this customer segment; validate whether current capacity matches demand."
        elif driver == "Department Name":
            action = "Audit inventory availability, picking/packing queues, and replenishment lead times for this department."
        else:
            action = "Review the operational workflow and hand-offs for this order-status group; identify where orders wait longest."
        recommendations.append({
            "Priority": "High" if rate >= overall_rate + 5 else "Medium",
            "Observation": f"{driver} = {level} has an observed late-delivery rate of {rate:.1f}% ({int(row['total_orders']):,} orders).",
            "Recommended action": action,
            "Measure": "Compare late-delivery rate and average delay before vs. after intervention",
        })

rec_df = pd.DataFrame(recommendations).drop_duplicates(subset=["Observation"])
st.dataframe(rec_df, use_container_width=True, hide_index=True)

st.markdown("### Executive summary")
summary = f"""
# Supply Chain Late Delivery – Executive Summary

**Report generated:** {datetime.now().strftime("%d %b %Y, %H:%M")}

**Scope:** {selected_region}

- Orders analysed: {len(scope):,}
- Orders flagged as actually delayed (Delay > 0): {late_count:,}
- Actual late-delivery rate: {overall_rate:.1f}%
- Average delay among late orders: {avg_late_delay:.2f} days
- Profit sum across selected orders: ${scope['Order Profit Per Order'].sum():,.2f}

## Recommended actions

""" + "\n".join(
    f"{i}. **{row['Priority']} priority – {row['Observation']}**\n   - {row['Recommended action']}\n   - Measure: {row['Measure']}"
    for i, row in rec_df.iterrows()
) + """

## Interpretation note
These findings describe associations in historical data. They do not establish that a factor causes delays. Validate operational causes with carrier, warehouse, inventory, and order-level investigations before changing policy.
"""

st.markdown(summary)
st.download_button(
    "Download executive report (Markdown)",
    summary.encode("utf-8"),
    "supply_chain_executive_report.md",
    "text/markdown",
)

if not drivers.empty:
    st.download_button(
        "Download driver analysis (CSV)",
        drivers.to_csv(index=False).encode("utf-8"),
        "late_delivery_driver_analysis.csv",
        "text/csv",
    )
