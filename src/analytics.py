import pandas as pd

FILTER_COLUMNS = {
    "Payment Type": "Type",
    "Order Region": "Order Region",
    "Customer Segment": "Customer Segment",
    "Shipping Mode": "Shipping Mode",
    "Department": "Department Name",
    "Category": "Category Name",
    "Order Status": "Order Status",
    "Order Day": "order_day",
}

def apply_filters(df, selections):
    filtered = df.copy()
    for label, column in FILTER_COLUMNS.items():
        selected = selections.get(label)
        if selected and column in filtered.columns:
            filtered = filtered[filtered[column].isin(selected)]
    for label, column in [("Order Month", "order_month"), ("Order Hour", "order_hour")]:
        selected = selections.get(label)
        if selected:
            filtered = filtered[filtered[column].isin(selected)]
    scheduled = selections.get("Scheduled Days")
    if scheduled:
        filtered = filtered[filtered["Days for shipment (scheduled)"].isin(scheduled)]
    return filtered


def delay_rate_by(df, column, min_orders=20):
    result = (
        df.groupby(column, dropna=False)
        .agg(
            total_orders=("Is_Delayed", "size"),
            late_orders=("Is_Delayed", "sum"),
            avg_delay=("Delay", "mean"),
        )
        .reset_index()
    )
    result["delay_pct"] = result["late_orders"] / result["total_orders"] * 100
    return result[result["total_orders"] >= min_orders].sort_values("delay_pct", ascending=False)
