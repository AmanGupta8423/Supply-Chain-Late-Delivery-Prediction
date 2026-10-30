from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "DataCoSupplyChainDataset.csv"

# These are the columns removed in the original notebook. The app only drops
# columns that are actually present, so the loader is tolerant of schema variants.
COLUMNS_TO_DROP = [
    "Product Description", "Product Image", "Customer Email", "Customer Password",
    "Customer Fname", "Customer Lname", "Customer Street", "Customer Zipcode",
    "Order Zipcode", "Longitude", "Latitude", "Order Item Cardprod Id",
    "Order Item Id", "Order Item Discount", "Order Item Discount Rate",
    "Order Item Product Price", "Order Item Quantity", "Order Item Total",
    "Category Id", "Department Id", "Order Id", "Order Customer Id", "Customer Id",
    "Product Card Id", "Product Category Id", "Benefit per order", "Product Status",
    "Customer City", "Order City", "Order Country", "Order State", "Customer State",
    "Market",
]

DATE_COLUMNS = ["order date (DateOrders)", "shipping date (DateOrders)"]

@st.cache_data(show_spinner=False)
def load_dataset(path: str = str(DATA_PATH)) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    data = pd.read_csv(file_path, encoding="latin-1")
    data = data.drop(columns=[c for c in COLUMNS_TO_DROP if c in data.columns])

    if "Delivery Status" in data.columns:
        data = data[data["Delivery Status"] != "Shipping canceled"].copy()

    for col in DATE_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce", dayfirst=False)

    required = [
        "Late_delivery_risk", "Type", "Days for shipment (scheduled)",
        "Category Name", "Customer Segment", "Department Name",
        "Order Region", "Shipping Mode", "order date (DateOrders)",
        "shipping date (DateOrders)", "Order Profit Per Order",
    ]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = data.dropna(subset=["Late_delivery_risk", "order date (DateOrders)"]).copy()
    data["Late_delivery_risk"] = data["Late_delivery_risk"].astype(int)

    # Recreate the time features used by the notebook.
    order_dt = data["order date (DateOrders)"]
    data["order_month"] = order_dt.dt.month
    data["order_day"] = order_dt.dt.day_name()
    data["order_hour"] = order_dt.dt.hour

    # Actual-delivery analytics, following the notebook's delay definition.
    data["Order Processing Time"] = (
        data["shipping date (DateOrders)"] - data["order date (DateOrders)"]
    ).dt.days
    data["Delay"] = (
        data["Order Processing Time"] - data["Days for shipment (scheduled)"]
    )
    data["Is_Delayed"] = (data["Delay"] > 0).astype(int)

    data["Profitability Flag"] = "Break-even"
    data.loc[data["Order Profit Per Order"] > 0, "Profitability Flag"] = "Profit"
    data.loc[data["Order Profit Per Order"] < 0, "Profitability Flag"] = "Loss"

    return data.reset_index(drop=True)
