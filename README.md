# Supply Chain Intelligence – Late Delivery Risk Prediction

A three-page Streamlit application based on the uploaded `yt.ipynb` notebook and the DataCo Supply Chain dataset.

## Pages

1. **Risk Prediction** – trains a Random Forest classification pipeline and predicts `Late_delivery_risk` from the same 10 input features used in the notebook.
2. **Analytics Dashboard** – interactive filters and historical delivery, profitability, regional, shipping-mode, and time-based analytics.
3. **Recommendations & Report** – identifies high-delay factor levels, suggests operational mitigations, and exports an executive report.

## Dataset setup

The notebook uses `DataCoSupplyChainDataset.csv` with Latin-1 encoding.

Place the CSV in the project root, alongside `app.py`:

```text
supply_chain_late_delivery_app/
├── app.py
├── DataCoSupplyChainDataset.csv
├── requirements.txt
├── README.md
├── pages/
└── src/
```

The dataset is not included in this repository ZIP. Add it locally before running the app. Check the dataset's licence and redistribution terms before uploading it to a public GitHub repository.

## Run locally in VS Code

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Streamlit

```bash
streamlit run app.py
```

Streamlit will open the application in your browser.

## Model details

- Target: `Late_delivery_risk`
- Features: `Type`, `Days for shipment (scheduled)`, `Category Name`, `Customer Segment`, `Department Name`, `Order Region`, `Shipping Mode`, `order_month`, `order_day`, `order_hour`
- Model: `RandomForestClassifier(random_state=42)`
- Encoding: frequency encoding for categorical features
- Imbalance handling: SMOTE inside an imbalanced-learn pipeline, so it is applied only to the training fold
- Evaluation: accuracy, precision, recall, F1 score, and confusion matrix are computed on a stratified hold-out test set

The notebook's reported baseline was Accuracy 0.79, Precision 0.84, and Recall 0.78. The app re-trains the model with the frequency encoder fitted only on the training fold to avoid the notebook's pre-split frequency-encoding leakage. Therefore, the app's measured metrics can differ from the notebook.

## Important interpretation notes

- The prediction page displays the model-estimated probability of class 1. It is not a guarantee and should not be described as certainty. The probability is not calibrated.
- The notebook's target `Late_delivery_risk` and its analytics flag `Delay > 0` are separate definitions in the source notebook. The app keeps that distinction: prediction uses `Late_delivery_risk`; historical analytics use actual dates and `Delay > 0`.
- The recommendations are rule-based summaries of historical associations, not causal claims. Validate causes with operational teams before making policy changes.
- The app's inputs are the notebook's existing feature set. This is a historical-data demonstration, not a production deployment or a real-time carrier integration.

## Suggested `.gitignore`

The included `.gitignore` excludes the dataset, local environments, caches, and model artifacts.
