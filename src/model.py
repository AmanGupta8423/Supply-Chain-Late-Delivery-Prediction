import numpy as np
import pandas as pd
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Exact feature set used in the user's notebook.
CATEGORICAL_FEATURES = [
    "Type",
    "Category Name",
    "Customer Segment",
    "Department Name",
    "Order Region",
    "Shipping Mode",
    "order_day",
]
NUMERIC_FEATURES = [
    "Days for shipment (scheduled)",
    "order_month",
    "order_hour",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "Late_delivery_risk"


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Train-fitted frequency encoder; unseen categories map to zero."""

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).copy()
        self.frequency_maps_ = []
        for col in frame.columns:
            frequencies = frame[col].value_counts(normalize=True, dropna=False)
            self.frequency_maps_.append(frequencies.to_dict())
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        encoded = np.zeros(frame.shape, dtype=float)
        for idx, mapping in enumerate(self.frequency_maps_):
            encoded[:, idx] = frame.iloc[:, idx].map(mapping).fillna(0).astype(float)
        return encoded


def build_pipeline():
    categorical_pipeline = SklearnPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("frequency_encoder", FrequencyEncoder()),
    ])
    numeric_pipeline = SklearnPipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    # SMOTE is applied only to the training fold by the imbalanced-learn pipeline.
    return ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])


@st.cache_resource(show_spinner="Training the Random Forest.")
def train_model(data_path: str):
    from src.data import load_dataset

    data = load_dataset(data_path)
    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "test_rows": len(y_test),
    }
    return pipeline, metrics


def predict_one(pipeline, input_values: dict):
    row = pd.DataFrame([input_values], columns=FEATURES)
    probability = float(pipeline.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    return prediction, probability


def get_feature_importance(pipeline):
    classifier = pipeline.named_steps["classifier"]
    importance = classifier.feature_importances_
    names = FEATURES
    return (
        pd.DataFrame({"Feature": names, "Importance": importance})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
