from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Loan_default.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Loan_default.csv was not found. Put this script in the same folder as your dataset."
    )

df = pd.read_csv(DATA_PATH)
if "Default" not in df.columns:
    raise ValueError("Expected a target column named 'Default' in Loan_default.csv.")

# Remove identifier and columns with excessive missing values.
df = df.drop(columns=["LoanID"], errors="ignore")
drop_cols = [c for c in df.columns if c != "Default" and df[c].isna().mean() > 0.50]
df = df.drop(columns=drop_cols)

# Create the same engineered feature used in the original notebook.
if {"LoanAmount", "Income"}.issubset(df.columns):
    df["loan_to_income_ratio"] = df["LoanAmount"] / (df["Income"].fillna(0) + 1)

df = df.dropna(subset=["Default"])
X = df.drop(columns=["Default"]).copy()
y = pd.to_numeric(df["Default"], errors="coerce")
valid = y.notna()
X, y = X.loc[valid].copy(), y.loc[valid].astype(int)

# Ensure target is binary.
target_values = sorted(y.unique().tolist())
if len(target_values) != 2:
    raise ValueError(f"Expected a binary Default target, found values: {target_values}")

categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numeric_cols = [c for c in X.columns if c not in categorical_cols]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=80, random_state=42, n_jobs=-1, class_weight="balanced_subsample",
        min_samples_leaf=2
    )
}

metrics = {}
trained = {}

for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    print(f"Training {name}...")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    result = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0, output_dict=True)
    }
    if hasattr(pipe, "predict_proba"):
        try:
            result["roc_auc"] = float(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]))
        except Exception:
            result["roc_auc"] = None
    metrics[name] = result
    trained[name] = pipe
    print(f"{name}: accuracy={result['accuracy']:.3f}, precision={result['precision']:.3f}, recall={result['recall']:.3f}, F1={result['f1']:.3f}")

# Save both pipelines. These files should only be loaded from your own trusted repository.
joblib.dump(trained["Random Forest"], MODEL_DIR / "random_forest_pipeline.joblib")
joblib.dump(trained["Logistic Regression"], MODEL_DIR / "logistic_regression_pipeline.joblib")

# Store safe UI metadata; use the dataset to generate dropdown choices in app.py.
metadata = {
    "feature_columns": X.columns.tolist(),
    "categorical_columns": categorical_cols,
    "numeric_columns": numeric_cols,
    "dropped_columns": drop_cols,
    "target_values": target_values,
    "metrics": metrics,
    "row_count": int(len(df)),
    "default_rate": float(y.mean())
}
with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nTraining complete.")
print(f"Saved models and metrics to: {MODEL_DIR}")
print("Note: recall for the Default class is especially important because the dataset may be imbalanced.")
