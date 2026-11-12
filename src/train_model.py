import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import PayerFeatureEngineer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "claims.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "claim_model.joblib"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "evaluation_report.json"
TARGET = "status"
CATEGORICAL_COLUMNS = ["payer", "provider", "cpt_code", "diagnosis_code"]
NUMERIC_COLUMNS = ["billed_amount", "patient_age"]


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS + ["payer_denial_rate", "payer_claim_count"]),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    return Pipeline(
        steps=[
            ("feature_engineer", PayerFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_and_evaluate() -> dict:
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y))
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    report = classification_report(y_test, predictions, digits=4, output_dict=True)
    matrix = confusion_matrix(y_test, predictions, labels=["Paid", "Partially_Paid", "Denied"])

    output = {
        "summary": {
            "n_rows": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "class_distribution": y.value_counts(normalize=True).to_dict(),
        },
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "labels": ["Paid", "Partially_Paid", "Denied"],
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    results = train_and_evaluate()
    print(json.dumps(results["summary"], indent=2))
    print(json.dumps(results["classification_report"]["macro avg"], indent=2))
    print(json.dumps({"confusion_matrix": results["confusion_matrix"]}, indent=2))
