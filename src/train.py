"""Train price and investment-proxy models and log them to local MLflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    _area_sqft,
    parse_amount,
    parse_price,
    prepare_features,
)

TRAIN_COLUMNS = [
    TARGET_COLUMN,
    "Price (in rupees)",
    "Carpet Area",
    "Title",
    "Super Area",
    "Bathroom",
    "Balcony",
    "Floor",
    "Car Parking",
    "location",
    "Status",
    "Transaction",
    "Furnishing",
    "facing",
    "overlooking",
    "Ownership",
]


def build_pipeline(estimator) -> Pipeline:
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def train(data_path: Path, max_rows: int | None = None, skip_mlflow: bool = False) -> dict:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    available_columns = set(pd.read_csv(data_path, nrows=0).columns)
    required = {TARGET_COLUMN, "Price (in rupees)", "Carpet Area", "location"}
    missing = required - available_columns
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    selected_columns = [column for column in TRAIN_COLUMNS if column in available_columns]
    raw = pd.read_csv(data_path, usecols=selected_columns, low_memory=False)

    if max_rows and len(raw) > max_rows:
        raw = raw.sample(n=max_rows, random_state=42).reset_index(drop=True)
    else:
        raw = raw.reset_index(drop=True)

    raw["_target_price"] = raw[TARGET_COLUMN].map(parse_amount)
    raw["_area_sqft"] = raw["Carpet Area"].map(_area_sqft)
    raw["_rate_per_sqft"] = raw["Price (in rupees)"].map(parse_price)
    derived_rate = raw["_target_price"] / raw["_area_sqft"]
    raw["_rate_per_sqft"] = raw["_rate_per_sqft"].where(
        raw["_rate_per_sqft"].gt(0), derived_rate
    )

    regression_data = raw.loc[raw["_target_price"].gt(0)].copy().reset_index(drop=True)
    if len(regression_data) < 100:
        raise ValueError("At least 100 valid total listing amounts are required.")
    regression_features = prepare_features(regression_data)
    regression_target = regression_data["_target_price"].astype(float)
    reg_train_idx, reg_test_idx = train_test_split(
        np.arange(len(regression_data)), test_size=0.2, random_state=42
    )
    x_train = regression_features.iloc[reg_train_idx]
    x_test = regression_features.iloc[reg_test_idx]
    y_train = regression_target.iloc[reg_train_idx]
    y_test = regression_target.iloc[reg_test_idx]

    regression_model = build_pipeline(
        ExtraTreesRegressor(
            n_estimators=80,
            max_depth=20,
            min_samples_leaf=5,
            max_features=0.9,
            n_jobs=-1,
            random_state=42,
        )
    )
    regression_model.fit(x_train, np.log1p(y_train))
    predicted_log = regression_model.predict(x_test)
    price_predictions = np.maximum(np.expm1(predicted_log), 0)
    regression_metrics = {
        "mae_rupees": float(mean_absolute_error(y_test, price_predictions)),
        "rmse_rupees": float(np.sqrt(mean_squared_error(y_test, price_predictions))),
        "r2": float(r2_score(y_test, price_predictions)),
    }

    classifier_data = raw.loc[raw["_rate_per_sqft"].gt(0)].copy().reset_index(drop=True)
    if len(classifier_data) < 100:
        raise ValueError("At least 100 valid price-per-area records are required.")
    classifier_features = prepare_features(classifier_data)
    cls_train_idx, cls_test_idx = train_test_split(
        np.arange(len(classifier_data)), test_size=0.2, random_state=42
    )
    x_cls_train = classifier_features.iloc[cls_train_idx]
    x_cls_test = classifier_features.iloc[cls_test_idx]
    train_ppsf = classifier_data.iloc[cls_train_idx]["_rate_per_sqft"].to_numpy(dtype=float)
    test_ppsf = classifier_data.iloc[cls_test_idx]["_rate_per_sqft"].to_numpy(dtype=float)
    train_locations = (
        classifier_data.iloc[cls_train_idx]["location"]
        .fillna("unknown").astype(str).str.strip().str.lower()
    )
    test_locations = (
        classifier_data.iloc[cls_test_idx]["location"]
        .fillna("unknown").astype(str).str.strip().str.lower()
    )
    threshold_frame = pd.DataFrame(
        {"location": train_locations.to_numpy(), "price_per_sqft": train_ppsf}
    )
    location_thresholds = threshold_frame.groupby("location")[
        "price_per_sqft"
    ].median()
    fallback_threshold = float(np.median(train_ppsf))
    train_thresholds = train_locations.map(location_thresholds).fillna(fallback_threshold)
    test_thresholds = test_locations.map(location_thresholds).fillna(fallback_threshold)
    y_class_train = (train_ppsf <= train_thresholds.to_numpy()).astype(int)
    y_class_test = (test_ppsf <= test_thresholds.to_numpy()).astype(int)

    classifier = build_pipeline(
        RandomForestClassifier(
            n_estimators=80,
            max_depth=16,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
    )
    classifier.fit(x_cls_train, y_class_train)
    class_predictions = classifier.predict(x_cls_test)
    class_probabilities = classifier.predict_proba(x_cls_test)[:, 1]
    classification_metrics = {
        "accuracy": float(accuracy_score(y_class_test, class_predictions)),
        "precision": float(precision_score(y_class_test, class_predictions, zero_division=0)),
        "recall": float(recall_score(y_class_test, class_predictions, zero_division=0)),
        "f1": float(f1_score(y_class_test, class_predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_class_test, class_probabilities))
        if len(np.unique(y_class_test)) == 2
        else None,
        "confusion_matrix": confusion_matrix(y_class_test, class_predictions).tolist(),
    }

    models_dir = PROJECT_ROOT / "models"
    reports_dir = PROJECT_ROOT / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(regression_model, models_dir / "price_regressor.joblib", compress=3)
    joblib.dump(classifier, models_dir / "investment_proxy_classifier.joblib", compress=3)
    thresholds = {
        "metric": "listing price per carpet-area square foot",
        "definition": "Below local median listing price per square foot",
        "fallback_rupees_per_sqft": fallback_threshold,
        "by_location_rupees_per_sqft": {
            str(key): float(value) for key, value in location_thresholds.items()
        },
    }
    _write_json(models_dir / "benchmark_thresholds.json", thresholds)

    metrics = {
        "dataset_file": data_path.name,
        "rows_used": int(len(regression_data)),
        "train_rows": int(len(reg_train_idx)),
        "test_rows": int(len(reg_test_idx)),
        "classifier_rows": int(len(classifier_data)),
        "classifier_train_rows": int(len(cls_train_idx)),
        "classifier_test_rows": int(len(cls_test_idx)),
        "regression": regression_metrics,
        "classification_proxy": classification_metrics,
        "classification_target": thresholds["definition"],
        "forecast_note": (
            "The dataset has listing prices, not five-year outcomes. The app applies "
            "a user-selected annual growth assumption to produce a scenario."
        ),
    }
    _write_json(models_dir / "metrics.json", _json_safe(metrics))
    _save_feature_importance(regression_model, models_dir / "feature_importance.json")
    _write_model_report(reports_dir / "model_metrics.md", metrics)

    if not skip_mlflow:
        _log_mlflow(
            regression_model,
            classifier,
            regression_metrics,
            classification_metrics,
            len(regression_data),
        )

    print(json.dumps(_json_safe(metrics), indent=2))
    print(f"Saved trained models under: {models_dir}")
    return metrics


def _save_feature_importance(model: Pipeline, output_path: Path) -> None:
    try:
        preprocessor = model.named_steps["preprocessor"]
        names = preprocessor.get_feature_names_out()
        values = model.named_steps["model"].feature_importances_
        ordered = sorted(
            zip(names, values), key=lambda item: float(item[1]), reverse=True
        )[:30]
        _write_json(
            output_path,
            [{"feature": str(name), "importance": float(value)} for name, value in ordered],
        )
    except (AttributeError, KeyError, ValueError):
        _write_json(output_path, [])


def _write_model_report(path: Path, metrics: dict) -> None:
    reg = metrics["regression"]
    clf = metrics["classification_proxy"]
    report = f"""# Model evaluation

Dataset: {metrics['dataset_file']}

Regression rows used: {metrics['rows_used']} ({metrics['train_rows']} train, {metrics['test_rows']} test)

Classifier rows used: {metrics['classifier_rows']} ({metrics['classifier_train_rows']} train, {metrics['classifier_test_rows']} test)

## Current total asking-price regression

- MAE: ₹{reg['mae_rupees']:,.0f}
- RMSE: ₹{reg['rmse_rupees']:,.0f}
- R²: {reg['r2']:.4f}

## Investment proxy classifier

Target: {metrics['classification_target']}.

- Accuracy: {clf['accuracy']:.4f}
- Precision: {clf['precision']:.4f}
- Recall: {clf['recall']:.4f}
- F1: {clf['f1']:.4f}
- ROC AUC: {clf['roc_auc'] if clf['roc_auc'] is not None else 'not available'}
- Confusion matrix: {clf['confusion_matrix']}

## Limitations

The classifier predicts a price-per-area affordability proxy, not future investment returns.
The source contains property listing amounts and per-area rates, not verified completed sale prices or five-year
outcomes. The five-year value shown in the app is a scenario based on the user's annual
growth assumption, not a supervised five-year model.
"""
    path.write_text(report, encoding="utf-8")


def _log_mlflow(regression_model, classifier, reg_metrics, clf_metrics, rows: int) -> None:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is required. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').resolve().as_posix()}",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("real-estate-investment-advisor")
    with mlflow.start_run(run_name="extra-trees-price-and-proxy-v1"):
        mlflow.log_param("dataset_rows", rows)
        mlflow.log_param("regression_model", "ExtraTreesRegressor on log price")
        mlflow.log_param("classifier", "RandomForestClassifier")
        mlflow.log_param("proxy_definition", "below location median price per sqft")
        mlflow.log_metrics(
            {
                "regression_mae_rupees": reg_metrics["mae_rupees"],
                "regression_rmse_rupees": reg_metrics["rmse_rupees"],
                "regression_r2": reg_metrics["r2"],
                "classification_accuracy": clf_metrics["accuracy"],
                "classification_precision": clf_metrics["precision"],
                "classification_recall": clf_metrics["recall"],
                "classification_f1": clf_metrics["f1"],
            }
        )
        trusted_types = ["numpy.dtype", "sklearn.tree._tree.Tree"]
        mlflow.sklearn.log_model(
            regression_model,
            name="price_regressor",
            skops_trusted_types=trusted_types,
        )
        mlflow.sklearn.log_model(
            classifier,
            name="investment_classifier",
            skops_trusted_types=trusted_types,
        )
        mlflow.log_artifact(str(PROJECT_ROOT / "models" / "metrics.json"))
        mlflow.log_artifact(str(PROJECT_ROOT / "models" / "benchmark_thresholds.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "India_Property_Listings.csv",
        help="Path to the public listings CSV.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for a quicker development run.",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Train and save models without logging to MLflow.",
    )
    args = parser.parse_args()
    train(args.data.resolve(), max_rows=args.max_rows, skip_mlflow=args.skip_mlflow)


if __name__ == "__main__":
    main()
