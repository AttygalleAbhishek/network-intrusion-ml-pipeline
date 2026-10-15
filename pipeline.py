"""Leak-free baseline pipeline for UNSW-NB15 and CIC-IDS-2017.

The pipeline keeps the test partition isolated before fitting imputers, encoders,
scalers, or models. It supports binary labels (``Label``/``label``) and the
multiclass ``attack_cat`` label used by UNSW-NB15.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder, RobustScaler


RANDOM_STATE = 42


def detect_target_column(df: pd.DataFrame, requested: str | None = None) -> str:
    """Resolve common target names used by the two supported datasets."""
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Target column {requested!r} is not present in the CSV.")
        return requested

    candidates = ("Label", "label", "attack_cat", "Attack_cat", "class", "Class")
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not detect a target column. Use --target with one of the CSV columns."
    )


def normalize_labels(y: pd.Series) -> pd.Series:
    """Normalize whitespace and missing labels without changing class meaning."""
    normalized = y.astype("string").str.strip()
    if normalized.isna().any() or (normalized == "").any():
        raise ValueError("The target column contains missing or empty labels.")
    return normalized.astype(str)


def split_first_partitioning(
    df: pd.DataFrame, target_col: str, test_size: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split before any fitted statistical operation to prevent test leakage."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if df[target_col].nunique(dropna=False) < 2:
        raise ValueError("The target must contain at least two classes.")

    X = df.drop(columns=[target_col]).copy()
    y = normalize_labels(df[target_col])
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def physics_informed_sanitization(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace non-finite values and remove constant columns using train only.

    Negative values are not globally clipped: a negative feature may be a valid
    encoded/category value. Domain-specific physical bounds belong in a dataset
    adapter, not in a generic benchmark pipeline.
    """
    train = X_train.copy()
    test = X_test.copy()
    numeric_cols = train.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        train[col] = train[col].replace([np.inf, -np.inf], np.nan)
        test[col] = test[col].replace([np.inf, -np.inf], np.nan)
        fill_value = train[col].median()
        if pd.isna(fill_value):
            fill_value = 0.0
        train[col] = train[col].fillna(fill_value)
        test[col] = test[col].fillna(fill_value)

    categorical_cols = train.select_dtypes(exclude=np.number).columns.tolist()
    for col in categorical_cols:
        train[col] = train[col].fillna("__missing__").astype(str)
        test[col] = test[col].fillna("__missing__").astype(str)

    variances = train[numeric_cols].var().fillna(0.0)
    dead_cols = variances[variances < 1e-12].index.tolist()
    train = train.drop(columns=dead_cols)
    test = test.drop(columns=dead_cols)
    return train, test


def robust_variance_transformation(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply signed log compression and train-fitted robust scaling to numbers.

    ``sign(x) * log1p(abs(x))`` is defined for both positive and negative
    benchmark columns, unlike an unconditional ``log1p(x)``.
    """
    train = X_train.copy()
    test = X_test.copy()
    numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        for frame in (train, test):
            values = frame[numeric_cols].to_numpy(dtype=float)
            frame[numeric_cols] = np.sign(values) * np.log1p(np.abs(values))
        scaler = RobustScaler()
        train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
        test[numeric_cols] = scaler.transform(test[numeric_cols])
    return train, test


def encode_tabular_categoricals(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ordinal-encode categoricals, assigning unseen test values to -1."""
    train = X_train.copy()
    test = X_test.copy()
    cat_cols = train.select_dtypes(exclude=np.number).columns.tolist()
    if cat_cols:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        train[cat_cols] = encoder.fit_transform(train[cat_cols])
        test[cat_cols] = encoder.transform(test[cat_cols])
    return train, test


def _average_precision(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Return binary or macro one-vs-rest average precision."""
    classes = np.unique(y_true)
    if len(classes) == 2:
        positive_index = list(classes).index(classes[1])
        return float(average_precision_score(y_true == classes[1], probabilities[:, positive_index]))

    binarizer = LabelBinarizer().fit(classes)
    encoded = binarizer.transform(y_true)
    if encoded.shape[1] == 1:
        encoded = np.column_stack([1 - encoded, encoded])
    return float(
        average_precision_score(encoded, probabilities, average="macro")
    )


def calculate_nids_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    model_name: str,
) -> dict[str, Any]:
    """Calculate metrics valid for both binary and multiclass classification."""
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    metrics: dict[str, Any] = {
        "model": model_name,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        ),
        "confusion_matrix": matrix.tolist(),
        "labels": labels.tolist(),
    }
    if y_prob is not None:
        metric_name = "pr_auc" if len(np.unique(y_true)) == 2 else "pr_auc_macro"
        metrics[metric_name] = _average_precision(y_true, y_prob)

    if len(labels) == 2:
        tn, fp, fn, tp = matrix.ravel()
        metrics["fpr"] = float(fp / (fp + tn)) if fp + tn else 0.0
        metrics["false_alarms"] = int(fp)
    else:
        false_positives = matrix.sum(axis=0) - np.diag(matrix)
        true_negatives = matrix.sum() - (matrix.sum(axis=0) + matrix.sum(axis=1) - np.diag(matrix))
        metrics["fpr_macro"] = float(
            np.mean(np.divide(false_positives, false_positives + true_negatives, out=np.zeros_like(false_positives, dtype=float), where=(false_positives + true_negatives) != 0))
        )

    print(f"\n{'=' * 48}\nEvaluation Metrics: {model_name}\n{'=' * 48}")
    print(f"MCC:             {metrics['mcc']:.4f}")
    if "pr_auc" in metrics:
        print(f"PR-AUC:          {metrics['pr_auc']:.4f}")
    if "pr_auc_macro" in metrics:
        print(f"Macro PR-AUC:    {metrics['pr_auc_macro']:.4f}")
    print(json.dumps({key: value for key, value in metrics.items() if key not in {"classification_report"}}, indent=2))
    print(classification_report(y_true, y_pred, zero_division=0))
    return metrics


def run_pipeline(
    csv_path: str | Path,
    target_col: str | None = None,
    test_size: float = 0.20,
    n_estimators: int = 100,
) -> dict[str, Any]:
    """Load a CSV, train both baselines, and return their evaluation results."""
    df = pd.read_csv(csv_path)
    target_col = detect_target_column(df, target_col)
    X_train, X_test, y_train, y_test = split_first_partitioning(
        df, target_col, test_size
    )
    X_train, X_test = physics_informed_sanitization(X_train, X_test)
    X_train, X_test = robust_variance_transformation(X_train, X_test)
    X_train, X_test = encode_tabular_categoricals(X_train, X_test)

    models = {
        "Cost-Sensitive Random Forest": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
    results = {"target": target_col, "classes": sorted(y_train.unique().tolist()), "models": {}}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        results["models"][name] = calculate_nids_metrics(
            y_test, predictions, probabilities, name
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--target", default=None, help="Target column; auto-detected by default.")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--estimators", type=int, default=100)
    args = parser.parse_args()
    run_pipeline(args.csv_path, args.target, args.test_size, args.estimators)


if __name__ == "__main__":
    main()