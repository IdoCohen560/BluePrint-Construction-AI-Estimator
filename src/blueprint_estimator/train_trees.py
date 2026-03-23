"""Decision Tree and Random Forest training with CV and metrics."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from blueprint_estimator.features import make_preprocessor


def train_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    categorical: list[str],
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict[str, Any]:
    pre = make_preprocessor(categorical)
    pipe = Pipeline(
        [
            ("prep", pre),
            (
                "clf",
                DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
            ),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    param_grid = {
        "clf__max_depth": [3, 5, 8, 12, None],
        "clf__min_samples_leaf": [1, 2, 4],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=cv, n_jobs=1)
    t0 = time.perf_counter()
    grid.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    y_pred = grid.predict(X_test)
    return {
        "model": grid.best_estimator_,
        "best_params": grid.best_params_,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "train_time_s": train_s,
        "X_test": X_test,
        "y_test": y_test,
    }


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    categorical: list[str],
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict[str, Any]:
    pre = make_preprocessor(categorical)
    pipe = Pipeline(
        [
            ("prep", pre),
            (
                "clf",
                RandomForestClassifier(random_state=random_state, class_weight="balanced"),
            ),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    param_grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [4, 8, 12, None],
        "clf__min_samples_leaf": [1, 2],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=cv, n_jobs=1)
    t0 = time.perf_counter()
    grid.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    y_pred = grid.predict(X_test)
    return {
        "model": grid.best_estimator_,
        "best_params": grid.best_params_,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "train_time_s": train_s,
        "X_test": X_test,
        "y_test": y_test,
    }
