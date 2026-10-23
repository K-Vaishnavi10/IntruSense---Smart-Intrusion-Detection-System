from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PreprocessingValidationError(ValueError):
    """Raised when preprocessing input or artifacts violate their schema."""


class PrecomputedMedianImputer(BaseEstimator, TransformerMixin):
    """Serializable sklearn transformer with medians computed out of core."""

    def __init__(self, statistics: Sequence[float], feature_names: Sequence[str]):
        self.statistics = statistics
        self.feature_names = feature_names

    def fit(self, X: pd.DataFrame | np.ndarray, _y: object = None):
        _ = _y
        if isinstance(X, pd.DataFrame):
            names = tuple(str(name) for name in X.columns)
            if names != tuple(self.feature_names):
                raise PreprocessingValidationError(
                    "Imputer input feature names or order do not match its schema"
                )
            self.feature_names_in_ = np.asarray(names, dtype=object)
        elif X.shape[1] != len(self.feature_names):
            raise PreprocessingValidationError(
                "Imputer input feature count does not match its schema"
            )
        self.n_features_in_ = len(self.feature_names)
        self.statistics_ = np.asarray(self.statistics, dtype=np.float64)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64).copy()
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise PreprocessingValidationError(
                "Imputer input shape does not match its fitted feature schema"
            )
        if isinstance(X, pd.DataFrame):
            names = tuple(str(name) for name in X.columns)
            if names != tuple(self.feature_names):
                raise PreprocessingValidationError(
                    "Imputer input feature names or order do not match its schema"
                )
        missing_rows, missing_columns = np.where(np.isnan(values))
        values[missing_rows, missing_columns] = self.statistics_[missing_columns]
        return values
