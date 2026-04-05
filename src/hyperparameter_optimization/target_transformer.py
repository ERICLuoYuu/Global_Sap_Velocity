"""
TargetTransformer: handles various target variable transformations for regression.

Extracted from test_hyperparameter_tuning_ML_spatial_stratified.py to enable
reuse across ML training, prediction, and standalone SHAP analysis scripts.
"""

import logging

import numpy as np
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer


class TargetTransformer:
    """
    A class to handle various target variable transformations for regression.

    Supported methods:
    - 'log1p': log(1 + x) transformation, inverse: exp(x) - 1
    - 'sqrt': square root transformation, inverse: x^2
    - 'box-cox': Box-Cox transformation (requires positive values), inverse uses fitted lambda
    - 'yeo-johnson': Yeo-Johnson transformation (works with negative values), uses sklearn
    - 'none': no transformation

    Usage:
        transformer = TargetTransformer(method='log1p')
        y_transformed = transformer.fit_transform(y_train)
        y_test_transformed = transformer.transform(y_test)
        y_pred_original = transformer.inverse_transform(y_pred_transformed)
    """

    VALID_METHODS = ["log1p", "sqrt", "box-cox", "yeo-johnson", "none"]

    def __init__(self, method: str = "log1p"):
        """
        Initialize the transformer.

        Args:
            method: Transformation method. One of 'log1p', 'sqrt', 'box-cox', 'yeo-johnson', 'none'
        """
        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method '{method}'. Must be one of {self.VALID_METHODS}")

        self.method = method
        self._fitted = False
        self._lambda = None  # For Box-Cox
        self._sklearn_transformer = None  # For Yeo-Johnson
        self._shift = 0  # Shift for handling zeros/negatives

    def fit(self, y: np.ndarray) -> "TargetTransformer":
        """
        Fit the transformer to the data.

        Args:
            y: Target values to fit on

        Returns:
            self
        """
        y = np.asarray(y).flatten()

        if self.method == "box-cox":
            # Box-Cox requires strictly positive values
            if np.any(y <= 0):
                self._shift = np.abs(y.min()) + 1e-6
                logging.info(f"Box-Cox: Shifting data by {self._shift:.4f} to ensure positive values")
            else:
                self._shift = 0

            y_shifted = y + self._shift
            _, self._lambda = boxcox(y_shifted)
            logging.info(f"Box-Cox: Fitted lambda = {self._lambda:.4f}")

        elif self.method == "yeo-johnson":
            self._sklearn_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
            self._sklearn_transformer.fit(y.reshape(-1, 1))
            logging.info(f"Yeo-Johnson: Fitted lambda = {self._sklearn_transformer.lambdas_[0]:.4f}")

        elif self.method == "sqrt":
            # Check for negative values
            if np.any(y < 0):
                self._shift = np.abs(y.min()) + 1e-6
                logging.info(f"Sqrt: Shifting data by {self._shift:.4f} to ensure non-negative values")
            else:
                self._shift = 0

        self._fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform the target values.

        Args:
            y: Target values to transform

        Returns:
            Transformed values
        """
        y = np.asarray(y).flatten()

        if self.method == "none":
            return y

        if self.method == "log1p":
            # Handle negative values by using signed log
            return np.log1p(np.maximum(y, 0))

        elif self.method == "sqrt":
            y_shifted = y + self._shift
            return np.sqrt(np.maximum(y_shifted, 0))

        elif self.method == "box-cox":
            if not self._fitted:
                raise RuntimeError("Transformer must be fitted before transform for Box-Cox")
            y_shifted = y + self._shift
            # Use the fitted lambda
            return boxcox(np.maximum(y_shifted, 1e-10), lmbda=self._lambda)

        elif self.method == "yeo-johnson":
            if not self._fitted or self._sklearn_transformer is None:
                raise RuntimeError("Transformer must be fitted before transform for Yeo-Johnson")
            return self._sklearn_transformer.transform(y.reshape(-1, 1)).flatten()

        return y

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            y: Target values to fit and transform

        Returns:
            Transformed values
        """
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse transform the values back to original scale.

        Args:
            y_transformed: Transformed values

        Returns:
            Values in original scale
        """
        y_transformed = np.asarray(y_transformed).flatten()

        if self.method == "none":
            return y_transformed

        elif self.method == "log1p":
            return np.expm1(y_transformed)

        elif self.method == "sqrt":
            y_squared = np.square(y_transformed)
            return y_squared - self._shift

        elif self.method == "box-cox":
            if self._lambda is None:
                raise RuntimeError("Transformer must be fitted before inverse_transform for Box-Cox")
            y_original = inv_boxcox(y_transformed, self._lambda)
            return y_original - self._shift

        elif self.method == "yeo-johnson":
            if self._sklearn_transformer is None:
                raise RuntimeError("Transformer must be fitted before inverse_transform for Yeo-Johnson")
            return self._sklearn_transformer.inverse_transform(y_transformed.reshape(-1, 1)).flatten()

        return y_transformed

    def get_params(self) -> dict:
        """Get transformer parameters for saving."""
        return {
            "method": self.method,
            "fitted": self._fitted,
            "lambda": self._lambda,
            "shift": self._shift,
        }

    @classmethod
    def from_params(cls, params: dict) -> "TargetTransformer":
        """Reconstruct transformer from saved parameters."""
        transformer = cls(method=params["method"])
        transformer._fitted = params["fitted"]
        transformer._lambda = params["lambda"]
        transformer._shift = params["shift"]
        return transformer

    def __repr__(self):
        return f"TargetTransformer(method='{self.method}', fitted={self._fitted})"
