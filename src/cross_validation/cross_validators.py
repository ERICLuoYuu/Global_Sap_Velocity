
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import numpy as np
from sklearn.model_selection import (
    KFold, 
    cross_val_score,
    GroupKFold,
    TimeSeriesSplit
)
import numpy as np
from sklearn.model_selection import (
    KFold, 
    # cross_val_score,
    GroupKFold,
    TimeSeriesSplit
)
from sklearn.base import BaseEstimator
import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List, Optional, Tuple, Callable
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import clone_model, Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit
import tensorflow as tf
from src.cross_validation.timeseries_split import GroupedTimeSeriesSplit


def cross_val_score(estimator, X, y, cv, groups, scoring='r2'):
    scores = []
    # cv is our GroupedTimeSeriesSplit instance
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        estimator.fit(X_train, y_train)
        
        # Get score
        if scoring == 'r2':
            score = estimator.score(X_test, y_test)
        scores.append(score)
    
    return np.array(scores)

class BaseCrossValidator(ABC):
    """
    Abstract base class for cross-validation strategies.
    
    Parameters
    ----------
    scoring : str, default='r2'
        Scoring metric to evaluate the predictions
    n_splits : int, default=5
        Number of folds for cross-validation
    random_state : int, default=42
        Random state for reproducibility
    """
    
    def __init__(
        self,
        scoring: str = 'r2',
        n_splits: int = 5,
        random_state: int = 42
    ):
        self.scoring = scoring
        self.n_splits = n_splits
        self.random_state = random_state
    
    @abstractmethod
    def get_scores(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> np.ndarray:
        """
        Abstract method to compute cross-validation scores.
        Must be implemented by subclasses.
        """
        pass

class MLCrossValidator(BaseCrossValidator):
    """
    Cross-validation implementation for scikit-learn models.
    
    Parameters
    ----------
    estimator : sklearn estimator object
        The estimator to use for modeling
    scoring : str, default='r2'
        Scoring metric to evaluate the predictions
    n_splits : int, default=5
        Number of folds for cross-validation
    random_state : int, default=42
        Random state for reproducibility
    """
    
    def __init__(
        self,
        estimator: BaseEstimator,
        scoring: str = 'r2',
        n_splits: int = 5,
        random_state: int = 42
    ):
        super().__init__(scoring, n_splits, random_state)
        self.estimator = estimator
    
    def random_cv(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        shuffle: bool = True
    ) -> np.ndarray:
        """
        Perform random K-fold cross-validation.
        """
        cv = KFold(
            n_splits=self.n_splits,
            shuffle=shuffle,
            random_state=self.random_state if shuffle else None
        )
        return self.get_scores(X, y, cv=cv)
    
    def spatial_cv(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        groups: Union[np.ndarray, List]
    ) -> np.ndarray:
        """
        Perform spatial cross-validation using location-based groups.
        """
        cv = GroupKFold(min(self.n_splits, len(np.unique(groups))))
        return self.get_scores(X, y, cv=cv, groups=groups)
    
    def temporal_cv(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        max_train_size: Optional[int] = None,
        groups: Union[np.ndarray, List, pd.Series] = None
    ) -> np.ndarray:
        """
        Perform temporal cross-validation using time series split.
        """
        cv = GroupedTimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=max_train_size
            
        )
        return self.get_scores(X, y, cv=cv, groups=groups)
    
    def spatial_stratified_cv(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        groups: Union[np.ndarray, List],
        strata: Union[np.ndarray, List]
    ) -> np.ndarray:
        """
        Perform spatial stratified cross-validation.
        """
        combined_groups = [f"{g}_{s}" for g, s in zip(groups, strata)]
        cv = GroupKFold(n_splits=self.n_splits)
        return self.get_scores(X, y, cv=cv, groups=combined_groups)
    
    def get_scores(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        cv: Union[KFold, GroupKFold, GroupedTimeSeriesSplit],
        groups: Optional[Union[np.ndarray, List, pd.Series]] = None,
    ) -> np.ndarray:
        """
        Compute cross-validation scores for ML models.
        """
        return cross_val_score(
            self.estimator,
            X,
            y,
            scoring=self.scoring,
            cv=cv,
            groups=groups
        )


class DLCrossValidator(BaseCrossValidator):
    """
    Cross-validation implementation for Keras deep learning models.
    
    Parameters
    ----------
    model : tensorflow.keras.Model
        The Keras model to use for modeling
    scoring : str or callable, default='val_loss'
        Metric to evaluate the predictions
    n_splits : int, default=5
        Number of folds for cross-validation
    random_state : int, default=42
        Random state for reproducibility
    batch_size : int, default=32
        Batch size for training
    epochs : int, default=100
        Maximum number of epochs for training
    patience : int, default=10
        Number of epochs with no improvement after which training will be stopped
    verbose : int, default=0
        Verbosity mode
    """
    
    def __init__(
        self,
        model: Model,
        scoring: Union[str, Callable] = 'val_loss',
        n_splits: int = 5,
        random_state: int = 42,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        verbose: int = 0
    ):
        super().__init__(scoring, n_splits, random_state)
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
    
    def _prepare_callbacks(self):
        """Prepare callbacks for model training."""
        return [
            EarlyStopping(
                monitor=self.scoring if isinstance(self.scoring, str) else 'val_loss',
                patience=self.patience,
                restore_best_weights=True
            )
        ]
    
    def _clone_and_compile_model(self):
        """Clone the model and compile it with the same configuration."""
        cloned_model = clone_model(self.model)
        cloned_model.compile(
            optimizer=self.model.optimizer,
            loss=self.model.loss,
            metrics=self.model.metrics
        )
        return cloned_model
    
    def random_cv(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        shuffle: bool = True
    ) -> Tuple[np.ndarray, List[Model]]:
        """
        Perform random K-fold cross-validation.
        
        Returns
        -------
        Tuple[np.ndarray, List[Model]]
            Cross-validation scores and trained models
        """
        cv = KFold(
            n_splits=self.n_splits,
            shuffle=shuffle,
            random_state=self.random_state if shuffle else None
        )
        return self.get_scores(X, y, cv=cv)
    
    def spatial_cv(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        groups: Union[np.ndarray, List]
    ) -> Tuple[np.ndarray, List[Model]]:
        """
        Perform spatial cross-validation using location-based groups.
        """
        cv = GroupKFold(min(self.n_splits, len(np.unique(groups))))
        return self.get_scores(X, y, cv=cv, groups=groups)
    
    def temporal_cv(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        max_train_size: Optional[int] = None,
        groups: Union[np.ndarray, List, pd.Series] = None
    ) -> Tuple[np.ndarray, List[Model]]:
        """
        Perform temporal cross-validation using time series split.
        """
        cv = GroupedTimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=max_train_size,
            
        )
        return self.get_scores(X, y, cv=cv, groups=groups)
    
    def spatial_stratified_cv(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        groups: Union[np.ndarray, List],
        strata: Union[np.ndarray, List]
    ) -> Tuple[np.ndarray, List[Model]]:
        """
        Perform spatial stratified cross-validation.
        """
        combined_groups = [f"{g}_{s}" for g, s in zip(groups, strata)]
        cv = GroupKFold(n_splits=self.n_splits)
        return self.get_scores(X, y, cv=cv, groups=combined_groups)
    
    def get_scores(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> Tuple[np.ndarray, List[Model]]:
        """
        Compute cross-validation scores for deep learning models.
        
        Returns
        -------
        Tuple[np.ndarray, List[Model]]
            Cross-validation scores and list of trained models
        """
        cv = kwargs.pop('cv')
        scores = []
        trained_models = []
        
        for train_idx, val_idx in cv.split(X, y, **kwargs):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone and compile the model for this fold
            model = self._clone_and_compile_model()
            
            # Train the model
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=self._prepare_callbacks(),
                verbose=self.verbose
            )
            
            # Get the score
            if isinstance(self.scoring, str):
                score = min(history.history[f'val_{self.scoring}'])
            else:
                score = self.scoring(y_val, model.predict(X_val))
            
            scores.append(score)
            trained_models.append(model)
        
        return np.array(scores), trained_models