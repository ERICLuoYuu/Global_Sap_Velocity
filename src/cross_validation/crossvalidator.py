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
from sklearn.base import BaseEstimator
import pandas as pd
from typing import Union, List, Optional, Tuple
from abc import ABC, abstractmethod
from typing import Union, List, Optional, Tuple, Callable
from tensorflow import keras
from tensorflow.keras.models import Model
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score, KFold, GroupKFold, TimeSeriesSplit
from src.cross_validation.timeseries_split import GroupedTimeSeriesSplit
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
        return cross_val_score(
            self.estimator,
            X,
            y,
            scoring=self.scoring,
            cv=cv
        )
    
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
        return cross_val_score(
            self.estimator,
            X,
            y,
            scoring=self.scoring,
            cv=cv,
            groups=groups
        )
    
    def temporal_cv(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        max_train_size: Optional[int] = None,
        groups: Union[np.ndarray, List] = None
    ) -> np.ndarray:
        """
        Perform temporal cross-validation using time series split.
        """
        cv = GroupedTimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=max_train_size,
            groups=groups
        )
        return cross_val_score(
            self.estimator,
            X,
            y,
            scoring=self.scoring,
            cv=cv
        )
    
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
        return cross_val_score(
            self.estimator,
            X,
            y,
            scoring=self.scoring,
            cv=cv,
            groups=combined_groups
        )
    
    def get_scores(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> np.ndarray:
        """
        Compute cross-validation scores for ML models.
        """
        return cross_val_score(
            self.estimator,
            X,
            y,
            scoring=self.scoring,
            **kwargs
        )
    


class DLCrossValidator(BaseCrossValidator):
    """
    Cross-validation implementation for Keras deep learning models using scikeras wrapper.
    
    Parameters
    ----------
    model : tensorflow.keras.Model
        The Keras model to use for modeling
    task : str, default='regression'
        Type of task, either 'regression' or 'classification'
    scoring : str, default='r2' for regression, 'accuracy' for classification
        Scoring metric to evaluate the predictions
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
        task: str = 'regression',
        scoring: str = None,
        n_splits: int = 5,
        random_state: int = 42,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        verbose: int = 0
    ):
        # Set default scoring based on task if not provided
        if scoring is None:
            scoring = 'r2' if task == 'regression' else 'accuracy'
            
        super().__init__(scoring, n_splits, random_state)
        self.model = model
        self.task = task
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.wrapped_model = self._wrap_model()
    
    def _create_model(self):
        """Create a clone of the model with same architecture and compile settings."""
        return keras.models.clone_model(self.model)

    def _wrap_model(self):
        """Wrap the Keras model with scikeras wrapper."""
        wrapper_class = KerasRegressor if self.task == 'regression' else KerasClassifier
        
        wrapped = wrapper_class(
            model=self._create_model,
            optimizer=self.model.optimizer.get_config(),
            loss=self.model.loss,
            metrics=self.model.metrics,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True
                )
            ]
        )
        return wrapped
    
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
        return cross_val_score(
            self.wrapped_model,
            X,
            y,
            scoring=self.scoring,
            cv=cv
        )
    
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
        return cross_val_score(
            self.wrapped_model,
            X,
            y,
            scoring=self.scoring,
            cv=cv,
            groups=groups
        )
    
    def temporal_cv(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        max_train_size: Optional[int] = None,
        groups: Union[np.ndarray, List] = None
    ) -> np.ndarray:
        """
        Perform temporal cross-validation using time series split.
        """
        cv = GroupedTimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=max_train_size,
            groups=groups
        )
        return cross_val_score(
            self.wrapped_model,
            X,
            y,
            scoring=self.scoring,
            cv=cv
        )
    
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
        return cross_val_score(
            self.wrapped_model,
            X,
            y,
            scoring=self.scoring,
            cv=cv,
            groups=combined_groups
        )
    
    def get_scores(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> np.ndarray:
        """
        Compute cross-validation scores for deep learning models.
        """
        return cross_val_score(
            self.wrapped_model,
            X,
            y,
            scoring=self.scoring,
            **kwargs
        )