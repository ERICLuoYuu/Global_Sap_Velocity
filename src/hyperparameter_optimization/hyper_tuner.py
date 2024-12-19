import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from src.cross_validation.timeseries_split import GroupedTimeSeriesSplit
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from abc import ABC, abstractmethod
from sklearn.model_selection import KFold, GroupKFold
from typing import Dict, Union, List, Optional, Tuple, Callable
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import itertools



class BaseOptimizer(ABC):
    """
    Abstract base class for hyperparameter optimization.
    
    Parameters
    ----------
    param_grid : dict
        Dictionary with parameters names as keys and lists of parameter settings
    scoring : str, default='r2'
        Scoring metric for optimization
    n_splits : int, default=5
        Number of folds for cross-validation
    random_state : int, default=42
        Random state for reproducibility
    n_jobs : int, default=-1
        Number of jobs to run in parallel
    verbose : int, default=1
        Controls the verbosity
    """
    
    def __init__(
        self,
        param_grid: Dict,
        scoring: str = 'r2',
        n_splits: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 1
    ):
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = None
    
    def _get_cv_splitter(
        self,
        split_type: str,
        **kwargs
    ):
        """Get the appropriate cross-validation splitter."""
        if split_type == 'spatial':
            return GroupKFold(n_splits=self.n_splits)
        elif split_type == 'temporal':
            return GroupedTimeSeriesSplit(
                n_splits=self.n_splits,
                max_train_size=kwargs.get('max_train_size', None),
                test_size=kwargs.get('test_size', None),
                gap=kwargs.get('gap', 0)
            )
        else:  # random
            return KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state
            )
    
    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        split_type: str = 'random',
        **kwargs
    ):
        """Abstract method to perform hyperparameter optimization."""
        pass

class MLOptimizer(BaseOptimizer):
    """
    Optimizer for scikit-learn compatible models.
    
    Parameters
    ----------
    model_type : str
        Type of model ('rf', 'xgb', or 'svm')
    task : str
        Type of task ('regression' or 'classification')
    param_grid : dict
        Dictionary with parameters names as keys and lists of parameter settings
    scoring : str, default=None
        Scoring metric for optimization
    n_splits : int, default=5
        Number of folds for cross-validation
    random_state : int, default=42
        Random state for reproducibility
    n_jobs : int, default=-1
        Number of jobs to run in parallel
    verbose : int, default=1
        Controls the verbosity
    """
    
    def __init__(
        self,
        model_type: str,
        task: str,
        param_grid: Dict,
        scoring: str = None,
        n_splits: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 1
    ):  
        if task.lower() not in ['regression', 'classification']:
            raise ValueError("Unknown task type. Choose 'regression' or 'classification'.")
        # Set default scoring based on task
        if scoring is None:
            scoring = 'r2' if task.lower() == 'regression' else 'accuracy'
            
        super().__init__(param_grid, scoring, n_splits, random_state, n_jobs, verbose)
        self.model_type = model_type.lower()
        self.task = task.lower()
        self.model = self._get_base_model()
    
    def _get_base_model(self):
        """Get the base model based on type and task."""
        models = {
            'rf': {
                'regression': RandomForestRegressor(random_state=self.random_state),
                'classification': RandomForestClassifier(random_state=self.random_state)
            },
            'xgb': {
                'regression': XGBRegressor(random_state=self.random_state),
                'classification': XGBClassifier(random_state=self.random_state)
            },
            'svm': {
                'regression': SVR(),
                'classification': SVC(probability=True)
            }
        }
        
        if self.model_type not in models:
            raise ValueError(f"Unknown model type: {self.model_type}")
        if self.task not in models[self.model_type]:
            raise ValueError(f"Unknown task: {self.task}")
        
        return models[self.model_type][self.task]
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        split_type: str = 'random',
        **kwargs
    ):
        """
        Perform grid search optimization.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        split_type : str, default='random'
            Type of cross-validation split ('random', 'spatial', or 'temporal')
        **kwargs : dict
            Additional arguments for cross-validation splitter
        
        Returns
        -------
        self : returns an instance of self
        """
        # Get appropriate CV splitter
        cv = self._get_cv_splitter(split_type, **kwargs)
        
        # Set up GridSearchCV
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        # Handle groups for spatial/temporal cross-validation
        if split_type in ['spatial', 'temporal']:
            if 'groups' not in kwargs:
                raise ValueError("groups parameter is required for spatial/temporal cross-validation")
            grid_search.fit(X, y, groups=kwargs['groups'])
        else:
            grid_search.fit(X, y)
        
        # Store results
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.best_estimator_ = grid_search.best_estimator_
        self.cv_results_ = grid_search.cv_results_
        
        return self

    def get_best_model(self):
        """Return the best model found during optimization."""
        if self.best_estimator_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.best_estimator_
    
import numpy as np
import pandas as pd
from typing import Dict, Union, List, Optional, Tuple, Callable
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import itertools

class DLOptimizer(BaseOptimizer):
    """
    Optimizer for deep learning models using Keras native grid search.
    
    Parameters
    ----------
    base_architecture : Callable
        Function that returns a Keras model architecture
    task : str
        Type of task ('regression' or 'classification')
    param_grid : dict
        Dictionary with parameters names as keys and lists of parameter settings.
        Supported parameter types:
        - 'architecture': model architecture parameters
        - 'optimizer': optimizer parameters
        - 'training': training parameters (batch_size, epochs, etc.)
    input_shape : tuple
        Shape of input data (excluding batch dimension)
    output_shape : int or tuple
        Shape of output (number of classes for classification)
    scoring : str, default=None
        Metric to monitor ('val_loss' by default)
    n_splits : int, default=5
        Number of folds for cross-validation
    random_state : int, default=42
        Random state for reproducibility
    verbose : int, default=1
        Controls the verbosity
    """
    
    def __init__(
        self,
        base_architecture: Callable,
        task: str,
        param_grid: Dict,
        input_shape: tuple,
        output_shape: Union[int, tuple],
        scoring: str = None,
        n_splits: int = 5,
        random_state: int = 42,
        verbose: int = 1
    ):
        # Set default scoring based on task
        if scoring is None:
            scoring = 'val_loss' if task.lower() == 'regression' else 'val_accuracy'
            
        super().__init__(param_grid, scoring, n_splits, random_state, 1, verbose)
        self.base_architecture = base_architecture
        self.task = task.lower()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.label_encoder = LabelEncoder() if task == 'classification' else None
        
        # Organize parameters by category
        self.architecture_params = param_grid.get('architecture', {})
        self.optimizer_params = param_grid.get('optimizer', {})
        self.training_params = param_grid.get('training', {})
    
    def _prepare_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
            
        # Handle classification targets
        if self.task == 'classification':
            y = self.label_encoder.fit_transform(y)
            y = keras.utils.to_categorical(y)
        
        return X, y
    
    def _create_model(self, architecture_params: Dict) -> keras.Model:
        """Create model with given architecture parameters."""
        model = self.base_architecture(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            **architecture_params
        )
        return model
    
    def _compile_model(
        self,
        model: keras.Model,
        optimizer_name: str,
        optimizer_params: Dict
    ) -> keras.Model:
        """Compile model with given optimizer parameters."""
        optimizer_class = getattr(keras.optimizers, optimizer_name)
        optimizer = optimizer_class(**optimizer_params)
        
        if self.task == 'regression':
            loss = 'mse'
            metrics = ['mae']
        else:
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        return model
    
    def _get_callbacks(self, fold: int) -> List[keras.callbacks.Callback]:
        """Get callbacks for training."""
        return [
            keras.callbacks.EarlyStopping(
                monitor=self.scoring,
                patience=self.training_params.get('patience', 10),
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                f'best_model_fold_{fold}.weights.h5',
                monitor=self.scoring,
                save_best_only=True,
                save_weights_only=True
            )
        ]
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        split_type: str = 'random',
        **kwargs
    ):
        """
        Perform grid search optimization using Keras native methods.
        
        Parameters
        ----------
        X : array-like
            Training data
        y : array-like
            Target values
        split_type : str, default='random'
            Type of cross-validation split ('random', 'spatial', or 'temporal')
        **kwargs : dict
            Additional arguments for cross-validation splitter
        
        Returns
        -------
        self : returns an instance of self
        """
        X, y = self._prepare_data(X, y)
        cv = self._get_cv_splitter(split_type, **kwargs)
        
        # Generate all parameter combinations
        param_combinations = []
        arch_keys, arch_values = zip(*self.architecture_params.items())
        opt_name = self.optimizer_params.get('name', ['adam'])[0]
        opt_params_keys, opt_params_values = zip(*{k: v for k, v in 
                                                 self.optimizer_params.items() 
                                                 if k != 'name'}.items())
        
        for arch_combo in itertools.product(*arch_values):
            arch_params = dict(zip(arch_keys, arch_combo))
            for opt_combo in itertools.product(*opt_params_values):
                opt_params = dict(zip(opt_params_keys, opt_combo))
                param_combinations.append((arch_params, opt_params))
        
        # Track results
        all_results = []
        best_score = float('inf') if 'loss' in self.scoring else float('-inf')
        best_params = None
        best_model = None
        
        # Perform grid search
        for arch_params, opt_params in param_combinations:
            fold_scores = []
            
            # Cross-validation loop
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, 
                                                               groups=kwargs.get('groups'))):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                # Convert to tensors with explicit types
                X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
                y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
                X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
                y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
                # Print shapes for debugging
                print(f"X_train shape: {X_train.shape}")
                print(f"y_train shape: {y_train.shape}")
                # Create and compile model
                model = self._create_model(arch_params)
                model = self._compile_model(model, opt_name, opt_params)
                
                # Train model
                history = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    batch_size=self.training_params.get('batch_size', 32),
                    epochs=self.training_params.get('epochs', 100),
                    callbacks=self._get_callbacks(fold),
                    verbose=self.verbose
                )
                
                # Get best score for this fold
                fold_score = min(history.history[self.scoring]) if 'loss' in self.scoring \
                    else max(history.history[self.scoring])
                fold_scores.append(fold_score)
            
            # Calculate mean score across folds
            mean_score = np.mean(fold_scores)
            all_results.append({
                'params': {'architecture': arch_params, 'optimizer': opt_params},
                'mean_score': mean_score,
                'std_score': np.std(fold_scores)
            })
            
            # Update best results
            is_better = (mean_score < best_score if 'loss' in self.scoring 
                        else mean_score > best_score)
            if is_better:
                best_score = mean_score
                best_params = {'architecture': arch_params, 'optimizer': opt_params}
                
                # Create best model
                best_model = self._create_model(arch_params)
                best_model = self._compile_model(best_model, opt_name, opt_params)
        
        # Store results
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.best_estimator_ = best_model
        self.cv_results_ = pd.DataFrame(all_results)
        
        return self

    def get_best_model(self) -> keras.Model:
        """Return the best model found during optimization."""
        if self.best_estimator_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.best_estimator_