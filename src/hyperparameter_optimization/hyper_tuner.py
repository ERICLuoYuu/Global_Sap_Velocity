import sys
from pathlib import Path
import os
from datetime import datetime
import json
import socket
import subprocess
import logging
# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from path_config import PathConfig, get_default_paths
from src.cross_validation.timeseries_split import GroupedTimeSeriesSplit
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid, ParameterSampler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from xgboost.sklearn import XGBRegressor, XGBClassifier
from abc import ABC, abstractmethod
from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit, PredefinedSplit, StratifiedGroupKFold
from typing import Dict, Union, List, Optional, Tuple, Callable
import tensorflow as tf
import os
from sklearn.metrics import get_scorer
from sklearn.base import BaseEstimator, RegressorMixin
# Set threading parameters before any other TensorFlow operations
#cpu_num = 10  # or whatever number you want to use
#tf.config.threading.set_intra_op_parallelism_threads(cpu_num)
#tf.config.threading.set_inter_op_parallelism_threads(cpu_num)
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import itertools
import warnings
# Set environment variables for determinism BEFORE importing TensorFlow
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = '42'
# Force TensorFlow to use deterministic algorithms (TF 2.6+)
try:
    tf.config.experimental.enable_op_determinism()
except:
    # Fallback for older TF versions
    logging.warning("Warning: Your TensorFlow version doesn't support enable_op_determinism()")
    # Try alternative approaches
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
# Limit TensorFlow to use only one thread for CPU operations
n_physical_cores = os.cpu_count() // 2  # Estimate of physical cores
#n_physical_cores = 1
tf.config.threading.set_inter_op_parallelism_threads(n_physical_cores)
tf.config.threading.set_intra_op_parallelism_threads(n_physical_cores)




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
        verbose: int = 1,
        search_type: str = 'grid',
        paths: Optional[PathConfig] = None
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
        self.search_type = search_type.lower()
        self.paths = paths if paths is not None else get_default_paths()
    
    def _get_cv_splitter(
        self,
        split_type: str,
        **kwargs
    ):
        """Get the appropriate cross-validation splitter."""
        if split_type == 'spatial':
            return GroupKFold(n_splits=self.n_splits)
        elif split_type == 'spatial_stratified':
            # Add validation for StratifiedGroupKFold
            n_splits = self.n_splits
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        elif split_type == 'temporal':
            return TimeSeriesSplit(
                n_splits=self.n_splits,
                max_train_size=kwargs.get('max_train_size', None),
                test_size=kwargs.get('test_size', None),
                gap=kwargs.get('gap', 0)
            )
        else:  # random
            is_shuffle = kwargs.get('is_shuffle', False)
            return KFold(
                n_splits=self.n_splits,
                shuffle=is_shuffle,
                random_state=self.random_state if is_shuffle else None
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


class MLOptimizer(BaseOptimizer, BaseEstimator, RegressorMixin):
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
    save_model_dir : str, default=None
        Directory to save the best model
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
        verbose: int = 1,
        save_model_dir: Optional[str] = None,
        search_type: str = 'grid'
    ):  
        if task.lower() not in ['regression', 'classification']:
            raise ValueError("Unknown task type. Choose 'regression' or 'classification'.")
        # Set default scoring based on task
        if scoring is None:
            scoring = 'r2' if task.lower() == 'regression' else 'accuracy'
            
        super().__init__(param_grid, scoring, n_splits, random_state, n_jobs, verbose, search_type)
        self.logger = logging.getLogger(f'{model_type}_optimizer')
        self.model_type = model_type.lower()
        self.task = task.lower()
        self.model = self._get_base_model()
        if save_model_dir is not None:
            self.save_model_dir = Path(save_model_dir + f'/{model_type}_{task}')
            if not self.save_model_dir.exists():
                self.save_model_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_model_dir = self.paths.models_root / f'{model_type}_{task}'
            if not self.save_model_dir.exists():
                self.save_model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized MLOptimizer with {model_type} model for {task} task")

    
    def _get_base_model(self):
        """Get the base model based on type and task."""
        models = {
            'rf': {
                'regression': RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs),
                'classification': RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
            },
            'xgb': {
            'regression': XGBRegressor(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                objective='reg:squarederror',
                eval_metric='rmse', 
                enable_categorical=True,
                tree_method='hist',  # Added for better compatibility
                early_stopping_rounds=50,  
            ),
            'classification': XGBClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                enable_categorical=True,
                tree_method='hist'
            )
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
    
    def _manual_search(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            cv_splitter,
            n_iterations: int = None,
            is_refit: bool = True,
            **kwargs
        ):
            """
            Performs a manual grid/random search, correctly handling the 'best_estimator_'
            attribute based on the 'is_refit' flag.
            """
            self.logger.info(f"Starting manual {self.search_type} search for {self.model_type} model.")
            self.logger.info(f"Parameter grid: {self.param_grid}")

            scorer = get_scorer(self.scoring)
            if self.search_type == 'random':
                if n_iterations is None:
                    raise ValueError("'n_iterations' must be set for random search.")
                param_iterable = ParameterSampler(
                    self.param_grid, n_iter=n_iterations, random_state=self.random_state
                )
            else:
                param_iterable = ParameterGrid(self.param_grid)

            cv_results = {
            'params': [], 'mean_test_score': [], 'std_test_score': [], 'all_fold_scores': [],
            'param_best_score': [], 'best_model_from_param': []
        }
            if self.model_type == 'xgb':
                cv_results['all_best_iterations'] = []
                        # Convert to numpy for robust indexing, in case they are pandas objects
            if isinstance(X, pd.DataFrame):
                X_np = X.values
            else:
                X_np = X # It's already a numpy array

            if isinstance(y, pd.Series):
                y_np = y.values
            else:
                y_np = y # It's already a numpy array

            # Outer Loop: Hyperparameters
            for params in param_iterable:
                self.logger.debug(f"Evaluating parameters: {params}")
                fold_scores = []
                fold_best_iterations = []
                param_best_score = -np.inf
                best_model_from_param = None
                y_stratify = kwargs.get('y_stratify', None)
                if y_stratify is None:
                    self.logger.debug("No stratification provided, using target variable for stratification.")
                    # If no stratification is provided, use the target variable directly
                    y_stratify = y_np
                # Inner Loop: CV Folds
                for train_idx, val_idx in cv_splitter.split(X, y_stratify, groups=kwargs.get('groups')):
                    
                    # Use the NumPy arrays with the generated indices for slicing.
                    # This is robust and avoids pandas indexing issues.
                    X_train_fold, y_train_fold = X_np[train_idx], y_np[train_idx]
                    X_val_fold, y_val_fold = X_np[val_idx], y_np[val_idx]

                    model = self._get_base_model()
                    model.set_params(**params)
                    
                    # Model-Specific Fitting
                    if self.model_type == 'xgb':
                        model.fit(
                            X_train_fold, y_train_fold,
                            eval_set=[(X_val_fold, y_val_fold)],
                            verbose=False
                        )
                        # access the best iteration if uses early stopping
                        try:
                            fold_best_iterations.append(model.best_iteration)
                            self.logger.debug(f"Best iteration for current fold: {model.best_iteration}")
                        except AttributeError:
                            self.logger.debug("No best iteration found for current fold.")
                            fold_best_iterations.append(None)
                    else:
                        model.fit(X_train_fold, y_train_fold)

                    score = scorer(model, X_val_fold, y_val_fold)
                    fold_scores.append(score)

                    # Check if this is the best single model seen so far ---
                    if score > param_best_score:
                        param_best_score = score
                        best_model_from_param = model
                        self.logger.debug(f"New best single model found. Score: {param_best_score:.4f} with params: {params}")


                # Aggregate results for this parameter set
                cv_results['params'].append(params)
                cv_results['mean_test_score'].append(np.mean(fold_scores))
                cv_results['std_test_score'].append(np.std(fold_scores))
                cv_results['all_fold_scores'].append(fold_scores)
                cv_results['param_best_score'].append(param_best_score)
                cv_results['best_model_from_param'].append(best_model_from_param)
                if self.model_type == 'xgb':
                    cv_results['all_best_iterations'].append(fold_best_iterations)

            # --- Find Best Parameters (based on AVERAGE score) ---
            best_idx = np.argmax(cv_results['mean_test_score'])
            self.best_params_ = cv_results['params'][best_idx]
            self.best_score_ = cv_results['mean_test_score'][best_idx]
            self.cv_results_ = cv_results
            
            self.logger.info(f"Search complete. Best average score: {self.best_score_:.4f}")
            self.logger.info(f"Best parameters (based on average score): {self.best_params_}")

            # --- UPDATED: Conditional Refitting and Assignment Block ---
            if is_refit:
                # --- REFIT LOGIC ---
                # Create a NEW model and train it on ALL data.
                final_params = self.best_params_.copy()
                if self.model_type == 'xgb':
                    best_iterations = cv_results['all_best_iterations'][best_idx]
                    # Only process if we actually have valid iterations
                    valid_iterations = [iter for iter in best_iterations if iter is not None]
                    optimal_n_estimators = None
                    if valid_iterations and len(valid_iterations) > 0:
                        optimal_n_estimators = int(round(np.median(valid_iterations)))
                        final_params['n_estimators'] = optimal_n_estimators
                        self.logger.info(f"Using median of {len(valid_iterations)} valid iterations: {optimal_n_estimators}")
                    else:
                        # Keep the original n_estimators from best_params
                        self.logger.info("No early stopping iterations found, keeping original n_estimators")
                    final_params['n_estimators'] = optimal_n_estimators if optimal_n_estimators is not None else final_params.get('n_estimators', 1000)
                    self.logger.info(f"Median best iterations: {best_iterations} -> Refitting with {final_params['n_estimators']} estimators.")
                    final_params['early_stopping_rounds'] = None
                
                self.logger.info("Refitting model with best parameters on the full dataset.")
                self.best_estimator_ = self._get_base_model()
                self.best_estimator_.set_params(**final_params)
                self.best_estimator_.fit(X, y, verbose=False)
            else:
                # --- NO REFIT LOGIC ---
                # Assign the best model saved from the folds directly.
                self.logger.info(
                    "Refitting is disabled. 'best_estimator_' is the single best model found in one of the CV folds."
                )
                self.best_estimator_ = cv_results['best_model_from_param'][best_idx]
                # Note: The score of `best_estimator_` is `param_best_score`, which may
                # be different from `self.best_score_` (the best average).

            return self


    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        is_cv: bool = True,
        is_refit: bool = True,
        split_type: str = 'random',
        n_iterations: int = None,
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs
    ):
        """
        Performs hyperparameter optimization by dispatching to a manual search loop.
        """
        try:
            # Ensure data is in pandas format for consistent .iloc slicing
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            if not isinstance(y, pd.Series):
                y = pd.Series(y, name='target')

            if is_cv:
                # --- Standard Cross-Validation ---
                cv_splitter = self._get_cv_splitter(split_type, **kwargs)
                self._manual_search(X, y, cv_splitter, n_iterations, is_refit, **kwargs)

            else:
                # --- Fixed Validation Set (Non-CV) ---
                self.logger.info(f"Optimizing with a fixed validation set.")
                if X_val is None or y_val is None:
                    self.logger.warning("Using fixed validation set splitted from training data.")
                    cv_splitter = self._get_cv_splitter(split_type, **kwargs)
                    y_stratify = kwargs.get('y_stratify', None)
                    # Split the data into training and validation sets
                    train_idx, val_idx = next(cv_splitter.split(X, y_stratify, groups=kwargs.get('groups')))
                    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
                    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
                    # Now, assign them to the variables you use later.
                    X = X_train_fold
                    y = y_train_fold
                    X_val = X_val_fold
                    y_val = y_val_fold
                else:
                    self.logger.info("Using fixed validation set provided by user.")
                    # Ensure validation data is also in pandas format
                    if not isinstance(X_val, pd.DataFrame):
                        X_val = pd.DataFrame(X_val)
                    if not isinstance(y_val, pd.Series):
                        y_val = pd.Series(y_val, name='target')

                # For PredefinedSplit, we combine data and provide an index
                X_combined = pd.concat([X, X_val], ignore_index=True)
                y_combined = pd.concat([y, y_val], ignore_index=True)
                
                # -1 for training, 0 for validation
                split_index = np.concatenate([np.full(len(X), -1, dtype=int), np.full(len(X_val), 0, dtype=int)])
                pds = PredefinedSplit(test_fold=split_index)
                
                # Pass the combined data and the PredefinedSplitter to the manual search
                self._manual_search(X_combined, y_combined, pds, n_iterations, is_refit, **kwargs)
            # self.save_best_model(path=self.save_model_dir)
                
            return self

        except Exception as e:
            self.logger.error(f"An error occurred during fitting: {e}", exc_info=True)
            raise

    def get_best_model(self):
        """Return the best model found during optimization."""
        if self.best_estimator_ is None:
            self.logger.error("Model has not been fitted yet. Call fit() first.")
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        return self.best_estimator_
    def save_best_model(self, path=None, filename=None, save_method='joblib'):
        """
        Save the best model found during optimization.
        
        Parameters
        ----------
        path : str, default=None
            Directory to save the model. If None, uses the instance's save_model_dir.
        filename : str, default=None
            Filename for the saved model. If None, generates a name based on model type, task and timestamp.
        save_method : str, default='joblib'
            Method to use for serializing the model. Options are 'joblib', 'pickle', or 'cloudpickle'.
            
        Returns
        -------
        str : Path to the saved model file
        """
        from datetime import datetime
        
        if self.best_estimator_ is None:
            self.logger.error("No best model to save. Run fit() first.")
            raise ValueError("No best model to save. Run fit() first.")
        
        # Set default path if not provided
        if path is None:
            path = self.save_model_dir
        else:
            path = Path(path)
        
        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        
        # Set file extension based on save method
        if save_method.lower() == 'joblib':
            extension = '.joblib'
        else:
            extension = '.pkl'
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_type}_{self.task}_model_{timestamp}{extension}"
        elif not (filename.endswith('.joblib') or filename.endswith('.pkl')):
            filename += extension
        
        # Full path to save the model
        model_path = path / filename
        
        # Save the model with the selected method
        self.logger.info(f"Saving best {self.model_type} model for {self.task} task to {model_path} using {save_method}")
        
        if save_method.lower() == 'joblib':
            import joblib
            joblib.dump(self.best_estimator_, model_path)
        elif save_method.lower() == 'cloudpickle':
            import cloudpickle
            with open(model_path, "wb") as f:
                cloudpickle.dump(self.best_estimator_, f, protocol=5)
        else:  # default to pickle
            from pickle import dump
            with open(model_path, "wb") as f:
                dump(self.best_estimator_, f, protocol=5)
        
        return str(model_path)

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
        model_type: str,
        base_architecture: Callable,
        task: str,
        param_grid: Dict,
        input_shape: tuple,
        output_shape: Union[int, tuple],
        scoring: str = None,
        n_splits: int = 5,
        random_state: int = 42,
        verbose: int = 1,
        back_up_dir: Optional[str] = None,
        save_model_dir: Optional[str] = None,
        search_type: str = 'grid'
    ):
        # Set default scoring based on task
        if scoring is None:
            scoring = 'val_loss' if task.lower() == 'regression' else 'val_accuracy'
            
        super().__init__(param_grid, scoring, n_splits, random_state, 1, verbose, search_type)
        self.logger = logging.getLogger(f'{model_type}_optimizer')
        self.base_architecture = base_architecture
        self.task = task.lower()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.label_encoder = LabelEncoder() if task == 'classification' else None
        self.best_model_weights = None
        
        if back_up_dir is not None:
            self.back_up_dir = Path(back_up_dir)
        else:
            self.back_up_dir = self.paths.hyper_tuning_tmp_dir / f'{model_type}_{task}'
        if not self.back_up_dir.exists():
            self.back_up_dir.mkdir(parents=True, exist_ok=True)
        
        if save_model_dir is not None:
            self.save_model_dir = Path(save_model_dir + f'/{model_type}_{task}')
            if not self.save_model_dir.exists():
                self.save_model_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_model_dir = self.paths.models_root / f'{model_type}_{task}'
            if not self.save_model_dir.exists():
                self.save_model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized DLOptimizer for {task} task")
        
        
        # Organize parameters by category
        self.architecture_params = param_grid.get('architecture', {})
        self.optimizer_params = param_grid.get('optimizer', {})
        self.training_params = param_grid.get('training', {})
        
        # Log parameter grid size
        arch_combinations = 1
        for v in self.architecture_params.values():
            if hasattr(v, '__len__'):
                arch_combinations *= len(v)
                
        opt_combinations = 1
        for k, v in self.optimizer_params.items():
            if k != 'name' and hasattr(v, '__len__'):
                opt_combinations *= len(v)
        
        # Count optimizer name combinations if it's a list
        opt_name_combinations = 1
        if 'name' in self.optimizer_params and hasattr(self.optimizer_params['name'], '__len__'):
            opt_name_combinations = len(self.optimizer_params['name'])
        
        # Count training param combinations
        train_combinations = 1
        for v in self.training_params.values():
            if hasattr(v, '__len__'):
                train_combinations *= len(v)
        
        total_combinations = arch_combinations * opt_combinations * opt_name_combinations * train_combinations
        self.logger.info(f"Parameter grid has {total_combinations} combinations")
    
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
            self.logger.info(f"Encoded classification targets with {len(self.label_encoder.classes_)} classes")
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
            loss = 'mean_squared_error'
            metrics = ['mean_absolute_error']
        else:
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        return model

    def _get_callbacks(self, is_refit: bool, fold: int, training_params: Dict) -> List[keras.callbacks.Callback]:
        """Get callbacks for training with specific training parameters."""
        if is_refit:
            callbacks = [
            keras.callbacks.ModelCheckpoint(
                self.back_up_dir/f'temp_best_model_refit.weights.h5',
                monitor=self.scoring,
                save_best_only=True,
                save_weights_only=True,
                mode='min' if 'loss' in self.scoring else 'max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=5,
                factor=0.5,
                min_lr=1e-6
            )
        ]  
        else:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor=self.scoring,
                    patience=training_params.get('patience', 10),
                    restore_best_weights=True
                ),
                keras.callbacks.ModelCheckpoint(
                    self.back_up_dir/f'temp_best_model_fold_{fold}.weights.h5',
                    monitor=self.scoring,
                    save_best_only=True,
                    save_weights_only=True,
                    mode='min' if 'loss' in self.scoring else 'max'
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    patience=5,
                    factor=0.5,
                    min_lr=1e-6
                )
            ]  
        return callbacks

    def _get_param_combinations(self):
        """Generate all parameter combinations for grid search."""
        param_combinations = []
        
        # Get keys and values for architecture parameters
        arch_keys, arch_values = zip(*self.architecture_params.items()) if self.architecture_params else ([], [])
        
        # Get optimizer name(s)
        opt_names = self.optimizer_params.get('name', ['adam'])
        if not isinstance(opt_names, list):
            opt_names = [opt_names]
            
        # Get remaining optimizer parameters
        opt_params = {k: v for k, v in self.optimizer_params.items() if k != 'name'}
        if opt_params:
            opt_keys, opt_values = zip(*opt_params.items())
        else:
            opt_keys, opt_values = [], []
            
        # Get training parameters
        train_params = {}
        train_keys = []
        train_values = []
        for k, v in self.training_params.items():
            if isinstance(v, list):
                train_keys.append(k)
                train_values.append(v)
            else:
                train_params[k] = v
                
        # Generate all combinations
        # Empty architecture case
        if not arch_keys:
            # Generate all optimizer name combinations
            for opt_name in opt_names:
                # Generate all optimizer parameter combinations
                if opt_params:
                    for opt_combo in itertools.product(*opt_values):
                        curr_opt_params = dict(zip(opt_keys, opt_combo))
                        
                        # Generate all training parameter combinations
                        if train_keys:
                            for train_combo in itertools.product(*train_values):
                                curr_train_params = dict(zip(train_keys, train_combo))
                                # Merge with fixed training parameters
                                curr_train_params.update(train_params)
                                param_combinations.append(({}, opt_name, curr_opt_params, curr_train_params))
                        else:
                            param_combinations.append(({}, opt_name, curr_opt_params, train_params))
                else:
                    # No optimizer params to grid search
                    if train_keys:
                        for train_combo in itertools.product(*train_values):
                            curr_train_params = dict(zip(train_keys, train_combo))
                            curr_train_params.update(train_params)
                            param_combinations.append(({}, opt_name, {}, curr_train_params))
                    else:
                        param_combinations.append(({}, opt_name, {}, train_params))
        else:
            # Architecture parameters exist
            for arch_combo in itertools.product(*arch_values):
                arch_params = dict(zip(arch_keys, arch_combo))
                
                # Generate all optimizer name combinations
                for opt_name in opt_names:
                    # Generate all optimizer parameter combinations
                    if opt_params:
                        for opt_combo in itertools.product(*opt_values):
                            curr_opt_params = dict(zip(opt_keys, opt_combo))
                            
                            # Generate all training parameter combinations
                            if train_keys:
                                for train_combo in itertools.product(*train_values):
                                    curr_train_params = dict(zip(train_keys, train_combo))
                                    # Merge with fixed training parameters
                                    curr_train_params.update(train_params)
                                    param_combinations.append((arch_params, opt_name, curr_opt_params, curr_train_params))
                            else:
                                param_combinations.append((arch_params, opt_name, curr_opt_params, train_params))
                    else:
                        # No optimizer params to grid search
                        if train_keys:
                            for train_combo in itertools.product(*train_values):
                                curr_train_params = dict(zip(train_keys, train_combo))
                                curr_train_params.update(train_params)
                                param_combinations.append((arch_params, opt_name, {}, curr_train_params))
                        else:
                            param_combinations.append((arch_params, opt_name, {}, train_params))
                
        return param_combinations
    
    
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        split_type: str = 'random',
        is_cv: bool = True,
        is_refit: bool = True,
        n_iterations: Optional[int] = None,
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
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
        is_cv : bool, default=True
            Whether to use cross-validation or direct validation set
        X_val : array-like, optional
            Validation data (required if is_cv=False)
        y_val : array-like, optional
            Validation target values (required if is_cv=False)
        **kwargs : dict
            Additional arguments for cross-validation splitter
        
        Returns
        -------
        self : returns an instance of self
        """
        try:
            if not is_cv and (X_val is None or y_val is None):
                raise ValueError("X_val and y_val must be provided when is_cv=False")
                
            self.logger.info(f"Starting hyperparameter {'search with ' + split_type + ' split type' if is_cv else 'optimization with direct validation set'}")
            self.logger.info(f"Data shapes: X {X.shape}, y {y.shape}")
            if not is_cv:
                self.logger.info(f"Validation data shapes: X_val {X_val.shape}, y_val {y_val.shape}")
            
            X, y = self._prepare_data(X, y)
            if not is_cv:
                X_val, y_val = self._prepare_data(X_val, y_val)
            else:
                cv = self._get_cv_splitter(split_type, **kwargs)
            
            param_combinations = self._get_param_combinations()
            if self.search_type == 'random' and n_iterations is not None:
                # Randomly sample n_iterations from the full parameter combinations
                if n_iterations > len(param_combinations):
                    warnings.warn(f"Requested {n_iterations} iterations, but only {len(param_combinations)} combinations available. Using all combinations.")
                    self.logger.warning(f"Requested {n_iterations} iterations, but only {len(param_combinations)} combinations available. Using all combinations.")
                    n_iterations = len(param_combinations)
                param_combinations = np.random.choice(param_combinations, size=n_iterations, replace=False).tolist()
            all_results = []
            best_score = float('inf') if 'loss' in self.scoring else float('-inf')
            best_params = None
            self.logger.info(f"Evaluating {len(param_combinations)} parameter combinations")   
            for combo_idx, (arch_params, opt_name, opt_params, train_params) in enumerate(param_combinations):
                # Clear TensorFlow state between iterations
                keras.backend.clear_session()
                
                self.logger.info(f"Parameter combination {combo_idx+1}/{len(param_combinations)}")
                self.logger.info(f"Architecture params: {arch_params}")
                self.logger.info(f"Optimizer: {opt_name} with params: {opt_params}")
                self.logger.info(f"Training params: {train_params}")
                
                if is_cv:
                    # Cross-validation logic
                    fold_scores = []
                    fold_histories = []
                    current_model = None
                    combo_best_score = float('inf') if 'loss' in self.scoring else float('-inf')
                    combo_epochs = []

                    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=kwargs.get('groups'))):
                        self.logger.info(f"Training fold {fold+1}/{self.n_splits}")
                        
                        # Clear backend to free memory between folds
                        keras.backend.clear_session()
                        
                        X_train, X_val_fold = X[train_idx], X[val_idx]
                        y_train, y_val_fold = y[train_idx], y[val_idx]
                        
                        # Convert to tensors with explicit types
                        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
                        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
                        X_val_fold = tf.convert_to_tensor(X_val_fold, dtype=tf.float32)
                        y_val_fold = tf.convert_to_tensor(y_val_fold, dtype=tf.float32)
                        
                        # Get current batch size from training parameters
                        batch_size = train_params.get('batch_size', 32)
                        
                        model = self._create_model(arch_params)
                        model = self._compile_model(
                            model, 
                            opt_name,
                            opt_params
                        )
                        
                        history = model.fit(
                            X_train, y_train,
                            validation_data=(X_val_fold, y_val_fold),
                            batch_size=batch_size,
                            epochs=train_params.get('epochs', 100),
                            callbacks=self._get_callbacks(is_refit, fold, train_params),
                            verbose=self.verbose
                        )
                        
                        # Store history and score
                        fold_histories.append(history.history)
                        best_fold_score = min(history.history[self.scoring]) if 'loss' in self.scoring \
                            else max(history.history[self.scoring])
                        fold_scores.append(best_fold_score)
                        
                        self.logger.info(f"Fold {fold+1} best score: {best_fold_score:.4f}")
                        fold_epochs = len(history.history[self.scoring])
                        combo_epochs.append(fold_epochs)
                        # Save model if it's the best so far
                        is_better = (('loss' in self.scoring and best_fold_score < combo_best_score) or 
                                    ('loss' not in self.scoring and best_fold_score > combo_best_score))
                        if current_model is None or is_better:
                            current_model = model
                    
                    # Calculate mean score across folds
                    mean_score = np.mean(fold_scores)
                    std_score = np.std(fold_scores)
                    median_epochs = int(np.median(combo_epochs))
                    self.logger.info(f"Combination {combo_idx+1} mean score: {mean_score:.4f} ± {std_score:.4f}")
                    
                    all_results.append({
                        'params': {
                            'architecture': arch_params, 
                            'optimizer_name': opt_name,
                            'optimizer': opt_params,
                            'training': train_params
                        },
                        'mean_score': mean_score,
                        'std_score': std_score,
                        'fold_scores': fold_scores,
                        'histories': fold_histories,
                        'median_epochs': median_epochs,
                    })
                    
                    # Update best results
                    is_better = (mean_score < best_score if 'loss' in self.scoring 
                                else mean_score > best_score)
                    if is_better:
                        self.logger.info(f"New best score: {mean_score:.4f}")
                        best_score = mean_score
                        best_params = {
                            'architecture': arch_params, 
                            'optimizer_name': opt_name,
                            'optimizer': opt_params,
                            'training': train_params
                        }
                        refit_epochs = int(np.median(combo_epochs))
                        self.logger.info(f"Saving best model weights for fold {fold+1} with {refit_epochs} epochs")
                        self.best_model_weights = current_model.get_weights()
                
                else:
                    # Direct validation set without cross-validation
                    self.logger.info("Training with direct validation set")
                    
                    # Clear backend to free memory
                    keras.backend.clear_session()
                    
                    # Convert to tensors with explicit types
                    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
                    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
                    X_val_tensor = tf.convert_to_tensor(X_val, dtype=tf.float32)
                    y_val_tensor = tf.convert_to_tensor(y_val, dtype=tf.float32)
                    combo_epochs = None
                    # Get current batch size from training parameters
                    batch_size = train_params.get('batch_size', 32)
                    
                    model = self._create_model(arch_params)
                    model = self._compile_model(
                        model, 
                        opt_name,
                        opt_params
                    )
                    
                    history = model.fit(
                        X_tensor, y_tensor,
                        validation_data=(X_val_tensor, y_val_tensor),
                        batch_size=batch_size,
                        epochs=train_params.get('epochs', 100),
                        callbacks=self._get_callbacks(is_refit, 0, train_params),  
                        verbose=self.verbose
                    )
                    
                    # Store history and score
                    best_val_score = min(history.history[self.scoring]) if 'loss' in self.scoring \
                        else max(history.history[self.scoring])
                    
                    self.logger.info(f"Combination {combo_idx+1} validation score: {best_val_score:.4f}")
                    combo_epochs = len(history.history[self.scoring])
                    all_results.append({
                        'params': {
                            'architecture': arch_params, 
                            'optimizer_name': opt_name,
                            'optimizer': opt_params,
                            'training': train_params
                        },
                        'validation_score': best_val_score,
                        'history': history.history,
                        'epochs': combo_epochs
                    })
                    
                    # Update best results
                    is_better = (best_val_score < best_score if 'loss' in self.scoring 
                                else best_val_score > best_score)
                    if is_better:
                        self.logger.info(f"New best score: {best_val_score:.4f}")
                        best_score = best_val_score
                        best_params = {
                            'architecture': arch_params, 
                            'optimizer_name': opt_name,
                            'optimizer': opt_params,
                            'training': train_params
                        }
                        refit_epochs = combo_epochs
                        self.logger.info(f"Saving best model weights with {refit_epochs} epochs")
                        self.best_model_weights = model.get_weights()
                    
        finally:
            # Cleanup temporary files
            self.logger.info("Cleaning up temporary files")
            if is_cv:
                for fold in range(self.n_splits):
                    try:
                        os.remove(self.back_up_dir/f'temp_best_model_fold_{fold}.weights.h5')
                    except:
                        pass

            else:
                try:
                    os.remove(self.back_up_dir/f'temp_best_model_fold_0.weights.h5')
                except:
                    pass
        
        try:
            # Create final best model and set weights
            self.logger.info("Creating final best model")
            keras.backend.clear_session()
            
            self.best_estimator_ = self._create_model(best_params['architecture'])
            self.best_estimator_ = self._compile_model(
                self.best_estimator_,
                best_params['optimizer_name'],
                best_params['optimizer']
            )
            if self.best_model_weights is not None:
                self.best_estimator_.set_weights(self.best_model_weights)
            
            self.best_params_ = best_params
            self.best_score_ = best_score
            self.cv_results_ = pd.DataFrame(all_results)
            # Save best model weights to disk
            # self.save_best_model(self.save_model_dir)
            self.logger.info(f"Hyperparameter {'search' if is_cv else 'optimization'} completed. Best score: {best_score:.4f}, best params: {best_params}")
            
            return self
        except Exception as e:
            self.logger.error(f"Error creating and saving the best model: {e}")
            raise ValueError(f"Error creating and saving the best model: {e}")
            

    def get_best_model(self) -> keras.Model:
        """Return the best model found during optimization."""
        if self.best_estimator_ is None:
            self.logger.error("Model has not been fitted yet. Call fit() first.")
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.best_estimator_
    # Add these methods to the DLOptimizer class
    def save_best_model(self, path=None, filename=None, format=None, weights_only=False):
        """
        Save the best model found during optimization.
        Compatible with both Keras 2 and Keras 3.
        
        Parameters
        ----------
        path : str, default=None
            Directory to save the model. If None, saves to './outputs/models/'
        filename : str, default=None
            Filename for the saved model. If None, generates a name based on task and timestamp.
        format : str, default=None
            Optional format hint. In Keras 3, this is only used to help generate the filename extension
            if no specific filename is provided. Options: 'keras', 'h5', or 'tf'.
        weights_only : bool, default=False
            If True, saves only the model weights, not the architecture.
            
        Returns
        -------
        str : Path to the saved model file or directory
        """
        if self.best_estimator_ is None:
            self.logger.error("No best model to save. Run fit() first.")
            raise ValueError("No best model to save. Run fit() first.")
        
        # Set default path if not provided
        if path is None:
            path = self.save_model_dir
        else:
            path = Path(path)
        
        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Determine appropriate extension based on format hint
            if format is None or format == 'keras':
                # Default to .keras format for Keras 3
                ext = '.keras'
            elif format == 'h5':
                ext = '.h5'
            elif format == 'tf':
                # For SavedModel format, no extension is needed as it's a directory
                ext = ''
            else:
                # Default to .keras for unknown formats
                ext = '.keras'
                
            if weights_only:
                filename = f"dl_{self.task}_weights_{timestamp}.h5"  # Weights always use h5
            else:
                filename = f"dl_{self.task}_model_{timestamp}{ext}"
        
        # For full path
        model_path = path / filename
        
        try:
            # Handle saving logic based on whether we're saving weights only
            if weights_only:
                self.logger.info(f"Saving best model weights to {model_path}")
                self.best_estimator_.save_weights(model_path)
            else:
                # Check if we're using Keras 3 (look for keras.src in the module path)
                is_keras3 = 'keras.src' in sys.modules
                
                if is_keras3:
                    # In Keras 3, the format is determined by the file extension
                    self.logger.info(f"Saving best model to {model_path} using Keras 3")
                    
                    # If the user specified 'tf' format but didn't provide a specific filename,
                    # handle SavedModel specially since it needs a directory, not a file
                    if format == 'tf' and ext == '':
                        model_path = path / f"dl_{self.task}_model_{timestamp}"
                        self.best_estimator_.save(model_path)
                    else:
                        self.best_estimator_.save(model_path)
                else:
                    # Keras 2 compatibility path
                    if str(model_path).endswith('.h5'):
                        self.logger.info(f"Saving best model in H5 format to {model_path}")
                        self.best_estimator_.save(model_path, save_format='h5')
                    elif str(model_path).endswith('.keras'):
                        self.logger.info(f"Saving best model in Keras format to {model_path}")
                        self.best_estimator_.save(model_path, save_format='keras')
                    else:
                        self.logger.info(f"Saving best model in SavedModel format to {model_path}")
                        self.best_estimator_.save(model_path, save_format='tf')
        
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
            
        return str(model_path)


