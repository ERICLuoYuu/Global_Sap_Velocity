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
from src.cross_validation.timeseries_split import GroupedTimeSeriesSplit
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from xgboost.sklearn import XGBRegressor, XGBClassifier
from abc import ABC, abstractmethod
from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit
from typing import Dict, Union, List, Optional, Tuple, Callable
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import itertools
import warnings


# Simple logging setup function
def setup_logging(log_dir=None):
    """Set up basic logging configuration"""
    if log_dir is None:
        log_dir = Path('./outputs/logs')
    
    log_dir = Path(log_dir)
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / 'optimizer.log'
    
    # Configure logging to file and console with basic format
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    return logging.getLogger()


def setup_distribution_strategy():
    """
    Set up the appropriate distribution strategy based on environment.
    
    Returns:
    --------
    strategy : tf.distribute.Strategy
        Either MirroredStrategy for single-node or MultiWorkerMirroredStrategy for multi-node
    is_multi_node : bool
        Whether we're running in a multi-node environment
    """
    # Check if running under SLURM
    if 'SLURM_JOB_NUM_NODES' in os.environ:
        num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', '1'))
        if num_nodes > 1:
            # Multi-node setup
            setup_tf_config_for_slurm()
            # Use NCCL for best GPU performance
            communication_options = tf.distribute.experimental.CommunicationOptions(
                implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
            strategy = tf.distribute.MultiWorkerMirroredStrategy(
                communication_options=communication_options)
            return strategy, True, num_nodes
    
    # Single node setup - use MirroredStrategy if GPUs are available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        strategy = tf.distribute.MirroredStrategy()
        return strategy, False, len(gpus)
    else:
        # Fall back to default strategy for CPU-only machines
        strategy = tf.distribute.get_strategy()
        return strategy, False, 0


def setup_tf_config_for_slurm():
    """
    Set up the TF_CONFIG environment variable for MultiWorkerMirroredStrategy using SLURM.
    """
    logger = logging.getLogger()
    
    # Get node list from SLURM
    try:
        node_list = os.environ.get('SLURM_NODELIST', '')
        if '[' in node_list:  # Handle compressed node list format (e.g., "node[1-3,5]")
            cmd = f"scontrol show hostnames {node_list}"
            nodes = subprocess.check_output(cmd, shell=True).decode().splitlines()
        else:
            nodes = node_list.split(',')
        
        # Get current node's hostname
        current_host = socket.gethostname()
        
        # Get current node's rank (assuming SLURM_PROCID represents the worker rank)
        if 'SLURM_PROCID' in os.environ:
            node_rank = int(os.environ['SLURM_PROCID'])
        else:
            # Try to determine which node we're on if SLURM_PROCID is not available
            node_rank = nodes.index(current_host) if current_host in nodes else 0
        
        # Create worker list with a consistent port
        port = 12345  # Choose an available port that's open on your cluster
        worker_hosts = [f"{node}:{port}" for node in nodes]
        
        # Create and set TF_CONFIG
        tf_config = {
            "cluster": {
                "worker": worker_hosts
            },
            "task": {
                "type": "worker",
                "index": node_rank
            }
        }
        
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
        logger.info(f"Set TF_CONFIG for node {node_rank} of {len(nodes)}")
        
    except Exception as e:
        logger.error(f"Failed to set up TF_CONFIG: {e}")
        # Fallback to no TF_CONFIG (single-worker)
        if "TF_CONFIG" in os.environ:
            del os.environ["TF_CONFIG"]


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
    log_dir : str, default=None
        Directory for log files
    """
    
    def __init__(
        self,
        param_grid: Dict,
        scoring: str = 'r2',
        n_splits: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 1,
        log_dir: Optional[str] = None,
        
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
        
        # Set up logging once for all instances
        self.logger = setup_logging(log_dir)
    
    def _get_cv_splitter(
        self,
        split_type: str,
        **kwargs
    ):
        """Get the appropriate cross-validation splitter."""
        if split_type == 'spatial':
            return GroupKFold(n_splits=self.n_splits)
        elif split_type == 'temporal':
            return TimeSeriesSplit(
                n_splits=self.n_splits,
                max_train_size=kwargs.get('max_train_size', None),
                test_size=kwargs.get('test_size', None),
                gap=kwargs.get('gap', 0)
            )
        else:  # random
            is_shuffle = kwargs.get('is_shuffle', False)
            if is_shuffle:
                return KFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_state
                )
            else:

                return KFold(
                    n_splits=self.n_splits,
                    shuffle=kwargs.get('is_shuffle', False),
                    
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
    log_dir : str, default=None
        Directory for log files
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
        log_dir: Optional[str] = None,
        save_model_dir: Optional[str] = None
    ):  
        if task.lower() not in ['regression', 'classification']:
            raise ValueError("Unknown task type. Choose 'regression' or 'classification'.")
        # Set default scoring based on task
        if scoring is None:
            scoring = 'r2' if task.lower() == 'regression' else 'accuracy'
            
        super().__init__(param_grid, scoring, n_splits, random_state, n_jobs, verbose, log_dir)
        self.model_type = model_type.lower()
        self.task = task.lower()
        self.model = self._get_base_model()
        if save_model_dir is not None:
            self.save_model_dir = Path(save_model_dir + f'/{model_type}_{task}')
            if not self.save_model_dir.exists():
                self.save_model_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_model_dir = Path('./outputs/models' + f'/{model_type}_{task}')
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
                tree_method='hist'  # Added for better compatibility
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
        try:
            # Get appropriate CV splitter
            cv = self._get_cv_splitter(split_type, **kwargs)
            
            self.logger.info(f"Starting grid search for {self.model_type} model with {split_type} split")
            
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
            if split_type in ['spatial']:
                if 'groups' not in kwargs:
                    raise ValueError("groups parameter is required for spatial cross-validation")
                grid_search.fit(X, y, groups=kwargs['groups'])
            else:
                grid_search.fit(X, y, )
            
            # Store results
            self.best_params_ = grid_search.best_params_
            self.best_score_ = grid_search.best_score_
            self.best_estimator_ = grid_search.best_estimator_
            self.cv_results_ = grid_search.cv_results_
            # save the best model
            self.save_best_model(path=self.save_model_dir)
            self.logger.info(f"Grid search completed. Best score: {self.best_score_:.4f}")
            self.logger.info(f"Best parameters: {self.best_params_}")
            
            return self
        except Exception as e:
            self.logger.error(f"Error in grid search: {e}")
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
    use_distribution : bool, default=True
        Whether to use distributed training strategies
    log_dir : str, default=None
        Directory for log files
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
        use_distribution: bool = True,
        log_dir: Optional[str] = None,
        back_up_dir: Optional[str] = None,
        save_model_dir: Optional[str] = None

    ):
        # Set default scoring based on task
        if scoring is None:
            scoring = 'val_loss' if task.lower() == 'regression' else 'val_accuracy'
            
        super().__init__(param_grid, scoring, n_splits, random_state, 1, verbose, log_dir)
        self.base_architecture = base_architecture
        self.task = task.lower()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.label_encoder = LabelEncoder() if task == 'classification' else None
        self.best_model_weights = None
        self.use_distribution = use_distribution
        if back_up_dir is not None:
            self.back_up_dir = Path(back_up_dir)
        else:
            self.back_up_dir = Path('./outputs/hyper_tuning/tmp')
        if not self.back_up_dir.exists():
            self.back_up_dir.mkdir(parents=True, exist_ok=True)
        
        if save_model_dir is not None:
            self.save_model_dir = Path(save_model_dir + f'/{model_type}_{task}')
            if not self.save_model_dir.exists():
                self.save_model_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_model_dir = Path('./outputs/models' + f'/{model_type}_{task}')
            if not self.save_model_dir.exists():
                self.save_model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized DLOptimizer for {task} task")
        
        # Set up distribution strategy if requested
        if self.use_distribution:
            self.strategy, self.is_multi_node, num_devices = setup_distribution_strategy()
            strategy_type = 'MultiWorkerMirroredStrategy' if self.is_multi_node else 'MirroredStrategy'
            self.logger.info(f"Using {strategy_type} with {self.strategy.num_replicas_in_sync} replicas")
            
            if self.is_multi_node:
                self.logger.info(f"Running on {num_devices} nodes in distributed mode")
            else:
                device_type = "GPUs" if num_devices > 0 else "CPU"
                self.logger.info(f"Running on {num_devices} {device_type} in single-node mode")
        else:
            self.strategy = None
            self.is_multi_node = False
            self.logger.info("Distribution strategy disabled")
        
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
        for v in self.optimizer_params.values():
            if hasattr(v, '__len__') and v != 'name':
                opt_combinations *= len(v)
                
        total_combinations = arch_combinations * opt_combinations
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
        if self.strategy:
            with self.strategy.scope():
                model = self.base_architecture(
                    input_shape=self.input_shape,
                    output_shape=self.output_shape,
                    **architecture_params
                )
                return model
        else:
            # No distribution strategy
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
            metrics=metrics
        )
        return model
    
    def _get_callbacks(self, fold: int) -> List[keras.callbacks.Callback]:
        """Get callbacks for training."""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=self.scoring,
                patience=self.training_params.get('patience', 10),
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
        
        # Add special callbacks for distributed training
        if self.strategy and self.is_multi_node:
            # Add BackupAndRestore for fault tolerance in multi-worker setup
            backup_dir = self.back_up_dir/f'backup_{fold}'
            if not os.path.exists(backup_dir):
                Path(backup_dir).mkdir(parents=True, exist_ok=True)
            os.makedirs(backup_dir, exist_ok=True)
            callbacks.append(
                keras.callbacks.BackupAndRestore(
                    backup_dir=backup_dir,
                    save_freq='epoch'  # Can also be an integer for step-based backup
                )
            )
            
        return callbacks
    
    def _get_param_combinations(self):
        """Generate all parameter combinations for grid search."""
        param_combinations = []
        arch_keys, arch_values = zip(*self.architecture_params.items()) if self.architecture_params else ([], [])
        opt_name = self.optimizer_params.get('name', ['adam'])[0]
        opt_params = {k: v for k, v in self.optimizer_params.items() if k != 'name'}
        
        if opt_params:
            opt_keys, opt_values = zip(*opt_params.items())
        else:
            opt_keys, opt_values = [], []
        
        # Handle case where architecture_params is empty
        if not arch_keys:
            if opt_params:
                for opt_combo in itertools.product(*opt_values):
                    opt_params = dict(zip(opt_keys, opt_combo))
                    param_combinations.append(({}, opt_params))
            else:
                param_combinations.append(({}, {}))
        else:
            for arch_combo in itertools.product(*arch_values):
                arch_params = dict(zip(arch_keys, arch_combo))
                if opt_params:
                    for opt_combo in itertools.product(*opt_values):
                        opt_params = dict(zip(opt_keys, opt_combo))
                        param_combinations.append((arch_params, opt_params))
                else:
                    param_combinations.append((arch_params, {}))
                
        return param_combinations
    
    def _create_distributed_dataset(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        batch_size: int
    ) -> tf.data.Dataset:
        """Create a distributed dataset for training."""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        # Get global batch size
        if self.strategy:
            global_batch_size = batch_size * self.strategy.num_replicas_in_sync
        else:
            global_batch_size = batch_size
            
        # Configure dataset for performance
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(global_batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        # Distribute dataset if using a strategy
        if self.strategy:
            return self.strategy.experimental_distribute_dataset(dataset)
        return dataset
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        split_type: str = 'random',
        is_cv: bool = True,
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
            all_results = []
            best_score = float('inf') if 'loss' in self.scoring else float('-inf')
            best_params = None
            
            self.logger.info(f"Evaluating {len(param_combinations)} parameter combinations")
            
            for combo_idx, (arch_params, opt_params) in enumerate(param_combinations):
                self.logger.info(f"Parameter combination {combo_idx+1}/{len(param_combinations)}")
                
                if is_cv:
                    # Original cross-validation logic
                    fold_scores = []
                    fold_histories = []
                    current_model = None
                    
                    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=kwargs.get('groups'))):
                        self.logger.info(f"Training fold {fold+1}/{self.n_splits}")
                        
                        # Clear backend to free memory
                        keras.backend.clear_session()
                        
                        X_train, X_val_fold = X[train_idx], X[val_idx]
                        y_train, y_val_fold = y[train_idx], y[val_idx]
                        
                        # Convert to tensors with explicit types
                        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
                        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
                        X_val_fold = tf.convert_to_tensor(X_val_fold, dtype=tf.float32)
                        y_val_fold = tf.convert_to_tensor(y_val_fold, dtype=tf.float32)
                        
                        # Get current batch size
                        batch_size = self.training_params.get('batch_size', 32)
                        
                        # Create and compile model
                        model = self._create_model(arch_params)
                        model = self._compile_model(model, 
                                                self.optimizer_params.get('name', ['adam'])[0],
                                                opt_params)
                        
                        # Create datasets optimized for distributed training
                        if self.strategy:
                            # For distributed training, we need to handle the dataset differently
                            train_dataset = self._create_distributed_dataset(X_train, y_train, batch_size)
                            val_dataset = self._create_distributed_dataset(X_val_fold, y_val_fold, batch_size)
                            
                            # When using a distributed dataset, don't pass X and y directly
                            history = model.fit(
                                train_dataset,
                                validation_data=val_dataset,
                                epochs=self.training_params.get('epochs', 100),
                                callbacks=self._get_callbacks(fold),
                                verbose=self.verbose
                            )
                        else:
                            # Standard training without distribution
                            history = model.fit(
                                X_train, y_train,
                                validation_data=(X_val_fold, y_val_fold),
                                batch_size=batch_size,
                                epochs=self.training_params.get('epochs', 100),
                                callbacks=self._get_callbacks(fold),
                                verbose=self.verbose
                            )
                        
                        # Store history and score
                        fold_histories.append(history.history)
                        best_fold_score = min(history.history[self.scoring]) if 'loss' in self.scoring \
                            else max(history.history[self.scoring])
                        fold_scores.append(best_fold_score)
                        
                        self.logger.info(f"Fold {fold+1} best score: {best_fold_score:.4f}")
                        
                        # Save model if it's the best so far
                        if current_model is None or \
                            (('loss' in self.scoring and best_fold_score < best_score) or \
                            ('loss' not in self.scoring and best_fold_score > best_score)):
                            current_model = model
                    
                    # Calculate mean score across folds
                    mean_score = np.mean(fold_scores)
                    std_score = np.std(fold_scores)
                    
                    self.logger.info(f"Combination {combo_idx+1} mean score: {mean_score:.4f} ± {std_score:.4f}")
                    
                    all_results.append({
                        'params': {'architecture': arch_params, 'optimizer': opt_params},
                        'mean_score': mean_score,
                        'std_score': std_score,
                        'fold_scores': fold_scores,
                        'histories': fold_histories
                    })
                    
                    # Update best results
                    is_better = (mean_score < best_score if 'loss' in self.scoring 
                                else mean_score > best_score)
                    if is_better:
                        self.logger.info(f"New best score: {mean_score:.4f}")
                        best_score = mean_score
                        best_params = {'architecture': arch_params, 'optimizer': opt_params}
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
                    
                    # Get current batch size
                    batch_size = self.training_params.get('batch_size', 32)
                    
                    # Create and compile model
                    model = self._create_model(arch_params)
                    model = self._compile_model(model, 
                                            self.optimizer_params.get('name', ['adam'])[0],
                                            opt_params)
                    
                    # Create datasets optimized for distributed training
                    if self.strategy:
                        # For distributed training, we need to handle the dataset differently
                        train_dataset = self._create_distributed_dataset(X_tensor, y_tensor, batch_size)
                        val_dataset = self._create_distributed_dataset(X_val_tensor, y_val_tensor, batch_size)
                        
                        # When using a distributed dataset, don't pass X and y directly
                        history = model.fit(
                            train_dataset,
                            validation_data=val_dataset,
                            epochs=self.training_params.get('epochs', 100),
                            callbacks=self._get_callbacks(0),  # Use 0 as fold number for single validation
                            verbose=self.verbose
                        )
                    else:
                        # Standard training without distribution
                        history = model.fit(
                            X_tensor, y_tensor,
                            validation_data=(X_val_tensor, y_val_tensor),
                            batch_size=batch_size,
                            epochs=self.training_params.get('epochs', 100),
                            callbacks=self._get_callbacks(0),  # Use 0 as fold number for single validation
                            verbose=self.verbose
                        )
                    
                    # Store history and score
                    best_val_score = min(history.history[self.scoring]) if 'loss' in self.scoring \
                        else max(history.history[self.scoring])
                    
                    self.logger.info(f"Combination {combo_idx+1} validation score: {best_val_score:.4f}")
                    
                    all_results.append({
                        'params': {'architecture': arch_params, 'optimizer': opt_params},
                        'validation_score': best_val_score,
                        'history': history.history
                    })
                    
                    # Update best results
                    is_better = (best_val_score < best_score if 'loss' in self.scoring 
                                else best_val_score > best_score)
                    if is_better:
                        self.logger.info(f"New best score: {best_val_score:.4f}")
                        best_score = best_val_score
                        best_params = {'architecture': arch_params, 'optimizer': opt_params}
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
                    # Clean up backup directories if using distributed training
                    if self.strategy and self.is_multi_node:
                        try:
                            backup_dir = self.back_up_dir/f'backup_{fold}'
                            if os.path.exists(backup_dir):
                                import shutil
                                shutil.rmtree(backup_dir)
                        except:
                            pass
            else:
                try:
                    os.remove(self.back_up_dir/f'temp_best_model_fold_0.weights.h5')
                except:
                    pass
                # Clean up backup directories if using distributed training
                if self.strategy and self.is_multi_node:
                    try:
                        backup_dir = self.back_up_dir/f'backup_0'
                        if os.path.exists(backup_dir):
                            import shutil
                            shutil.rmtree(backup_dir)
                    except:
                        pass
        
        try:
            # Create final best model and set weights
            self.logger.info("Creating final best model")
            if self.strategy:
                with self.strategy.scope():
                    self.best_estimator_ = self._create_model(best_params['architecture'])
                    self.best_estimator_ = self._compile_model(
                        self.best_estimator_,
                        self.optimizer_params.get('name', ['adam'])[0],
                        best_params['optimizer']
                    )
                    if self.best_model_weights is not None:
                        self.best_estimator_.set_weights(self.best_model_weights)
            else:
                self.best_estimator_ = self._create_model(best_params['architecture'])
                self.best_estimator_ = self._compile_model(
                    self.best_estimator_,
                    self.optimizer_params.get('name', ['adam'])[0],
                    best_params['optimizer']
                )
                if self.best_model_weights is not None:
                    self.best_estimator_.set_weights(self.best_model_weights)
            
            self.best_params_ = best_params
            self.best_score_ = best_score
            self.cv_results_ = pd.DataFrame(all_results)
            # Save best model weights to disk
            if self.is_multi_node:
                self.save_best_model_distributed(self.save_model_dir)
            else:
                self.save_best_model(self.save_model_dir)
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
            path = Path('./outputs/models')
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


# Updated distributed saving method for Keras 3 compatibility
def save_best_model_distributed(self, path=None, filename=None, format=None):
    """
    Save the best model found during optimization in a distributed environment.
    Only the chief worker (rank 0) will save the model. Compatible with Keras 3.
    
    Parameters
    ----------
    path : str, default=None
        Directory to save the model. If None, saves to './outputs/models/'
    filename : str, default=None
        Filename for the saved model. If None, generates a name based on task and timestamp.
    format : str, default=None
        Optional format hint for the file extension if filename is not provided.
        
    Returns
    -------
    str : Path to the saved model file or directory, or None if not the chief worker
    """
    # Only save from the chief worker in a distributed setting
    if not self.is_multi_node or (self.is_multi_node and 
                                  tf.distribute.get_replica_context().replica_id_in_sync_group == 0):
        return self.save_best_model(path, filename, format)
    else:
        self.logger.info("Skipping model saving on non-chief worker")
        return None