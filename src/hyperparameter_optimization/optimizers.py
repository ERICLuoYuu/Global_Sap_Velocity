import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Callable
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

class HyperparameterOptimizer:
    """
    A class for hyperparameter optimization using various methods:
    - Grid Search
    - Random Search
    - Bayesian Optimization
    - Hyperband
    
    Supports multiple ML algorithms:
    - SVM
    - Random Forest
    - XGBoost
    - Artificial Neural Network
    """
    
    def __init__(self, 
                 algo_type: str,
                 cross_validator,
                 metric: str = 'accuracy',
                 random_state: int = 42):
        """
        Initialize the optimizer
        
        Parameters:
        -----------
        algo_type : str
            Type of ML algorithm ('svm', 'rf', 'xgb', 'ann')
        cross_validator : CrossValidator
            Instance of CrossValidator class for performance evaluation
        metric : str
            Metric to optimize for
        random_state : int
            Random seed for reproducibility
        """
        self.algo_type = algo_type.lower()
        self.cross_validator = cross_validator
        self.metric = metric
        self.random_state = random_state
        
        # Define default parameter grids for each algorithm
        self.param_grids = {
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            },
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgb': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0]
            },
            'ann': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
        }
        
        # Initialize the base model
        self.model = self._get_base_model()
    
    def _get_base_model(self):
        """Create base model based on algorithm type"""
        if self.algo_type == 'svm':
            return SVC(random_state=self.random_state)
        elif self.algo_type == 'rf':
            return RandomForestClassifier(random_state=self.random_state)
        elif self.algo_type == 'xgb':
            return XGBClassifier(random_state=self.random_state)
        elif self.algo_type == 'ann':
            return MLPClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown algorithm type: {self.algo_type}")
    
    def _get_param_grid(self):
        """Get parameter grid for the selected algorithm"""
        return self.param_grids[self.algo_type]
    
    def _create_objective(self, X, y):
        """Create objective function for Bayesian optimization"""
        def objective(trial):
            params = self._get_optuna_params(trial)
            model = self._get_base_model()
            model.set_params(**params)
            
            # Use cross_validator to get scores
            if hasattr(X, 'spatial_groups'):
                scores = self.cross_validator.spatial_cv(
                    model, X, y, X.spatial_groups, scoring=self.metric
                )
            else:
                scores = self.cross_validator.random_cv(
                    model, X, y, scoring=self.metric
                )
            
            return -np.mean(scores)  # Negative because Optuna minimizes
        
        return objective
    
    def _get_optuna_params(self, trial):
        """Define parameter space for Optuna"""
        if self.algo_type == 'svm':
            return {
                'C': trial.suggest_loguniform('C', 1e-2, 1e2),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
                'gamma': trial.suggest_loguniform('gamma', 1e-3, 1e1)
            }
        elif self.algo_type == 'rf':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
            }
        # Add more parameter spaces for other algorithms...
        elif self.algo_type == 'xgb':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e0),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0)
            }
        elif self.algo_type == 'ann':
            return {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50)]),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'alpha': trial.suggest_loguniform('alpha', 1e-4, 1e-2),
                'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-3, 1e-1)
            }
        else:
            raise ValueError(f"Unknown algorithm type: {self.algo_type}")

    def grid_search(self, X, y, param_grid: Optional[Dict] = None):
        """
        Perform grid search optimization
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        param_grid : dict, optional
            Custom parameter grid. If None, uses default grid
            
        Returns:
        --------
        dict
            Best parameters and score
        """
        if param_grid is None:
            param_grid = self._get_param_grid()
            
        if hasattr(X, 'spatial_groups'):
            cv = self.cross_validator.spatial_cv
        else:
            cv = self.cross_validator.random_cv
            
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring=self.metric,
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    
    def random_search(self, X, y, n_iter: int = 100, param_distributions: Optional[Dict] = None):
        """
        Perform random search optimization
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        n_iter : int
            Number of parameter settings to try
        param_distributions : dict, optional
            Custom parameter distributions. If None, uses default grid
            
        Returns:
        --------
        dict
            Best parameters and score
        """
        if param_distributions is None:
            param_distributions = self._get_param_grid()
            
        if hasattr(X, 'spatial_groups'):
            cv = self.cross_validator.spatial_cv
        else:
            cv = self.cross_validator.random_cv
            
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=self.metric,
            n_jobs=-1,
            random_state=self.random_state
        )
        
        random_search.fit(X, y)
        
        return {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_
        }
    
    def bayesian_optimization(self, X, y, n_trials: int = 100):
        """
        Perform Bayesian optimization using Optuna
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        n_trials : int
            Number of trials
            
        Returns:
        --------
        dict
            Best parameters and score
        """
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state)
        )
        
        objective = self._create_objective(X, y)
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_score': -study.best_value  # Convert back from minimization
        }
    
    def hyperband(self, X, y, max_epochs: int = 100, factor: int = 3):
        """
        Perform Hyperband optimization
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        max_epochs : int
            Maximum iterations per configuration
        factor : int
            Reduction factor for successive halving
            
        Returns:
        --------
        dict
            Best parameters and score
        """
        # Calculate hyperband parameters
        s_max = int(np.log(max_epochs) / np.log(factor))
        B = (s_max + 1) * max_epochs
        
        best_score = float('-inf')
        best_params = None
        
        for s in reversed(range(s_max + 1)):
            n = int(np.ceil(int(B/max_epochs/(s+1)) * factor**s))
            r = max_epochs * factor**(-s)
            
            # Generate n random configurations
            configs = []
            for _ in range(n):
                trial = optuna.trial.Trial(None)
                configs.append(self._get_optuna_params(trial))
            
            # Successive halving with increasing resources
            for i in range(s + 1):
                n_i = n * factor**(-i)
                r_i = r * factor**i
                
                # Evaluate each configuration
                scores = []
                for config in configs[:int(n_i)]:
                    model = self._get_base_model()
                    model.set_params(**config)
                    
                    if hasattr(X, 'spatial_groups'):
                        score = np.mean(self.cross_validator.spatial_cv(
                            model, X, y, X.spatial_groups, scoring=self.metric
                        ))
                    else:
                        score = np.mean(self.cross_validator.random_cv(
                            model, X, y, scoring=self.metric
                        ))
                    
                    scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_params = config
                
                # Select top configurations
                indices = np.argsort(scores)[-int(n_i/factor):]
                configs = [configs[i] for i in indices]
        
        return {
            'best_params': best_params,
            'best_score': best_score
        }