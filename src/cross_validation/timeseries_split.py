
from typing import Dict, Union, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd


class GroupedTimeSeriesSplit:
    """
    Time Series cross-validator that handles multiple measurements per timestamp.
    All measurements with the same timestamp will always be in the same fold.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits
    max_train_size : int, optional
        Maximum size for a single training set
    test_size : int, optional
        Used to limit the size of the test set
    gap : int, default=0
        Number of timestamps to exclude between train and test sets
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0
    ):
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
    
    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series]] = None
    ):
        """Generate indices to split data into training and test sets."""
        print(groups)
        if groups is None:
            raise ValueError("The 'groups' parameter should contain timestamp values.")
        
        # Convert groups to numpy array
        if isinstance(groups, pd.Series):
            groups = groups.to_numpy()
        
        # Get unique timestamps and their indices
        unique_times, unique_indices = np.unique(groups, return_inverse=True)
        n_times = len(unique_times)
        
        # Calculate fold size based on number of unique timestamps
        fold_size = n_times // self.n_splits
        test_size = self.test_size if self.test_size else fold_size
        
        for i in range(self.n_splits):
            # Calculate indices for test set
            if i < self.n_splits - 1:
                test_start = i * fold_size
                test_end = test_start + test_size
            else:
                # Last fold might be larger to include remaining samples
                test_start = i * fold_size
                test_end = n_times
            
            # Apply gap
            train_end = max(0, test_start - self.gap)
            
            # Apply max_train_size
            if self.max_train_size:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0
            
            # Get time indices for train and test
            time_train_idx = np.arange(train_start, train_end)
            time_test_idx = np.arange(test_start, test_end)
            
            # Convert to sample indices
            train_mask = np.isin(unique_indices, time_train_idx)
            test_mask = np.isin(unique_indices, time_test_idx)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
    
    def get_n_splits(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits