
from typing import Dict, Union, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd


class GroupedTimeSeriesSplit:
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
    
    def _calculate_n_splits(self, groups):
        """Helper method to calculate the actual number of possible splits"""
        unique_times = np.unique(groups)
        n_times = len(unique_times)
        fold_size = n_times // self.n_splits
        if fold_size == 0:
            raise ValueError(f"Too few unique timestamps ({n_times}) for {self.n_splits} splits")
        
        test_size = self.test_size if self.test_size else fold_size
        possible_splits = (n_times - test_size) // (test_size) + 1
        print(f"test_size:{test_size} and n_splits:{self.n_splits}")
        return min(self.n_splits, possible_splits)
    
    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series]] = None
    ):
        if groups is None:
            raise ValueError("The 'groups' parameter should contain timestamp values.")
        
        # Convert groups to numpy array
        if isinstance(groups, pd.Series):
            groups = groups.to_numpy()
        
        # Get unique timestamps and their indices
        unique_times, unique_indices = np.unique(groups, return_inverse=True)
        n_times = len(unique_times)
        
        # Calculate actual number of splits possible
        n_splits = self._calculate_n_splits(groups)
        
        # Calculate sizes
        fold_size = n_times // self.n_splits
        test_size = self.test_size if self.test_size else fold_size
        
        # Generate exactly n_splits splits
        for i in range(n_splits):
            # Calculate test indices
            test_start = i * test_size
            test_end = min(test_start + test_size, n_times)
            
            # Calculate train indices
            train_end = max(0, test_start - self.gap)
            train_start = 0 if self.max_train_size is None else max(0, train_end - self.max_train_size)
            
            # Get time indices for train and test
            time_train_idx = np.arange(train_start, train_end)
            time_test_idx = np.arange(test_start, test_end)
            
            # Convert to sample indices
            train_mask = np.isin(unique_indices, time_train_idx)
            test_mask = np.isin(unique_indices, time_test_idx)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            yield train_indices, test_indices
    
    def get_n_splits(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> int:
        """Returns the number of splitting iterations."""
        if groups is None:
            return self.n_splits
        return self._calculate_n_splits(groups)