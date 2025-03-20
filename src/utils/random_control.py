"""
Global Randomization Control System for ML Pipelines
"""

import os
import random
import numpy as np
import tensorflow as tf
from functools import wraps
from contextlib import contextmanager

# Initialize the global random state manager
class RandomStateManager:
    """Singleton class to manage random states across libraries."""
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RandomStateManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, seed=None):
        if self._initialized:
            return
            
        self._master_seed = seed if seed is not None else 42
        self._random_generators = {}
        self._initialized = True
        
        # Create random generators
        self._random_generators['python'] = random.Random(self._master_seed)
        self._random_generators['numpy'] = np.random.RandomState(self._master_seed)
        
        # Set global seed immediately
        self.set_global_seed(self._master_seed)
    
    @property
    def master_seed(self):
        """Get the master seed."""
        return self._master_seed
    
    def set_global_seed(self, seed=None):
        """Set all random seeds for reproducibility."""
        seed_to_use = seed if seed is not None else self._master_seed
        self._master_seed = seed_to_use
        
        # Set environment variables
        os.environ['PYTHONHASHSEED'] = str(seed_to_use)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        
        # Set Python's random seed
        random.seed(seed_to_use)
        self._random_generators['python'] = random.Random(seed_to_use)
        
        # Set NumPy's random seed
        np.random.seed(seed_to_use)
        self._random_generators['numpy'] = np.random.RandomState(seed_to_use)
        
        # Set TensorFlow's random seed
        tf.random.set_seed(seed_to_use)
        
        # Configure TensorFlow for determinism
        if hasattr(tf.config.experimental, 'enable_op_determinism'):
            tf.config.experimental.enable_op_determinism()
        
        # Control threading
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        print(f"Global random seed set to {seed_to_use} for reproducibility")

# Create the global instance
random_state_manager = RandomStateManager()

# Define the decorator for deterministic execution
def deterministic(func):
    """Decorator to ensure functions execute deterministically."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a function-specific seed
        func_name_seed = sum(ord(c) for c in func.__name__)
        derived_seed = (random_state_manager.master_seed * 1000 + func_name_seed) % 2**32
        
        # Save current seed
        current_seed = random_state_manager.master_seed
        
        # Set the derived seed
        random_state_manager.set_global_seed(derived_seed)
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Restore the original seed
        random_state_manager.set_global_seed(current_seed)
        
        return result
    return wrapper

# Export these public functions
def set_seed(seed=None):
    """Set the global random seed."""
    random_state_manager.set_global_seed(seed)

def get_seed():
    """Get the current global random seed."""
    return random_state_manager.master_seed

def get_initializer(initializer_type='glorot_uniform'):
    """Get a TensorFlow initializer with deterministic seed."""
    # Get our master seed as an integer
    seed = int(random_state_manager.master_seed)
    
    try:
        # Try the modern Keras approach first
        if hasattr(tf.keras.random, 'SeedGenerator'):
            seed_gen = tf.keras.random.SeedGenerator(seed=seed)
            
            if initializer_type == 'glorot_uniform':
                return tf.keras.initializers.GlorotUniform(seed=seed_gen)
            elif initializer_type == 'glorot_normal':
                return tf.keras.initializers.GlorotNormal(seed=seed_gen)
            elif initializer_type == 'he_uniform':
                return tf.keras.initializers.HeUniform(seed=seed_gen)
            elif initializer_type == 'he_normal':
                return tf.keras.initializers.HeNormal(seed=seed_gen)
            else:
                return tf.keras.initializers.GlorotUniform(seed=seed_gen)
        else:
            # Fall back to the older approach
            if initializer_type == 'glorot_uniform':
                return tf.keras.initializers.GlorotUniform(seed=seed)
            elif initializer_type == 'glorot_normal':
                return tf.keras.initializers.GlorotNormal(seed=seed)
            elif initializer_type == 'he_uniform':
                return tf.keras.initializers.HeUniform(seed=seed)
            elif initializer_type == 'he_normal':
                return tf.keras.initializers.HeNormal(seed=seed)
            else:
                return tf.keras.initializers.GlorotUniform(seed=seed)
    except Exception as e:
        print(f"Warning: Error creating initializer: {e}")
        print("Falling back to default initializer")
        return tf.keras.initializers.GlorotUniform()

# Simple test function
def test_determinism():
    """Test if the randomization control is working properly."""
    # Set seed and get some random numbers
    set_seed(42)
    nums1 = [random.random() for _ in range(5)]
    
    # Reset seed and get random numbers again
    set_seed(42)
    nums2 = [random.random() for _ in range(5)]
    
    # Check if they match
    are_identical = nums1 == nums2
    print(f"Determinism test {'passed' if are_identical else 'failed'}")
    return are_identical

# Make sure the functions are in the module's namespace
__all__ = ['set_seed', 'get_seed', 'get_initializer', 'deterministic', 
           'random_state_manager', 'test_determinism']