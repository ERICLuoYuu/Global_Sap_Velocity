import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from src.hyperparameter_optimization.hyper_tuner import DLOptimizer
from src.hyperparameter_optimization.hyper_tuner import MLOptimizer
from tensorflow import keras
# Define the base architecture function
def create_nn(input_shape, output_shape, n_layers, units, dropout_rate):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    
    for _ in range(n_layers):
        model.add(keras.layers.Dense(units, activation='relu'))
        model.add(keras.layers.Dropout(dropout_rate))
    
    if isinstance(output_shape, int) and output_shape > 1:
        model.add(keras.layers.Dense(output_shape, activation='softmax'))
    else:
        model.add(keras.layers.Dense(1))
    
    return model

# Define parameter grid
param_grid = {
    'architecture': {
        'n_layers': [2, 3],
        'units': [64, 128],
        'dropout_rate': [0.2, 0.3]
    },
    'optimizer': {
        'name': ['adam'],
        'learning_rate': [0.001, 0.0001]
    },
    'training': {
        'batch_size': 32,
        'epochs': 100,
        'patience': 10
    }
}

# Create optimizer
optimizer = DLOptimizer(
    base_architecture=create_nn,
    task='regression',
    param_grid=param_grid,
    input_shape=(10,),
    output_shape=1,
    scoring='val_loss'
)

# Temporal cross-validation with multiple measurements per timestamp
optimizer.fit(
    X,
    y,
    split_type='temporal',
    groups=timestamps,  # timestamps for each sample
    gap=1,  # optional gap between train and test
    max_train_size=1000  # optional limit on training size
)

# Get best model and parameters
best_model = optimizer.get_best_model()
best_params = optimizer.best_params_