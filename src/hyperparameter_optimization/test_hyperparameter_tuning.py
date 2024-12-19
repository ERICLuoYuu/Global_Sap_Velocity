import sys
from pathlib import Path
import pandas as pd
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


# Create stratification variable (e.g., climate zones)
# Here we create three climate zones based on latitude
# strata = pd.qcut(lat, q=3, labels=['cold', 'temperate', 'warm'])
test_data = pd.read_csv('./data/processed/merged/merged_data.csv')
test_data = test_data.dropna().set_index('TIMESTAMP').sort_index()
X = test_data.drop(columns=['sap_velocity', 'lat', 'long']).values
y = test_data['sap_velocity'].values


# Perform spatial stratified cross-validation
time_groups = test_data.sort_index().index

# Define parameter grid
param_grid = {
    'architecture': {
        'n_layers': [2, 3],
        'units': [64, 128],
        'dropout_rate': [0.2, 0.3]
    },
    'optimizer': {
        'name': ['Adam'],
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
    input_shape=(X.shape[1],),
    output_shape=1,
    scoring='val_loss'
)

# Temporal cross-validation with multiple measurements per timestamp
optimizer.fit(
    X,
    y,
    split_type='random',
    groups=None,  # timestamps for each sample
    gap=1,  # optional gap between train and test
    max_train_size=1000  # optional limit on training size
)

# Get best model and parameters
best_model = optimizer.get_best_model()
best_params = optimizer.best_params_

# Print results
print(f"best_params: {best_params}")
print(f"best_model: {best_model}")

"""""
# do optimatization for ML model
# Define parameter grid
# for rf model
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

optimizer = MLOptimizer( param_grid=param_grid, scoring='r2', model_type='rf', task='regression')
optimizer.fit(X, y, split_type='random')
# print results
print(optimizer.best_params_)
print(optimizer.best_estimator_)
print(optimizer.best_score_)
print(optimizer.cv_results_)
"""""