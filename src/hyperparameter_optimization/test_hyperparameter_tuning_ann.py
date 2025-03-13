import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pathlib import Path
import sys
# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from src.hyperparameter_optimization.hyper_tuner import DLOptimizer
def add_time_features(df, datetime_column=None):
    """
    Create cyclical time features from a datetime column or index.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the datetime column or with datetime index
    datetime_column : str, optional
        Name of the datetime column. If None, uses the DataFrame index
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with additional cyclical time features
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Get the datetime series from column or index
    if datetime_column is not None:
        if datetime_column not in df.columns:
            raise ValueError(f"Column '{datetime_column}' not found in DataFrame")
        date_time = pd.to_datetime(df[datetime_column])
    else:
        # Use index if no column specified
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                date_time = pd.to_datetime(df.index)
            except:
                raise ValueError("DataFrame index cannot be converted to datetime and no datetime_column specified")
        else:
            date_time = df.index
    
    # Convert datetime to timestamp in seconds
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    
    # Define day and year in seconds
    day = 24 * 60 * 60  # seconds in a day
    year = 365.2425 * day  # seconds in a year
    
    # Create cyclical features for day and year
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    
    # Add week features (7 day cycle)
    week = 7 * day
    df['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
    df['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
    
    # Add month features (30.44 day cycle)
    month = 30.44 * day
    df['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
    df['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))
    
    
    return df
# Load and preprocess data
data = pd.read_csv('./outputs/processed_data/merged/site/gap_filled_size1/by_biome/temperate_forest_merged_data.csv').dropna().set_index('TIMESTAMP').sort_index()
# Add time features
data = add_time_features(data)
data = data[['ws_mean', 'vpd_mean', 'sw_in_mean', 'ta_mean', 'sap_velocity', 'Day sin', 'Day cos', 'Year sin', 'Year cos', 'Week sin', 'Week cos', 'Month sin', 'Month cos']]
training_portion = 0.8
#training_data = data.loc[:data.index[int(len(data)*training_portion)]]
#test_data = data.loc[data.index[int(len(data)*training_portion)+1:]]
# Use positional indexing instead of label-based indexing
training_data = data.iloc[:int(len(data)*training_portion)]
test_data = data.iloc[int(len(data)*training_portion)+1:]
#training_data = data.sample(frac=0.8, random_state=42)
#test_data = data.drop(training_data.index)
# Print data info
print("Training data info:")
print(training_data.info())
print("\nSummary statistics:")
print(training_data.describe())
print("\nCheck for NaN values:")
print(training_data.isnull().sum())

# Select features and target
train_features = training_data[['ws_mean', 'vpd_mean', 'sw_in_mean', 'ta_mean', 'Day sin', 'Day cos']]
train_labels = training_data['sap_velocity']

# Clean data and handle NaN values
train_features = train_features.replace([np.inf, -np.inf], np.nan)
train_labels = train_labels.replace([np.inf, -np.inf], np.nan)

# Remove rows with NaN values
mask = ~(train_features.isna().any(axis=1) | train_labels.isna())
train_features = train_features[mask]
train_labels = train_labels[mask]

test_features = test_data[['ws_mean', 'vpd_mean', 'sw_in_mean', 'ta_mean', 'Day sin', 'Day cos']]
test_labels = test_data['sap_velocity']

mask = ~(test_features.isna().any(axis=1) | test_labels.isna())
test_features = test_features[mask]
test_labels = test_labels[mask]

# standardize the data
scaler = StandardScaler()
train_labels = scaler.fit_transform(train_labels.values.reshape(-1, 1))
test_labels = scaler.transform(test_labels.values.reshape(-1, 1))

def create_nn(input_shape, output_shape, n_layers, units, dropout_rate):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    
    for i in range(n_layers):
        model.add(keras.layers.Dense(
            units,
            activation='relu',
            kernel_initializer='glorot_uniform'
        ))
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate))
    
    model.add(keras.layers.Dense(1))
    return model

# Define parameter grid similar to first script's architecture
param_grid = {
    'architecture': {
        'n_layers': [3],
        'units': [64],
        'dropout_rate': [0.2]
    },
    'optimizer': {   
        
        'name': ['Adam'],
        'learning_rate': [0.001]
    },
    'training': {
        'batch_size': 64,
        'epochs': 100,
        'patience': 10
    }
}

# Create and fit optimizer
optimizer = DLOptimizer(
    base_architecture=create_nn,
    task='regression',
    model_type='nn',
    param_grid=param_grid,
    input_shape=(train_features.shape[1],),
    output_shape=1,
    scoring='val_loss'
)

# Fit the optimizer
optimizer.fit(
    train_features,
    train_labels,
    split_type='temporal',
    groups=None
)

# Get the best model
best_model = optimizer.get_best_model()

# Plot predictions vs actual
test_predictions = best_model.predict(test_features).flatten()
train_predictions = best_model.predict(train_features).flatten()


# reverse the normalization
test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
train_predictions = scaler.inverse_transform(train_predictions.reshape(-1, 1)).flatten()
test_labels = scaler.inverse_transform(test_labels.reshape(-1,1)).flatten()
train_labels = scaler.inverse_transform(train_labels.reshape(-1,1)).flatten()

plt.figure(figsize=(10, 10))
plt.scatter(test_labels, test_predictions, alpha=0.5)
plt.xlabel('True Values [Sap Velocity]')
plt.ylabel('Predictions [Sap Velocity]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])
plt.savefig('./plots/predictions_vs_actual.png')
plt.close()

# Plot train predictions vs actual
plt.figure(figsize=(10, 10))
plt.scatter(train_labels, train_predictions, alpha=0.5)
plt.xlabel('True Values [Sap Velocity]')
plt.ylabel('Predictions [Sap Velocity]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])
plt.savefig('./plots/train_predictions_vs_actual.png')
plt.close()

# Plot error distribution
error = test_predictions - test_labels
plt.figure(figsize=(10, 7))
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [Sap Velocity]')
plt.ylabel('Count')
plt.savefig('./plots/error_distribution.png')
plt.close()

# Plot error distribution for training data
error_train = train_predictions - train_labels
plt.figure(figsize=(10, 7))
plt.hist(error_train, bins=25)
plt.xlabel('Prediction Error [Sap Velocity]')
plt.ylabel('Count')
plt.savefig('./plots/error_distribution_train.png')
plt.close()

# Calculate and print R-squared scores
r2 = r2_score(test_labels, test_predictions)
r2_train = r2_score(train_labels, train_predictions)
print(f"R² Score (test): {r2:.3f}")
print(f"R² Score (train): {r2_train:.3f}")

# Print optimization summary
print("\nOptimization Summary:")
print(f"Best parameters: {optimizer.best_params_}")
print(f"Best validation score: {optimizer.best_score_}")

# Evaluate the model
test_results = best_model.evaluate(test_features, test_labels, verbose=0)
print(f"Test loss (MSE): {test_results[0]:.3f}")
print(f"Test MAE: {test_results[1]:.3f}")