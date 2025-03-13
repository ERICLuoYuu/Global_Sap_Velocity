import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
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


training_data = pd.read_csv('./outputs/processed_data/merged/site/training/training_data.csv')
training_data = training_data[['TIMESTAMP', 'ws_mean', 'vpd_mean', 'sw_in_mean', 'ta_mean', 'sap_velocity']].dropna().set_index('TIMESTAMP').sort_index()
test_data = pd.read_csv('./outputs/processed_data/merged/site/testing/testing_data.csv')
test_data = test_data[['TIMESTAMP', 'ws_mean', 'vpd_mean', 'sw_in_mean', 'ta_mean', 'sap_velocity']].dropna().set_index('TIMESTAMP').sort_index()
print(training_data.shape)
X = training_data[['ws_mean', 'vpd_mean', 'sw_in_mean', 'ta_mean']]
y = training_data['sap_velocity'].values
X_test = test_data[['ws_mean', 'vpd_mean', 'sw_in_mean', 'ta_mean']]
y_test = test_data['sap_velocity'].values

# Perform spatial stratified cross-validation
time_groups = test_data.sort_index().index

# Define parameter grid
param_grid = {
    'architecture': {
        'n_layers': [2, 3, 5],
        'units': [64, 128, 256],
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
    scoring='val_loss',
    model_type='nn'
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
# use the best model to predict
y_pred = best_model.predict(X_test)
# plot the results
fig, ax = plt.subplots()
ax.plot(test_data.index, y_test, label='True')
ax.plot(test_data.index, y_pred, label='Predicted')
ax.legend()

# calculate the r2 score and mse and label on the plot

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
ax.set_title(f"R2: {r2:.2f}, MSE: {mse:.2f}")
plt.show()

# Print results
print(f"best_params: {best_params}")
print(f"best_model: {best_model}")

"""

# do optimatization for ML model
# Define parameter grid
# for rf model
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

optimizer = MLOptimizer( param_grid=param_grid, scoring='r2', model_type='rf', task='regression')
optimizer.fit(X, y, split_type='random')
"""
"""
param_grid = {
    # Core Tree Structure
    'max_depth': [3,  5,  7],
    'min_child_weight': [1, 3, 5],
    
    # Learning Parameters
    'learning_rate': [0.01, 0.001],
    'n_estimators': [100, 200, 300, 500],
    
    # Sampling Parameters
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
optimizer = MLOptimizer( param_grid=param_grid, scoring='neg_mean_squared_error', model_type='xgb', task='regression')
optimizer.fit(X, y, split_type='random')

# print results
print(optimizer.best_params_)
print(optimizer.best_estimator_)
print(optimizer.best_estimator_.score(X, y))
print(optimizer.best_score_)
print(optimizer.cv_results_)
# use the best model to predict on the test data
y_pred = optimizer.best_estimator_.predict(X_test)
# use the best model to predict on the training data
y_train_pred = optimizer.best_estimator_.predict(X)
# plot the results
fig, ax = plt.subplots(2,1)
ax[0].plot(test_data.index, y_test, label='True')
ax[0].plot(test_data.index, y_pred, label='Predicted')
ax[0].legend()
ax[0].set_title('Test Data')
ax[1].plot(training_data.index, y, label='True')
ax[1].plot(training_data.index, y_train_pred, label='Predicted')
ax[1].legend()
ax[1].set_title('Training Data')


# calculate the r2 score and mse and label on the plot

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2_train = r2_score(y, y_train_pred)
mse_train = mean_squared_error(y, y_train_pred)
ax[0].text(0.5, 0.5, f"R2: {r2:.2f}, MSE: {mse:.2f}", transform=ax[0].transAxes)
ax[1].text(0.5, 0.5, f"R2: {r2_train:.2f}, MSE: {mse_train:.2f}", transform=ax[1].transAxes)
# set a suptitle
fig.suptitle('rf Model Results')
# layout the plot
plt.tight_layout()
plt.show()

# Print results
print(f"r2: {r2}, mse: {mse}")
print(f"r2_train: {r2_train}, mse_train: {mse_train}")
# save the plot
fig.savefig('./plots/rf_result.png')

# make scatter plots for the predicted and true values
fig1, ax = plt.subplots(2,1)
ax[0].scatter(y_test, y_pred)
# add a line for the perfect fit
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# add a line to fit the data
m, b = np.polyfit(y_test, y_pred, 1)
ax[0].plot(y_test, m*y_test + b, color='red')
ax[0].set_title('Test Data')
ax[0].set_xlabel('True')
ax[0].set_ylabel('Predicted')
ax[0].text(0.5, 0.5, f"R2: {r2:.2f}, MSE: {mse:.2f}", transform=ax[0].transAxes)
ax[1].scatter(y, y_train_pred)
# add a line for the perfect fit
ax[1].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# add a line to fit the data
m, b = np.polyfit(y, y_train_pred, 1)
ax[1].plot(y, m*y + b, color='red')
ax[1].set_title('Training Data')
ax[1].set_xlabel('True')
ax[1].set_ylabel('Predicted')
ax[1].text(0.5, 0.5, f"R2: {r2_train:.2f}, MSE: {mse_train:.2f}", transform=ax[1].transAxes)
# set a suptitle
fig1.suptitle('Scatter Plot of Predicted and True Values for rf Model')
# layout the plot
plt.tight_layout()

plt.show()
# save the plot
fig1.savefig('./plots/rf_scatter.png')
"""