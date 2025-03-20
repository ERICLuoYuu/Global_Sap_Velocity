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
data = pd.read_csv('./outputs/processed_data/merged/site/gap_filled_size1_with_era5/all_biomes_merged_data.csv').set_index('TIMESTAMP').sort_index().dropna()
data = data[['sap_velocity','vpd', 'ta','swc_shallow', 'ppfd_in', 'volumetric_soil_water_layer_4', 'leaf_area_index_low_vegetation', 'soil_temperature_level_3', 'volumetric_soil_water_layer_3', 'biome', 'swc_deep', 'u_component_of_wind_10m', 'v_component_of_wind_10m',]]
# Convert biome to categorical if it exists
if 'biome' in data.columns:
    data['biome'] = data['biome'].astype('category')
# get dummy variables for the biome column
data = pd.get_dummies(data, columns=['biome'])
print(data.columns)
data = add_time_features(data)

training_portion = 0.8
#training_data = data.loc[:data.index[int(len(data)*training_portion)]]
#test_data = data.loc[data.index[int(len(data)*training_portion)+1:]]
# Use positional indexing instead of label-based indexing
training_data = data.iloc[:int(len(data)*training_portion)]
test_data = data.iloc[int(len(data)*training_portion):]
#training_data = data.sample(frac=0.8, random_state=42).sort_index()
#b test_data = data.drop(training_data.index).sort_index()
# training_data = training_data[['TIMESTAMP', 'ws_mean', 'vpd_mean', 'sw_in_mean', 'ta_mean', 'sap_velocity']].dropna().set_index('TIMESTAMP')
# test_data = test_data[['TIMESTAMP', 'ws_mean', 'vpd_mean', 'sw_in_mean', 'ta_mean', 'sap_velocity']].dropna().set_index('TIMESTAMP')
print(training_data.shape)
X = training_data.drop(columns=['sap_velocity'])
y = training_data['sap_velocity'].values
X_test = test_data.drop(columns=['sap_velocity'])
y_test = test_data['sap_velocity'].values
# Perform spatial stratified cross-validation
time_groups = test_data.sort_index().index


# do optimatization for ML model
# Define parameter grid
param_grid = {
    'n_estimators': [500],  # Fewer trees can sometimes help reduce overfitting
    'max_depth': [7],  # Reduce max_depth to prevent trees from becoming too specific
    'min_samples_split': [5],  # Higher values prevent creating too many small splits
    'min_samples_leaf': [2,],  # Higher values ensure terminal nodes aren't too specific
    'max_features': ['sqrt',],  # Controls feature randomness in tree building
    'bootstrap': [True],
    'random_state': [42],
    'oob_score': [True]  # Enable out-of-bag evaluation for better generalization assessment
}

optimizer = MLOptimizer( param_grid=param_grid, scoring='r2', model_type='rf', task='regression')
optimizer.fit(X, y, split_type='temporal')

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
ax[0].text(0.2, 0.8, f"R2: {r2:.2f}, MSE: {mse:.2f}", transform=ax[0].transAxes)
ax[1].scatter(y, y_train_pred)
# add a line for the perfect fit
ax[1].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# add a line to fit the data
m, b = np.polyfit(y, y_train_pred, 1)
ax[1].plot(y, m*y + b, color='red')
ax[1].set_title('Training Data')
ax[1].set_xlabel('True')
ax[1].set_ylabel('Predicted')
ax[1].text(0.2, 0.8, f"R2: {r2_train:.2f}, MSE: {mse_train:.2f}", transform=ax[1].transAxes)
# set a suptitle
fig1.suptitle('Scatter Plot of Predicted and True Values for rf Model')
# layout the plot
plt.tight_layout()

plt.show()
# save the plot
fig1.savefig('./plots/rf_scatter.png')