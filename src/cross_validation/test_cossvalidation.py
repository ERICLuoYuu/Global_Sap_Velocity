import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from cross_validators import BaseCrossValidator, MLCrossValidator, DLCrossValidator
# from src.tools import create_spatial_groups
import pandas as pd



test_data = pd.read_csv('./data/processed/merged/site/merged_data.csv')
test_data = test_data.dropna().set_index('TIMESTAMP')

# Generate spatial coordinates and groups
lat = test_data['lat'].values
lon = test_data['long'].values

# Create spatial groups using the grid method
'''
groups, _ = create_spatial_groups(
    lat=lat,
    lon=lon,
    method='grid',
    lat_grid_size=0.1,
    lon_grid_size=0.1
)
'''
# Create stratification variable (e.g., climate zones)
# Here we create three climate zones based on latitude
# strata = pd.qcut(lat, q=3, labels=['cold', 'temperate', 'warm'])

# Generate sample features and target
X = test_data.drop(columns=['sap_velocity', 'lat', 'long']).values
y = test_data['sap_velocity'].values

# Initialize cross-validator
rf = RandomForestRegressor(n_estimators=100, random_state=42)
cv = MLCrossValidator(estimator=rf, scoring='r2', n_splits=10)

# Perform spatial stratified cross-validation

# spatial_scores = cv.spatial_cv(X, y, groups=groups)
random_scores = cv.random_cv(X, y)
time_groups = test_data.sort_index().index

temporal_scores = cv.temporal_cv(X, y, groups=time_groups)
# print("Spatial Stratified CV Scores:", spatial_scores)
# print("Mean CV Score:", spatial_scores.mean())
# print("CV Score Standard Deviation:", spatial_scores.std())

print("\nRandom CV Scores:", random_scores)
print("Mean CV Score:", random_scores.mean())
print("CV Score Standard Deviation:", random_scores.std())

print("\nTemporal CV Scores:", temporal_scores)
print("Mean CV Score:", temporal_scores.mean())
print("CV Score Standard Deviation:", temporal_scores.std())
