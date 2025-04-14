#!/usr/bin/env python3
# test_hyperparameter_tuning_ensemble.py - Create ensemble from individually trained models
# 
# This script loads saved models from each individual hyperparameter tuning script,
# generates predictions for each model, and creates an ensemble prediction.

import os
import sys
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Add parent directory to system path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import custom modules
from src.utils.random_control import set_seed
from src.hyperparameter_optimization.timeseries_processor import TimeSeriesSegmenter, SegmentedWindowGenerator

# Set random seed for reproducibility
set_seed(42)

# Set environment variables for determinism BEFORE importing TensorFlow
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = '42'

# Import TensorFlow after setting environment variables
import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)
tf.config.experimental.enable_op_determinism()

def add_time_features(df, datetime_column=None):
    """
    Create cyclical time features from a datetime column or index.
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

def load_data(input_width=8, label_width=1, shift=1, batch_size=32, exclude_label=True):
    """
    Load and prepare data for time series prediction.
    Uses the same data processing as the individual model scripts.
    
    Returns:
    --------
    dict
        Dictionary containing datasets, scalers, and other info needed for evaluation
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Load data
    print("Loading data...")
    data_dir = Path('./outputs/processed_data/merged/site/gap_filled_size1_hourly_after_filter')
    data_list = list(data_dir.glob('*merged.csv'))
    
    if not data_list:
        print(f"No CSV files found in {data_dir}")
        try:
            # Fallback to single file if no directory is found
            data = pd.read_csv('./outputs/processed_data/merged/site/gap_filled_size1_with_era5/test.csv')
            print("Using fallback test.csv file")
        except:
            print("ERROR: Could not find any data files")
            return None
    else:
        print(f"Found {len(data_list)} data files")
    
    # Define the columns we want to use
    used_cols = ['sap_velocity', 'ext_rad', 'sw_in', 'ta', 'ws', 'vpd','ppfd_in', 'biome', 'Day sin', 'Week sin', 'Month sin', 'Year sin']
    all_biome_types = set()  # Will collect all unique biome types
    all_possible_biome_types = ['Boreal forest', 'Subtropical desert', 'Temperate forest', 'Temperate grassland desert', 'Temperate rain forest', 'Tropical forest savanna', 'Tropical rain forest', 'Tundra', 'Woodland/Shrubland']
    # Sort data files for deterministic processing order
    data_list = sorted(data_list)
    INPUT_WIDTH = input_width
    SHIFT = shift
    all_segments = []  # To store all segments
    # Process each data file
    for data_file in data_list:
        print(f"Processing {data_file.name}")
        try:
            df = pd.read_csv(data_file, parse_dates=['TIMESTAMP'])
            df.set_index('TIMESTAMP', inplace=True)
            df = add_time_features(df)
            
            # Verify columns exist
            missing_cols = [col for col in used_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns in {data_file.name}: {missing_cols}")
                continue
                
            # Fix potential case/whitespace issues
            df.columns = [col.strip().lower() for col in df.columns]
            used_cols_lower = [col.lower() for col in used_cols]
            
            # Select only the columns we need
            available_cols = [col for col in used_cols_lower if col in df.columns]
            df = df[available_cols]
            
            if 'biome' in used_cols_lower:
                if 'biome' in df.columns:
                    # Get non-biome columns 
                    orig_cols = [col for col in used_cols_lower if col != 'biome']
                    
                    # Track which biome types are found in the data
                    all_biome_types.update(df['biome'].unique())
                    
                    # Create a temporary biome column Series
                    biome_series = df['biome']
                    
                    # Initialize a DataFrame with zeros for all possible biomes
                    biome_df = pd.DataFrame(0.0, index=df.index, columns=all_possible_biome_types)
                    
                    # For each row, set the corresponding biome column to 1.0
                    for idx, biome in biome_series.items():
                        if biome in all_possible_biome_types:
                            biome_df.loc[idx, biome] = 1.0
                    
                    # Join with original data
                    df = df[orig_cols].join(biome_df)
                else:
                    print(f"Warning: Missing biome column in {data_file.name}")
                    continue
            
            # Clean data (only once)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            if len(df) < INPUT_WIDTH + SHIFT:
                print(f"Warning: {data_file.name} has too few records after cleaning: {len(df)}")
                continue
            
            # Segment the data (just like in CNN-LSTM)
            segments = TimeSeriesSegmenter.segment_time_series(
                df, 
                gap_threshold=2, 
                unit='hours', 
                min_segment_length=INPUT_WIDTH + SHIFT
            )
            
            print(f"  Created {len(segments)} segments")
            all_segments.extend(segments)
            
        except Exception as e:
            print(f"Error processing {data_file.name}: {e}")

    if not all_segments:
        print("No valid segments found. Falling back to single file mode.")
        try:
            # Fallback to single file processing
            data = pd.read_csv('./outputs/processed_data/merged/site/gap_filled_size1_with_era5/test.csv')
            data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
            data = data.sort_values('TIMESTAMP').set_index('TIMESTAMP')
            data = add_time_features(data)
            
            # Select only needed columns
            columns = ['sap_velocity', 'sw_in', 'ext_rad', 'ta', 'ws', 'vpd', 'rh', 'ppfd_in']
            # Use available columns
            columns = [col for col in columns if col in data.columns]
            data = data[columns].dropna()
            
            # Create a single segment
            all_segments = [data]
        except:
            print("ERROR: Could not process any data.")
            return (0, 0), (0, 0)  # Return dummy values

    print(f"Total segments collected: {len(all_segments)}")
    
    if 'biome' in used_cols:
        # Sort all biome types for consistent column order
        all_biome_types = sorted(all_biome_types)
        
        # Log what biome types were actually found in the data
        print(f"Biome types found in data: {all_biome_types}")
        print(f"Missing biome types: {set(all_possible_biome_types) - set(all_biome_types)}")

    # Now standardize the data
    all_data = pd.concat(all_segments)

    # Create scalers
    feature_scaler = StandardScaler()
    label_scaler = StandardScaler()

       # Get numerical and categorical feature columns
    if 'biome' in used_cols:
        numerical_features = [col for col in all_data.columns 
                            if col != 'sap_velocity' and col not in all_possible_biome_types]
        categorical_features = [col for col in all_possible_biome_types if col in all_data.columns]
    else:
        numerical_features = [col for col in all_data.columns if col != 'sap_velocity']
        categorical_features = []

    feature_columns = numerical_features + categorical_features
    # Only fit scaler on numerical features
    feature_scaler.fit(all_data[numerical_features])
    label_scaler.fit(all_data[['sap_velocity']])
 
    # Apply scaling to each segment
    scaled_segments = []
    for segment in all_segments:
        segment_copy = segment.copy()
        # Only transform numerical features
        segment_copy[numerical_features] = feature_scaler.transform(segment[numerical_features])
        # Leave categorical features as is
        segment_copy['sap_velocity'] = label_scaler.transform(segment[['sap_velocity']])
        print(segment_copy.columns)
        scaled_segments.append(segment_copy)
    
    # Create windowed datasets with deterministic options
    print("Creating windowed datasets...")
    try:
        # First attempt with regular split
        datasets = SegmentedWindowGenerator.create_complete_dataset_from_segments(
            segments=scaled_segments,
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            label_columns=['sap_velocity'],
            train_val_test_split=(0.8, 0.1, 0.1),
            batch_size=batch_size,
            exclude_labels_from_inputs=exclude_label
        )
    except ValueError as e:
        print(f"Error with default split ratio: {e}")
        print("Trying with an alternative split ratio...")
        
        # Second attempt with more balanced split
        datasets = SegmentedWindowGenerator.create_complete_dataset_from_segments(
            segments=scaled_segments,
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            label_columns=['sap_velocity'],
            train_val_test_split=(0.8, 0.1, 0.1),
            batch_size=batch_size,
            exclude_labels_from_inputs=False  # Include labels in inputs as a fallback
        )
    
    train_ds = datasets['train']
    val_ds = datasets['val']
    test_ds = datasets['test']

    # Check dataset properties
    if train_ds is None:
        print("ERROR: Could not create training dataset.")
        return None
    
    # Print dataset shapes for debugging
    print("Dataset shapes:")
    for features_batch, labels_batch in train_ds.take(1):
        print(f"Features batch shape: {features_batch.shape}")
        print(f"Labels batch shape: {labels_batch.shape}")
        input_shape = (features_batch.shape[1], features_batch.shape[2])
        n_features = features_batch.shape[2]
        break
        
    # Return as a dictionary for easy reference
    return {
        'train_ds': train_ds,
        'val_ds': val_ds,
        'test_ds': test_ds,
        'feature_scaler': feature_scaler,
        'label_scaler': label_scaler,
        'input_shape': input_shape,
        'n_features': n_features,
        'feature_columns': feature_columns,
        'all_biome_types': all_biome_types
    }

def convert_tf_dataset_to_xy(dataset):
    """
    Convert TensorFlow dataset to X and y numpy arrays with deterministic behavior.
    """
    features_list = []
    labels_list = []
    
    # Iterate through all batches in the dataset
    for features_batch, labels_batch in dataset:
        # Convert TensorFlow tensors to numpy arrays
        features_list.append(features_batch.numpy())
        labels_list.append(labels_batch.numpy())
    
    # Concatenate batches
    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    
    # If labels are shape (batch, seq_len, features) and you need (batch, features)
    if y.ndim > 2:
        y = y.reshape(y.shape[0], -1)
    
    return X, y

def convert_tf_dataset_to_xy_flat(dataset):
    """
    Convert TensorFlow dataset to flattened X and y arrays for traditional ML models.
    """
    X, y = convert_tf_dataset_to_xy(dataset)
    
    # Reshape for traditional ML (flatten window dimension)
    X_reshaped = X.reshape(X.shape[0], -1)
    
    return X_reshaped, y.flatten()

def get_predictions(model, data, model_type, label_scaler):
    """
    Get predictions from a model based on model type.
    
    Parameters:
    -----------
    model : object
        Trained model (either traditional ML or deep learning)
    data : dict
        Dictionary containing datasets
    model_type : str
        Type of model ('ann', 'lstm', 'transformer', 'cnn_lstm', 'rf', 'svr', 'xgb')
    label_scaler : object
        Scaler for the target variable
    
    Returns:
    --------
    tuple
        (test_predictions, test_labels, train_predictions, train_labels)
    """
    print(f"Getting predictions for {model_type} model...")
    
    # Deep learning models
    if model_type in ['ann', 'lstm', 'transformer', 'cnn_lstm']:
        # For test set
        all_test_predictions = []
        all_test_labels = []
        
        for features, labels in data['test_ds']:
            batch_predictions = model.predict(features, verbose=0)
            
            # For single-step prediction, reshape appropriately
            if len(batch_predictions.shape) == 3:  # Shape is [batch, time_steps, features]
                batch_predictions = batch_predictions[:, -1, :]  # Get last time step
                
            if len(labels.shape) == 3:  # Shape is [batch, time_steps, features]
                labels = labels[:, -1, :]  # Get last time step
                
            # Flatten and add to collections
            all_test_predictions.extend(batch_predictions.flatten())
            all_test_labels.extend(labels.numpy().flatten())
        
        # For train set
        all_train_predictions = []
        all_train_labels = []
        
        for features, labels in data['train_ds']:
            batch_predictions = model.predict(features, verbose=0)
            
            # For single-step prediction, reshape appropriately
            if len(batch_predictions.shape) == 3:
                batch_predictions = batch_predictions[:, -1, :]
                
            if len(labels.shape) == 3:
                labels = labels[:, -1, :]
                
            # Flatten and add to collections
            all_train_predictions.extend(batch_predictions.flatten())
            all_train_labels.extend(labels.numpy().flatten())
        
        # Convert to numpy arrays
        test_predictions = np.array(all_test_predictions).reshape(-1, 1)
        test_labels = np.array(all_test_labels).reshape(-1, 1)
        train_predictions = np.array(all_train_predictions).reshape(-1, 1)
        train_labels = np.array(all_train_labels).reshape(-1, 1)
        
    # Traditional ML models
    else:
        # Prepare the data
        X_test, y_test = convert_tf_dataset_to_xy_flat(data['test_ds'])
        X_train, y_train = convert_tf_dataset_to_xy_flat(data['train_ds'])
        
        # Make predictions
        test_predictions = model.predict(X_test).reshape(-1, 1)
        train_predictions = model.predict(X_train).reshape(-1, 1)
        
        # Reshape labels
        test_labels = y_test.reshape(-1, 1)
        train_labels = y_train.reshape(-1, 1)
    
    # Inverse transform to get original scale
    test_predictions = label_scaler.inverse_transform(test_predictions)
    test_labels = label_scaler.inverse_transform(test_labels)
    train_predictions = label_scaler.inverse_transform(train_predictions)
    train_labels = label_scaler.inverse_transform(train_labels)
    
    return test_predictions.flatten(), test_labels.flatten(), train_predictions.flatten(), train_labels.flatten()

def load_models(models_dir=None, model_types=None):
    """
    Load saved models for each model type.
    
    Parameters:
    -----------
    models_dir : str, optional
        Directory containing model files. If None, uses './outputs/models/'
    model_types : list, optional
        List of model types to load. If None, loads all available models.
    
    Returns:
    --------
    dict
        Dictionary mapping model types to loaded models
    """
    if models_dir is None:
        models_dir = './outputs/models'
    
    if model_types is None:
        model_types = ['ann', 'lstm', 'transformer', 'cnn_lstm', 'rf', 'svr', 'xgb']
    
    models_dir = Path(models_dir)
    loaded_models = {}
    
    for model_type in model_types:
        model_path = None
        
        # Define model directory
        type_dir = models_dir / (model_type + '_regression')
        
        # Check if directory exists
        if not type_dir.exists():
            print(f"Warning: Directory for {model_type} not found at {type_dir}")
            continue
        
        # Find the most recent model file
        if model_type in ['rf', 'svr', 'xgb']:
            # Traditional ML models use .joblib format
            model_files = list(type_dir.glob('*.joblib'))
            
            if model_files:
                # Sort by modification time (most recent first)
                model_path = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                
                # Load the model
                try:
                    print(f"Loading {model_type} model from {model_path}")
                    model = joblib.load(model_path)
                    loaded_models[model_type] = model
                except Exception as e:
                    print(f"Error loading {model_type} model: {e}")
            else:
                print(f"No .joblib files found for {model_type}")
                
        else:
            # Deep learning models use .keras format
            model_files = list(type_dir.glob('*.keras'))
            
            if not model_files:
                # Check for saved model directories (TensorFlow format)
                model_dirs = [d for d in type_dir.iterdir() if d.is_dir() and (d/'saved_model.pb').exists()]
                if model_dirs:
                    model_path = sorted(model_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                else:
                    print(f"No .keras files or SavedModel directories found for {model_type}")
                    continue
            else:
                # Use the most recent .keras file
                model_path = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            
            # Load the model
            try:
                print(f"Loading {model_type} model from {model_path}")
                model = tf.keras.models.load_model(model_path)
                loaded_models[model_type] = model
            except Exception as e:
                print(f"Error loading {model_type} model: {e}")
    
    print(f"Successfully loaded {len(loaded_models)} models")
    return loaded_models

def evaluate_models(loaded_models, data):
    """
    Generate and evaluate predictions for all models.
    
    Parameters:
    -----------
    loaded_models : dict
        Dictionary mapping model types to loaded models
    data : dict
        Dictionary containing datasets and other info
    
    Returns:
    --------
    dict
        Dictionary of model predictions and metrics
    """
    predictions = {}
    metrics = {}
    
    for model_type, model in loaded_models.items():
        print(f"\nEvaluating {model_type} model...")
        try:
            # Get predictions
            test_pred, test_labels, train_pred, train_labels = get_predictions(
                model, data, model_type, data['label_scaler']
            )
            
            # Calculate metrics
            test_r2 = r2_score(test_labels, test_pred)
            test_rmse = np.sqrt(mean_squared_error(test_labels, test_pred))
            test_mae = mean_absolute_error(test_labels, test_pred)
            
            train_r2 = r2_score(train_labels, train_pred)
            train_rmse = np.sqrt(mean_squared_error(train_labels, train_pred))
            train_mae = mean_absolute_error(train_labels, train_pred)
            
            # Store predictions
            predictions[model_type] = {
                'test_pred': test_pred,
                'test_labels': test_labels,
                'train_pred': train_pred,
                'train_labels': train_labels
            }
            
            # Store metrics
            metrics[model_type] = {
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'train_rmse': train_rmse,
                'train_mae': train_mae
            }
            
            print(f"{model_type.upper()} Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {model_type} model: {e}")
    
    return {'predictions': predictions, 'metrics': metrics}

def create_ensemble(evaluation_results):
    """
    Create ensemble predictions from individual model predictions.
    
    Parameters:
    -----------
    evaluation_results : dict
        Dictionary of model predictions and metrics
    
    Returns:
    --------
    dict
        Dictionary of ensemble predictions and metrics
    """
    predictions = evaluation_results['predictions']
    
    if not predictions:
        print("No model predictions available for ensemble")
        return None
    
    # Collect predictions
    test_preds = []
    test_labels = None
    train_preds = []
    train_labels = None
    
    for model_type, pred_dict in predictions.items():
        test_preds.append(pred_dict['test_pred'])
        train_preds.append(pred_dict['train_pred'])
        
        # Store labels (should be the same for all models)
        if test_labels is None:
            test_labels = pred_dict['test_labels']
            train_labels = pred_dict['train_labels']
    
    # Convert to numpy arrays
    test_preds = np.array(test_preds)
    train_preds = np.array(train_preds)
    
    # Create ensemble predictions (median)
    ensemble_test_pred = np.median(test_preds, axis=0)
    ensemble_train_pred = np.median(train_preds, axis=0)
    
    # Calculate metrics
    test_r2 = r2_score(test_labels, ensemble_test_pred)
    test_rmse = np.sqrt(mean_squared_error(test_labels, ensemble_test_pred))
    test_mae = mean_absolute_error(test_labels, ensemble_test_pred)
    
    train_r2 = r2_score(train_labels, ensemble_train_pred)
    train_rmse = np.sqrt(mean_squared_error(train_labels, ensemble_train_pred))
    train_mae = mean_absolute_error(train_labels, ensemble_train_pred)
    
    print("\nEnsemble Model Evaluation:")
    print(f"Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
    print(f"Train R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
    
    # Store ensemble results
    ensemble_results = {
        'predictions': {
            'test_pred': ensemble_test_pred,
            'test_labels': test_labels,
            'train_pred': ensemble_train_pred,
            'train_labels': train_labels
        },
        'metrics': {
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'train_mae': train_mae
        },
        'component_models': list(predictions.keys())
    }
    
    plt.figure(figsize=(15, 6))
    component_models = ensemble_results['component_models']
    for i, model_type in enumerate(component_models):
        plt.scatter(test_labels, predictions[model_type]['test_pred'] - test_labels, 
                alpha=0.3, label=model_type)
    plt.scatter(test_labels, ensemble_test_pred - test_labels, 
            alpha=0.7, color='black', label='Ensemble')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.legend()
    plt.title('Error by True Value')
    plt.xlabel('True Values')
    plt.ylabel('Prediction Error')
    return ensemble_results

def save_results(evaluation_results, ensemble_results, output_dir=None, run_id=None):
    """
    Save evaluation results and create visualizations.
    """
    if output_dir is None:
        output_dir = './outputs/ensemble'
    
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(output_dir)
    results_dir = output_dir / 'results'
    plot_dir = output_dir / 'plots'
    
    # Create directories
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics for all models
    metrics = {model: evaluation_results['metrics'][model] for model in evaluation_results['metrics']}
    metrics['ensemble'] = ensemble_results['metrics']
    
    with open(results_dir / f"all_metrics_{run_id}.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save predictions for each individual model
    for model_type, pred_dict in evaluation_results['predictions'].items():
        # Save test predictions for this model
        test_df = pd.DataFrame({
            'true': pred_dict['test_labels'],
            'pred': pred_dict['test_pred']
        })
        test_df.to_csv(results_dir / f"{model_type}_test_predictions_{run_id}.csv", index=False)
        
        # Save train predictions for this model
        train_df = pd.DataFrame({
            'true': pred_dict['train_labels'],
            'pred': pred_dict['train_pred']
        })
        train_df.to_csv(results_dir / f"{model_type}_train_predictions_{run_id}.csv", index=False)
    
    # Save ensemble predictions
    # Test set
    ensemble_test_df = pd.DataFrame({
        'true': ensemble_results['predictions']['test_labels'],
        'pred': ensemble_results['predictions']['test_pred']
    })
    ensemble_test_df.to_csv(results_dir / f"ensemble_test_predictions_{run_id}.csv", index=False)
    
    # Train set
    ensemble_train_df = pd.DataFrame({
        'true': ensemble_results['predictions']['train_labels'],
        'pred': ensemble_results['predictions']['train_pred']
    })
    ensemble_train_df.to_csv(results_dir / f"ensemble_train_predictions_{run_id}.csv", index=False)
    
    # Create comparative visualizations
    create_comparison_plots(metrics, plot_dir, run_id)
    create_prediction_plots(ensemble_results, plot_dir, run_id, metrics)
    
    print(f"Results saved to {output_dir}")

def create_comparison_plots(metrics, plot_dir, run_id):
    """Create comparative plots of all models."""
    # Create metrics DataFrame
    df_data = []
    
    for model, model_metrics in metrics.items():
        df_data.append({
            'Model': model.upper(),
            'R² (Test)': model_metrics['test_r2'],
            'RMSE (Test)': model_metrics['test_rmse'],
            'MAE (Test)': model_metrics['test_mae']
        })
    
    df = pd.DataFrame(df_data)
    
    # Sort by R² (best models first)
    df = df.sort_values('R² (Test)', ascending=False)
    
    # Save metrics to CSV
    df.to_csv(plot_dir / f"model_comparison_{run_id}.csv", index=False)
    
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # 1. R² Comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Model', y='R² (Test)', data=df)
    plt.title('Model Comparison: R² Score', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add values on bars
    for i, v in enumerate(df['R² (Test)']):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(plot_dir / f"model_comparison_r2_{run_id}.png")
    plt.close()
    
    # 2. RMSE Comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Model', y='RMSE (Test)', data=df)
    plt.title('Model Comparison: RMSE', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add values on bars
    for i, v in enumerate(df['RMSE (Test)']):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(plot_dir / f"model_comparison_rmse_{run_id}.png")
    plt.close()
    
    # 3. MAE Comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Model', y='MAE (Test)', data=df)
    plt.title('Model Comparison: MAE', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add values on bars
    for i, v in enumerate(df['MAE (Test)']):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(plot_dir / f"model_comparison_mae_{run_id}.png")
    plt.close()

def create_prediction_plots(ensemble_results, plot_dir, run_id, metrics):
    """Create plots for ensemble predictions."""
    # Extract data
    test_pred = ensemble_results['predictions']['test_pred']
    test_labels = ensemble_results['predictions']['test_labels']
    train_pred = ensemble_results['predictions']['train_pred']
    train_labels = ensemble_results['predictions']['train_labels']
    
    # 1. Test predictions scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(test_labels, test_pred, alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(max(test_labels), max(test_pred))
    min_val = min(min(test_labels), min(test_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Ensemble: Test Predictions vs Actual', fontsize=14)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_dir / f"ensemble_test_scatter_{run_id}.png")
    plt.close()
    
    # 2. Error distribution
    plt.figure(figsize=(10, 6))
    errors = test_pred - test_labels
    plt.hist(errors, bins=30, alpha=0.7)
    plt.title('Ensemble: Prediction Error Distribution', fontsize=14)
    plt.xlabel('Prediction Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_dir / f"ensemble_error_hist_{run_id}.png")
    plt.close()
    
    # 3. Time series plot (subset)
    n_samples = min(200, len(test_labels))
    plt.figure(figsize=(12, 6))
    plt.plot(range(n_samples), test_labels[:n_samples], 'b-', label='Actual')
    plt.plot(range(n_samples), test_pred[:n_samples], 'r-', label='Predicted')
    plt.title('Ensemble: Time Series Prediction (Test Set Subset)', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_dir / f"ensemble_timeseries_{run_id}.png")
    plt.close()
    
    # 4. Ensemble vs. Best Individual Model
    component_models = ensemble_results['component_models']
    if component_models:
        # Find best component model
        best_model = None
        best_r2 = -float('inf')
        
        for model_type in component_models:
            model_r2 = metrics[model_type]['test_r2']
            if model_r2 > best_r2:
                best_r2 = model_r2
                best_model = model_type
        
        if best_model:
            comparison = pd.DataFrame({
                'Model': ['Best Individual', 'Ensemble'],
                'R²': [metrics[best_model]['test_r2'], ensemble_results['metrics']['test_r2']],
                'RMSE': [metrics[best_model]['test_rmse'], ensemble_results['metrics']['test_rmse']],
                'MAE': [metrics[best_model]['test_mae'], ensemble_results['metrics']['test_mae']]
            })
            
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='Model', y='R²', data=comparison)
            plt.title(f'Ensemble vs. Best Individual Model ({best_model.upper()}): R²', fontsize=14)
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('R²', fontsize=12)
            
            # Add values on bars
            for i, v in enumerate(comparison['R²']):
                ax.text(i, v + 0.02, f"{v:.3f}", ha='center')
            
            plt.tight_layout()
            plt.savefig(plot_dir / f"ensemble_vs_best_r2_{run_id}.png")
            plt.close()
    
    print(f"Plots saved to {plot_dir}")

def main():
    parser = argparse.ArgumentParser(description="Create ensemble model from individual trained models")
    parser.add_argument('--models-dir', type=str, default='./outputs/models',
                      help='Directory containing saved models')
    parser.add_argument('--output-dir', type=str, default='./outputs/models/ensemble',
                      help='Directory to save ensemble results')
    parser.add_argument('--run-id', type=str, default=None,
                      help='Unique identifier for this run')
    parser.add_argument('--model-types', type=str, nargs='+',
                      default=['ann', 'cnn_lstm', 'rf', 'xgb'],
                      help='Model types to include in ensemble')
    
    args = parser.parse_args()
    
    # Generate run ID if not provided
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting ensemble creation (Run ID: {args.run_id})")
    print(f"Models directory: {args.models_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model types to include: {args.model_types}")
    
    # Load data
    print("\n1. Loading and preparing data...")
    data = load_data()
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Load models
    print("\n2. Loading saved models...")
    models = load_models(args.models_dir, args.model_types)
    if not models:
        print("No models loaded. Exiting.")
        return
    
    # Evaluate individual models
    print("\n3. Evaluating individual models...")
    evaluation_results = evaluate_models(models, data)
    
    # Create ensemble
    print("\n4. Creating ensemble model...")
    ensemble_results = create_ensemble(evaluation_results)
    if ensemble_results is None:
        print("Failed to create ensemble. Exiting.")
        return
    
    # Save results
    print("\n5. Saving results and creating visualizations...")
    save_results(evaluation_results, ensemble_results, args.output_dir, args.run_id)
    
    print("\nEnsemble creation completed successfully.")

if __name__ == "__main__":
    # Set deterministic behavior globally
    set_seed(42)
    main()