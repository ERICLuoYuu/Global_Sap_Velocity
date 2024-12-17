import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from GHGA_selector import GHGA_selector


# Main example
def main():
    training_data = pd.read_csv('data/processed/merged/merged_data.csv')
    training_data = training_data.drop(columns=['TIMESTAMP']).dropna()
    print("Dataset shape:", training_data.shape)
    y = training_data['sap_velocity']
    x = training_data.drop(columns=['sap_velocity'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    # Initialize and fit GHGA selector
    selector = GHGA_selector(
        population_size=10,
        generations=2,
        mutation_rate=0.1,
        local_search_prob=0.1,
        elite_size=1,
        random_state=42
    )
    
    # Fit selector
    selector.fit(X_train_scaled, y_train)
    
    # Get selected features
    selected_features = selector.get_feature_names()
    print("\nSelected features:", selected_features)
    print("Number of selected features:", len(selected_features))
    
    # Get feature importance
    feature_importance = selector.get_feature_importance()
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_importance.index, y=feature_importance.values)
    plt.xticks(rotation=45)
    plt.title('Feature Importance Scores')
    plt.tight_layout()
    plt.show()
    
    # Plot convergence curve
    selector.plot_convergence()
    
    # Get summary statistics
    summary = selector.summary()
    print("\nSelection Summary:")
    print("Total features:", summary['n_total_features'])
    print("Selected features:", summary['n_selected_features'])
    print("Best fitness score:", round(summary['best_fitness'], 4))
    print("\nConvergence info:")
    print("Initial fitness:", round(summary['convergence']['initial_fitness'], 4))
    print("Final fitness:", round(summary['convergence']['final_fitness'], 4))
    
    # Transform data using selected features
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    print("\nTransformed data shapes:")
    print("Training data:", X_train_selected.shape)
    print("Test data:", X_test_selected.shape)

if __name__ == "__main__":
    main()