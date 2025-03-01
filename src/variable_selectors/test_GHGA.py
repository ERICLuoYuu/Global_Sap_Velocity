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
    all_data = pd.read_csv('data/processed/merged/site/merged_data.csv')
    print(all_data.shape)
    # randomly split the data into training and testing
    training_data = all_data.sample(frac=0.8, random_state=42)
    print(training_data.shape)
    testing_data = all_data.drop(training_data.index)
    # save the testing data and training data
    training_data.to_csv('data/processed/merged/site/training/training_data.csv', index=False)
    testing_data.to_csv('data/processed/merged/site/testing/testing_data.csv', index=False)
    
    training_data = training_data.set_index('TIMESTAMP').sort_index()
    print("training shape:",training_data.shape)
    training_data = training_data[['ta_mean', 'ws_mean', 'precip_sum', 'vpd_mean', 'sw_in_mean', 'sap_velocity']]
    training_data = training_data.dropna()
    print("Dataset shape:", training_data.shape)
    y = training_data['sap_velocity']
    X = training_data.drop(columns=['sap_velocity'])
    
    
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    
    
    # Initialize and fit GHGA selector
    selector = GHGA_selector(
        population_size=10,
        generations=10,
        mutation_rate=0.1,
        local_search_prob=0.1,
        elite_size=1,
        random_state=42,
        
    )
    
    # Fit selector
    selector.fit(X_train_scaled, y)
    
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
    # save the feature importance plot
    plt.savefig('plots/feature_importance_plot.png')

    # Plot convergence curve
    selector.plot_convergence(save_path='plots/convergence_plot.png')
    
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
    
    print("\nTransformed data shapes:")
    print("Training data:", X_train_selected.shape)
    

if __name__ == "__main__":
    main()