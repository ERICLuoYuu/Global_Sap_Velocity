"""
Plot data distributions for different folds in cross-validation.
This script provides visualization functions to understand data distribution across CV folds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

class FoldDistributionAnalyzer:
    """
    Analyzer for tracking and visualizing data distributions across CV folds.
    """
    
    def __init__(self, output_dir='./outputs/distributions'):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save distribution plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store distributions for each fold
        self.fold_data = []
        
    def capture_fold_data(self, fold_idx, X_train, y_train, X_test, y_test, 
                         feature_names=None, groups_train=None, groups_test=None):
        """
        Capture data from a single fold for later analysis.
        
        Parameters:
        -----------
        fold_idx : int
            Fold index
        X_train, X_test : numpy arrays
            Feature data for train and test
        y_train, y_test : numpy arrays
            Target data for train and test
        feature_names : list, optional
            Names of features
        groups_train, groups_test : numpy arrays, optional
            Spatial group assignments
        """
        fold_info = {
            'fold_idx': fold_idx,
            'X_train': X_train,
            'y_train': y_train.flatten(),
            'X_test': X_test,
            'y_test': y_test.flatten(),
            'groups_train': groups_train,
            'groups_test': groups_test,
            'feature_names': feature_names
        }
        self.fold_data.append(fold_info)
        
    def plot_target_distributions(self, figsize=(16, 10)):
        """
        Plot target variable distributions across all folds.
        Shows both train and test distributions for each fold.
        """
        n_folds = len(self.fold_data)
        fig, axes = plt.subplots(n_folds, 2, figsize=figsize)
        
        if n_folds == 1:
            axes = axes.reshape(1, -1)
        
        for i, fold_info in enumerate(self.fold_data):
            fold_idx = fold_info['fold_idx']
            y_train = fold_info['y_train']
            y_test = fold_info['y_test']
            
            # Train distribution
            axes[i, 0].hist(y_train, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[i, 0].axvline(np.mean(y_train), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(y_train):.3f}')
            axes[i, 0].axvline(np.median(y_train), color='green', linestyle='--',
                              label=f'Median: {np.median(y_train):.3f}')
            axes[i, 0].set_title(f'Fold {fold_idx + 1} - Train (n={len(y_train)})')
            axes[i, 0].set_xlabel('Target Value')
            axes[i, 0].set_ylabel('Frequency')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Test distribution
            axes[i, 1].hist(y_test, bins=50, alpha=0.7, color='orange', edgecolor='black')
            axes[i, 1].axvline(np.mean(y_test), color='red', linestyle='--',
                              label=f'Mean: {np.mean(y_test):.3f}')
            axes[i, 1].axvline(np.median(y_test), color='green', linestyle='--',
                              label=f'Median: {np.median(y_test):.3f}')
            axes[i, 1].set_title(f'Fold {fold_idx + 1} - Test (n={len(y_test)})')
            axes[i, 1].set_xlabel('Target Value')
            axes[i, 1].set_ylabel('Frequency')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'target_distributions_by_fold.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / 'target_distributions_by_fold.png'}")
        
    def plot_target_distribution_comparison(self, figsize=(14, 6)):
        """
        Compare target distributions across all folds in a single plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Collect all distributions
        for fold_info in self.fold_data:
            fold_idx = fold_info['fold_idx']
            axes[0].hist(fold_info['y_train'], bins=50, alpha=0.5, 
                        label=f"Fold {fold_idx + 1}", edgecolor='black')
            axes[1].hist(fold_info['y_test'], bins=50, alpha=0.5,
                        label=f"Fold {fold_idx + 1}", edgecolor='black')
        
        axes[0].set_title('Train Set - Target Distribution Across Folds')
        axes[0].set_xlabel('Target Value')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('Test Set - Target Distribution Across Folds')
        axes[1].set_xlabel('Target Value')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'target_distribution_overlay.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / 'target_distribution_overlay.png'}")
        
    def plot_train_test_comparison(self, figsize=(15, 5)):
        """
        Plot train vs test distribution for each fold side by side using violin plots.
        """
        n_folds = len(self.fold_data)
        fig, axes = plt.subplots(1, n_folds, figsize=figsize, sharey=True)
        
        if n_folds == 1:
            axes = [axes]
        
        for i, fold_info in enumerate(self.fold_data):
            fold_idx = fold_info['fold_idx']
            
            # Prepare data for violin plot
            train_df = pd.DataFrame({
                'Value': fold_info['y_train'],
                'Set': 'Train'
            })
            test_df = pd.DataFrame({
                'Value': fold_info['y_test'],
                'Set': 'Test'
            })
            combined_df = pd.concat([train_df, test_df])
            
            # Create violin plot
            sns.violinplot(data=combined_df, x='Set', y='Value', ax=axes[i], palette=['blue', 'orange'])
            axes[i].set_title(f'Fold {fold_idx + 1}')
            axes[i].set_xlabel('')
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add sample sizes
            axes[i].text(0, axes[i].get_ylim()[0], f'n={len(fold_info["y_train"])}',
                        ha='center', va='top', fontsize=9)
            axes[i].text(1, axes[i].get_ylim()[0], f'n={len(fold_info["y_test"])}',
                        ha='center', va='top', fontsize=9)
        
        axes[0].set_ylabel('Target Value')
        plt.suptitle('Train vs Test Distribution Comparison Across Folds', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'train_test_violin_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / 'train_test_violin_comparison.png'}")
        
    def plot_distribution_statistics(self):
        """
        Create a summary table of distribution statistics for each fold.
        """
        stats_data = []
        
        for fold_info in self.fold_data:
            fold_idx = fold_info['fold_idx']
            y_train = fold_info['y_train']
            y_test = fold_info['y_test']
            
            # Calculate statistics
            train_stats = {
                'Fold': fold_idx + 1,
                'Set': 'Train',
                'Count': len(y_train),
                'Mean': np.mean(y_train),
                'Std': np.std(y_train),
                'Min': np.min(y_train),
                'Q25': np.percentile(y_train, 25),
                'Median': np.median(y_train),
                'Q75': np.percentile(y_train, 75),
                'Max': np.max(y_train),
                'Skewness': stats.skew(y_train),
                'Kurtosis': stats.kurtosis(y_train)
            }
            
            test_stats = {
                'Fold': fold_idx + 1,
                'Set': 'Test',
                'Count': len(y_test),
                'Mean': np.mean(y_test),
                'Std': np.std(y_test),
                'Min': np.min(y_test),
                'Q25': np.percentile(y_test, 25),
                'Median': np.median(y_test),
                'Q75': np.percentile(y_test, 75),
                'Max': np.max(y_test),
                'Skewness': stats.skew(y_test),
                'Kurtosis': stats.kurtosis(y_test)
            }
            
            stats_data.extend([train_stats, test_stats])
        
        stats_df = pd.DataFrame(stats_data)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=stats_df.round(3).values,
                        colLabels=stats_df.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.06, 0.06, 0.07, 0.09, 0.09, 0.08, 0.08, 0.09, 0.08, 0.08, 0.11, 0.11])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color code the header
        for i in range(len(stats_df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code train/test rows
        for i in range(1, len(stats_df) + 1):
            if stats_df.iloc[i-1]['Set'] == 'Train':
                for j in range(len(stats_df.columns)):
                    table[(i, j)].set_facecolor('#d4e6f1')
            else:
                for j in range(len(stats_df.columns)):
                    table[(i, j)].set_facecolor('#fdebd0')
        
        plt.title('Distribution Statistics Across Folds', fontsize=14, weight='bold', pad=20)
        plt.savefig(self.output_dir / 'distribution_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save as CSV
        stats_df.to_csv(self.output_dir / 'distribution_statistics.csv', index=False)
        print(f"Saved: {self.output_dir / 'distribution_statistics.png'}")
        print(f"Saved: {self.output_dir / 'distribution_statistics.csv'}")
        
        return stats_df
    
    def plot_feature_distributions(self, max_features=10, figsize=(16, 12)):
        """
        Plot distributions of features across folds.
        Shows the first max_features features.
        """
        if len(self.fold_data) == 0:
            print("No fold data captured yet.")
            return
        
        # Get feature information from first fold
        first_fold = self.fold_data[0]
        X_sample = first_fold['X_train']
        
        # Handle windowed data (3D) vs non-windowed data (2D)
        if len(X_sample.shape) == 3:
            # Windowed: (samples, timesteps, features)
            n_features = X_sample.shape[2]
            # Flatten to (samples * timesteps, features)
            feature_names = first_fold['feature_names'] if first_fold['feature_names'] else [f'Feature {i}' for i in range(n_features)]
        else:
            # Non-windowed: (samples, features)
            n_features = X_sample.shape[1]
            feature_names = first_fold['feature_names'] if first_fold['feature_names'] else [f'Feature {i}' for i in range(n_features)]
        
        # Limit number of features to plot
        n_to_plot = min(max_features, n_features)
        n_folds = len(self.fold_data)
        
        fig, axes = plt.subplots(n_to_plot, n_folds, figsize=figsize)
        
        if n_to_plot == 1:
            axes = axes.reshape(1, -1)
        if n_folds == 1:
            axes = axes.reshape(-1, 1)
        
        for feat_idx in range(n_to_plot):
            for fold_idx, fold_info in enumerate(self.fold_data):
                X_train = fold_info['X_train']
                X_test = fold_info['X_test']
                
                # Extract feature data
                if len(X_train.shape) == 3:
                    # Flatten windowed data
                    train_feature = X_train[:, :, feat_idx].flatten()
                    test_feature = X_test[:, :, feat_idx].flatten()
                else:
                    train_feature = X_train[:, feat_idx]
                    test_feature = X_test[:, feat_idx]
                
                ax = axes[feat_idx, fold_idx]
                
                # Plot histograms
                ax.hist(train_feature, bins=30, alpha=0.5, color='blue', label='Train', density=True)
                ax.hist(test_feature, bins=30, alpha=0.5, color='orange', label='Test', density=True)
                
                if feat_idx == 0:
                    ax.set_title(f'Fold {fold_info["fold_idx"] + 1}', fontsize=10)
                if fold_idx == 0:
                    ax.set_ylabel(feature_names[feat_idx], fontsize=9)
                if feat_idx == n_to_plot - 1 and fold_idx == n_folds // 2:
                    ax.legend(loc='upper right', fontsize=8)
                
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
        
        plt.suptitle('Feature Distributions Across Folds (Train vs Test)', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / 'feature_distributions.png'}")
    
    def check_distribution_shift(self):
        """
        Perform statistical tests to check for distribution shifts between train and test.
        Uses Kolmogorov-Smirnov test.
        """
        results = []
        
        for fold_info in self.fold_data:
            fold_idx = fold_info['fold_idx']
            y_train = fold_info['y_train']
            y_test = fold_info['y_test']
            
            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(y_train, y_test)
            
            # Calculate difference in means and stds
            mean_diff = abs(np.mean(y_train) - np.mean(y_test))
            std_diff = abs(np.std(y_train) - np.std(y_test))
            
            results.append({
                'Fold': fold_idx + 1,
                'KS_Statistic': ks_statistic,
                'P_Value': p_value,
                'Significant_Shift': 'Yes' if p_value < 0.05 else 'No',
                'Mean_Difference': mean_diff,
                'Std_Difference': std_diff
            })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'distribution_shift_tests.csv', index=False)
        print(f"\nDistribution Shift Analysis:")
        print(results_df.to_string(index=False))
        print(f"\nSaved: {self.output_dir / 'distribution_shift_tests.csv'}")
        
        return results_df
    
    def generate_all_plots(self):
        """
        Generate all available plots and statistics.
        """
        print("\n" + "="*60)
        print("Generating Distribution Analysis Plots")
        print("="*60 + "\n")
        
        self.plot_target_distributions()
        self.plot_target_distribution_comparison()
        self.plot_train_test_comparison()
        self.plot_distribution_statistics()
        self.plot_feature_distributions()
        self.check_distribution_shift()
        
        print("\n" + "="*60)
        print(f"All plots saved to: {self.output_dir}")
        print("="*60 + "\n")


# Example usage showing how to integrate with your code
if __name__ == "__main__":
    # This is an example of how to use it
    # In your actual code, you would integrate this into your CV loop
    
    # Create analyzer
    analyzer = FoldDistributionAnalyzer(output_dir='./outputs/distributions')
    
    # Example with dummy data
    np.random.seed(42)
    n_folds = 3
    
    for fold_idx in range(n_folds):
        # Simulate some data with slight distribution shift
        X_train = np.random.randn(1000, 10) + fold_idx * 0.1
        y_train = np.random.randn(1000) * 2 + 5 + fold_idx * 0.2
        X_test = np.random.randn(200, 10) + fold_idx * 0.15
        y_test = np.random.randn(200) * 2 + 5 + fold_idx * 0.3
        
        # Capture fold data
        analyzer.capture_fold_data(
            fold_idx=fold_idx,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=[f'Feature_{i}' for i in range(10)]
        )
    
    # Generate all plots
    analyzer.generate_all_plots()