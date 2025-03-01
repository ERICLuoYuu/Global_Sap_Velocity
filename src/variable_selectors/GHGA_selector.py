import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from scipy.special import softmax
from typing import List, Dict, Union, Optional
import warnings
from sklearn.exceptions import NotFittedError
from src.cross_validation.cross_validators import MLCrossValidator

class GHGA_selector:
    """
    Guided Hybrid Genetic Algorithm for feature selection.
    
    This class implements a sophisticated feature selection approach that combines
    genetic algorithms with domain-specific guidance and local search optimization.
    
    Parameters:
    -----------
    population_size : int, default=50
        Size of the population in each generation
    generations : int, default=100
        Number of generations to evolve
    mutation_rate : float, default=0.1
        Probability of mutation for each gene
    local_search_prob : float, default=0.1
        Probability of applying local search to each offspring
    elite_size : int, default=1
        Number of best solutions to preserve across generations
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        local_search_prob: float = 0.1,
        elite_size: int = 1,
        random_state: Optional[int] = None,
        cv_method: str = 'random',
        groups: Union[np.ndarray, pd.Series, List] = None
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.local_search_prob = local_search_prob
        self.elite_size = elite_size
        self.cv_method = cv_method
        self.groups = groups
        
        if random_state is not None:
            np.random.seed(random_state)
            
        self.best_solution_ = None
        self.best_solutions_ = []
        self.best_fitness_ = float('-inf')
        self.feature_names_ = None
        self.feature_importance_ = None
        # List to store the best fitness score of each generation
        self.convergence_metrics_ = {
            'best_fitness': [],     # Best fitness in each generation
            'mean_fitness': [],     # Average fitness in each generation
            'population_diversity': [],  # Diversity measure of population
            'selected_features_count': []  # Number of selected features in best solution
        }
    
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray]
    ) -> 'GHGA_selector':
        """
        Fit the GHGA selector to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : object
            Returns self.
        """
        # Convert input data to DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
            
        self.feature_names_ = X.columns.tolist()
        self.n_features_ = len(self.feature_names_)
        
        # Store data for fitness evaluation
        self.X_ = X
        self.y_ = y
        
        # Initialize population
        population = self._initialize_population()
        
        # Main evolution loop
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._fitness_function(chrom) for chrom in population]
            
            # Update convergence metrics
            self._update_convergence_metrics(population, fitness_scores)

            # Update best solution
            max_idx = np.argmax(fitness_scores)
            self.best_solutions_.append(population[max_idx].copy())
            if fitness_scores[max_idx] > self.best_fitness_:
                self.best_fitness_ = fitness_scores[max_idx]
                self.best_solution_ = population[max_idx].copy()
            
            print(self.population_size, len(fitness_scores), len(population))
            
            # Selection
            parents_idx = np.random.choice(
                len(population),
                size=self.population_size,
                p=softmax(fitness_scores)
            )
            parents = population[parents_idx]
            
            # Create new population
            new_population = []
            for i in range(0, self.population_size - self.elite_size, 2):
                parent1, parent2 = parents[i], parents[i+1]
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)
                
                # Local search
                if np.random.random() < self.local_search_prob:
                    child1 = self._local_search(child1)
                if np.random.random() < self.local_search_prob:
                    child2 = self._local_search(child2)
                    
                new_population.extend([child1, child2])
            
            # Add elite solutions
            elite_idx = np.argsort(fitness_scores)[-self.elite_size:]
            elite_solutions = population[elite_idx]
            new_population.extend(elite_solutions)
            
            population = np.array(new_population)
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        return self
    # Add a method to update convergence metrics
    def _update_convergence_metrics(self, population, fitness_scores):
        """
        Update convergence metrics for the current generation.
        
        Parameters:
        -----------
        population : np.ndarray
            Current population of solutions
        fitness_scores : List[float]
            Fitness scores for current population
        """
        # Best fitness
        best_fitness = np.max(fitness_scores)
        self.convergence_metrics_['best_fitness'].append(best_fitness)
        
        # Mean fitness
        mean_fitness = np.mean(fitness_scores)
        self.convergence_metrics_['mean_fitness'].append(mean_fitness)
        
        # Population diversity (using Hamming distance)
        diversity = np.mean([
            np.mean([np.sum(p1 != p2) for p2 in population])
            for p1 in population
        ])
        self.convergence_metrics_['population_diversity'].append(diversity)
        
        # Number of selected features in best solution
        best_solution = population[np.argmax(fitness_scores)]
        n_selected = np.sum(best_solution)
        self.convergence_metrics_['selected_features_count'].append(n_selected)

    def _initialize_population(self) -> np.ndarray:
        """
        Initialize population with both random and guided solutions.
        
        Returns:
        --------
        np.ndarray
            Initial population of chromosomes
        """
        population = []
        
        # Create random solutions
        for _ in range(self.population_size - 1):  # -1 to leave room for guided solution
            chromosome = np.random.randint(2, size=self.n_features_)
            population.append(chromosome)
        
        # Add guided solution based on correlation
        if isinstance(self.X_, pd.DataFrame) and isinstance(self.y_, pd.Series):
            correlations = self.X_.corrwith(self.y_).abs()
            # Select features with correlation above median
            median_corr = correlations.median()
            guided_chromosome = np.zeros(self.n_features_)
            for i, corr in enumerate(correlations):
                if corr >= median_corr:
                    guided_chromosome[i] = 1
            population.append(guided_chromosome)
        else:
            # If not pandas objects, add random solution
            population.append(np.random.randint(2, size=self.n_features_))
        
        return np.array(population)
    
    def _fitness_function(self, chromosome: np.ndarray) -> float:
        """
        Evaluate solution quality using cross-validation.
        
        Parameters:
        -----------
        chromosome : np.ndarray
            Binary array indicating selected features
            
        Returns:
        --------
        float
            Fitness score
        """
        # Get selected features
        selected = np.where(chromosome == 1)[0]
        if len(selected) == 0:
            return float('-inf')
        
        # Extract selected features
        X_selected = self.X_.iloc[:, selected] if isinstance(self.X_, pd.DataFrame) else self.X_[:, selected]
        
        try:
            # Perform cross-validation
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            cross_validator = MLCrossValidator(estimator=model, scoring='r2', n_splits=5)
            if self.cv_method == 'temporal':
                scores = cross_validator.temporal_cv(X_selected, self.y_, groups = self.groups)
            elif self.cv_method == 'random':
                scores = cross_validator.random_cv(X_selected, self.y_)
            elif self.cv_method == 'spatial':
                scores = cross_validator.spatial_cv(X_selected, self.y_, self.groups)
            
            print(scores)
            
            # Calculate fitness with penalty for number of features
            cv_score = np.mean(scores)
            penalty = 0.01 * len(selected) / self.n_features_  # Normalized penalty
            
            return cv_score - penalty
            
        except Exception as e:
            warnings.warn(f"Error in fitness evaluation: {str(e)}")
            return float('-inf')
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        """
        Perform two-point crossover between parents.
        
        Parameters:
        -----------
        parent1, parent2 : np.ndarray
            Parent chromosomes
            
        Returns:
        --------
        tuple
            Two child chromosomes
        """
        points = sorted(np.random.choice(self.n_features_, 2, replace=False))
        
        child1 = np.concatenate([
            parent1[:points[0]],
            parent2[points[0]:points[1]],
            parent1[points[1]:]
        ])
        
        child2 = np.concatenate([
            parent2[:points[0]],
            parent1[points[0]:points[1]],
            parent2[points[1]:]
        ])
        
        return child1, child2
    
    def _mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Perform bit-flip mutation.
        
        Parameters:
        -----------
        chromosome : np.ndarray
            Input chromosome
            
        Returns:
        --------
        np.ndarray
            Mutated chromosome
        """
        for i in range(len(chromosome)):
            if np.random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome
    
    def _local_search(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Implement hill climbing to improve solution.
        
        Parameters:
        -----------
        chromosome : np.ndarray
            Input chromosome
            
        Returns:
        --------
        np.ndarray
            Improved chromosome
        """
        improved = True
        best_fitness = self._fitness_function(chromosome)
        
        while improved:
            improved = False
            for i in range(len(chromosome)):
                # Try flipping each bit
                chromosome[i] = 1 - chromosome[i]
                new_fitness = self._fitness_function(chromosome)
                
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    improved = True
                else:
                    # Revert if no improvement
                    chromosome[i] = 1 - chromosome[i]
        
        return chromosome
    
    def _calculate_feature_importance(self):
        """
        Calculate feature importance scores based on multiple criteria:
        1. Selection frequency across top solutions
        2. Correlation with target variable
        3. Impact on model performance
        """
        # Initialize importance scores
        self.feature_importance_ = pd.Series(np.zeros(self.n_features_), 
                                        index=self.feature_names_)
        
        # Get top solutions (e.g., top 10% of solutions encountered)
        n_top = max(int(self.population_size * 0.1), 1)
        top_solutions = []
        top_fitness = []
        
        # Evaluate current population
        for chromosome in self.best_solutions_:
            fitness = self._fitness_function(chromosome)
            if len(top_solutions) < n_top:
                top_solutions.append(chromosome)
                top_fitness.append(fitness)
            else:
                min_idx = np.argmin(top_fitness)
                if fitness > top_fitness[min_idx]:
                    top_solutions[min_idx] = chromosome
                    top_fitness[min_idx] = fitness
        
        # Calculate importance based on multiple criteria
        for i, feature in enumerate(self.feature_names_):
            # 1. Selection frequency in top solutions
            selection_freq = sum(solution[i] for solution in top_solutions) / len(top_solutions)
            
            # 2. Correlation with target (if available)
            if isinstance(self.X_, pd.DataFrame) and isinstance(self.y_, pd.Series):
                correlation = abs(self.X_[feature].corr(self.y_))
            else:
                correlation = abs(np.corrcoef(self.X_[:, i], self.y_)[0, 1])
                
            # 3. Performance impact
            # Evaluate performance drop when removing the feature
            base_mask = self.best_solution_.copy()
            if base_mask[i] == 1:  # If feature is selected in best solution
                base_mask[i] = 0  # Remove feature
                performance_impact = self.best_fitness_ - self._fitness_function(base_mask)
            else:
                performance_impact = 0
                
            # Combine metrics into final importance score
            # Weights can be adjusted based on specific needs
            importance = (0.4 * selection_freq + 
                        0.3 * correlation + 
                        0.3 * max(0, performance_impact))
            
            self.feature_importance_[feature] = importance
        
        # Normalize importance scores to [0, 1] range
        self.feature_importance_ = (self.feature_importance_ - self.feature_importance_.min()) / \
                                (self.feature_importance_.max() - self.feature_importance_.min())

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        Get a boolean mask or integer indices of selected features.
        
        Parameters:
        -----------
        indices : bool, default=False
            If True, returns integer indices of selected features.
            If False, returns a boolean mask indicating selected features.
            
        Returns:
        --------
        np.ndarray or List[int]
            Selected feature mask or indices
            
        Raises:
        -------
        NotFittedError
            If the selector hasn't been fitted yet
        """
        if self.best_solution_ is None:
            raise NotFittedError("Selector has not been fitted yet.")
            
        if indices:
            return np.where(self.best_solution_ == 1)[0].tolist()
        return self.best_solution_ == 1
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of selected features.
        
        Returns:
        --------
        List[str]
            Names of selected features
            
        Raises:
        -------
        NotFittedError
            If the selector hasn't been fitted yet
        """
        if self.feature_names_ is None:
            raise NotFittedError("Selector has not been fitted yet.")
            
        mask = self.get_support()
        return [self.feature_names_[i] for i in range(len(mask)) if mask[i]]
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Reduce X to selected features.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        Union[pd.DataFrame, np.ndarray]
            X with selected features
            
        Raises:
        -------
        NotFittedError
            If the selector hasn't been fitted yet
        ValueError
            If X has wrong shape
        """
        if self.best_solution_ is None:
            raise NotFittedError("Selector has not been fitted yet.")
            
        # Check input dimensions
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, but GHGA_selector was "
                           f"trained with {self.n_features_} features.")
        
        # Get selected feature mask
        mask = self.get_support()
        
        # Transform data
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, mask]
        return X[:, mask]
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], 
                     y: Union[pd.Series, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit the selector and reduce X to selected features.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        Union[pd.DataFrame, np.ndarray]
            X with selected features
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get importance scores for all features.
        
        Returns:
        --------
        pd.Series
            Feature importance scores
            
        Raises:
        -------
        NotFittedError
            If the selector hasn't been fitted yet
        """
        if self.feature_importance_ is None:
            raise NotFittedError("Selector has not been fitted yet.")
        return self.feature_importance_
    
    def get_params(self) -> Dict:
        """
        Get parameters of the selector.
        
        Returns:
        --------
        Dict
            Parameter names mapped to their values
        """
        return {
            'population_size': self.population_size,
            'generations': self.generations,
            'mutation_rate': self.mutation_rate,
            'local_search_prob': self.local_search_prob,
            'elite_size': self.elite_size
        }
    
    def get_convergence_curve(self) -> Dict[str, List[float]]:
        """
        Get the convergence metrics tracked during optimization.
        
        Returns:
        --------
        Dict[str, List[float]]
            Dictionary containing different convergence metrics:
            - best_fitness: Best fitness score in each generation
            - mean_fitness: Average fitness score in each generation
            - population_diversity: Measure of population diversity
            - selected_features_count: Number of selected features in best solution
            
        Raises:
        -------
        NotFittedError
            If the selector hasn't been fitted yet
        """
        if not self.convergence_metrics_['best_fitness']:
            raise NotFittedError("Selector has not been fitted yet.")
        return self.convergence_metrics_
    
    def plot_convergence(self, plot_type: str = 'matplotlib', save_path: str = None):
        """
        Plot convergence metrics using either matplotlib or plotly.
        
        Parameters:
        -----------
        plot_type : str, default='matplotlib'
            Type of plot to generate ('matplotlib' or 'plotly')
        save_path : str, optional
            Path to save the plot. If None, displays the plot
        
        Returns:
        --------
        None or go.Figure
            Returns plotly figure object if plot_type is 'plotly'
        """
        if not self.convergence_metrics_['best_fitness']:
            raise NotFittedError("No convergence data available. Run fit() first.")
        
        generations = range(len(self.convergence_metrics_['best_fitness']))
        
        if plot_type == 'matplotlib':
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Fitness plot
            ax1.plot(generations, self.convergence_metrics_['best_fitness'], 
                    label='Best Fitness', color='blue')
            ax1.plot(generations, self.convergence_metrics_['mean_fitness'], 
                    label='Mean Fitness', color='orange', alpha=0.7)
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness Score')
            ax1.set_title('Fitness Convergence over Generations')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Diversity and feature count plot
            ax2.plot(generations, self.convergence_metrics_['population_diversity'], 
                    label='Population Diversity', color='green')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Diversity Score')
            ax2.set_title('Population Diversity and Selected Features')
            ax2.grid(True, alpha=0.3)
            
            # Add feature count on secondary y-axis
            ax3 = ax2.twinx()
            ax3.plot(generations, self.convergence_metrics_['selected_features_count'], 
                    label='Selected Features', color='red', linestyle='--')
            ax3.set_ylabel('Number of Selected Features')
            
            # Combine legends
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax3.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        elif plot_type == 'plotly':
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(rows=2, cols=1, 
                            subplot_titles=('Fitness Convergence', 
                                            'Population Diversity and Selected Features'))
            
            # Fitness plot
            fig.add_trace(
                go.Scatter(x=generations, y=self.convergence_metrics_['best_fitness'],
                        name='Best Fitness', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=generations, y=self.convergence_metrics_['mean_fitness'],
                        name='Mean Fitness', line=dict(color='orange')),
                row=1, col=1
            )
            
            # Diversity plot
            fig.add_trace(
                go.Scatter(x=generations, y=self.convergence_metrics_['population_diversity'],
                        name='Population Diversity', line=dict(color='green')),
                row=2, col=1
            )
            
            # Feature count plot
            fig.add_trace(
                go.Scatter(x=generations, y=self.convergence_metrics_['selected_features_count'],
                        name='Selected Features', line=dict(color='red', dash='dash'),
                        yaxis='y3'),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text='GHGA Convergence Metrics',
                yaxis=dict(title='Fitness Score'),
                yaxis2=dict(title='Diversity Score'),
                yaxis3=dict(title='Number of Selected Features',
                        overlaying='y2', side='right')
            )
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
        
        else:
            raise ValueError("plot_type must be either 'matplotlib' or 'plotly'")
    
    def summary(self) -> Dict:
        """
        Get a summary of the feature selection results.
        
        Returns:
        --------
        Dict
            Summary statistics of the selection process
            
        Raises:
        -------
        NotFittedError
            If the selector hasn't been fitted yet
        """
        if self.best_solution_ is None:
            raise NotFittedError("Selector has not been fitted yet.")
            
        selected_features = self.get_feature_names()
        
        return {
            'n_total_features': self.n_features_,
            'n_selected_features': len(selected_features),
            'selected_features': selected_features,
            'best_fitness': self.best_fitness_,
            'feature_importance': self.get_feature_importance(),
            'convergence': {
                'initial_fitness': self.convergence_metrics_['best_fitness'][0],
                'final_fitness': self.convergence_metrics_['best_fitness'][-1],
                'n_generations': len(self.convergence_metrics_['best_fitness'])
            }
        }