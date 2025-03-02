import sys
from pathlib import Path
import numpy as np
import logging
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from scipy.special import softmax
from typing import List, Dict, Union, Optional, Tuple, Callable
import warnings
from functools import lru_cache
import time
import os
import pickle
from datetime import datetime
from sklearn.exceptions import NotFittedError
from src.cross_validation.cross_validators import MLCrossValidator

# Parallel processing imports
import multiprocessing
from joblib import Parallel, delayed

# Optional GPU imports - will be checked at runtime
try:
    import cupy as cp
    import cudf
    import cuml
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

class GHGASelector:
    """
    Guided Hybrid Genetic Algorithm for feature selection.
    Optimized for high-performance computing environments.
    
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
    feature_penalty : float, default=0.01
        Penalty coefficient for the number of selected features
    max_stored_solutions : int, default=100
        Maximum number of best solutions to store
    random_state : int, optional
        Random seed for reproducibility
    cv_method : str, default='random'
        Cross-validation method ('random', 'temporal', 'spatial')
    log_dir : str, optional
        Directory to store log files
    groups : array-like, optional
        Group labels for samples used in spatial or temporal cross-validation
    n_jobs : int, default=-1
        Number of CPU cores to use (-1 means all cores)
    use_gpu : bool, default=False
        Whether to use GPU acceleration if available
    gpu_id : int, default=0
        ID of the GPU to use if multiple are available
    checkpoint_interval : int, default=10
        Save checkpoint every N generations (0 to disable)
    checkpoint_dir : str, optional
        Directory to save checkpoints
    fitness_batch_size : int, default=10
        Number of chromosomes to evaluate in parallel
    """
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        local_search_prob: float = 0.1,
        elite_size: int = 1,
        feature_penalty: float = 0.01,
        max_stored_solutions: int = 100,
        random_state: Optional[int] = None,
        cv_method: str = 'random',
        log_dir: Optional[str] = None,
        groups: Optional[Union[np.ndarray, pd.Series, List]] = None,
        n_jobs: int = -1,
        use_gpu: bool = False,
        gpu_id: int = 0,
        checkpoint_interval: int = 10,
        checkpoint_dir: Optional[str] = None,
        fitness_batch_size: int = 10
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.local_search_prob = local_search_prob
        self.elite_size = elite_size
        self.feature_penalty = feature_penalty
        self.max_stored_solutions = max_stored_solutions
        self.cv_method = cv_method
        self.groups = groups
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu and HAS_GPU
        self.gpu_id = gpu_id
        self.checkpoint_interval = checkpoint_interval
        self.fitness_batch_size = fitness_batch_size
        
        # Setup GPU if requested and available
        if self.use_gpu:
            if HAS_GPU:
                try:
                    cp.cuda.Device(self.gpu_id).use()
                    self.xp = cp  # Use cupy for array operations
                    self.logger.info(f"Using GPU {self.gpu_id}")
                except Exception as e:
                    self.use_gpu = False
                    self.xp = np
                    warnings.warn(f"Failed to initialize GPU {self.gpu_id}: {str(e)}. Falling back to CPU.")
            else:
                self.use_gpu = False
                self.xp = np
                warnings.warn("GPU requested but required libraries not found. Install cupy, cudf, and cuml for GPU support.")
        else:
            self.xp = np
        
        # Determine number of jobs for parallel processing
        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        
        # Setup logging
        if log_dir is not None:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path('./outputs/logs/var_select_logs')
            
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / 'ghga_selector.log'
        
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('GHGASelector')
        
        # Setup checkpoint directory
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = Path('./outputs/checkpoints')
        
        if self.checkpoint_interval > 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Checkpoints will be saved to {self.checkpoint_dir} every {self.checkpoint_interval} generations")

        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
            if self.use_gpu:
                cp.random.seed(random_state)
            self.random_state = random_state
        else:
            self.random_state = None
            
        # Initialize state variables
        self.best_solution_ = None
        self.best_solutions_ = []
        self.best_fitness_ = float('-inf')
        self.feature_names_ = None
        self.feature_importance_ = None
        self._fitness_cache = {}  # Cache for fitness values
        self.current_generation = 0
        
        # Convergence metrics
        self.convergence_metrics_ = {
            'best_fitness': [],     # Best fitness in each generation
            'mean_fitness': [],     # Average fitness in each generation
            'population_diversity': [],  # Diversity measure of population
            'selected_features_count': [],  # Number of selected features in best solution
            'generation_time': []   # Time taken for each generation
        }
        
        # Log initialization
        self.logger.info(f"Initialized GHGASelector with {self.n_jobs} CPU cores and GPU={self.use_gpu}")
    
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        resume_from: Optional[str] = None
    ) -> 'GHGASelector':
        """
        Fit the GHGA selector to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        resume_from : str, optional
            Path to checkpoint file to resume from
            
        Returns:
        --------
        self : object
            Returns self.
        """
        start_time = time.time()
        self.logger.info(f"Starting GHGA feature selection with {self.population_size} population size and {self.generations} generations")
        
        # Clear fitness cache
        self._fitness_cache = {}
        
        # Convert input data to DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
            
        self.feature_names_ = X.columns.tolist()
        self.n_features_ = len(self.feature_names_)
        self.logger.info(f"Initialized selection with {self.n_features_} features")
        
        # Transfer data to GPU if using it
        if self.use_gpu:
            try:
                self.logger.info("Moving data to GPU...")
                self.X_ = X  # Keep original data for reference
                self.y_ = y
                # Note: We're not actually moving to GPU here because the MLCrossValidator
                # might not support GPU data. We'll convert specific operations as needed.
            except Exception as e:
                self.use_gpu = False
                self.xp = np
                self.logger.warning(f"Failed to move data to GPU: {str(e)}. Falling back to CPU.")
                self.X_ = X
                self.y_ = y
        else:
            self.X_ = X
            self.y_ = y
        
        # Resume from checkpoint if specified
        if resume_from:
            try:
                self._load_checkpoint(resume_from)
                self.logger.info(f"Resumed from checkpoint: {resume_from}")
                population = self.current_population
                start_gen = self.current_generation
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint: {str(e)}. Starting from beginning.")
                population = self._initialize_population()
                start_gen = 0
        else:
            # Initialize population
            population = self._initialize_population()
            start_gen = 0
            
        self.logger.info(f"Population initialized with {len(population)} chromosomes")
        
        # Main evolution loop
        for generation in range(start_gen, self.generations):
            gen_start_time = time.time()
            self.current_generation = generation
            
            self.logger.info(f"Generation {generation+1}/{self.generations} started")
            
            # Evaluate fitness in parallel
            fitness_scores = self._parallel_fitness_evaluation(population)
            
            # Update convergence metrics
            self._update_convergence_metrics(population, fitness_scores)
            
            # Save current population for checkpointing
            self.current_population = population
            
            # Check if we have valid solutions
            if np.all(np.isinf(fitness_scores)):
                self.logger.warning("All solutions have -inf fitness. Reinitializing population.")
                population = self._initialize_population()
                continue
            
            # Update best solution
            max_idx = np.argmax(fitness_scores)
            
            # Store best solution for this generation (limiting number stored)
            if len(self.best_solutions_) >= self.max_stored_solutions:
                self.best_solutions_.pop(0)  # Remove oldest solution
            self.best_solutions_.append(population[max_idx].copy())
            
            current_gen_best_fitness = fitness_scores[max_idx]
            if fitness_scores[max_idx] > self.best_fitness_:
                old_fitness = self.best_fitness_
                self.best_fitness_ = current_gen_best_fitness
                self.best_solution_ = population[max_idx].copy()
                n_selected = np.sum(self.best_solution_)
                self.logger.info(f"New best solution found in generation {generation+1}: fitness improved from {old_fitness:.4f} to {self.best_fitness_:.4f} with {n_selected} features")
            
            # Log generation results
            avg_fitness = np.mean(fitness_scores[~np.isinf(fitness_scores)]) if np.any(~np.isinf(fitness_scores)) else float('-inf')
            gen_time = time.time() - gen_start_time
            self.convergence_metrics_['generation_time'].append(gen_time)
            self.logger.info(f"Generation {generation+1} completed in {gen_time:.2f}s: best fitness = {current_gen_best_fitness:.4f}, avg fitness = {avg_fitness:.4f}")
            
            # Save checkpoint if needed
            if self.checkpoint_interval > 0 and (generation + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint()
            
            # Check if we've reached the final generation
            if generation == self.generations - 1:
                break
                
            # Selection - use tournament selection which is more robust than softmax
            parents = self._tournament_selection(population, fitness_scores)
            
            # Create new population
            new_population = []
            
            # Elitism - preserve best solutions
            elite_idx = np.argsort(fitness_scores)[-self.elite_size:]
            elite_solutions = [population[i].copy() for i in elite_idx]
            
            # Perform crossover and mutation in parallel batches
            offspring = self._parallel_offspring_generation(parents)
            new_population.extend(offspring)
            
            # Add elite solutions
            new_population.extend(elite_solutions)
            
            # Ensure we maintain the correct population size
            while len(new_population) < self.population_size:
                # Add random solutions if needed
                new_solution = np.random.randint(2, size=self.n_features_)
                if np.sum(new_solution) == 0:  # Ensure at least one feature
                    new_solution[np.random.randint(0, self.n_features_)] = 1
                new_population.append(new_solution)
                
            # Truncate if too many solutions
            if len(new_population) > self.population_size:
                new_population = new_population[:self.population_size]
                
            population = np.array(new_population)
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        selected_features = self.get_feature_names()
        total_time = time.time() - start_time
        self.logger.info(f"GHGA selection completed in {total_time:.2f}s: {len(selected_features)}/{self.n_features_} features selected with best fitness {self.best_fitness_:.4f}")
        self.logger.info(f"Selected features: {', '.join(selected_features)}")
        
        # Save final checkpoint
        if self.checkpoint_interval > 0:
            self._save_checkpoint(final=True)
        
        return self
    
    def _parallel_fitness_evaluation(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness of the population in parallel.
        
        Parameters:
        -----------
        population : np.ndarray
            The current population
            
        Returns:
        --------
        np.ndarray
            Fitness scores for each chromosome
        """
        # Convert population to tuples for caching
        pop_tuples = [tuple(chrom.tolist()) for chrom in population]
        
        # Check which chromosomes need evaluation
        to_evaluate = []
        cached_fitness = []
        
        for i, chrom_tuple in enumerate(pop_tuples):
            if chrom_tuple in self._fitness_cache:
                cached_fitness.append((i, self._fitness_cache[chrom_tuple]))
            else:
                to_evaluate.append((i, chrom_tuple))
        
        # If we need to evaluate chromosomes
        if to_evaluate:
            # Split into batches for parallel processing
            batch_size = min(self.fitness_batch_size, len(to_evaluate))
            batches = [to_evaluate[i:i + batch_size] for i in range(0, len(to_evaluate), batch_size)]
            
            # Process batches in parallel
            batch_results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._evaluate_batch)(batch) for batch in batches
            )
            
            # Flatten results
            evaluated_fitness = []
            for batch_result in batch_results:
                evaluated_fitness.extend(batch_result)
        else:
            evaluated_fitness = []
            
        # Combine cached and newly evaluated fitness
        all_fitness = cached_fitness + evaluated_fitness
        all_fitness.sort(key=lambda x: x[0])  # Sort by chromosome index
        
        return np.array([f for _, f in all_fitness])
    
    def _evaluate_batch(self, batch: List[Tuple[int, tuple]]) -> List[Tuple[int, float]]:
        """
        Evaluate a batch of chromosomes.
        
        Parameters:
        -----------
        batch : List[Tuple[int, tuple]]
            List of (index, chromosome_tuple) pairs
            
        Returns:
        --------
        List[Tuple[int, float]]
            List of (index, fitness) pairs
        """
        results = []
        
        for idx, chrom_tuple in batch:
            # Convert tuple to array
            chromosome = np.array(chrom_tuple)
            fitness = self._fitness_function(chromosome)
            
            # Cache the result
            self._fitness_cache[chrom_tuple] = fitness
            
            results.append((idx, fitness))
            
        return results
    
    def _parallel_offspring_generation(self, parents: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate offspring using parallel processing.
        
        Parameters:
        -----------
        parents : List[np.ndarray]
            List of parent chromosomes
            
        Returns:
        --------
        List[np.ndarray]
            List of offspring chromosomes
        """
        # Prepare parent pairs
        parent_pairs = []
        for i in range(0, len(parents) - 1, 2):
            if i + 1 < len(parents):
                parent_pairs.append((parents[i], parents[i+1]))
        
        # Generate offspring in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._generate_offspring_pair)(parent1, parent2) 
            for parent1, parent2 in parent_pairs
        )
        
        # Flatten results
        offspring = []
        for pair in results:
            offspring.extend(pair)
            
        return offspring
    
    def _generate_offspring_pair(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a pair of offspring from two parents.
        
        Parameters:
        -----------
        parent1, parent2 : np.ndarray
            Parent chromosomes
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Two offspring chromosomes
        """
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
            
        return child1, child2
        
    def _evaluate_fitness(self, chromosome_tuple: tuple) -> float:
        """
        Evaluate the fitness of a chromosome with caching.
        
        Parameters:
        -----------
        chromosome_tuple : tuple
            Binary tuple indicating selected features
            
        Returns:
        --------
        float
            Fitness score
        """
        # Check if we've already calculated this fitness
        if chromosome_tuple in self._fitness_cache:
            return self._fitness_cache[chromosome_tuple]
        
        # Convert tuple to array
        chromosome = np.array(chromosome_tuple)
        fitness = self._fitness_function(chromosome)
        
        # Cache the result
        self._fitness_cache[chromosome_tuple] = fitness
        return fitness
    
    def _tournament_selection(self, population: np.ndarray, fitness_scores: np.ndarray, tournament_size: int = 3) -> List[np.ndarray]:
        """
        Perform tournament selection.
        
        Parameters:
        -----------
        population : np.ndarray
            The current population
        fitness_scores : np.ndarray
            The fitness scores for the population
        tournament_size : int, default=3
            Size of each tournament
            
        Returns:
        --------
        List[np.ndarray]
            Selected parents
        """
        selected_parents = []
        
        for _ in range(self.population_size):
            # Select random individuals for the tournament
            tournament_idx = np.random.choice(len(population), size=tournament_size, replace=False)
            tournament_fitness = fitness_scores[tournament_idx]
            
            # Get the best individual from the tournament
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected_parents.append(population[winner_idx].copy())
            
        return selected_parents
    
    def _update_convergence_metrics(self, population: np.ndarray, fitness_scores: np.ndarray) -> None:
        """
        Update convergence metrics for the current generation.
        
        Parameters:
        -----------
        population : np.ndarray
            Current population of solutions
        fitness_scores : np.ndarray
            Fitness scores for current population
        """
        # Best fitness
        best_fitness = np.max(fitness_scores) if not np.all(np.isinf(fitness_scores)) else float('-inf')
        self.convergence_metrics_['best_fitness'].append(best_fitness)
        
        # Mean fitness (excluding -inf values)
        valid_fitness = fitness_scores[~np.isinf(fitness_scores)]
        mean_fitness = np.mean(valid_fitness) if valid_fitness.size > 0 else float('-inf')
        self.convergence_metrics_['mean_fitness'].append(mean_fitness)
        
        # Population diversity (using Hamming distance)
        # More efficient calculation using numpy operations
        pop_size = population.shape[0]
        diversity = 0
        if pop_size > 1:
            # Calculate sum of Hamming distances between all pairs
            if self.use_gpu and population.size > 1000:  # Only use GPU for larger populations
                try:
                    pop_gpu = cp.array(population)
                    # Vectorized Hamming distance calculation on GPU
                    total_distance = 0
                    for i in range(pop_size):
                        distances = cp.sum(pop_gpu[i] != pop_gpu, axis=1)
                        total_distance += cp.sum(distances)
                    
                    # Average distance (excluding self-comparisons)
                    diversity = cp.asnumpy(total_distance / (pop_size * (pop_size - 1)))
                except:
                    # Fall back to CPU if GPU calculation fails
                    total_distance = 0
                    for i in range(pop_size):
                        distances = np.sum(population[i] != population, axis=1)
                        total_distance += np.sum(distances)
                    
                    diversity = total_distance / (pop_size * (pop_size - 1))
            else:
                # CPU calculation
                total_distance = 0
                for i in range(pop_size):
                    distances = np.sum(population[i] != population, axis=1)
                    total_distance += np.sum(distances)
                
                diversity = total_distance / (pop_size * (pop_size - 1))
        
        self.convergence_metrics_['population_diversity'].append(diversity)
        
        # Number of selected features in best solution
        if not np.all(np.isinf(fitness_scores)):
            best_solution = population[np.argmax(fitness_scores)]
            n_selected = np.sum(best_solution)
        else:
            n_selected = 0
        self.convergence_metrics_['selected_features_count'].append(n_selected)

    def _initialize_population(self) -> np.ndarray:
        """
        Initialize population with both random and guided solutions.
        
        Returns:
        --------
        np.ndarray
            Initial population of chromosomes
        """
        self.logger.info("Initializing population")
        population = []
        
        # Create random solutions
        for _ in range(self.population_size - 3):  # Leave room for guided solutions
            # Ensure each solution has at least one feature
            chromosome = np.random.randint(2, size=self.n_features_)
            if np.sum(chromosome) == 0:
                random_idx = np.random.randint(0, self.n_features_)
                chromosome[random_idx] = 1
            population.append(chromosome)
        
        # Add guided solution based on correlation
        try:
            guided_chromosome = np.zeros(self.n_features_, dtype=int)
            
            if isinstance(self.X_, pd.DataFrame) and isinstance(self.y_, pd.Series):
                # Pandas method for correlation
                correlations = self.X_.corrwith(self.y_).abs()
                correlations = correlations.fillna(0)  # Replace NaNs with 0
                
                # Select features with correlation above median
                median_corr = correlations.median()
                for i, corr in enumerate(correlations):
                    if corr >= median_corr:
                        guided_chromosome[i] = 1
                
                # Ensure at least one feature is selected
                if np.sum(guided_chromosome) == 0:
                    top_idx = np.argmax(correlations.values)
                    guided_chromosome[top_idx] = 1
            else:
                # NumPy method for correlation
                correlations = np.zeros(self.n_features_)
                for i in range(self.n_features_):
                    try:
                        corr = np.corrcoef(self.X_[:, i], self.y_)[0, 1]
                        correlations[i] = abs(corr) if not np.isnan(corr) else 0
                    except Exception:
                        correlations[i] = 0
                
                # Select features with correlation above median
                median_corr = np.median(correlations)
                for i, corr in enumerate(correlations):
                    if corr >= median_corr:
                        guided_chromosome[i] = 1
                
                # Ensure at least one feature is selected
                if np.sum(guided_chromosome) == 0:
                    top_idx = np.argmax(correlations)
                    guided_chromosome[top_idx] = 1
            
            population.append(guided_chromosome)
            self.logger.info("Guided solution added based on correlation")
        except Exception as e:
            self.logger.warning(f"Failed to create correlation-based solution: {str(e)}")
            # Add random solution as fallback
            random_solution = np.random.randint(2, size=self.n_features_)
            if np.sum(random_solution) == 0:
                random_solution[np.random.randint(0, self.n_features_)] = 1
            population.append(random_solution)
        
        # Add a sparse solution (few features)
        sparse_solution = np.zeros(self.n_features_, dtype=int)
        n_to_select = max(1, int(0.1 * self.n_features_))  # Select 10% of features or at least 1
        selected_indices = np.random.choice(self.n_features_, size=n_to_select, replace=False)
        sparse_solution[selected_indices] = 1
        population.append(sparse_solution)
        self.logger.info("Sparse solution added with few features")
        
        # Add a dense solution (many features)
        dense_solution = np.ones(self.n_features_, dtype=int)
        n_to_deselect = max(0, int(0.2 * self.n_features_))  # Deselect 20% of features
        if n_to_deselect > 0:
            deselected_indices = np.random.choice(self.n_features_, size=n_to_deselect, replace=False)
            dense_solution[deselected_indices] = 0
        population.append(dense_solution)
        self.logger.info("Dense solution added with many features")
        
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
            self.logger.debug("Empty feature set encountered, assigning minimum fitness")
            return float('-inf')
        
        # Extract selected features
        X_selected = self.X_.iloc[:, selected] if isinstance(self.X_, pd.DataFrame) else self.X_[:, selected]
        
        try:
            # Perform cross-validation
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            cross_validator = MLCrossValidator(estimator=model, scoring='r2', n_splits=5)
            
            if self.cv_method == 'temporal':
                scores = cross_validator.temporal_cv(X_selected, self.y_, groups=self.groups)
            elif self.cv_method == 'random':
                scores = cross_validator.random_cv(X_selected, self.y_)
            elif self.cv_method == 'spatial':
                scores = cross_validator.spatial_cv(X_selected, self.y_, self.groups)
            else:
                raise ValueError(f"Unknown CV method: {self.cv_method}")
            
            # Calculate fitness with penalty for number of features
            cv_score = np.mean(scores)
            penalty = self.feature_penalty * len(selected) / self.n_features_  # Normalized penalty
            
            return cv_score - penalty
            
        except Exception as e:
            self.logger.error(f"Error in fitness evaluation: {str(e)}", exc_info=True)
            return float('-inf')
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform two-point crossover between parents.
        
        Parameters:
        -----------
        parent1, parent2 : np.ndarray
            Parent chromosomes
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Two child chromosomes
        """
        # Get random crossover points
        points = sorted(np.random.choice(self.n_features_, 2, replace=False))
        
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
        
        # Perform crossover
        child1[points[0]:points[1]] = parent2[points[0]:points[1]]
        child2[points[0]:points[1]] = parent1[points[0]:points[1]]
        
        # Ensure at least one feature is selected
        if np.sum(child1) == 0:
            random_idx = np.random.randint(0, self.n_features_)
            child1[random_idx] = 1
        if np.sum(child2) == 0:
            random_idx = np.random.randint(0, self.n_features_)
            child2[random_idx] = 1
            
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
        # Create a copy to avoid modifying the original
        mutated = chromosome.copy()
        
        # Apply mutation
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        
        # Ensure at least one feature is selected
        if np.sum(mutated) == 0:
            random_idx = np.random.randint(0, self.n_features_)
            mutated[random_idx] = 1
            
        return mutated
    
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
        # Create a copy to avoid modifying the original
        improved_chrom = chromosome.copy()
        
        # Get initial fitness
        best_fitness = self._evaluate_fitness(tuple(improved_chrom.tolist()))
        n_selected_initial = np.sum(improved_chrom)
       
        self.logger.debug(f"Starting local search with initial fitness: {best_fitness:.4f} and {n_selected_initial} features")
        
        # Try to improve by flipping bits one by one
        improved = True
        max_iterations = min(20, self.n_features_)  # Limit iterations to avoid getting stuck
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Identify features to try first (randomly shuffled)
            feature_indices = np.random.permutation(self.n_features_)
            
            for i in feature_indices:
                # Try flipping this bit
                improved_chrom[i] = 1 - improved_chrom[i]
                
                # Check if we still have at least one feature
                if np.sum(improved_chrom) == 0:
                    improved_chrom[i] = 1  # Revert if this would lead to no features
                    continue
                
                # Calculate new fitness
                new_fitness = self._evaluate_fitness(tuple(improved_chrom.tolist()))
                
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    improved = True
                    break  # Found an improvement, continue to next iteration
                else:
                    # Revert if no improvement
                    improved_chrom[i] = 1 - improved_chrom[i]
        
        self.logger.debug(f"Local search completed after {iteration} iterations with final fitness: {best_fitness:.4f}")
        return improved_chrom
    
    def _calculate_feature_importance(self) -> None:
        """
        Calculate feature importance scores based on multiple criteria:
        1. Selection frequency across top solutions
        2. Correlation with target variable
        3. Impact on model performance
        """
        # Initialize importance scores
        self.logger.info("Calculating feature importance scores")
        self.feature_importance_ = pd.Series(np.zeros(self.n_features_), 
                                        index=self.feature_names_)
        
        # Get top solutions (either all solutions or top 10%)
        n_top = min(len(self.best_solutions_), max(int(len(self.best_solutions_) * 0.1), 1))
        
        # Sort solutions by fitness and take top n_top
        top_solutions = []
        top_fitness = []
        
        for solution in self.best_solutions_:
            fitness = self._evaluate_fitness(tuple(solution.tolist()))
            if len(top_solutions) < n_top:
                top_solutions.append(solution)
                top_fitness.append(fitness)
            else:
                min_idx = np.argmin(top_fitness)
                if fitness > top_fitness[min_idx]:
                    top_solutions[min_idx] = solution
                    top_fitness[min_idx] = fitness
        
        # Calculate importance in parallel
        feature_importance_values = Parallel(n_jobs=self.n_jobs)(
            delayed(self._calculate_feature_importance_single)(i, feature, top_solutions)
            for i, feature in enumerate(self.feature_names_)
        )
        
        # Update feature importance series
        for i, importance in enumerate(feature_importance_values):
            self.feature_importance_[self.feature_names_[i]] = importance
        
        # Normalize importance scores to [0, 1] range
        if self.feature_importance_.max() > self.feature_importance_.min():
            self.feature_importance_ = (self.feature_importance_ - self.feature_importance_.min()) / \
                                    (self.feature_importance_.max() - self.feature_importance_.min())
        
        top_features = self.feature_importance_.nlargest(min(5, len(self.feature_importance_))).index.tolist()
        self.logger.info(f"Feature importance calculation completed. Top features: {', '.join(top_features)}")
    
    def _calculate_feature_importance_single(self, index: int, feature: str, top_solutions: List[np.ndarray]) -> float:
        """
        Calculate importance for a single feature.
        
        Parameters:
        -----------
        index : int
            Feature index
        feature : str
            Feature name
        top_solutions : List[np.ndarray]
            List of top solutions
            
        Returns:
        --------
        float
            Feature importance score
        """
        # 1. Selection frequency in top solutions
        selection_freq = sum(solution[index] for solution in top_solutions) / len(top_solutions)
        
        # 2. Correlation with target (if available)
        correlation = 0
        try:
            if isinstance(self.X_, pd.DataFrame) and isinstance(self.y_, pd.Series):
                correlation = abs(self.X_[feature].corr(self.y_))
                if pd.isna(correlation):
                    correlation = 0
            else:
                correlation = abs(np.corrcoef(self.X_[:, index], self.y_)[0, 1])
                if np.isnan(correlation):
                    correlation = 0
        except Exception as e:
            self.logger.warning(f"Error calculating correlation for feature {feature}: {str(e)}")
            correlation = 0
            
        # 3. Performance impact
        # Evaluate performance drop when removing the feature
        performance_impact = 0
        try:
            if self.best_solution_ is not None:
                base_mask = self.best_solution_.copy()
                if base_mask[index] == 1:  # If feature is selected in best solution
                    base_mask[index] = 0  # Remove feature
                    if np.sum(base_mask) > 0:  # Ensure we still have features
                        base_fitness = self._evaluate_fitness(tuple(base_mask.tolist()))
                        performance_impact = self.best_fitness_ - base_fitness
        except Exception as e:
            self.logger.warning(f"Error calculating performance impact for feature {feature}: {str(e)}")
            
        # Combine metrics into final importance score
        # Weights can be adjusted based on specific needs
        importance = (0.4 * selection_freq + 
                    0.3 * correlation + 
                    0.3 * max(0, performance_impact))
        
        return importance
    
    def _save_checkpoint(self, final: bool = False) -> str:
        """
        Save the current state to a checkpoint file.
        
        Parameters:
        -----------
        final : bool, default=False
            Whether this is the final checkpoint
            
        Returns:
        --------
        str
            Path to the saved checkpoint file
        """
        if final:
            filename = f"ghga_final_gen{self.current_generation}_fitness{self.best_fitness_:.4f}.pkl"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ghga_checkpoint_gen{self.current_generation}_{timestamp}.pkl"
            
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare data to save
        checkpoint_data = {
            'current_generation': self.current_generation,
            'best_solution': self.best_solution_,
            'best_solutions': self.best_solutions_,
            'best_fitness': self.best_fitness_,
            'feature_names': self.feature_names_,
            'n_features': self.n_features_,
            'convergence_metrics': self.convergence_metrics_,
            'current_population': self.current_population,
            'fitness_cache': self._fitness_cache,
            'params': self.get_params()
        }
        
        # Save to file
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        return str(checkpoint_path)
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load state from a checkpoint file.
        
        Parameters:
        -----------
        checkpoint_path : str
            Path to the checkpoint file
        """
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
            
        # Restore state
        self.current_generation = checkpoint_data['current_generation']
        self.best_solution_ = checkpoint_data['best_solution']
        self.best_solutions_ = checkpoint_data['best_solutions']
        self.best_fitness_ = checkpoint_data['best_fitness']
        self.feature_names_ = checkpoint_data['feature_names']
        self.n_features_ = checkpoint_data['n_features']
        self.convergence_metrics_ = checkpoint_data['convergence_metrics']
        self.current_population = checkpoint_data['current_population']
        self._fitness_cache = checkpoint_data['fitness_cache']
        
        # No need to restore parameters as they should be the same as when initialized
    
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
            raise ValueError(f"X has {X.shape[1]} features, but GHGASelector was "
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
    
    def get_params(self, deep: bool = True) -> Dict:
        """
        Get parameters of the selector.
        
        Parameters:
        -----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
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
            'elite_size': self.elite_size,
            'feature_penalty': self.feature_penalty,
            'max_stored_solutions': self.max_stored_solutions,
            'random_state': self.random_state,
            'cv_method': self.cv_method,
            'n_jobs': self.n_jobs,
            'use_gpu': self.use_gpu,
            'gpu_id': self.gpu_id,
            'checkpoint_interval': self.checkpoint_interval,
            'fitness_batch_size': self.fitness_batch_size
        }
    
    def set_params(self, **parameters) -> 'GHGASelector':
        """
        Set the parameters of this estimator.
        
        Parameters:
        -----------
        **parameters : dict
            Estimator parameters
            
        Returns:
        --------
        self : object
            Estimator instance
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
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
            - generation_time: Time taken for each generation
            
        Raises:
        -------
        NotFittedError
            If the selector hasn't been fitted yet
        """
        if not self.convergence_metrics_['best_fitness']:
            raise NotFittedError("Selector has not been fitted yet.")
        return self.convergence_metrics_
    
    def plot_convergence(self, plot_type: str = 'matplotlib', save_path: Optional[str] = None):
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
        None or plotly.graph_objects.Figure
            Returns plotly figure object if plot_type is 'plotly'
        
        Raises:
        -------
        NotFittedError
            If the selector hasn't been fitted yet
        ValueError
            If plot_type is not 'matplotlib' or 'plotly'
        """
        if not self.convergence_metrics_['best_fitness']:
            raise NotFittedError("No convergence data available. Run fit() first.")
        
        generations = range(len(self.convergence_metrics_['best_fitness']))
        
        if plot_type == 'matplotlib':
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                self.logger.error("Matplotlib is required for plotting but is not installed.")
                raise ImportError("Matplotlib is required for plotting. Please install it with 'pip install matplotlib'.")
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
            
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
            ax2_right = ax2.twinx()
            ax2_right.plot(generations, self.convergence_metrics_['selected_features_count'], 
                    label='Selected Features', color='red', linestyle='--')
            ax2_right.set_ylabel('Number of Selected Features')
            
            # Combine legends for second plot
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_right.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # Computation time plot
            ax3.bar(generations, self.convergence_metrics_['generation_time'], 
                   color='purple', alpha=0.7)
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Time (seconds)')
            ax3.set_title('Computation Time per Generation')
            ax3.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
                self.logger.info(f"Convergence plot saved to {save_path}")
            else:
                plt.show()
                
        elif plot_type == 'plotly':
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
            except ImportError:
                self.logger.error("Plotly is required for interactive plotting but is not installed.")
                raise ImportError("Plotly is required for interactive plotting. Please install it with 'pip install plotly'.")
            
            fig = make_subplots(rows=3, cols=1, 
                            subplot_titles=('Fitness Convergence', 
                                            'Population Diversity and Selected Features',
                                            'Computation Time per Generation'))
            
            # Fitness plot
            fig.add_trace(
                go.Scatter(x=list(generations), y=self.convergence_metrics_['best_fitness'],
                        name='Best Fitness', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=list(generations), y=self.convergence_metrics_['mean_fitness'],
                        name='Mean Fitness', line=dict(color='orange')),
                row=1, col=1
            )
            
            # Diversity plot
            fig.add_trace(
                go.Scatter(x=list(generations), y=self.convergence_metrics_['population_diversity'],
                        name='Population Diversity', line=dict(color='green')),
                row=2, col=1
            )
            
            # Feature count plot
            fig.add_trace(
                go.Scatter(x=list(generations), y=self.convergence_metrics_['selected_features_count'],
                        name='Selected Features', line=dict(color='red', dash='dash'),
                        yaxis='y3'),
                row=2, col=1
            )
            
            # Computation time plot
            fig.add_trace(
                go.Bar(x=list(generations), y=self.convergence_metrics_['generation_time'],
                      name='Computation Time', marker_color='purple'),
                row=3, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=1000,
                showlegend=True,
                title_text='GHGA Convergence Metrics',
                yaxis=dict(title='Fitness Score'),
                yaxis2=dict(title='Diversity Score'),
                yaxis3=dict(title='Number of Selected Features',
                        overlaying='y2', side='right'),
                yaxis4=dict(title='Time (seconds)')
            )
            
            if save_path:
                fig.write_html(save_path)
                self.logger.info(f"Interactive convergence plot saved to {save_path}")
            
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
        
        # Calculate total runtime
        total_time = sum(self.convergence_metrics_['generation_time'])
        
        summary = {
            'n_total_features': self.n_features_,
            'input_features': self.feature_names_,
            'n_selected_features': len(selected_features),
            'selected_features': selected_features,
            'best_fitness': self.best_fitness_,
            'feature_importance': self.get_feature_importance(),
            'convergence': {
                'initial_fitness': self.convergence_metrics_['best_fitness'][0] if self.convergence_metrics_['best_fitness'] else None,
                'final_fitness': self.convergence_metrics_['best_fitness'][-1] if self.convergence_metrics_['best_fitness'] else None,
                'n_generations': len(self.convergence_metrics_['best_fitness']),
                'total_runtime': total_time,
                'avg_generation_time': total_time / max(1, len(self.convergence_metrics_['generation_time']))
            },
            'hardware_info': {
                'used_gpu': self.use_gpu,
                'n_jobs': self.n_jobs
            }
        }
        
        self.logger.info(f"Generated summary: {len(selected_features)} features selected with fitness {self.best_fitness_:.4f}")
        
        return summary