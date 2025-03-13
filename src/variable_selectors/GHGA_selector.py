import sys
from pathlib import Path
import numpy as np
import logging
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from scipy.special import softmax
from typing import List, Dict, Union, Optional, Tuple, Callable, Any
import warnings
from functools import lru_cache
import time
import os
import pickle
from datetime import datetime
from sklearn.exceptions import NotFittedError
parent_dir = str(Path(__file__).parent.parent.parent)
print(parent_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from src.cross_validation.cross_validators import MLCrossValidator

# Parallel processing imports
import multiprocessing
from joblib import Parallel, delayed

# Optional GPU imports - will be checked at runtime
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# Optional cuDF and cuML imports - will be checked at runtime
try:
    import cudf
    import cuml
    from cuml.ensemble import RandomForestRegressor as cuRFR
    HAS_CUDF_CUML = True
except ImportError:
    HAS_CUDF_CUML = False

# Determine overall GPU availability
HAS_GPU = HAS_CUPY  # At minimum, we need CuPy

class GHGASelector:
    """
    Guided Hybrid Genetic Algorithm for feature selection.
    Optimized for high-performance computing environments with improved GPU utilization.
    
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
    gpu_mem_limit : float, default=0.8
        Maximum fraction of GPU memory to use (0.0-1.0)
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
        gpu_mem_limit: float = 0.8,
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
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.gpu_mem_limit = gpu_mem_limit
        self.checkpoint_interval = checkpoint_interval
        self.fitness_batch_size = fitness_batch_size
        
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
        self.logger = logging.getLogger('ghga_selector')
        self.logger.addHandler(logging.StreamHandler())

        
        # Check and setup GPU capabilities
        self._setup_gpu()
         
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
            if self.use_gpu and self.has_cupy:
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
        
        # Log initialization with GPU status
        gpu_info = f"GPU={self.use_gpu} (cupy={self.has_cupy}, cudf/cuml={self.has_cudf_cuml})"
        self.logger.info(f"Initialized GHGASelector with {self.n_jobs} CPU cores and {gpu_info}")
    
    def _setup_gpu(self):
        """
        Setup and detect GPU capabilities, setting appropriate flags.
        """
        # Check if GPU libraries are available
        self.has_cupy = HAS_CUPY
        self.has_cudf_cuml = HAS_CUDF_CUML
        
        # Overall GPU availability
        gpu_available = self.has_cupy
        
        # Initialize GPU if requested and available
        self.use_gpu = self.use_gpu and gpu_available
        
        if self.use_gpu:
            try:
                if self.has_cupy:
                    # Setup GPU device
                    cp.cuda.Device(self.gpu_id).use()
                    
                    # Get memory info for logging
                    try:
                        mem_info = cp.cuda.runtime.memGetInfo()
                        free_mem = mem_info[0] / 1024**3  # Convert to GB
                        total_mem = mem_info[1] / 1024**3  # Convert to GB
                        self.logger.info(f"GPU {self.gpu_id} memory: {free_mem:.2f}GB free / {total_mem:.2f}GB total")
                    except Exception as e:
                        self.logger.warning(f"Could not get GPU memory info: {str(e)}")
                    
                    # Set xp as cupy for array operations
                    self.xp = cp
                    self.logger.info(f"Successfully initialized GPU {self.gpu_id} with CuPy")
            except Exception as e:
                self.use_gpu = False
                self.xp = np
                self.logger.warning(f"Failed to initialize GPU {self.gpu_id}: {str(e)}. Falling back to CPU.")
        else:
            self.xp = np
            if not gpu_available and self.use_gpu:
                self.logger.warning("GPU requested but required libraries not found. Install cupy for basic GPU support.")
    
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
        self.X_cpu = X  # Always keep a CPU copy for methods that don't support GPU
        self.y_cpu = y
        
        if self.use_gpu:
            try:
                self.logger.info("Setting up data for GPU processing...")
                
                # For CuPy-based operations (most of our custom code)
                if self.has_cupy:
                    # Keep original references
                    self.X_ = X
                    self.y_ = y
                    # We'll convert data to GPU arrays as needed in specific operations
                
                # For cuDF/cuML operations if they're available
                if self.has_cudf_cuml:
                    self.logger.info("Converting data to cuDF format...")
                    try:
                        # Convert to cuDF dataframes
                        self.X_gpu = cudf.DataFrame.from_pandas(X)
                        self.y_gpu = cudf.Series.from_pandas(y)
                        self.logger.info("Data successfully converted to cuDF format")
                    except Exception as e:
                        self.logger.warning(f"Failed to convert data to cuDF: {str(e)}. Specific cuDF/cuML operations will use CPU.")
                        self.X_gpu = None
                        self.y_gpu = None
            except Exception as e:
                self.logger.warning(f"Error in GPU data setup: {str(e)}. Falling back to CPU for data operations.")
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
        stagnation_count = 0
        best_fitness_history = []
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
            
            # Track best fitness for stagnation detection
            best_fitness_history.append(current_gen_best_fitness)
            # Check for stagnation (no improvement in last 5 generations)
            if len(best_fitness_history) > 5:
                if all(abs(best_fitness_history[-1] - bf) < 1e-6 for bf in best_fitness_history[-5:]):
                    # increase mutation rate
                    self.mutation_rate = min(0.5, self.mutation_rate + 0.1)
                    stagnation_count += 1
                    
                    # If stagnated for too long, inject diversity
                    if stagnation_count >= 10:
                        self.logger.warning(f"Stagnation detected for {stagnation_count} cycles, program stops.")
                        break
                else:
                    stagnation_count = 0

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
            
            # Perform crossover and mutation in parallel batches (now with GPU acceleration where appropriate)
            offspring = self._parallel_offspring_generation(parents)
            new_population.extend(offspring)
            
            # Add elite solutions
            new_population.extend(elite_solutions)
            
            # Ensure we maintain the correct population size
            while len(new_population) < self.population_size:
                # Add random solutions if needed
                new_solution = self._fast_random_chromosome()
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
    
    def _fast_random_chromosome(self) -> np.ndarray:
        """
        Generate a random chromosome efficiently, using GPU if available.
        
        Returns:
        --------
        np.ndarray
            A random binary chromosome with at least one feature selected
        """
        if self.use_gpu and self.has_cupy:
            try:
                # Generate random chromosome on GPU
                chromosome = cp.random.randint(2, size=self.n_features_).get()
                if np.sum(chromosome) == 0:  # Ensure at least one feature
                    chromosome[cp.random.randint(0, self.n_features_).get()] = 1
                return chromosome
            except:
                # Fall back to CPU if GPU fails
                pass
                
        # CPU fallback
        chromosome = np.random.randint(2, size=self.n_features_)
        if np.sum(chromosome) == 0:  # Ensure at least one feature
            chromosome[np.random.randint(0, self.n_features_)] = 1
        return chromosome
    
    def _parallel_fitness_evaluation(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness of the population in parallel with GPU acceleration when possible.
        
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
            # Use GPU batch processing if available and beneficial
            if self.use_gpu and self.has_cupy and len(to_evaluate) > 10:
                evaluated_fitness = self._evaluate_batch_gpu(to_evaluate)
            else:
                # Split into batches for parallel CPU processing
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
    
    def _evaluate_batch_gpu(self, batch: List[Tuple[int, tuple]]) -> List[Tuple[int, float]]:
        """
        Evaluate a batch of chromosomes using GPU acceleration.
        
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
        
        # Extract chromosomes and create a 2D array
        indices = [idx for idx, _ in batch]
        chromosomes = np.array([np.array(chrom) for _, chrom in batch])
        
        try:
            # Move chromosomes to GPU
            chromosomes_gpu = cp.array(chromosomes)
            
            # Process in smaller sub-batches to avoid GPU memory issues
            sub_batch_size = min(20, len(chromosomes))
            for i in range(0, len(chromosomes), sub_batch_size):
                end_idx = min(i + sub_batch_size, len(chromosomes))
                sub_batch = chromosomes_gpu[i:end_idx]
                
                # Process each chromosome in the sub-batch
                for j in range(sub_batch.shape[0]):
                    idx = indices[i + j]
                    chrom = cp.asnumpy(sub_batch[j])
                    chrom_tuple = tuple(chrom.tolist())
                    
                    # Calculate fitness (this uses CPU for now as _fitness_function is not GPU-optimized)
                    fitness = self._fitness_function(chrom)
                    
                    # Cache the result
                    self._fitness_cache[chrom_tuple] = fitness
                    results.append((idx, fitness))
                
            return results
        except Exception as e:
            self.logger.warning(f"GPU batch evaluation failed: {str(e)}. Falling back to CPU.")
            
            # Fall back to CPU evaluation
            for idx, chrom_tuple in batch:
                chromosome = np.array(chrom_tuple)
                fitness = self._fitness_function(chromosome)
                self._fitness_cache[chrom_tuple] = fitness
                results.append((idx, fitness))
                
            return results
    
    def _evaluate_batch(self, batch: List[Tuple[int, tuple]]) -> List[Tuple[int, float]]:
        """
        Evaluate a batch of chromosomes on CPU.
        
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
        Generate offspring using parallel processing with GPU acceleration for batch operations.
        
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
        
        # Use GPU batch processing if available and beneficial
        if self.use_gpu and self.has_cupy and len(parent_pairs) >= 10:
            try:
                return self._generate_offspring_batch_gpu(parent_pairs)
            except Exception as e:
                self.logger.warning(f"GPU offspring generation failed: {str(e)}. Falling back to CPU.")
        
        # Generate offspring in parallel on CPU
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._generate_offspring_pair)(parent1, parent2) 
            for parent1, parent2 in parent_pairs
        )
        
        # Flatten results
        offspring = []
        for pair in results:
            offspring.extend(pair)
            
        return offspring
    
    def _generate_offspring_batch_gpu(self, parent_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        """
        Generate offspring in a batch using GPU acceleration.
        
        Parameters:
        -----------
        parent_pairs : List[Tuple[np.ndarray, np.ndarray]]
            List of parent pairs
            
        Returns:
        --------
        List[np.ndarray]
            Generated offspring
        """
        offspring = []
        
        # Extract parents and create arrays
        parent1_array = np.array([p1 for p1, _ in parent_pairs])
        parent2_array = np.array([p2 for _, p2 in parent_pairs])
        
        try:
            # Move to GPU
            parent1_gpu = cp.array(parent1_array)
            parent2_gpu = cp.array(parent2_array)
            
            # Process in smaller batches to avoid memory issues
            batch_size = min(20, len(parent_pairs))
            for i in range(0, len(parent_pairs), batch_size):
                end_idx = min(i + batch_size, len(parent_pairs))
                
                # Get sub-batch
                p1_batch = parent1_gpu[i:end_idx]
                p2_batch = parent2_gpu[i:end_idx]
                
                # For each pair in batch
                for j in range(p1_batch.shape[0]):
                    p1 = cp.asnumpy(p1_batch[j])
                    p2 = cp.asnumpy(p2_batch[j])
                    
                    # Process this pair (currently happens on CPU)
                    # In future iterations, these operations could be further GPU-optimized
                    child1, child2 = self._crossover(p1, p2)
                    
                    # Mutation
                    child1 = self._mutation(child1)
                    child2 = self._mutation(child2)
                    
                    # Local search
                    if np.random.random() < self.local_search_prob:
                        child1 = self._local_search(child1)
                    if np.random.random() < self.local_search_prob:
                        child2 = self._local_search(child2)
                    
                    offspring.extend([child1, child2])
            
            return offspring
            
        except Exception as e:
            self.logger.warning(f"Error in GPU offspring generation: {str(e)}. Falling back to CPU.")
            
            # Fall back to CPU processing
            for parent1, parent2 in parent_pairs:
                child1, child2 = self._generate_offspring_pair(parent1, parent2)
                offspring.extend([child1, child2])
                
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
        Perform tournament selection with GPU acceleration if available.
        
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
        
        # Use GPU for tournament selection if available
        if self.use_gpu and self.has_cupy:
            try:
                # Move population and fitness scores to GPU
                population_gpu = cp.array(population)
                fitness_scores_gpu = cp.array(fitness_scores)
                
                for _ in range(self.population_size):
                    # Select random individuals for the tournament
                    tournament_idx = cp.random.choice(len(population), size=tournament_size, replace=False)
                    tournament_fitness = fitness_scores_gpu[tournament_idx]
                    
                    # Get the best individual from the tournament
                    winner_idx = tournament_idx[cp.argmax(tournament_fitness)]
                    selected_parents.append(cp.asnumpy(population_gpu[winner_idx]).copy())
                
                return selected_parents
            except Exception as e:
                self.logger.warning(f"GPU tournament selection failed: {str(e)}. Falling back to CPU.")
        
        # CPU fallback
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
        Update convergence metrics for the current generation with GPU acceleration.
        
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
        pop_size = population.shape[0]
        diversity = 0
        
        if pop_size > 1:
            # Calculate sum of Hamming distances between all pairs
            if self.use_gpu and self.has_cupy and population.size >= 1000:  # Only use GPU for larger populations
                try:
                    pop_gpu = cp.array(population)
                    
                    # More efficient GPU implementation of diversity calculation
                    total_distance = 0
                    # Process in chunks to avoid memory issues
                    chunk_size = min(50, pop_size)
                    
                    for i in range(0, pop_size, chunk_size):
                        end_i = min(i + chunk_size, pop_size)
                        chunk_i = pop_gpu[i:end_i]
                        
                        for j in range(0, pop_size, chunk_size):
                            end_j = min(j + chunk_size, pop_size)
                            chunk_j = pop_gpu[j:end_j]
                            
                            # Compute pairwise Hamming distances for this block
                            for k in range(chunk_i.shape[0]):
                                # Use broadcasting for efficient comparison
                                distances = cp.sum(chunk_i[k:k+1] != chunk_j, axis=1)
                                total_distance += cp.sum(distances)
                    
                    # Correct for self-comparisons
                    total_distance -= pop_size  # Subtract self-comparisons
                    
                    # Average distance
                    diversity = cp.asnumpy(total_distance / (pop_size * (pop_size - 1)))
                except Exception as e:
                    self.logger.warning(f"GPU diversity calculation failed: {str(e)}. Falling back to CPU.")
                    # Fall back to CPU calculation
                    total_distance = 0
                    for i in range(pop_size):
                        distances = np.sum(population[i] != population, axis=1)
                        total_distance += np.sum(distances)
                    
                    # Correct for self-comparisons
                    total_distance -= pop_size
                    
                    diversity = total_distance / (pop_size * (pop_size - 1))
            else:
                # CPU calculation
                total_distance = 0
                for i in range(pop_size):
                    distances = np.sum(population[i] != population, axis=1)
                    total_distance += np.sum(distances)
                
                # Correct for self-comparisons
                total_distance -= pop_size
                
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
        Initialize population with both random and guided solutions using GPU when available.
        
        Returns:
        --------
        np.ndarray
            Initial population of chromosomes
        """
        self.logger.info("Initializing population")
        population = []
        
        # Create random solutions - utilize GPU for batch generation if available
        if self.use_gpu and self.has_cupy:
            try:
                # Generate random solutions in batch on GPU
                n_random = self.population_size - 3  # Leave room for guided solutions
                
                # Generate binary matrix on GPU
                random_pop = cp.random.randint(0, 2, (n_random, self.n_features_))
                
                # Find rows with all zeros
                row_sums = cp.sum(random_pop, axis=1)
                zero_indices = cp.where(row_sums == 0)[0]
                
                # Handle rows with all zeros (ensure at least one feature)
                if zero_indices.size > 0:
                    random_indices = cp.random.randint(0, self.n_features_, size=zero_indices.size)
                    for i, row_idx in enumerate(zero_indices):
                        random_pop[row_idx, random_indices[i]] = 1
                
                # Copy back to CPU
                population = [cp.asnumpy(random_pop[i]) for i in range(n_random)]
                
                self.logger.info(f"Generated {n_random} random solutions using GPU")
                
            except Exception as e:
                self.logger.warning(f"GPU population initialization failed: {str(e)}. Falling back to CPU.")
                # Fall back to CPU initialization for random solutions
                for _ in range(self.population_size - 3):
                    chromosome = np.random.randint(2, size=self.n_features_)
                    if np.sum(chromosome) == 0:
                        random_idx = np.random.randint(0, self.n_features_)
                        chromosome[random_idx] = 1
                    population.append(chromosome)
        else:
            # CPU initialization for random solutions
            for _ in range(self.population_size - 3):
                chromosome = np.random.randint(2, size=self.n_features_)
                if np.sum(chromosome) == 0:
                    random_idx = np.random.randint(0, self.n_features_)
                    chromosome[random_idx] = 1
                population.append(chromosome)
        
        # Add guided solution based on correlation
        try:
            guided_chromosome = np.zeros(self.n_features_, dtype=int)
            
            # Try using cuDF for correlation calculation if available
            if self.use_gpu and self.has_cudf_cuml and hasattr(self, 'X_gpu') and self.X_gpu is not None:
                try:
                    # Use cuDF for correlation calculation
                    correlations = self.X_gpu.corrwith(self.y_gpu).abs().to_pandas()
                    correlations = correlations.fillna(0)
                    
                    # Select features with correlation above median
                    median_corr = correlations.median()
                    for i, corr in enumerate(correlations):
                        if corr >= median_corr:
                            guided_chromosome[i] = 1
                    
                    # Ensure at least one feature is selected
                    if np.sum(guided_chromosome) == 0:
                        top_idx = np.argmax(correlations.values)
                        guided_chromosome[top_idx] = 1
                        
                    self.logger.info("Created correlation-based solution using GPU acceleration")
                except Exception as e:
                    self.logger.warning(f"GPU correlation calculation failed: {str(e)}. Using CPU.")
                    raise  # Fall through to CPU method
            
            # If no GPU or GPU failed, use pandas/numpy
            if np.sum(guided_chromosome) == 0:  # If not set by GPU method
                if isinstance(self.X_cpu, pd.DataFrame) and isinstance(self.y_cpu, pd.Series):
                    # Pandas method for correlation
                    correlations = self.X_cpu.corrwith(self.y_cpu).abs()
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
                            corr = np.corrcoef(self.X_cpu[:, i], self.y_cpu)[0, 1]
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
        
        if self.use_gpu and self.has_cupy:
            try:
                selected_indices = cp.random.choice(self.n_features_, size=n_to_select, replace=False).get()
            except:
                selected_indices = np.random.choice(self.n_features_, size=n_to_select, replace=False)
        else:
            selected_indices = np.random.choice(self.n_features_, size=n_to_select, replace=False)
            
        sparse_solution[selected_indices] = 1
        population.append(sparse_solution)
        self.logger.info("Sparse solution added with few features")
        
        # Add a dense solution (many features)
        dense_solution = np.ones(self.n_features_, dtype=int)
        n_to_deselect = max(0, int(0.2 * self.n_features_))  # Deselect 20% of features
        
        if n_to_deselect > 0:
            if self.use_gpu and self.has_cupy:
                try:
                    deselected_indices = cp.random.choice(self.n_features_, size=n_to_deselect, replace=False).get()
                except:
                    deselected_indices = np.random.choice(self.n_features_, size=n_to_deselect, replace=False)
            else:
                deselected_indices = np.random.choice(self.n_features_, size=n_to_deselect, replace=False)
                
            dense_solution[deselected_indices] = 0
        
        population.append(dense_solution)
        self.logger.info("Dense solution added with many features")
        
        return np.array(population)
    
    def _fitness_function(self, chromosome: np.ndarray) -> float:
        """
        Evaluate solution quality using cross-validation with correlation-based penalty.
        
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
        
        try:
            # GPU PATH
            if self.use_gpu and self.has_cudf_cuml and hasattr(self, 'X_gpu') and self.X_gpu is not None:
                try:
                    # Extract selected features using cuDF
                    X_selected = self.X_gpu.iloc[:, selected]
                    
                    # Use cuML for RandomForest
                    model = cuRFR(n_estimators=100, random_state=self.random_state)
                    
                    # Convert to pandas for cross-validation
                    X_selected_pd = X_selected.to_pandas()
                    
                    cross_validator = MLCrossValidator(estimator=model, scoring='r2', n_splits=5)
                    
                    if self.cv_method == 'temporal':
                        scores = cross_validator.temporal_cv(X_selected_pd, self.y_cpu, groups=self.groups)
                    elif self.cv_method == 'random':
                        scores = cross_validator.random_cv(X_selected_pd, self.y_cpu)
                    elif self.cv_method == 'spatial':
                        scores = cross_validator.spatial_cv(X_selected_pd, self.y_cpu, self.groups)
                    else:
                        raise ValueError(f"Unknown CV method: {self.cv_method}")
                    
                    cv_score = np.mean(scores)
                    
                    # For correlation penalty, we'll use pandas (CPU)
                    X_selected_corr = self.X_cpu.iloc[:, selected]
                except Exception as e:
                    self.logger.warning(f"GPU fitness calculation failed: {str(e)}. Falling back to CPU.")
                    # Fall through to CPU implementation
                    raise  # Re-raise to fall through to CPU path
            
            # CPU PATH
            else:  # CPU implementation (primary or fallback from GPU exception)
                # Extract selected features
                X_selected = self.X_cpu.iloc[:, selected] if isinstance(self.X_cpu, pd.DataFrame) else self.X_cpu[:, selected]
                X_selected_corr = X_selected  # For correlation calculation later
                
                # Perform cross-validation
                model = RandomForestRegressor(
                    random_state=self.random_state, 
                    n_jobs=self.n_jobs,
                    n_estimators=100
                )
                cross_validator = MLCrossValidator(estimator=model, scoring='r2', n_splits=5)
                
                if self.cv_method == 'temporal':
                    scores = cross_validator.temporal_cv(X_selected, self.y_cpu, groups=self.groups)
                elif self.cv_method == 'random':
                    scores = cross_validator.random_cv(X_selected, self.y_cpu)
                elif self.cv_method == 'spatial':
                    scores = cross_validator.spatial_cv(X_selected, self.y_cpu, self.groups)
                else:
                    raise ValueError(f"Unknown CV method: {self.cv_method}")
                
                cv_score = np.mean(scores)
            
            # CORRELATION PENALTY CALCULATION (after either GPU or CPU path)
            # Basic feature count penalty
            basic_penalty = self.feature_penalty * len(selected) / self.n_features_
            
            # Only calculate correlation penalty if we have multiple features
            correlation_penalty = 0.0
            if len(selected) > 1:
                try:
                    # Ensure we have a DataFrame for correlation calculation
                    if not isinstance(X_selected_corr, pd.DataFrame):
                        X_selected_corr = pd.DataFrame(X_selected_corr)
                    
                    # Calculate correlation matrix (absolute values)
                    correlation_matrix = np.abs(X_selected_corr.corr().values)
                    np.fill_diagonal(correlation_matrix, 0)  # Ignore self-correlations
                    
                    # Calculate average correlation across features
                    mean_correlation = np.mean(correlation_matrix)
                    
                    # Apply stronger penalties for highly correlated feature sets
                    if mean_correlation > 0.7:
                        correlation_penalty = self.feature_penalty * 0.8 * mean_correlation
                    else:
                        correlation_penalty = self.feature_penalty * 0.5 * mean_correlation
                        
                    # Apply a threshold for minimal correlation
                    if mean_correlation < 0.2:
                        correlation_penalty = 0
                except Exception as e:
                    self.logger.warning(f"Correlation calculation failed: {str(e)}. Using only basic penalty.")
                    correlation_penalty = 0.0
            
            
                
            # Combine penalties and calculate final fitness
            total_penalty = basic_penalty + correlation_penalty
            fitness = cv_score - total_penalty
            
            return fitness
            
        except Exception as e:
            self.logger.error(f"Error in fitness evaluation: {str(e)}", exc_info=True)
            return float('-inf')
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform two-point crossover between parents with GPU acceleration when appropriate.
        
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
        if self.use_gpu and self.has_cupy:
            try:
                points = sorted(cp.random.choice(self.n_features_, 2, replace=False).get())
            except:
                points = sorted(np.random.choice(self.n_features_, 2, replace=False))
        else:
            points = sorted(np.random.choice(self.n_features_, 2, replace=False))
        
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
        
        # Perform crossover
        child1[points[0]:points[1]] = parent2[points[0]:points[1]]
        child2[points[0]:points[1]] = parent1[points[0]:points[1]]
        
        # Ensure at least one feature is selected
        if np.sum(child1) == 0:
            if self.use_gpu and self.has_cupy:
                try:
                    random_idx = cp.random.randint(0, self.n_features_).get()
                except:
                    random_idx = np.random.randint(0, self.n_features_)
            else:
                random_idx = np.random.randint(0, self.n_features_)
            child1[random_idx] = 1
            
        if np.sum(child2) == 0:
            if self.use_gpu and self.has_cupy:
                try:
                    random_idx = cp.random.randint(0, self.n_features_).get()
                except:
                    random_idx = np.random.randint(0, self.n_features_)
            else:
                random_idx = np.random.randint(0, self.n_features_)
            child2[random_idx] = 1
            
        return child1, child2
    
    def _mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Perform bit-flip mutation with GPU acceleration when appropriate.
        
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
        
        if self.use_gpu and self.has_cupy:
            try:
                # Generate mutation mask on GPU
                random_values = cp.random.random(self.n_features_).get()
                mutation_mask = random_values < self.mutation_rate
                
                # Apply mutations
                if np.any(mutation_mask):
                    mutated[mutation_mask] = 1 - mutated[mutation_mask]
                
                # Ensure at least one feature is selected
                if np.sum(mutated) == 0:
                    random_idx = cp.random.randint(0, self.n_features_).get()
                    mutated[random_idx] = 1
                
                return mutated
            except:
                # Fall back to CPU if GPU fails
                pass
        
        # CPU implementation
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
        Implement hill climbing to improve solution with GPU acceleration where possible.
        
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
        max_iterations = min(10, self.n_features_)  # Limit iterations to avoid getting stuck
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Identify features to try first (randomly shuffled)
            if self.use_gpu and self.has_cupy:
                try:
                    feature_indices = cp.random.permutation(self.n_features_).get()
                except:
                    feature_indices = np.random.permutation(self.n_features_)
            else:
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
        Calculate feature importance scores with GPU acceleration where possible.
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
            # Try GPU correlation if available
            if self.use_gpu and self.has_cudf_cuml and hasattr(self, 'X_gpu') and self.X_gpu is not None:
                try:
                    # Use cuDF for correlation calculation
                    feature_series = self.X_gpu[feature]
                    correlation = abs(feature_series.corr(self.y_gpu))
                    if pd.isna(correlation):
                        correlation = 0
                except:
                    # Fall back to CPU correlation
                    if isinstance(self.X_cpu, pd.DataFrame) and isinstance(self.y_cpu, pd.Series):
                        correlation = abs(self.X_cpu[feature].corr(self.y_cpu))
                        if pd.isna(correlation):
                            correlation = 0
                    else:
                        correlation = abs(np.corrcoef(self.X_cpu[:, index], self.y_cpu)[0, 1])
                        if np.isnan(correlation):
                            correlation = 0
            else:
                # CPU correlation
                if isinstance(self.X_cpu, pd.DataFrame) and isinstance(self.y_cpu, pd.Series):
                    correlation = abs(self.X_cpu[feature].corr(self.y_cpu))
                    if pd.isna(correlation):
                        correlation = 0
                else:
                    correlation = abs(np.corrcoef(self.X_cpu[:, index], self.y_cpu)[0, 1])
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
        
        # Transform data - try using GPU if available
        if self.use_gpu and self.has_cudf_cuml and isinstance(X, pd.DataFrame):
            try:
                # Convert to cuDF DataFrame
                X_gpu = cudf.DataFrame.from_pandas(X)
                
                # Select features
                X_transformed = X_gpu.iloc[:, mask]
                
                # Convert back to pandas for compatibility
                return X_transformed.to_pandas()
            except:
                # Fall back to CPU if GPU transform fails
                pass
                
        # CPU transform
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
            'gpu_mem_limit': self.gpu_mem_limit,
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
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get information about the GPU configuration and usage.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary with GPU information
        """
        gpu_info = {
            "gpu_enabled": self.use_gpu,
            "gpu_id": self.gpu_id if self.use_gpu else None,
            "cupy_available": self.has_cupy,
            "cudf_cuml_available": self.has_cudf_cuml,
        }
        
        # Add memory info if available
        if self.use_gpu and self.has_cupy:
            try:
                mem_info = cp.cuda.runtime.memGetInfo()
                gpu_info["free_memory_gb"] = mem_info[0] / 1024**3
                gpu_info["total_memory_gb"] = mem_info[1] / 1024**3
                gpu_info["memory_usage_percent"] = 100 * (1 - mem_info[0] / mem_info[1])
            except:
                gpu_info["memory_info"] = "Not available"
        
        return gpu_info
    
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
        
        # Get GPU info
        gpu_info = self.get_gpu_info() if hasattr(self, 'use_gpu') and self.use_gpu else {
            "gpu_enabled": False
        }
        
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
                'gpu': gpu_info,
                'n_jobs': self.n_jobs
            }
        }
        
        self.logger.info(f"Generated summary: {len(selected_features)} features selected with fitness {self.best_fitness_:.4f}")
        
        return summary