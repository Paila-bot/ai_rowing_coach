import numpy as np
from typing import List, Tuple, Dict, Optional

class EvolutionaryOptimizer:
    """Evolutionary algorithm for optimizing model parameters"""

    def __init__(self, param_ranges: Dict[str, Tuple[float, float]],
                 population_size: int = 50, mutation_rate: float = 0.1):
        self.param_ranges = param_ranges
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.param_names = list(param_ranges.keys())
        self.n_params = len(self.param_names)

        # Initialize population
        self.population = self._initialize_population()
        self.fitness_scores = np.zeros(population_size)

    def _initialize_population(self) -> np.ndarray:
        """Initialize random population within parameter ranges"""
        population = np.zeros((self.population_size, self.n_params))

        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.param_ranges[param_name]
            population[:, i] = np.random.uniform(min_val, max_val, self.population_size)

        return population

    def optimize(self, fitness_function, n_generations: int = 100) -> Dict[str, float]:
        """Run evolutionary optimization"""
        best_fitness = -np.inf
        best_params = None

        for generation in range(n_generations):
            # Evaluate fitness for all individuals
            for i in range(self.population_size):
                params = self._array_to_params(self.population[i])
                self.fitness_scores[i] = fitness_function(params)

            # Track best solution
            current_best_idx = np.argmax(self.fitness_scores)
            current_best_fitness = self.fitness_scores[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_params = self._array_to_params(self.population[current_best_idx])

            # Create next generation
            self.population = self._create_next_generation()

        return best_params

    def _array_to_params(self, param_array: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary"""
        return {name: param_array[i] for i, name in enumerate(self.param_names)}

    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to array"""
        return np.array([params[name] for name in self.param_names])

    def _create_next_generation(self) -> np.ndarray:
        """Create next generation using selection, crossover, and mutation"""
        new_population = np.zeros_like(self.population)

        # Elite selection - keep best 10%
        elite_count = self.population_size // 10
        elite_indices = np.argsort(self.fitness_scores)[-elite_count:]
        new_population[:elite_count] = self.population[elite_indices]

        # Generate rest through crossover and mutation
        for i in range(elite_count, self.population_size):
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            child = self._crossover(parent1, parent2)

            # Mutation
            child = self._mutate(child)

            new_population[i] = child

        return new_population

    def _tournament_selection(self, tournament_size: int = 3) -> np.ndarray:
        """Select individual using tournament selection"""
        tournament_indices = np.random.choice(
            self.population_size, tournament_size, replace=False)
        tournament_fitness = self.fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]

        return self.population[winner_idx].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform uniform crossover"""
        child = np.zeros_like(parent1)
        crossover_mask = np.random.random(self.n_params) < 0.5

        child[crossover_mask] = parent1[crossover_mask]
        child[~crossover_mask] = parent2[~crossover_mask]

        return child

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Apply Gaussian mutation"""
        mutated = individual.copy()

        for i in range(self.n_params):
            if np.random.random() < self.mutation_rate:
                min_val, max_val = self.param_ranges[self.param_names[i]]
                mutation_strength = (max_val - min_val) * 0.1

                mutated[i] += np.random.normal(0, mutation_strength)
                mutated[i] = np.clip(mutated[i], min_val, max_val)

        return mutated