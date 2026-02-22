"""
Neuroevolution module for Risiko.

Implements a genetic algorithm to evolve neural network weights:
- Tournament selection
- Uniform crossover
- Gaussian mutation with adaptive sigma
- Elitism
- Parallel fitness evaluation via multiprocessing
"""

import numpy as np
import pickle
import time as _time
from multiprocessing import Pool, cpu_count
from neural_net import NeuralAgent, RandomAgent
from game import (RiskGame, GameState, NUM_PLAYERS, INPUT_DIM,
                  territory_count, compute_fitness_details)


class Individual:
    """A single individual in the population = a set of NN weights."""

    def __init__(self, genome: np.ndarray):
        self.genome = genome.astype(np.float32)
        self.fitness = 0.0

    def copy(self) -> "Individual":
        ind = Individual(self.genome.copy())
        ind.fitness = self.fitness
        return ind


# ── Standalone function for multiprocessing (must be at module level) ────────

def _evaluate_individual(args: tuple) -> float:
    """
    Evaluate a single individual's fitness by playing games.
    This is a module-level function so it can be pickled for multiprocessing.
    """
    genome, opponent_genomes, games_per_eval, input_dim, seed = args
    rng = np.random.default_rng(seed)
    game = RiskGame(rng=rng)

    # Create the agent for this individual
    def make_agent(g):
        a = NeuralAgent(input_dim=input_dim, rng=rng)
        a.set_params(g)
        return a

    total_score = 0.0
    for g_idx in range(games_per_eval):
        # Pick 3 opponents (cycling through available opponents)
        opp_indices = [(g_idx * 3 + j) % len(opponent_genomes) for j in range(3)]
        player_slot = g_idx % NUM_PLAYERS

        agents = []
        opp_iter = iter(opp_indices)
        for slot in range(NUM_PLAYERS):
            if slot == player_slot:
                agents.append(make_agent(genome))
            else:
                agents.append(make_agent(opponent_genomes[next(opp_iter)]))

        winner, final_state = game.play_game(agents)
        details = compute_fitness_details(final_state, player_slot)

        # Rich multi-component fitness
        score = 0.0
        score += details["territory_frac"] * 0.30        # territory control
        score += details["continent_progress"] * 0.20    # continent progress
        score += details["army_ratio"] * 0.15             # army advantage
        score += details["border_strength"] * 0.10        # border defense

        if winner == player_slot:
            score += 1.0                                  # win bonus
            score += (1.0 - final_state.turn / 150) * 0.25  # speed bonus
        elif details["territory_frac"] == 0:
            score -= 0.2                                  # elimination penalty

        total_score += score

    return total_score / games_per_eval


class NeuroEvolution:
    """Genetic algorithm for evolving Risiko-playing neural networks."""

    def __init__(
        self,
        population_size: int = 48,
        games_per_eval: int = 3,
        tournament_size: int = 5,
        mutation_rate: float = 0.1,
        mutation_sigma: float = 0.2,
        crossover_rate: float = 0.7,
        elite_fraction: float = 0.05,
        input_dim: int = INPUT_DIM,
        seed: int = 42,
        n_workers: int = 0,  # 0 = auto-detect
    ):
        self.pop_size = population_size
        self.games_per_eval = games_per_eval
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.crossover_rate = crossover_rate
        self.elite_count = max(1, int(population_size * elite_fraction))
        self.input_dim = input_dim
        self.rng = np.random.default_rng(seed)
        self.n_workers = n_workers if n_workers > 0 else max(1, cpu_count() - 1)

        template = NeuralAgent(input_dim=input_dim)
        self.param_count = template.param_count()
        print(f"[Evolution] Parametri per agente: {self.param_count}")
        print(f"[Evolution] Workers paralleli: {self.n_workers}")

        # Initialize population
        self.population: list[Individual] = []
        for _ in range(population_size):
            genome = self.rng.standard_normal(self.param_count).astype(np.float32) * 0.5
            self.population.append(Individual(genome))

        self.history: list[dict] = []
        self.generation = 0

    def _create_agent(self, genome: np.ndarray) -> NeuralAgent:
        agent = NeuralAgent(input_dim=self.input_dim, rng=self.rng)
        agent.set_params(genome)
        return agent

    def evaluate_fitness(self):
        """Evaluate fitness of all individuals in parallel using multiprocessing."""
        # Collect all genomes for opponent sampling
        all_genomes = [ind.genome for ind in self.population]

        # Build args for each individual
        eval_args = []
        for i, ind in enumerate(self.population):
            # Opponents = everyone except self
            opp_genomes = all_genomes[:i] + all_genomes[i+1:]
            seed = int(self.rng.integers(0, 2**31))
            eval_args.append((
                ind.genome, opp_genomes, self.games_per_eval,
                self.input_dim, seed
            ))

        # Parallel evaluation
        if self.n_workers > 1:
            with Pool(self.n_workers) as pool:
                fitnesses = pool.map(_evaluate_individual, eval_args)
        else:
            fitnesses = [_evaluate_individual(a) for a in eval_args]

        for ind, fit in zip(self.population, fitnesses):
            ind.fitness = fit

    def evaluate_vs_random(self, genome: np.ndarray, n_games: int = 10) -> float:
        """Evaluate an agent against RandomAgents. Returns win rate."""
        rng = np.random.default_rng()
        game = RiskGame(rng=rng)
        wins = 0
        for _ in range(n_games):
            slot = int(rng.integers(0, NUM_PLAYERS))
            agents = []
            for s in range(NUM_PLAYERS):
                if s == slot:
                    agents.append(self._create_agent(genome))
                else:
                    agents.append(RandomAgent(rng=rng))
            winner, _ = game.play_game(agents)
            if winner == slot:
                wins += 1
        return wins / n_games

    def tournament_select(self) -> Individual:
        candidates = self.rng.choice(len(self.population), size=self.tournament_size, replace=False)
        best = max(candidates, key=lambda i: self.population[i].fitness)
        return self.population[best]

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        mask = self.rng.random(self.param_count) < 0.5
        child_genome = np.where(mask, parent1.genome, parent2.genome)
        return Individual(child_genome)

    def mutate(self, individual: Individual) -> Individual:
        genome = individual.genome.copy()
        mutation_mask = self.rng.random(self.param_count) < self.mutation_rate
        noise = self.rng.standard_normal(self.param_count).astype(np.float32) * self.mutation_sigma
        genome += mutation_mask * noise
        return Individual(genome)

    def evolve_generation(self):
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        fitnesses = [ind.fitness for ind in self.population]
        stats = {
            "generation": self.generation,
            "best_fitness": fitnesses[0],
            "avg_fitness": float(np.mean(fitnesses)),
            "worst_fitness": fitnesses[-1],
            "std_fitness": float(np.std(fitnesses)),
        }
        self.history.append(stats)

        new_population: list[Individual] = []
        for i in range(self.elite_count):
            new_population.append(self.population[i].copy())

        while len(new_population) < self.pop_size:
            parent1 = self.tournament_select()
            if self.rng.random() < self.crossover_rate:
                parent2 = self.tournament_select()
                child = self.crossover(parent1, parent2)
            else:
                child = parent1.copy()
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population[:self.pop_size]
        self.generation += 1

    def run(self, n_generations: int = 30, save_path: str = "best_agent.pkl",
            log_path: str = "evolution_log.pkl"):
        """Run the full evolution loop."""
        print(f"\n{'='*60}")
        print(f"  NEUROEVOLUZIONE RISIKO — {self.pop_size} individui, {n_generations} generazioni")
        print(f"  Partite per valutazione: {self.games_per_eval}")
        print(f"  Mutazione: rate={self.mutation_rate}, σ={self.mutation_sigma}")
        print(f"  Workers: {self.n_workers}")
        print(f"{'='*60}\n")

        for gen in range(n_generations):
            t0 = _time.time()
            self.evaluate_fitness()
            self.evolve_generation()
            elapsed = _time.time() - t0

            stats = self.history[-1]
            print(
                f"Gen {stats['generation']:3d} | "
                f"Best: {stats['best_fitness']:.3f} | "
                f"Avg: {stats['avg_fitness']:.3f} | "
                f"Std: {stats['std_fitness']:.3f} | "
                f"{elapsed:.1f}s"
            )

            if (gen + 1) % 10 == 0 or gen == n_generations - 1:
                best_genome = self.population[0].genome
                self._save_agent(best_genome, save_path)
                self._save_log(log_path)
                win_rate = self.evaluate_vs_random(best_genome, n_games=10)
                print(f"        └─ Win rate vs Random: {win_rate:.0%}")

        print(f"\n✓ Evoluzione completata. Miglior agente salvato in '{save_path}'")
        return self.population[0]

    def _save_agent(self, genome: np.ndarray, path: str):
        data = {
            "genome": genome,
            "input_dim": self.input_dim,
            "generation": self.generation,
            "fitness": self.population[0].fitness,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _save_log(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.history, f)

    @staticmethod
    def load_agent(path: str) -> NeuralAgent:
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = NeuralAgent(input_dim=data["input_dim"])
        agent.set_params(data["genome"])
        print(f"Agente caricato: generazione {data['generation']}, fitness {data['fitness']:.3f}")
        return agent
