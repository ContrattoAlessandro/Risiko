"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for Risiko.

CMA-ES is a derivative-free optimization algorithm that maintains
a multivariate normal distribution over the search space. It adapts
the covariance matrix to model correlations between parameters,
making it much more efficient than simple GA for neural network
weight optimization.

Run via: python main_cmaes.py
"""

import numpy as np
import cma
import pickle
import time as _time
from multiprocessing import Pool, cpu_count
from neural_net import NeuralAgent, RandomAgent
from game import (RiskGame, NUM_PLAYERS, INPUT_DIM,
                  territory_count, compute_fitness_details)


# ── Module-level worker for multiprocessing ──────────────────────────────────

def _evaluate_solution(args: tuple) -> float:
    """
    Evaluate a single CMA-ES solution (weight vector) by playing Risk games.
    Module-level function for multiprocessing compatibility.
    Returns NEGATIVE fitness (CMA-ES minimizes by default).
    """
    genome, games_per_eval, input_dim, seed = args
    rng = np.random.default_rng(seed)
    game = RiskGame(rng=rng)

    agent = NeuralAgent(input_dim=input_dim, rng=rng, hidden_layers=[])
    agent.set_params(genome.astype(np.float32))

    total_score = 0.0
    for g in range(games_per_eval):
        slot = g % NUM_PLAYERS
        agents = []
        for s in range(NUM_PLAYERS):
            if s == slot:
                agents.append(agent)
            else:
                agents.append(RandomAgent(rng=rng))

        winner, final_state = game.play_game(agents)
        details = compute_fitness_details(final_state, slot)

        # Rich multi-component fitness
        score = 0.0
        score += details["territory_frac"] * 0.30
        score += details["continent_progress"] * 0.20
        score += details["army_ratio"] * 0.15
        score += details["border_strength"] * 0.10

        if winner == slot:
            score += 1.0
            score += (1.0 - final_state.turn / 150) * 0.25
        elif details["territory_frac"] == 0:
            score -= 0.2

        total_score += score

    fitness = total_score / games_per_eval
    return -fitness  # CMA-ES minimizes, so negate


def evaluate_vs_random(genome: np.ndarray, input_dim: int,
                       n_games: int = 20) -> float:
    """Evaluate an agent against RandomAgents. Returns win rate."""
    rng = np.random.default_rng()
    game = RiskGame(rng=rng)
    agent = NeuralAgent(input_dim=input_dim, rng=rng, hidden_layers=[])
    agent.set_params(genome.astype(np.float32))

    wins = 0
    for _ in range(n_games):
        slot = int(rng.integers(0, NUM_PLAYERS))
        agents = []
        for s in range(NUM_PLAYERS):
            if s == slot:
                agents.append(agent)
            else:
                agents.append(RandomAgent(rng=rng))
        winner, _ = game.play_game(agents)
        if winner == slot:
            wins += 1
    return wins / n_games


def run_cmaes(
    n_generations: int = 50,
    games_per_eval: int = 8,
    sigma0: float = 0.5,
    input_dim: int = INPUT_DIM,
    save_path: str = "cmaes_best.pkl",
    log_path: str = "cmaes_log.pkl",
    popsize: int = 0,  # 0 = let CMA-ES choose
):
    """
    Run CMA-ES evolution for Risiko.

    Args:
        n_generations: Number of generations to run.
        games_per_eval: Games per fitness evaluation.
        sigma0: Initial step size (standard deviation).
        input_dim: Neural network input dimension.
        save_path: Path to save the best agent.
        log_path: Path to save training log.
        popsize: Population size (0 = CMA-ES default ~4+3*ln(N)).
    """
    n_workers = max(1, cpu_count() - 1)

    # Create template to get parameter count (using no hidden layers for fewer parameters)
    template = NeuralAgent(input_dim=input_dim, hidden_layers=[])
    n_params = template.param_count()

    # Initial mean: small random values
    x0 = np.random.default_rng(42).standard_normal(n_params) * 0.1

    # CMA-ES options
    opts = {
        "maxiter": n_generations,
        "verb_disp": 0,           # we print our own output
        "verb_log": 0,
        "tolfun": 1e-11,          # don't stop early on fitness
        "tolx": 1e-12,            # don't stop early on step size
        "CMA_diagonal": True,     # separable CMA: O(N) instead of O(N²)
    }
    if popsize > 0:
        opts["popsize"] = popsize

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    actual_popsize = es.popsize

    print(f"\n{'='*60}")
    print(f"  CMA-ES RISIKO (separable/diagonal)")
    print(f"  Parametri: {n_params} | Popolazione: {actual_popsize}")
    print(f"  Partite per valutazione: {games_per_eval}")
    print(f"  Sigma iniziale: {sigma0}")
    print(f"  Workers paralleli: {n_workers}")
    print(f"{'='*60}\n")

    history = []
    best_fitness_ever = float("-inf")
    best_genome_ever = None

    gen = 0
    # Persistent pool — avoids respawning processes each generation
    pool = Pool(n_workers) if n_workers > 1 else None
    try:
        while not es.stop():
            t0 = _time.time()

            # Ask CMA-ES for candidate solutions
            solutions = es.ask()

            # Evaluate in parallel
            eval_args = [
                (sol, games_per_eval, input_dim,
                 int(np.random.default_rng().integers(0, 2**31)))
                for sol in solutions
            ]

            if pool is not None:
                neg_fitnesses = pool.map(_evaluate_solution, eval_args)
            else:
                neg_fitnesses = [_evaluate_solution(a) for a in eval_args]

            # Tell CMA-ES the results
            es.tell(solutions, neg_fitnesses)

            # Track stats (convert back to positive fitness)
            fitnesses = [-f for f in neg_fitnesses]
            best_fit = max(fitnesses)
            avg_fit = float(np.mean(fitnesses))
            std_fit = float(np.std(fitnesses))
            elapsed = _time.time() - t0

            # Track best ever
            if best_fit > best_fitness_ever:
                best_fitness_ever = best_fit
                best_idx = fitnesses.index(best_fit)
                best_genome_ever = solutions[best_idx].copy()

            gen_stats = {
                "generation": gen,
                "best_fitness": best_fit,
                "best_ever": best_fitness_ever,
                "avg_fitness": avg_fit,
                "std_fitness": std_fit,
                "sigma": float(es.sigma),
                "elapsed": elapsed,
            }
            history.append(gen_stats)

            print(
                f"Gen {gen:3d} | "
                f"Best: {best_fit:.3f} (ever: {best_fitness_ever:.3f}) | "
                f"Avg: {avg_fit:.3f} | "
                f"σ: {es.sigma:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Periodic win-rate evaluation
            if (gen + 1) % 10 == 0 or gen == 0:
                wr = evaluate_vs_random(best_genome_ever, input_dim, n_games=20)
                print(f"        └─ Win rate vs Random: {wr:.0%}")

            # Periodic save
            if (gen + 1) % 10 == 0:
                _save_agent(best_genome_ever, input_dim, best_fitness_ever,
                            gen, history, save_path)
                _save_log(history, log_path)

            gen += 1

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Final save
    _save_agent(best_genome_ever, input_dim, best_fitness_ever,
                gen, history, save_path)
    _save_log(history, log_path)

    print(f"\n✓ CMA-ES completato dopo {gen} generazioni.")
    print(f"  Miglior fitness: {best_fitness_ever:.4f}")
    print(f"  Agente salvato in '{save_path}'")

    # Final win rate
    wr = evaluate_vs_random(best_genome_ever, input_dim, n_games=30)
    print(f"  Win rate finale vs Random: {wr:.0%}")

    return best_genome_ever, history


def _save_agent(genome, input_dim, fitness, generation, history, path):
    data = {
        "genome": genome.astype(np.float32),
        "input_dim": input_dim,
        "fitness": fitness,
        "generation": generation,
        "history": history,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _save_log(history, path):
    with open(path, "wb") as f:
        pickle.dump(history, f)


def load_cmaes_agent(path: str = "cmaes_best.pkl") -> NeuralAgent:
    """Load a CMA-ES trained agent from file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    agent = NeuralAgent(input_dim=data["input_dim"], hidden_layers=[])
    agent.set_params(data["genome"])
    print(f"CMA-ES agente caricato: gen {data['generation']}, "
          f"fitness {data['fitness']:.4f}")
    return agent
