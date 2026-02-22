"""
NEAT (NeuroEvolution of Augmenting Topologies) agent for Risiko.

Uses the neat-python library to evolve both network topology and weights.
Unlike the fixed-topology neuroevolution, NEAT starts with minimal networks
and complexifies them over generations through speciation.
"""

import neat
import numpy as np
import pickle
import os
import time as _time
from multiprocessing import Pool, cpu_count
from game import (RiskGame, GameState, NUM_TERRITORIES, NUM_PLAYERS,
                  INPUT_DIM, MAX_TURNS, territory_count)
from neural_net import RandomAgent


class NEATAgent:
    """
    Agent that uses a NEAT-evolved network for Risk decisions.
    Same interface as NeuralAgent: reinforce(), attack(), fortify().
    """

    def __init__(self, net: neat.nn.FeedForwardNetwork, rng=None):
        self.net = net
        self.rng = rng or np.random.default_rng()

    def _get_output(self, state_encoded: np.ndarray) -> np.ndarray:
        """Run the NEAT network and return output array."""
        output = self.net.activate(state_encoded.tolist())
        return np.array(output, dtype=np.float32)

    def reinforce(self, state_encoded: np.ndarray, n_armies: int,
                  owned_territories: np.ndarray) -> np.ndarray:
        """Distribute reinforcement armies."""
        output = self._get_output(state_encoded)
        scores = output[:42]

        # Softmax over owned territories
        owned_scores = scores[owned_territories]
        owned_scores = owned_scores - np.max(owned_scores)
        exp_scores = np.exp(owned_scores)
        probs = exp_scores / (np.sum(exp_scores) + 1e-10)

        distribution = np.zeros(len(owned_territories), dtype=np.int32)
        raw = probs * n_armies
        for i in range(len(owned_territories)):
            distribution[i] = int(raw[i])
        remaining = n_armies - distribution.sum()
        fractions = raw - distribution.astype(np.float64)
        order = np.argsort(-fractions)
        for i in range(remaining):
            distribution[order[i % len(order)]] += 1
        return distribution

    def attack(self, state_encoded: np.ndarray,
               valid_attacks: list[tuple[int, int]]) -> tuple[int, int] | None:
        """Decide attack action."""
        if not valid_attacks:
            return None

        output = self._get_output(state_encoded)
        src_scores = output[:42]
        tgt_scores = output[42:84]
        stop_score = output[84]

        best_score = stop_score
        best_attack = None

        for frm, to in valid_attacks:
            score = src_scores[frm] + tgt_scores[to]
            if score > best_score:
                best_score = score
                best_attack = (frm, to)

        return best_attack

    def fortify(self, state_encoded: np.ndarray,
                valid_fortifications: list[tuple[int, int]]) -> tuple[int, int, int] | None:
        """Decide fortification."""
        if not valid_fortifications:
            return None

        output = self._get_output(state_encoded)
        src_scores = output[:42]
        tgt_scores = output[42:84]
        stop_score = output[84]

        best_score = stop_score
        best_fort = None

        for frm, to in valid_fortifications:
            score = src_scores[frm] + tgt_scores[to]
            if score > best_score:
                best_score = score
                best_fort = (frm, to)

        if best_fort is None:
            return None

        sigmoid_val = 1.0 / (1.0 + np.exp(-best_score))
        return (best_fort[0], best_fort[1], max(1, int(sigmoid_val * 10)))


# ── Module-level worker for multiprocessing ──────────────────────────────────

def _eval_genome_worker(args):
    """Evaluate a single NEAT genome. Module-level for pickling."""
    genome_data, config_path, games_per_eval, seed = args
    rng = np.random.default_rng(seed)

    # Reconstruct config and genome in worker process
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    genome = genome_data
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    agent = NEATAgent(net, rng=rng)
    game = RiskGame(rng=rng)

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
        territories = territory_count(final_state, slot)

        if winner == slot:
            total_score += 1.0 + territories / 42.0
        else:
            total_score += territories / 42.0 * 0.5

    return total_score / games_per_eval


def evaluate_genome(genome, config, games_per_eval=3, seed=None):
    """
    Evaluate a single NEAT genome by playing Risk games.
    Kept for compatibility / single-process fallback.
    """
    rng = np.random.default_rng(seed)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    agent = NEATAgent(net, rng=rng)
    game = RiskGame(rng=rng)

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
        territories = territory_count(final_state, slot)

        if winner == slot:
            total_score += 1.0 + territories / 42.0
        else:
            total_score += territories / 42.0 * 0.5

    return total_score / games_per_eval


def run_neat(config_path: str = "neat_config.txt", n_generations: int = 30,
             games_per_eval: int = 3, save_path: str = "neat_best.pkl"):
    """
    Run NEAT evolution for Risiko with parallel genome evaluation.
    """
    config_path = os.path.abspath(config_path)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    n_workers = max(1, cpu_count() - 1)
    pop = neat.Population(config)

    # Reporters for progress
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    history = []
    best_win_rate = 0.0

    def eval_genomes(genomes, config):
        nonlocal best_win_rate
        t0 = _time.time()

        # Parallel evaluation
        args = [
            (genome, config_path, games_per_eval,
             int(np.random.default_rng().integers(0, 2**31)))
            for _, genome in genomes
        ]

        if n_workers > 1 and len(genomes) > 4:
            with Pool(n_workers) as pool:
                fitnesses = pool.map(_eval_genome_worker, args)
        else:
            fitnesses = [_eval_genome_worker(a) for a in args]

        for (_, genome), fit in zip(genomes, fitnesses):
            genome.fitness = fit

        elapsed = _time.time() - t0
        fit_values = [g.fitness for _, g in genomes]
        gen_stats = {
            "generation": len(history),
            "best_fitness": max(fit_values),
            "avg_fitness": float(np.mean(fit_values)),
            "std_fitness": float(np.std(fit_values)),
            "n_species": len(pop.species.species),
            "elapsed": elapsed,
        }
        history.append(gen_stats)

        print(f"  ⏱ {elapsed:.1f}s | Species: {gen_stats['n_species']}")

        # Periodic eval vs random
        if len(history) % 10 == 0:
            best_genome = max(genomes, key=lambda x: x[1].fitness)[1]
            net = neat.nn.FeedForwardNetwork.create(best_genome, config)
            agent = NEATAgent(net)
            wins = 0
            rng = np.random.default_rng()
            game = RiskGame(rng=rng)
            for _ in range(10):
                slot = int(rng.integers(0, NUM_PLAYERS))
                agents_list = []
                for s in range(NUM_PLAYERS):
                    if s == slot:
                        agents_list.append(agent)
                    else:
                        agents_list.append(RandomAgent(rng=rng))
                w, _ = game.play_game(agents_list)
                if w == slot:
                    wins += 1
            wr = wins / 10
            if wr > best_win_rate:
                best_win_rate = wr
            print(f"  └─ Win rate vs Random: {wr:.0%}")

    print(f"\n{'='*60}")
    print(f"  NEAT RISIKO — {config.pop_size} individui, {n_generations} generazioni")
    print(f"  Partite per valutazione: {games_per_eval}")
    print(f"  Workers paralleli: {n_workers}")
    print(f"{'='*60}\n")

    winner = pop.run(eval_genomes, n_generations)

    # Save best
    with open(save_path, "wb") as f:
        pickle.dump({
            "genome": winner,
            "config": config,
            "history": history,
            "best_win_rate": best_win_rate,
        }, f)
    print(f"\n✓ NEAT completato. Miglior genoma salvato in '{save_path}'")
    print(f"  Nodi: {len(winner.nodes)}, Connessioni: {len(winner.connections)}")

    return winner, config, history


def load_neat_agent(path: str = "neat_best.pkl") -> NEATAgent:
    """Load a trained NEAT agent from file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(data["genome"], data["config"])
    genome = data["genome"]
    print(f"NEAT agente caricato: {len(genome.nodes)} nodi, {len(genome.connections)} connessioni")
    return NEATAgent(net)
