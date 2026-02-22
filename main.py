"""
Main entry point for the Risiko Neuroevolution project.

Runs the evolutionary optimization to train neural network agents.
"""

import time
import sys


def calibrate_input_dim() -> int:
    """Run a dummy game to determine the exact input dimension."""
    import numpy as np
    from game import RiskGame
    rng = np.random.default_rng(0)
    game = RiskGame(rng=rng)
    state = game.reset()
    encoded = game.encode_state(state, 0)
    return len(encoded)


def demo_game(agent_path: str = "best_agent.pkl"):
    """Play a demo game with the best trained agent vs 3 random agents."""
    import numpy as np
    from game import RiskGame, NUM_PLAYERS
    from neural_net import RandomAgent
    from evolution import NeuroEvolution

    agent = NeuroEvolution.load_agent(agent_path)
    rng = np.random.default_rng()
    game = RiskGame(rng=rng)
    state = game.reset()

    agents = [agent, RandomAgent(rng=rng), RandomAgent(rng=rng), RandomAgent(rng=rng)]

    print("\n" + "=" * 60)
    print("  PARTITA DIMOSTRATIVA: Agente Evoluto vs 3 Random")
    print("=" * 60)

    while not state.game_over:
        player = state.current_player
        state = game.play_turn(state, agents)

        if state.turn % 10 == 0 and player == 0:
            print(f"\nTurno {state.turn}:")
            for p in range(NUM_PLAYERS):
                t = int(np.sum(state.owner == p))
                a = int(np.sum(state.armies[state.owner == p]))
                marker = " ★" if p == 0 else ""
                status = " [ELIMINATO]" if state.eliminated[p] else ""
                print(f"  Giocatore {p}{marker}: {t} territori, {a} armate{status}")

    print(f"\n{'='*60}")
    if state.winner == 0:
        print(f"  ★ VITTORIA dell'agente evoluto al turno {state.turn}!")
    else:
        print(f"  Giocatore {state.winner} ha vinto al turno {state.turn}")
    for p in range(NUM_PLAYERS):
        t = int(np.sum(state.owner == p))
        marker = " ★" if p == 0 else ""
        print(f"    Giocatore {p}{marker}: {t} territori")
    print(f"{'='*60}\n")


def main():
    input_dim = calibrate_input_dim()
    print(f"Dimensione input rete neurale: {input_dim}")

    config = {
        "population_size": 48,
        "games_per_eval": 8,
        "tournament_size": 5,
        "mutation_rate": 0.1,
        "mutation_sigma": 0.2,
        "crossover_rate": 0.7,
        "elite_fraction": 0.05,
        "input_dim": input_dim,
        "seed": 42,
    }

    n_generations = 30

    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_game()
            return
        elif sys.argv[1] == "quick":
            config["population_size"] = 12
            config["games_per_eval"] = 2
            n_generations = 5
            print(">>> Modalità veloce: 12 individui, 5 generazioni")

    from evolution import NeuroEvolution

    start_time = time.time()
    evo = NeuroEvolution(**config)
    best = evo.run(n_generations=n_generations)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\nTempo totale: {minutes}m {seconds}s")

    print("\n" + "-" * 60)
    print("Partita dimostrativa con il miglior agente...")
    demo_game()


# IMPORTANT: on Windows, multiprocessing requires the __name__ guard
if __name__ == "__main__":
    main()
