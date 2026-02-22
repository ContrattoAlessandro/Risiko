"""
Main entry point for the CMA-ES version of Risiko.

CMA-ES adapts a covariance matrix to efficiently explore the
weight space —  much more sample-efficient than simple GA.

Run: python main_cmaes.py
     python main_cmaes.py quick    (fast test)
     python main_cmaes.py demo     (play demo game)
"""

import time
import sys
import numpy as np


def main():
    from cmaes_agent import run_cmaes, load_cmaes_agent
    from game import NUM_PLAYERS, INPUT_DIM
    from neural_net import RandomAgent, NeuralAgent

    n_generations = 50
    games_per_eval = 8

    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_game()
            return
        elif sys.argv[1] == "quick":
            n_generations = 10
            games_per_eval = 4
            print(">>> Modalità veloce: 10 generazioni, 4 partite/eval")

    # Calibrate input dim
    from game import RiskGame
    rng = np.random.default_rng(0)
    game = RiskGame(rng=rng)
    state = game.reset()
    input_dim = len(game.encode_state(state, 0))
    print(f"Dimensione input rete neurale: {input_dim}")

    start_time = time.time()
    best_genome, history = run_cmaes(
        n_generations=n_generations,
        games_per_eval=games_per_eval,
        input_dim=input_dim,
    )

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\nTempo totale: {minutes}m {seconds}s")

    print("\nPartita dimostrativa...")
    demo_game()


def demo_game(model_path: str = "cmaes_best.pkl"):
    """Play a demo game with CMA-ES agent vs 3 random agents."""
    from game import RiskGame, NUM_PLAYERS
    from cmaes_agent import load_cmaes_agent
    from neural_net import RandomAgent

    try:
        agent = load_cmaes_agent(model_path)
    except FileNotFoundError:
        print(f"File '{model_path}' non trovato.")
        return

    rng = np.random.default_rng()
    game = RiskGame(rng=rng)
    state = game.reset()

    agents = [agent, RandomAgent(rng=rng), RandomAgent(rng=rng), RandomAgent(rng=rng)]

    print("\n" + "=" * 60)
    print("  PARTITA DIMOSTRATIVA: Agente CMA-ES vs 3 Random")
    print("=" * 60)

    while not state.game_over:
        player = state.current_player
        state = game.play_turn(state, agents)

        if state.turn % 10 == 0 and player == 0:
            print(f"\nTurno {state.turn}:")
            for p in range(NUM_PLAYERS):
                t = int(np.sum(state.owner == p))
                marker = " ★" if p == 0 else ""
                status = " [ELIMINATO]" if state.eliminated[p] else ""
                print(f"  Giocatore {p}{marker}: {t} territori{status}")

    print(f"\n{'='*60}")
    if state.winner == 0:
        print(f"  ★ VITTORIA dell'agente CMA-ES al turno {state.turn}!")
    else:
        print(f"  Giocatore {state.winner} ha vinto al turno {state.turn}")
    for p in range(NUM_PLAYERS):
        t = int(np.sum(state.owner == p))
        marker = " ★" if p == 0 else ""
        print(f"    Giocatore {p}{marker}: {t} territori")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
