"""
Main entry point for the NEAT version of Risiko.

Evolves both network topology and weights using NEAT-python.
Run: python main_neat.py
      python main_neat.py quick    (fast test)
      python main_neat.py demo     (play demo game)
"""

import time
import sys
import numpy as np


def main():
    from neat_agent import run_neat, load_neat_agent
    from game import NUM_PLAYERS
    from neural_net import RandomAgent

    n_generations = 30
    games_per_eval = 3

    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_game()
            return
        elif sys.argv[1] == "quick":
            n_generations = 5
            games_per_eval = 2
            print(">>> Modalità veloce: 5 generazioni")

    start_time = time.time()
    winner, config, history = run_neat(
        n_generations=n_generations,
        games_per_eval=games_per_eval,
    )

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\nTempo totale: {minutes}m {seconds}s")
    print(f"Topologia finale: {len(winner.nodes)} nodi, {len(winner.connections)} connessioni")

    print("\nPartita dimostrativa...")
    demo_game()


def demo_game(model_path: str = "neat_best.pkl"):
    """Play a demo game with NEAT agent vs 3 random agents."""
    from game import RiskGame, NUM_PLAYERS
    from neat_agent import load_neat_agent
    from neural_net import RandomAgent

    try:
        agent = load_neat_agent(model_path)
    except FileNotFoundError:
        print(f"File '{model_path}' non trovato.")
        return

    rng = np.random.default_rng()
    game = RiskGame(rng=rng)
    state = game.reset()

    agents = [agent, RandomAgent(rng=rng), RandomAgent(rng=rng), RandomAgent(rng=rng)]

    print("\n" + "=" * 60)
    print("  PARTITA DIMOSTRATIVA: Agente NEAT vs 3 Random")
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
        print(f"  ★ VITTORIA dell'agente NEAT al turno {state.turn}!")
    else:
        print(f"  Giocatore {state.winner} ha vinto al turno {state.turn}")
    for p in range(NUM_PLAYERS):
        t = int(np.sum(state.owner == p))
        marker = " ★" if p == 0 else ""
        print(f"    Giocatore {p}{marker}: {t} territori")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
