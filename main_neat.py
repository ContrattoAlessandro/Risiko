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
    games_per_eval = 20

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

    # --- Print details for CMA-ES translation ---
    analyze_and_print_architecture(winner, config)

    print("\nPartita dimostrativa...")
    demo_game()


def analyze_and_print_architecture(genome, config):
    print("\n" + "=" * 60)
    print("  ANALISI ARCHITETTURA RETE NEAT (PER CMA-ES)")
    print("=" * 60)
    
    # Extract keys
    input_keys = config.genome_config.input_keys
    output_keys = config.genome_config.output_keys
    
    # Identify hidden nodes
    hidden_nodes = [k for k in genome.nodes.keys() if k not in output_keys]
    
    print(f"Input: {len(input_keys)} nodi")
    print(f"Output: {len(output_keys)} nodi")
    print(f"Nodi Nascosti (Hidden): {len(hidden_nodes)} nodi")
    
    enabled_connections = [cg for cg in genome.connections.values() if cg.enabled]
    print(f"Connessioni Attive: {len(enabled_connections)} (su {len(genome.connections)} totali)")
    
    activations = {}
    for n in genome.nodes.values():
        act = n.activation
        activations[act] = activations.get(act, 0) + 1
    
    print("\nFunzioni di Attivazione usate nei nodi (nascosti + output):")
    for act, count in activations.items():
        print(f"  - {act}: {count} nodi")
        
    print("\n--- Suggerimento per CMA-ES ---")
    print("Poiché NEAT evolve topologie arbitrarie (es. connessioni dirette Input->Output, nodi non stratificati),")
    print("è difficile tradurla in una classica rete Dense a layer perfetti rigorosi.")
    print("Tuttavia, puoi replicarne approssimativamente la *capacità* in CMA-ES creando una rete Dense così:")
    print(f"Input Layer    : {len(input_keys)} neuroni")
    if len(hidden_nodes) > 0:
        print(f"Hidden Layer 1 : {len(hidden_nodes)} neuroni")
    print(f"Output Layer   : {len(output_keys)} neuroni")
    print("\nEsempio di topologia ('layer_sizes') da inserire per CMA-ES:")
    if len(hidden_nodes) > 0:
        print(f"[{len(input_keys)}, {len(hidden_nodes)}, {len(output_keys)}]")
    else:
        print(f"[{len(input_keys)}, {len(output_keys)}]  (Nessun hidden layer, corrisponde a un modello lineare)")
    print("=" * 60 + "\n")


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
