"""
Run a simple tournament between the MCTS Agent and Random Agents or Neural Agents.
Observe how the MCTS Agent performs without ANY training!
"""

import time
import numpy as np
from game import RiskGame, NUM_PLAYERS
from mcts_agent import MCTSAgent
from neural_net import RandomAgent
from cmaes_agent import load_cmaes_agent

def main():
    rng = np.random.default_rng(42)
    game = RiskGame(rng=rng)
    
    # 1 Agent MCTS vs 3 Agent Casuali
    mcts_agent = MCTSAgent(time_limit=0.25, rng=rng)
    random_agents = [RandomAgent(rng=rng) for _ in range(3)]
    
    agents = [mcts_agent] + random_agents
    
    print("\n" + "=" * 60)
    print("  TORNEO MCTS vs 3 RANDOM")
    print("=" * 60)
    
    state = game.reset()
    start_time = time.time()
    
    while not state.game_over:
        player = state.current_player
        state = game.play_turn(state, agents)
        
        if state.turn % 5 == 0 and player == 0:
            print(f"Turno {state.turn}:")
            for p in range(NUM_PLAYERS):
                t = int(np.sum(state.owner == p))
                marker = " [MCTS]" if p == 0 else ""
                status = " (Eliminato)" if state.eliminated[p] else ""
                print(f"  Giocatore {p}{marker}: {t} territori{status}")
                
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    if state.winner == 0:
        print(f"  ★ VITTORIA dell'agente MCTS al turno {state.turn}!")
    else:
        print(f"  Vittoria del Giocatore {state.winner} al turno {state.turn}")
        
    for p in range(NUM_PLAYERS):
        t = int(np.sum(state.owner == p))
        marker = " [MCTS]" if p == 0 else ""
        print(f"    Giocatore {p}{marker}: {t} territori")
    print(f"  Tempo partita: {elapsed:.2f}s")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
