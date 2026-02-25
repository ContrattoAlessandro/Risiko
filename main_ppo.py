"""
Main entry point for training and testing the PPO agent in Risiko.
Run:
    python main_ppo.py
    python main_ppo.py quick
    python main_ppo.py demo
"""

import time
import sys
import numpy as np
import torch
import os

from game import RiskGame, NUM_PLAYERS, compute_fitness_details
from ppo_agent import PPOAgent
from neural_net import RandomAgent

def evaluate_vs_random(agent, n_games=20):
    """Evaluate PPO agent against RandomAgents. Returns win rate."""
    rng = np.random.default_rng()
    game = RiskGame(rng=rng)
    
    agent.set_eval() # No gradients
    
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
            
    agent.set_train()
    return wins / n_games

def train_ppo(episodes=500, save_path="ppo_best.pth", input_dim=144):
    rng = np.random.default_rng()
    game = RiskGame(rng=rng)
    
    agent = PPOAgent(input_dim=input_dim, rng=rng)
    
    # Check for existing
    if os.path.exists(save_path):
        agent.load(save_path)
        
    print(f"\n{'='*60}")
    print(f"  TRAINING PPO RISIKO")
    print(f"  Device: {agent.device}")
    print(f"  Episodes: {episodes}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    best_wr = -1.0
    history = []
    
    for ep in range(1, episodes + 1):
        state = game.reset()
        slot = 0 # PPO is always player 0 in training games for simplicity
        
        agents = [agent, RandomAgent(rng=rng), RandomAgent(rng=rng), RandomAgent(rng=rng)]
        
        # Play one full episode
        while not state.game_over:
            state = game.play_turn(state, agents)
            
        # Game over, compute terminal reward for the PPO agent
        details = compute_fitness_details(state, slot)
        
        # Reward shaping
        reward = 0.0
        reward += details["territory_frac"] * 0.40
        reward += details["continent_progress"] * 0.30
        reward += details["army_ratio"] * 0.20
        
        if state.winner == slot:
            reward += 1.0
            # Time bonus
            reward += (1.0 - state.turn / 150) * 0.25
        elif details["territory_frac"] == 0:
            reward -= 0.2
            
        # Assign this terminal reward to the last step of the agent
        agent.store_reward(reward, done=True)
        
        # Perform PPO Update
        metrics = agent.update()
        
        if ep % 10 == 0:
            elapsed = time.time() - start_time
            if metrics:
                print(f"Ep {ep:4d} | Turn: {state.turn:3d} | Reward: {reward:.3f} | "
                      f"Actor Loss: {metrics['actor_loss']:.3f} | "
                      f"Critic Loss: {metrics['critic_loss']:.3f} | "
                      f"Entropy: {metrics['entropy']:.3f} | {elapsed:.1f}s")
            else:
                print(f"Ep {ep:4d} | No steps taken by agent.")
                
            # Periodically evaluate
            if ep % 50 == 0:
                wr = evaluate_vs_random(agent, n_games=20)
                print(f"        └─ Win rate vs Random: {wr:.0%}")
                if wr >= best_wr:
                    best_wr = wr
                    agent.save(save_path)
                    print(f"        └─ Saved new best (WR: {wr:.0%})")

    # Final save
    agent.save(save_path.replace(".pth", "_final.pth"))
    print(f"\n✓ PPO completato dopo {episodes} episodi.")
    
    return agent

def demo_game(model_path="ppo_best.pth", input_dim=144):
    """Play a demo game with PPO agent vs 3 random agents."""
    agent = PPOAgent(input_dim=input_dim)
    agent.load(model_path)
    agent.set_eval()
    
    rng = np.random.default_rng()
    game = RiskGame(rng=rng)
    state = game.reset()
    
    agents = [agent, RandomAgent(rng=rng), RandomAgent(rng=rng), RandomAgent(rng=rng)]

    print("\n" + "=" * 60)
    print("  PARTITA DIMOSTRATIVA: Agente PPO vs 3 Random")
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
        print(f"  ★ VITTORIA dell'agente PPO al turno {state.turn}!")
    else:
        print(f"  Giocatore {state.winner} ha vinto al turno {state.turn}")
    for p in range(NUM_PLAYERS):
        t = int(np.sum(state.owner == p))
        marker = " ★" if p == 0 else ""
        print(f"    Giocatore {p}{marker}: {t} territori")
    print(f"{'='*60}\n")


def main():
    n_episodes = 500
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_game()
            return
        elif sys.argv[1] == "quick":
            n_episodes = 20
            print(">>> Modalità veloce PPO: 20 episodi")

    # Calibrate input dim
    rng = np.random.default_rng(0)
    game = RiskGame(rng=rng)
    state = game.reset()
    input_dim = len(game.encode_state(state, 0))
    print(f"Dimensione input rete neurale: {input_dim}")

    train_ppo(episodes=n_episodes, input_dim=input_dim)


if __name__ == "__main__":
    main()
