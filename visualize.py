"""
Visualization of training progress for both neuroevolution and PPO.

Usage:
    python visualize.py                # plot neuroevolution log
    python visualize.py ppo            # plot PPO log
    python visualize.py compare        # compare both
"""

import pickle
import sys
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib non trovato. Installa con: pip install matplotlib")


def load_evo_log(path: str = "evolution_log.pkl") -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_ppo_log(path: str = "ppo_agent.pt") -> list[dict]:
    import torch
    data = torch.load(path, weights_only=False)
    return data.get("history", [])


def plot_evolution(history: list[dict], save_path: str = "fitness_plot.png"):
    """Plot neuroevolution fitness over generations."""
    if not HAS_MPL or not history:
        return

    generations = [h["generation"] for h in history]
    best = [h["best_fitness"] for h in history]
    avg = [h["avg_fitness"] for h in history]
    std = [h["std_fitness"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(generations, best, "g-", lw=2, label="Migliore")
    ax.plot(generations, avg, "b-", lw=1.5, label="Media")
    ax.fill_between(generations,
                     np.array(avg) - np.array(std),
                     np.array(avg) + np.array(std),
                     alpha=0.2, color="blue", label="±1σ")
    ax.set_xlabel("Generazione")
    ax.set_ylabel("Fitness")
    ax.set_title("Neuroevoluzione — Fitness")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(generations, std, "purple", lw=1.5)
    ax.set_xlabel("Generazione")
    ax.set_ylabel("σ Fitness")
    ax.set_title("Diversità Popolazione")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Grafico salvato in '{save_path}'")


def plot_ppo(history: list[dict], save_path: str = "ppo_plot.png"):
    """Plot PPO training metrics."""
    if not HAS_MPL or not history:
        print("Nessun dato PPO disponibile.")
        return

    iters = [h["iteration"] for h in history]
    win_rates = [h.get("win_rate", 0) for h in history]
    policy_loss = [h.get("policy_loss", 0) for h in history]
    entropy = [h.get("entropy", 0) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(iters, win_rates, "g-o", lw=2, markersize=4)
    ax.axhline(y=0.25, color="red", ls="--", alpha=0.5, label="Random (25%)")
    ax.set_xlabel("Iterazione")
    ax.set_ylabel("Win Rate vs Random")
    ax.set_title("PPO — Win Rate")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(iters, policy_loss, "b-", lw=1.5)
    ax.set_xlabel("Iterazione")
    ax.set_ylabel("Policy Loss")
    ax.set_title("PPO — Policy Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(iters, entropy, "orange", lw=1.5)
    ax.set_xlabel("Iterazione")
    ax.set_ylabel("Entropia")
    ax.set_title("PPO — Entropia Policy")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Grafico salvato in '{save_path}'")


def plot_comparison(save_path: str = "comparison_plot.png"):
    """Compare neuroevolution and PPO win rates."""
    if not HAS_MPL:
        return

    # Load both logs
    try:
        evo_log = load_evo_log()
        has_evo = True
    except FileNotFoundError:
        has_evo = False
        print("Log neuroevoluzione non trovato.")

    try:
        ppo_log = load_ppo_log()
        has_ppo = bool(ppo_log)
    except FileNotFoundError:
        has_ppo = False
        print("Log PPO non trovato.")

    if not has_evo and not has_ppo:
        print("Nessun dato da confrontare.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if has_evo:
        gens = [h["generation"] for h in evo_log]
        best = [h["best_fitness"] for h in evo_log]
        # Normalize to approx win rate scale (best_fitness max ~2)
        ax.plot(gens, [b / 2.0 for b in best], "b-", lw=2,
                label="Neuroevoluzione (best fitness / 2)")

    if has_ppo:
        iters = [h["iteration"] for h in ppo_log]
        wr = [h.get("win_rate", 0) for h in ppo_log]
        ax.plot(iters, wr, "r-", lw=2, label="PPO (win rate)")

    ax.axhline(y=0.25, color="gray", ls="--", alpha=0.5, label="Baseline (random)")
    ax.set_xlabel("Iterazione / Generazione")
    ax.set_ylabel("Performance")
    ax.set_title("Confronto: Neuroevoluzione vs PPO")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Grafico salvato in '{save_path}'")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "evo"

    if mode == "ppo":
        try:
            history = load_ppo_log()
            plot_ppo(history)
        except FileNotFoundError:
            print("File 'ppo_agent.pt' non trovato. Esegui prima 'python main_ppo.py'.")
    elif mode == "compare":
        plot_comparison()
    else:
        try:
            history = load_evo_log()
            print_summary(history)
            plot_evolution(history)
        except FileNotFoundError:
            print("File 'evolution_log.pkl' non trovato. Esegui prima 'python main.py'.")


def print_summary(history: list[dict]):
    if not history:
        return
    first, last = history[0], history[-1]
    print(f"\n{'='*50}")
    print(f"  RIEPILOGO EVOLUZIONE")
    print(f"{'='*50}")
    print(f"  Generazioni:     {len(history)}")
    print(f"  Fitness iniziale: best={first['best_fitness']:.3f}, avg={first['avg_fitness']:.3f}")
    print(f"  Fitness finale:   best={last['best_fitness']:.3f}, avg={last['avg_fitness']:.3f}")
    print(f"  Miglioramento:    {last['best_fitness'] - first['best_fitness']:+.3f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
