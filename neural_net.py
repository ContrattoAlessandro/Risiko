"""
Neural network agent for Risiko.

Feed-forward neural network with weights evolved by neuroevolution.
No backpropagation — weights are set externally.
"""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-10)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


class NeuralNetwork:
    """
    Simple feed-forward neural network.
    Architecture: input → hidden1 (ReLU) → hidden2 (ReLU) → output
    """

    def __init__(self, layer_sizes: list[int]):
        """
        layer_sizes: e.g. [input_dim, 64, 32, output_dim]
        """
        self.layer_sizes = layer_sizes
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            w = np.zeros((layer_sizes[i], layer_sizes[i + 1]), dtype=np.float32)
            b = np.zeros(layer_sizes[i + 1], dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = relu(x)
        return x

    def get_params(self) -> np.ndarray:
        """Flatten all weights and biases into a single 1D vector."""
        params = []
        for w, b in zip(self.weights, self.biases):
            params.append(w.ravel())
            params.append(b.ravel())
        return np.concatenate(params).astype(np.float32)

    def set_params(self, params: np.ndarray):
        """Set weights and biases from a flat 1D vector."""
        idx = 0
        for i in range(len(self.weights)):
            w_size = self.weights[i].size
            self.weights[i] = params[idx:idx + w_size].reshape(self.weights[i].shape).copy()
            idx += w_size
            b_size = self.biases[i].size
            self.biases[i] = params[idx:idx + b_size].copy()
            idx += b_size

    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))


class NeuralAgent:
    """
    Agent that uses a neural network to make decisions for Risiko.
    Handles all three game phases: reinforce, attack, fortify.
    """

    # Network architecture
    INPUT_DIM = 144   # 42*3 + 6 + 4 + 4 = 140... will be calibrated at runtime
    HIDDEN1 = 128
    HIDDEN2 = 64
    OUTPUT_DIM = 87   # 42 (reinforce) + 42 (attack src) + 1 (stop) + 1 (fortify stop) + 1 (spare)
    # Actually we use the same network for all phases with different output interpretations

    def __init__(self, input_dim: int = 144, rng: np.random.Generator = None, hidden_layers: list[int] = None):
        self.input_dim = input_dim
        self.rng = rng or np.random.default_rng()
        # Single network with large enough output for all decision types
        # Output: 42 + 42 + 1 = 85 (territory scores + attack/fortify targets + stop signal)
        self.output_dim = 42 + 42 + 1  # 85
        
        if hidden_layers is None:
            hidden_layers = [128, 64] # Default architecture
            
        self.network = NeuralNetwork([input_dim] + hidden_layers + [self.output_dim])

    def get_params(self) -> np.ndarray:
        return self.network.get_params()

    def set_params(self, params: np.ndarray):
        self.network.set_params(params)

    def param_count(self) -> int:
        return self.network.param_count()

    def _forward(self, state_encoded: np.ndarray) -> np.ndarray:
        """Run the network and return raw output."""
        return self.network.forward(state_encoded)

    def reinforce(self, state_encoded: np.ndarray, n_armies: int,
                  owned_territories: np.ndarray) -> np.ndarray:
        """
        Decide how to distribute reinforcement armies.
        Returns an array of armies to add per owned territory.
        """
        output = self._forward(state_encoded)
        # Use first 42 outputs as territory preference scores
        scores = output[:42]

        # Mask to only owned territories
        mask = np.full(42, -1e10)
        mask[owned_territories] = 0.0
        scores = scores + mask

        # Convert to probability distribution
        probs = softmax(scores[owned_territories])

        # Distribute armies proportionally
        distribution = np.zeros(len(owned_territories), dtype=np.int32)
        remaining = n_armies
        raw_alloc = probs * n_armies

        # Integer allocation with remainder handling
        for i in range(len(owned_territories)):
            distribution[i] = int(raw_alloc[i])
        remaining = n_armies - distribution.sum()
        # Distribute remaining 1 at a time to highest fractional parts
        fractions = raw_alloc - distribution.astype(np.float64)
        order = np.argsort(-fractions)
        for i in range(remaining):
            distribution[order[i % len(order)]] += 1

        return distribution

    def attack(self, state_encoded: np.ndarray,
               valid_attacks: list[tuple[int, int]]) -> tuple[int, int] | None:
        """
        Decide whether and where to attack.
        Returns (from_territory, to_territory) or None to stop attacking.
        """
        if not valid_attacks:
            return None

        output = self._forward(state_encoded)
        # Use outputs 0:42 as source territory scores, 42:84 as target scores
        src_scores = output[:42]
        tgt_scores = output[42:84]
        stop_score = output[84]

        # Score each valid attack as src_score[from] + tgt_score[to]
        best_score = float('-inf')
        best_attack = None
        for frm, to in valid_attacks:
            score = src_scores[frm] + tgt_scores[to]
            if score > best_score:
                best_score = score
                best_attack = (frm, to)

        # Compare with stop score
        if best_attack is not None and best_score > stop_score:
            return best_attack
        return None

    def fortify(self, state_encoded: np.ndarray,
                valid_fortifications: list[tuple[int, int]]) -> tuple[int, int, int] | None:
        """
        Decide whether and how to fortify.
        Returns (from_territory, to_territory, n_armies) or None.
        """
        if not valid_fortifications:
            return None

        output = self._forward(state_encoded)
        src_scores = output[:42]
        tgt_scores = output[42:84]
        stop_score = output[84]

        best_score = float('-inf')
        best_fort = None
        for frm, to in valid_fortifications:
            score = src_scores[frm] + tgt_scores[to]
            if score > best_score:
                best_score = score
                best_fort = (frm, to)

        if best_fort is not None and best_score > stop_score:
            frm, to = best_fort
            # How many armies to move: use sigmoid on score to get 0-1 ratio
            move_ratio = float(sigmoid(np.array([best_score]))[0])
            return (frm, to, max(1, int(move_ratio * 50)))  # up to ~50 armies
        return None


class RandomAgent:
    """
    Simple random agent for baseline comparison.
    """

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()

    def reinforce(self, state_encoded: np.ndarray, n_armies: int,
                  owned_territories: np.ndarray) -> np.ndarray:
        distribution = np.zeros(len(owned_territories), dtype=np.int32)
        for _ in range(n_armies):
            idx = self.rng.integers(0, len(owned_territories))
            distribution[idx] += 1
        # Strict: total must equal n_armies (this loop guarantees it)
        assert int(np.sum(distribution)) == n_armies
        return distribution

    def attack(self, state_encoded: np.ndarray,
               valid_attacks: list[tuple[int, int]]) -> tuple[int, int] | None:
        if not valid_attacks or self.rng.random() < 0.3:
            return None
        idx = self.rng.integers(0, len(valid_attacks))
        return valid_attacks[idx]

    def fortify(self, state_encoded: np.ndarray,
                valid_fortifications: list[tuple[int, int]]) -> tuple[int, int, int] | None:
        if not valid_fortifications or self.rng.random() < 0.5:
            return None
        idx = self.rng.integers(0, len(valid_fortifications))
        frm, to = valid_fortifications[idx]
        # Clamp to valid range: 1 to (source armies - 1), will be validated by engine
        return (frm, to, max(1, self.rng.integers(1, 4)))
