"""
Full UCT Monte Carlo Tree Search Agent for Risiko.
Evaluates actions by building a lookahead tree using UCB1 selection and playout heuristics.
"""

import time
import math
import copy
import numpy as np
from game import GameState, RiskGame, fast_resolve_combat, compute_fitness_details, _ADJ_BOOL

class MCTSNode:
    def __init__(self, state: GameState, move, parent=None, untried_moves=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        # untried_moves: list of valid attacks + [None] for "stop attacking"
        self.untried_moves = untried_moves if untried_moves is not None else []

    def ucb1(self, exploration_weight=1.414):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)


class MCTSAgent:
    """
    Full MCTS Agent with UCT formulation for the Attack Phase.
    """
    requires_full_state = True
    
    def __init__(self, time_limit=0.5, rng=None):
        """
        time_limit: Time in seconds to spend building the UCT tree for EVERY single attack decision.
        """
        self.time_limit = time_limit
        self.rng = rng or np.random.default_rng()
        self.dummy_game = RiskGame(rng=self.rng)
        
    def reinforce(self, encoded: np.ndarray, n_armies: int, owned_territories: np.ndarray, state: GameState) -> np.ndarray:
        """Evaluate random reinforcement distributions using rollout, keeping it flat for speed."""
        best_dist = None
        best_score = -9999.0
        n_allocations = min(20, max(5, len(owned_territories)))
        
        for _ in range(n_allocations):
            p = np.ones(len(owned_territories)) / len(owned_territories)
            alloc = self.rng.multinomial(n_armies, p)
            
            s = copy.deepcopy(state)
            for i, t in enumerate(owned_territories):
                s.armies[t] += alloc[i]
                
            score = self._simulate_playout(s)
            
            if score > best_score:
                best_score = score
                best_dist = alloc
                
        if best_dist is None:
            best_dist = np.zeros(len(owned_territories), dtype=np.int32)
            best_dist[0] = n_armies
            
        return best_dist

    def attack(self, encoded: np.ndarray, valid_attacks: list, state: GameState) -> tuple | None:
        """Upper Confidence bound applied to Trees (UCT) calculation for the optimal attack."""
        if not valid_attacks:
            return None
            
        # Define root node with all possible actions
        moves = valid_attacks + [None] 
        root = MCTSNode(state=state, move=None, parent=None, untried_moves=moves)
        
        start_time = time.time()
        
        # Build the tree within the time limit
        while time.time() - start_time < self.time_limit:
            node = root
            
            # 1. SELECTION: Traverse down finding fully expanded nodes
            while not node.untried_moves and node.children:
                node = max(node.children, key=lambda c: c.ucb1())
                
            # 2. EXPANSION: Add a new child if possible
            if node.untried_moves:
                move_idx = self.rng.integers(len(node.untried_moves))
                move = node.untried_moves.pop(move_idx)
                
                new_state = copy.deepcopy(node.state)
                if move is not None:
                    # Resolve single attack to a terminal board consequence
                    self._blitz(new_state, move[0], move[1])
                    # Re-calculate legal moves from this new branch
                    next_moves = self.dummy_game.get_valid_attacks(new_state, new_state.current_player) + [None]
                else:
                    # Player stops attacking
                    next_moves = []
                    
                child = MCTSNode(state=new_state, move=move, parent=node, untried_moves=next_moves)
                node.children.append(child)
                node = child
                
            # 3. SIMULATION: Random playout heuristics from the new node
            score = self._simulate_playout(copy.deepcopy(node.state))
            
            # 4. BACKPROPAGATION: Send the evaluation up the branch
            while node is not None:
                node.visits += 1
                node.value += score
                node = node.parent
                
        # Return the most visited action (Standard UCT metric)
        if not root.children:
            return None
        
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def fortify(self, encoded: np.ndarray, valid_fortifications: list, state: GameState) -> tuple | None:
        """Fortify using a simple randomized test or heuristic."""
        if not valid_fortifications:
            return None
        if self.rng.random() < 0.5:
            return None
        idx = self.rng.choice(len(valid_fortifications))
        frm, to = valid_fortifications[idx]
        return (frm, to, max(1, int(state.armies[frm]-1) // 2))

    def _blitz(self, s: GameState, frm: int, to: int):
        """Mechanically execute an attack until territory is taken or out of armies."""
        player = s.current_player
        while s.armies[frm] > 1 and s.owner[to] != player:
            atk_dice = min(3, int(s.armies[frm] - 1))
            def_dice = min(3, int(s.armies[to]))
            a_loss, d_loss = fast_resolve_combat(atk_dice, def_dice)
            s.armies[frm] -= a_loss
            s.armies[to] -= d_loss
            
            # Conquest
            if s.armies[to] <= 0:
                s.owner[to] = player
                s.armies[to] = atk_dice  
                s.armies[frm] -= atk_dice
                break
                
    def _simulate_playout(self, s: GameState) -> float:
        """Simulate a pure random attack sequence to evaluate a branch end position."""
        player = s.current_player
        
        # Playout: Simulate up to 5 additional tactical attacks randomly
        for _ in range(5):
            my_can = (s.owner == player) & (s.armies >= 2)
            enemy = s.owner != player
            pairs = my_can[:, None] & enemy[None, :] & _ADJ_BOOL
            idx = np.argwhere(pairs)
            if len(idx) == 0:
                break
                
            atk = idx[self.rng.choice(len(idx))]
            self._blitz(s, atk[0], atk[1])
            
        # Convert post-simulation state to Value
        details = compute_fitness_details(s, player)
        score = details["territory_frac"] * 0.40 + details["continent_progress"] * 0.30 + details["army_ratio"] * 0.20
        return score
