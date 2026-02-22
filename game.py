"""
Risiko (Risk) game engine for 4 players.

Implements the full board with 42 territories, 6 continents,
dice combat, territory cards, and all turn phases.
Optimized with vectorized numpy operations for neuroevolution speed.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── Territory and continent definitions ──────────────────────────────────────

CONTINENTS = {
    "Nord America": {
        "territories": [
            "Alaska", "Territori del Nord Ovest", "Groenlandia", "Alberta",
            "Ontario", "Quebec", "Stati Uniti Occidentali",
            "Stati Uniti Orientali", "America Centrale"
        ],
        "bonus": 5,
    },
    "Sud America": {
        "territories": ["Venezuela", "Peru", "Brasile", "Argentina"],
        "bonus": 2,
    },
    "Europa": {
        "territories": [
            "Islanda", "Scandinavia", "Gran Bretagna", "Europa Settentrionale",
            "Europa Occidentale", "Europa Meridionale", "Ucraina"
        ],
        "bonus": 5,
    },
    "Africa": {
        "territories": [
            "Africa del Nord", "Egitto", "Africa Orientale",
            "Congo", "Africa del Sud", "Madagascar"
        ],
        "bonus": 3,
    },
    "Asia": {
        "territories": [
            "Urali", "Siberia", "Jacuzia", "Kamchatka", "Irkutsk",
            "Mongolia", "Giappone", "Afghanistan", "Cina", "India",
            "Siam", "Medio Oriente"
        ],
        "bonus": 7,
    },
    "Oceania": {
        "territories": [
            "Indonesia", "Nuova Guinea", "Australia Occidentale",
            "Australia Orientale"
        ],
        "bonus": 2,
    },
}

ALL_TERRITORIES: list[str] = []
for cont_data in CONTINENTS.values():
    for t in cont_data["territories"]:
        if t not in ALL_TERRITORIES:
            ALL_TERRITORIES.append(t)

TERRITORY_INDEX = {name: i for i, name in enumerate(ALL_TERRITORIES)}
NUM_TERRITORIES = len(ALL_TERRITORIES)  # 42

ADJACENCIES: dict[str, list[str]] = {
    "Alaska": ["Territori del Nord Ovest", "Alberta", "Kamchatka"],
    "Territori del Nord Ovest": ["Alaska", "Alberta", "Ontario", "Groenlandia"],
    "Groenlandia": ["Territori del Nord Ovest", "Ontario", "Quebec", "Islanda"],
    "Alberta": ["Alaska", "Territori del Nord Ovest", "Ontario", "Stati Uniti Occidentali"],
    "Ontario": ["Territori del Nord Ovest", "Alberta", "Groenlandia", "Quebec",
                 "Stati Uniti Occidentali", "Stati Uniti Orientali"],
    "Quebec": ["Ontario", "Groenlandia", "Stati Uniti Orientali"],
    "Stati Uniti Occidentali": ["Alberta", "Ontario", "Stati Uniti Orientali", "America Centrale"],
    "Stati Uniti Orientali": ["Ontario", "Quebec", "Stati Uniti Occidentali", "America Centrale"],
    "America Centrale": ["Stati Uniti Occidentali", "Stati Uniti Orientali", "Venezuela"],
    "Venezuela": ["America Centrale", "Peru", "Brasile"],
    "Peru": ["Venezuela", "Brasile", "Argentina"],
    "Brasile": ["Venezuela", "Peru", "Argentina", "Africa del Nord"],
    "Argentina": ["Peru", "Brasile"],
    "Islanda": ["Groenlandia", "Scandinavia", "Gran Bretagna"],
    "Scandinavia": ["Islanda", "Gran Bretagna", "Europa Settentrionale", "Ucraina"],
    "Gran Bretagna": ["Islanda", "Scandinavia", "Europa Settentrionale", "Europa Occidentale"],
    "Europa Settentrionale": ["Scandinavia", "Gran Bretagna", "Europa Occidentale",
                               "Europa Meridionale", "Ucraina"],
    "Europa Occidentale": ["Gran Bretagna", "Europa Settentrionale", "Europa Meridionale",
                            "Africa del Nord"],
    "Europa Meridionale": ["Europa Settentrionale", "Europa Occidentale", "Ucraina",
                            "Africa del Nord", "Egitto", "Medio Oriente"],
    "Ucraina": ["Scandinavia", "Europa Settentrionale", "Europa Meridionale",
                 "Urali", "Afghanistan", "Medio Oriente"],
    "Africa del Nord": ["Brasile", "Europa Occidentale", "Europa Meridionale",
                         "Egitto", "Africa Orientale", "Congo"],
    "Egitto": ["Europa Meridionale", "Africa del Nord", "Africa Orientale", "Medio Oriente"],
    "Africa Orientale": ["Africa del Nord", "Egitto", "Congo", "Africa del Sud", "Madagascar"],
    "Congo": ["Africa del Nord", "Africa Orientale", "Africa del Sud"],
    "Africa del Sud": ["Congo", "Africa Orientale", "Madagascar"],
    "Madagascar": ["Africa Orientale", "Africa del Sud"],
    "Urali": ["Ucraina", "Siberia", "Cina", "Afghanistan"],
    "Siberia": ["Urali", "Jacuzia", "Irkutsk", "Mongolia", "Cina"],
    "Jacuzia": ["Siberia", "Irkutsk", "Kamchatka"],
    "Kamchatka": ["Alaska", "Jacuzia", "Irkutsk", "Mongolia", "Giappone"],
    "Irkutsk": ["Siberia", "Jacuzia", "Kamchatka", "Mongolia"],
    "Mongolia": ["Siberia", "Irkutsk", "Kamchatka", "Giappone", "Cina"],
    "Giappone": ["Kamchatka", "Mongolia"],
    "Afghanistan": ["Ucraina", "Urali", "Cina", "India", "Medio Oriente"],
    "Cina": ["Urali", "Siberia", "Mongolia", "Afghanistan", "India", "Siam"],
    "India": ["Afghanistan", "Cina", "Siam", "Medio Oriente"],
    "Siam": ["India", "Cina", "Indonesia"],
    "Medio Oriente": ["Europa Meridionale", "Ucraina", "Egitto", "Afghanistan", "India"],
    "Indonesia": ["Siam", "Nuova Guinea", "Australia Occidentale"],
    "Nuova Guinea": ["Indonesia", "Australia Occidentale", "Australia Orientale"],
    "Australia Occidentale": ["Indonesia", "Nuova Guinea", "Australia Orientale"],
    "Australia Orientale": ["Nuova Guinea", "Australia Occidentale"],
}

# ── Precomputed structures (module-level, computed once) ─────────────────────

# Adjacency as index lists
ADJ_INDICES: list[np.ndarray] = []
for i in range(NUM_TERRITORIES):
    t_name = ALL_TERRITORIES[i]
    ADJ_INDICES.append(np.array([TERRITORY_INDEX[n] for n in ADJACENCIES[t_name]], dtype=np.int32))

# Adjacency matrix (42x42) for vectorized threat computation
ADJ_MATRIX = np.zeros((NUM_TERRITORIES, NUM_TERRITORIES), dtype=np.float32)
for i, neighbors in enumerate(ADJ_INDICES):
    ADJ_MATRIX[i, neighbors] = 1.0

# Neighbor count per territory for normalization
ADJ_COUNT = ADJ_MATRIX.sum(axis=1, keepdims=True)  # (42, 1)
ADJ_COUNT = np.maximum(ADJ_COUNT, 1.0)

# Boolean adjacency for vectorized valid-move queries
_ADJ_BOOL = ADJ_MATRIX.astype(bool)

# Continent data as arrays
CONTINENT_INDICES: list[np.ndarray] = []
CONTINENT_BONUSES: list[int] = []
for cont_data in CONTINENTS.values():
    CONTINENT_INDICES.append(np.array([TERRITORY_INDEX[t] for t in cont_data["territories"]], dtype=np.int32))
    CONTINENT_BONUSES.append(cont_data["bonus"])
NUM_CONTINENTS = len(CONTINENT_INDICES)

CARD_SET_BONUSES = [4, 6, 8, 10, 12, 15]
NUM_PLAYERS = 4
MAX_TURNS = 150

# Feature dimension (precomputed)
# 42*3 (territory) + 6 (continent) + 4 (global) + 4 (relative) = 140
INPUT_DIM = NUM_TERRITORIES * 3 + NUM_CONTINENTS + 4 + NUM_PLAYERS


# ── Game State ───────────────────────────────────────────────────────────────

@dataclass
class GameState:
    """Full state of a Risiko game."""
    owner: np.ndarray = field(default_factory=lambda: np.zeros(NUM_TERRITORIES, dtype=np.int32))
    armies: np.ndarray = field(default_factory=lambda: np.ones(NUM_TERRITORIES, dtype=np.int32))
    cards: list[list[int]] = field(default_factory=lambda: [[] for _ in range(NUM_PLAYERS)])
    sets_traded: list[int] = field(default_factory=lambda: [0] * NUM_PLAYERS)
    current_player: int = 0
    turn: int = 0
    game_over: bool = False
    winner: Optional[int] = None
    conquered_this_turn: bool = False
    eliminated: list[bool] = field(default_factory=lambda: [False] * NUM_PLAYERS)


# ── Game Engine ──────────────────────────────────────────────────────────────

class RiskGame:
    """Risiko game engine for 4 players, optimized for neuroevolution."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    def reset(self) -> GameState:
        state = GameState()
        territory_order = self.rng.permutation(NUM_TERRITORIES)
        for i, t_idx in enumerate(territory_order):
            state.owner[t_idx] = i % NUM_PLAYERS
            state.armies[t_idx] = 1
        # Distribute initial 30 armies per player
        for p in range(NUM_PLAYERS):
            owned = np.where(state.owner == p)[0]
            remaining = 30 - len(owned)
            if remaining > 0:
                targets = self.rng.choice(owned, size=remaining)
                for t in targets:
                    state.armies[t] += 1
        # 2 initial cards per player
        for p in range(NUM_PLAYERS):
            state.cards[p] = [int(self.rng.integers(0, 3)) for _ in range(2)]
        return state

    def calc_reinforcements(self, state: GameState, player: int) -> int:
        owned_mask = state.owner == player
        base = max(int(np.sum(owned_mask)) // 3, 3)
        bonus = 0
        for i in range(NUM_CONTINENTS):
            if np.all(owned_mask[CONTINENT_INDICES[i]]):
                bonus += CONTINENT_BONUSES[i]
        return base + bonus

    def check_and_trade_cards(self, state: GameState, player: int) -> int:
        total_bonus = 0
        cards = state.cards[player]
        while len(cards) >= 3:
            counts = [0, 0, 0]
            for c in cards:
                counts[c] += 1
            traded = False
            for ct in range(3):
                if counts[ct] >= 3:
                    removed = 0
                    new_cards = []
                    for c in cards:
                        if c == ct and removed < 3:
                            removed += 1
                        else:
                            new_cards.append(c)
                    cards = new_cards
                    traded = True
                    break
            if not traded and counts[0] >= 1 and counts[1] >= 1 and counts[2] >= 1:
                used = [False, False, False]
                new_cards = []
                for c in cards:
                    if not used[c]:
                        used[c] = True
                    else:
                        new_cards.append(c)
                cards = new_cards
                traded = True
            if not traded:
                break
            si = state.sets_traded[player]
            if si < len(CARD_SET_BONUSES):
                total_bonus += CARD_SET_BONUSES[si]
            else:
                total_bonus += CARD_SET_BONUSES[-1] + (si - len(CARD_SET_BONUSES) + 1) * 5
            state.sets_traded[player] += 1
        state.cards[player] = cards
        return total_bonus

    def resolve_combat(self, atk_armies: int, def_armies: int) -> tuple[int, int]:
        """Vectorized dice combat."""
        atk_dice = min(3, atk_armies - 1)
        def_dice = min(2, def_armies)
        if atk_dice <= 0 or def_dice <= 0:
            return 0, 0
        atk_rolls = np.sort(self.rng.integers(1, 7, size=atk_dice))[::-1]
        def_rolls = np.sort(self.rng.integers(1, 7, size=def_dice))[::-1]
        n = min(atk_dice, def_dice)
        atk_wins = int(np.sum(atk_rolls[:n] > def_rolls[:n]))
        return n - atk_wins, atk_wins

    # ── State encoding (fully vectorized, no Python loops) ───────────────

    def encode_state(self, state: GameState, player: int) -> np.ndarray:
        """Encode game state as NN input. Fully vectorized, returns 140 floats."""
        owner = state.owner
        armies = state.armies.astype(np.float32)
        max_a = max(float(np.max(armies)), 1.0)
        inv_max_a = 1.0 / max_a

        # (42,) boolean masks
        is_mine = (owner == player).astype(np.float32)
        enemy_mask = 1.0 - is_mine

        # Normalized armies (42,)
        army_norm = armies * inv_max_a

        # Enemy neighbor threat via matrix multiply — fully vectorized, no loop!
        # enemy_armies[j] = armies[j] if enemy, else 0
        enemy_armies = armies * enemy_mask  # (42,)
        # threat[i] = sum of enemy armies in neighbors of i, normalized
        threat = (ADJ_MATRIX @ enemy_armies) * inv_max_a / ADJ_COUNT.ravel()  # (42,)

        # Interleave into (126,) = [is_mine_0, army_0, threat_0, is_mine_1, ...]
        territory_features = np.empty(NUM_TERRITORIES * 3, dtype=np.float32)
        territory_features[0::3] = is_mine
        territory_features[1::3] = army_norm
        territory_features[2::3] = threat

        # Continent control (6,)
        continent_ctrl = np.empty(NUM_CONTINENTS, dtype=np.float32)
        for i in range(NUM_CONTINENTS):
            continent_ctrl[i] = np.mean(is_mine[CONTINENT_INDICES[i]])

        # Global stats (4,)
        total_a = float(np.sum(armies))
        my_a = float(np.sum(armies * is_mine))
        my_t = float(np.sum(is_mine))
        global_feats = np.array([
            my_a / max(total_a, 1.0),
            my_t / NUM_TERRITORIES,
            len(state.cards[player]) / 10.0,
            state.turn / MAX_TURNS,
        ], dtype=np.float32)

        # Relative strength (4,)
        rel = np.empty(NUM_PLAYERS, dtype=np.float32)
        for p in range(NUM_PLAYERS):
            if p == player:
                rel[p] = 0.5
            else:
                p_a = float(np.sum(armies[owner == p]))
                rel[p] = my_a / max(my_a + p_a, 1.0)

        return np.concatenate([territory_features, continent_ctrl, global_feats, rel])

    # ── Valid moves ──────────────────────────────────────────────────────

    def get_valid_attacks(self, state: GameState, player: int) -> list[tuple[int, int]]:
        my_can = (state.owner == player) & (state.armies >= 2)
        enemy = state.owner != player
        # Boolean matrix: source_i can attack target_j
        pairs = my_can[:, None] & enemy[None, :] & _ADJ_BOOL
        idx = np.argwhere(pairs)
        return [(int(a), int(b)) for a, b in idx]

    def get_valid_fortifications(self, state: GameState, player: int) -> list[tuple[int, int]]:
        my_can = (state.owner == player) & (state.armies >= 2)
        my_all = state.owner == player
        pairs = my_can[:, None] & my_all[None, :] & _ADJ_BOOL
        # Exclude self-loops (diagonal)
        np.fill_diagonal(pairs, False)
        idx = np.argwhere(pairs)
        return [(int(a), int(b)) for a, b in idx]

    # ── Turn execution ───────────────────────────────────────────────────

    def play_turn(self, state: GameState, agents: list) -> GameState:
        if state.game_over:
            return state

        player = state.current_player
        if state.eliminated[player]:
            self._advance_turn(state)
            return state

        agent = agents[player]
        state.conquered_this_turn = False

        # Phase 1: Cards + Reinforcement
        card_bonus = self.check_and_trade_cards(state, player)
        n_reinforcements = self.calc_reinforcements(state, player) + card_bonus
        owned_territories = np.where(state.owner == player)[0]

        # Encode once for reinforce + initial attack decision
        encoded = self.encode_state(state, player)

        if len(owned_territories) > 0:
            distribution = agent.reinforce(encoded, n_reinforcements, owned_territories)
            for t_idx, n in zip(owned_territories, distribution):
                state.armies[t_idx] += n

        # Phase 2: Attack (max 10 rounds)
        conquests_since_encode = 0
        for _ in range(10):
            valid_attacks = self.get_valid_attacks(state, player)
            if not valid_attacks:
                break
            attack_choice = agent.attack(encoded, valid_attacks)
            if attack_choice is None:
                break

            atk_from, atk_to = attack_choice
            atk_losses, def_losses = self.resolve_combat(
                state.armies[atk_from], state.armies[atk_to]
            )
            state.armies[atk_from] -= atk_losses
            state.armies[atk_to] -= def_losses

            if state.armies[atk_to] <= 0:
                state.conquered_this_turn = True
                defender = state.owner[atk_to]
                move = max(1, min(3, state.armies[atk_from] - 1))
                state.armies[atk_to] = move
                state.armies[atk_from] -= move
                state.owner[atk_to] = player

                if np.sum(state.owner == defender) == 0:
                    state.eliminated[defender] = True
                    state.cards[player].extend(state.cards[defender])
                    state.cards[defender] = []

                if np.all(state.owner == player):
                    state.game_over = True
                    state.winner = player
                    return state

                # Re-encode only every 3 conquests (reduces overhead)
                conquests_since_encode += 1
                if conquests_since_encode >= 3:
                    encoded = self.encode_state(state, player)
                    conquests_since_encode = 0

        # Phase 3: Draw card
        if state.conquered_this_turn:
            state.cards[player].append(int(self.rng.integers(0, 3)))

        # Phase 4: Fortification (reuse last encoded)
        valid_forts = self.get_valid_fortifications(state, player)
        if valid_forts:
            fort_choice = agent.fortify(encoded, valid_forts)
            if fort_choice is not None:
                f_from, f_to, f_n = fort_choice
                f_n = max(0, min(f_n, state.armies[f_from] - 1))
                state.armies[f_from] -= f_n
                state.armies[f_to] += f_n

        self._advance_turn(state)
        return state

    def _advance_turn(self, state: GameState):
        for _ in range(NUM_PLAYERS):
            state.current_player = (state.current_player + 1) % NUM_PLAYERS
            if state.current_player == 0:
                state.turn += 1
                if state.turn >= MAX_TURNS:
                    state.game_over = True
                    counts = np.array([np.sum(state.owner == p) for p in range(NUM_PLAYERS)])
                    state.winner = int(np.argmax(counts))
                    return
            if not state.eliminated[state.current_player]:
                return

    def play_game(self, agents: list) -> tuple[int, GameState]:
        state = self.reset()
        while not state.game_over:
            state = self.play_turn(state, agents)
        return state.winner, state


def territory_count(state: GameState, player: int) -> int:
    return int(np.sum(state.owner == player))


def compute_fitness_details(state: GameState, player: int) -> dict:
    """Compute detailed fitness components for a player at end of game."""
    owned_mask = state.owner == player
    n_territories = int(np.sum(owned_mask))
    total_armies = int(np.sum(state.armies))
    my_armies = int(np.sum(state.armies[owned_mask]))

    # Continent control ratio (0-1): fraction of continents fully owned
    continents_owned = 0
    continent_progress = 0.0
    for i in range(NUM_CONTINENTS):
        ci = CONTINENT_INDICES[i]
        frac = float(np.mean(owned_mask[ci]))
        continent_progress += frac
        if frac == 1.0:
            continents_owned += 1
    continent_ratio = continents_owned / NUM_CONTINENTS
    continent_progress /= NUM_CONTINENTS  # average % of each continent

    # Army advantage (0-1): my armies vs total
    army_ratio = my_armies / max(total_armies, 1)

    # Border strength: average army ratio at borders (my border armies / enemy neighbor armies)
    border_strength = 0.0
    border_count = 0
    my_terr = np.where(owned_mask)[0]
    for t in my_terr:
        for n in ADJ_INDICES[t]:
            if state.owner[n] != player:
                ratio = state.armies[t] / max(state.armies[n], 1)
                border_strength += min(ratio, 3.0) / 3.0  # clamp to [0, 1]
                border_count += 1
    if border_count > 0:
        border_strength /= border_count

    return {
        "territories": n_territories,
        "territory_frac": n_territories / NUM_TERRITORIES,
        "continent_ratio": continent_ratio,
        "continent_progress": continent_progress,
        "army_ratio": army_ratio,
        "border_strength": border_strength,
    }

