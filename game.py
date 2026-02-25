"""
Risiko (Risk) game engine for 4 players.

Implements the full board with 42 territories, 6 continents,
dice combat, territory cards, and all turn phases.
Strict state machine: every action is validated against the current phase.
"""

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional
from numba import njit


# ── Numba Optimized Components ───────────────────────────────────────────────

@njit(cache=True)
def seed_numba(seed: int):
    """Seed the internal Numba PRNG."""
    np.random.seed(seed)

@njit(cache=True)
def fast_resolve_combat(atk_dice: int, def_dice: int) -> tuple:
    """Super fast dice combat resolution without array allocations."""
    # Roll 1 to 3 dice for attacker
    a1 = np.random.randint(1, 7)
    a2 = np.random.randint(1, 7) if atk_dice > 1 else 0
    a3 = np.random.randint(1, 7) if atk_dice > 2 else 0
    
    # Roll 1 to 3 dice for defender
    d1 = np.random.randint(1, 7)
    d2 = np.random.randint(1, 7) if def_dice > 1 else 0
    d3 = np.random.randint(1, 7) if def_dice > 2 else 0
    
    # Sort attacker (descending)
    if a1 < a2: a1, a2 = a2, a1
    if a1 < a3: a1, a3 = a3, a1
    if a2 < a3: a2, a3 = a3, a2
    
    # Sort defender (descending)
    if d1 < d2: d1, d2 = d2, d1
    if d1 < d3: d1, d3 = d3, d1
    if d2 < d3: d2, d3 = d3, d2
    
    atk_losses = 0
    def_losses = 0
    
    # Highest dice face off
    if a1 > d1:
        def_losses += 1
    else:
        atk_losses += 1
        
    # Second highest dice face off (if both rolled >= 2 dice)
    if atk_dice > 1 and def_dice > 1:
        if a2 > d2:
            def_losses += 1
        else:
            atk_losses += 1
            
    return atk_losses, def_losses


# ── Exceptions ───────────────────────────────────────────────────────────────

class InvalidActionError(Exception):
    """Raised when an agent attempts an action not valid in the current phase."""
    pass


# ── Game Phases ──────────────────────────────────────────────────────────────

class GamePhase(Enum):
    REINFORCE = auto()   # Phase 1: place reinforcements (cards + base + continent)
    ATTACK = auto()      # Phase 2: declare attacks (optional, repeatable)
    FORTIFY = auto()     # Phase 3: single strategic move (optional)
    DRAW = auto()        # Phase 4: draw a card if conquered this turn
    GAME_OVER = auto()   # Terminal state

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

# Tris bonuses by combination type (card types: 0=Cannone, 1=Fante, 2=Cavaliere)
TRIS_BONUSES = {
    (0, 0, 0): 4,   # 3 Cannoni
    (1, 1, 1): 6,   # 3 Fanti
    (2, 2, 2): 8,   # 3 Cavalieri
    (0, 1, 2): 10,  # Misto (1 di ciascuno)
}
NUM_PLAYERS = 4
WIN_THRESHOLD = 24  # Territories needed to win
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
    current_player: int = 0
    current_phase: GamePhase = GamePhase.REINFORCE
    turn: int = 0
    game_over: bool = False
    winner: Optional[int] = None
    conquered_this_turn: bool = False
    eliminated: list[bool] = field(default_factory=lambda: [False] * NUM_PLAYERS)
    # Reinforcement pool: armies that MUST be placed before moving to ATTACK
    reinforcements_remaining: int = 0
    # Last attack dice count: used for minimum occupation move after conquest
    last_attack_dice: int = 0


# ── Game Engine ──────────────────────────────────────────────────────────────

class RiskGame:
    """Risiko game engine for 4 players, optimized for neuroevolution."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        # Ensure Numba's PRNG is synchronized with the agent's seed
        seed_numba(int(self.rng.integers(0, 2**31)))

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

    def _find_valid_tris(self, cards: list[int]) -> Optional[tuple[int, ...]]:
        """Find the first valid tris combination in a hand of cards.
        Returns a tuple of 3 card types to trade, or None."""
        counts = [0, 0, 0]
        for c in cards:
            counts[c] += 1
        # Check 3 of a kind (prefer higher bonus: Cavalieri > Fanti > Cannoni)
        for ct in [2, 1, 0]:
            if counts[ct] >= 3:
                return (ct, ct, ct)
        # Check mixed (one of each)
        if counts[0] >= 1 and counts[1] >= 1 and counts[2] >= 1:
            return (0, 1, 2)
        return None

    def check_and_trade_cards(self, state: GameState, player: int) -> int:
        """
        Auto-trade cards for the player. Uses combo-based bonuses:
        3 Cannoni (0) = +4, 3 Fanti (1) = +6, 3 Cavalieri (2) = +8, Misto = +10.
        Returns total bonus armies.
        """
        total_bonus = 0
        cards = state.cards[player]
        while len(cards) >= 3:
            tris = self._find_valid_tris(cards)
            if tris is None:
                break
            # Remove the 3 cards from hand
            combo_key = tuple(sorted(tris))
            for card_type in tris:
                cards.remove(card_type)
            # Apply combo-specific bonus
            bonus = TRIS_BONUSES.get(combo_key, 0)
            total_bonus += bonus
        state.cards[player] = cards
        return total_bonus

    def resolve_combat(self, atk_dice: int, def_dice: int) -> tuple[int, int]:
        """
        Resolve dice combat.
        atk_dice: number of dice the attacker rolls (1-3, already validated).
        def_dice: number of dice the defender rolls (1-3, already validated).
        Returns (attacker_losses, defender_losses).
        Tie goes to defender (attacker loses if roll <= defender roll).
        """
        if atk_dice <= 0 or def_dice <= 0:
            return 0, 0
        return fast_resolve_combat(atk_dice, def_dice)

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
        """Return list of (source, target) pairs valid for attack."""
        my_can = (state.owner == player) & (state.armies >= 2)
        enemy = state.owner != player
        # Boolean matrix: source_i can attack target_j
        pairs = my_can[:, None] & enemy[None, :] & _ADJ_BOOL
        idx = np.argwhere(pairs)
        return [(int(a), int(b)) for a, b in idx]

    def validate_attack(self, state: GameState, player: int,
                        atk_from: int, atk_to: int, atk_dice: int):
        """
        Strictly validate an attack declaration.
        Raises InvalidActionError if any rule is violated.
        """
        if state.current_phase != GamePhase.ATTACK:
            raise InvalidActionError(
                f"Cannot attack in phase {state.current_phase.name}. "
                f"Must be in ATTACK phase.")
        if state.current_player != player:
            raise InvalidActionError(
                f"Player {player} cannot act: it is Player {state.current_player}'s turn.")
        if state.owner[atk_from] != player:
            raise InvalidActionError(
                f"Territory {ALL_TERRITORIES[atk_from]} does not belong to Player {player}.")
        if state.owner[atk_to] == player:
            raise InvalidActionError(
                f"Cannot attack own territory {ALL_TERRITORIES[atk_to]}.")
        if not _ADJ_BOOL[atk_from, atk_to]:
            raise InvalidActionError(
                f"Territories {ALL_TERRITORIES[atk_from]} and {ALL_TERRITORIES[atk_to]} "
                f"are not adjacent.")
        if state.armies[atk_from] < 2:
            raise InvalidActionError(
                f"Territory {ALL_TERRITORIES[atk_from]} needs at least 2 armies to attack "
                f"(has {state.armies[atk_from]}).")
        max_atk_dice = min(3, state.armies[atk_from] - 1)
        if not (1 <= atk_dice <= max_atk_dice):
            raise InvalidActionError(
                f"Attacker dice must be 1-{max_atk_dice} (requested {atk_dice}). "
                f"Max = min(3, armies_on_source - 1) = min(3, {state.armies[atk_from] - 1}).")

    def validate_defense(self, state: GameState, defender: int,
                         atk_to: int, def_dice: int):
        """
        Validate the defender's dice choice.
        Raises InvalidActionError if invalid.
        """
        max_def_dice = min(3, int(state.armies[atk_to]))
        if not (1 <= def_dice <= max_def_dice):
            raise InvalidActionError(
                f"Defender dice must be 1-{max_def_dice} (requested {def_dice}). "
                f"Max = min(3, armies_on_territory) = min(3, {state.armies[atk_to]}).")

    def validate_fortify(self, state: GameState, player: int,
                         f_from: int, f_to: int, f_n: int):
        """
        Strictly validate a fortification move.
        Raises InvalidActionError if any rule is violated.
        """
        if state.current_phase != GamePhase.FORTIFY:
            raise InvalidActionError(
                f"Cannot fortify in phase {state.current_phase.name}. "
                f"Must be in FORTIFY phase.")
        if state.current_player != player:
            raise InvalidActionError(
                f"Player {player} cannot act: it is Player {state.current_player}'s turn.")
        if state.owner[f_from] != player:
            raise InvalidActionError(
                f"Source territory {ALL_TERRITORIES[f_from]} does not belong to Player {player}.")
        if state.owner[f_to] != player:
            raise InvalidActionError(
                f"Destination territory {ALL_TERRITORIES[f_to]} does not belong to Player {player}.")
        if not _ADJ_BOOL[f_from, f_to]:
            raise InvalidActionError(
                f"Territories {ALL_TERRITORIES[f_from]} and {ALL_TERRITORIES[f_to]} "
                f"are not adjacent.")
        max_move = state.armies[f_from] - 1
        if max_move < 1:
            raise InvalidActionError(
                f"Source territory {ALL_TERRITORIES[f_from]} has only {state.armies[f_from]} "
                f"army — cannot move any (must keep at least 1).")
        if not (1 <= f_n <= max_move):
            raise InvalidActionError(
                f"Must move between 1 and {max_move} armies (requested {f_n}). "
                f"Source must keep at least 1 army.")

    def get_valid_fortifications(self, state: GameState, player: int) -> list[tuple[int, int]]:
        """Return list of (source, target) pairs valid for fortification."""
        my_can = (state.owner == player) & (state.armies >= 2)
        my_all = state.owner == player
        pairs = my_can[:, None] & my_all[None, :] & _ADJ_BOOL
        # Exclude self-loops (diagonal)
        np.fill_diagonal(pairs, False)
        idx = np.argwhere(pairs)
        return [(int(a), int(b)) for a, b in idx]

    # ── Turn execution (strict state machine) ─────────────────────────────

    def play_turn(self, state: GameState, agents: list) -> GameState:
        """
        Execute a full turn for the current player with STRICT phase enforcement.
        Phases: REINFORCE → ATTACK → FORTIFY → DRAW → next player.
        Every action is validated; InvalidActionError is raised on violations.
        """
        if state.game_over or state.current_phase == GamePhase.GAME_OVER:
            return state

        player = state.current_player
        if state.eliminated[player]:
            self._advance_turn(state)
            return state

        agent = agents[player]
        state.conquered_this_turn = False

        # ── PHASE 1: REINFORCE ──────────────────────────────────────────
        state.current_phase = GamePhase.REINFORCE

        # Calculate total reinforcements: base + continent bonus + card trade-in
        card_bonus = self.check_and_trade_cards(state, player)
        n_reinforcements = self.calc_reinforcements(state, player) + card_bonus
        state.reinforcements_remaining = n_reinforcements
        owned_territories = np.where(state.owner == player)[0]

        encoded = self.encode_state(state, player)

        if len(owned_territories) > 0 and n_reinforcements > 0:
            if getattr(agent, 'requires_full_state', False):
                distribution = agent.reinforce(encoded, n_reinforcements, owned_territories, state)
            else:
                distribution = agent.reinforce(encoded, n_reinforcements, owned_territories)

            # STRICT VALIDATION: total placed must equal total available
            total_placed = int(np.sum(distribution))
            if total_placed != n_reinforcements:
                raise InvalidActionError(
                    f"Player {player}: must place exactly {n_reinforcements} reinforcements, "
                    f"but placed {total_placed}. Pool must be emptied before ATTACK phase.")

            # STRICT VALIDATION: no negative placements
            if np.any(distribution < 0):
                raise InvalidActionError(
                    f"Player {player}: negative army placement is not allowed.")

            for t_idx, n in zip(owned_territories, distribution):
                state.armies[t_idx] += n

        state.reinforcements_remaining = 0

        # ── PHASE 2: ATTACK ─────────────────────────────────────────────
        state.current_phase = GamePhase.ATTACK

        conquests_since_encode = 0
        for _ in range(10):
            valid_attacks = self.get_valid_attacks(state, player)
            if not valid_attacks:
                break
            if getattr(agent, 'requires_full_state', False):
                attack_choice = agent.attack(encoded, valid_attacks, state)
            else:
                attack_choice = agent.attack(encoded, valid_attacks)
            if attack_choice is None:
                break

            atk_from, atk_to = attack_choice

            # Determine dice counts
            # Attacker: agent chooses (via choose_attack_dice if available, else max)
            max_atk_dice = min(3, state.armies[atk_from] - 1)
            if hasattr(agent, 'choose_attack_dice'):
                atk_dice = agent.choose_attack_dice(encoded, atk_from, atk_to, max_atk_dice)
            else:
                atk_dice = max_atk_dice  # backward compatible default

            # STRICT VALIDATION of the attack
            self.validate_attack(state, player, atk_from, atk_to, atk_dice)

            # Defender: agent chooses (via choose_defense_dice if available, else max)
            defender = state.owner[atk_to]
            max_def_dice = min(3, int(state.armies[atk_to]))
            defender_agent = agents[defender]
            if hasattr(defender_agent, 'choose_defense_dice'):
                def_dice = defender_agent.choose_defense_dice(
                    self.encode_state(state, defender), atk_to, max_def_dice)
            else:
                def_dice = max_def_dice  # backward compatible default

            # STRICT VALIDATION of defense dice
            self.validate_defense(state, defender, atk_to, def_dice)

            # Store dice count for occupation minimum
            state.last_attack_dice = atk_dice

            # Resolve combat
            atk_losses, def_losses = self.resolve_combat(atk_dice, def_dice)
            state.armies[atk_from] -= atk_losses
            state.armies[atk_to] -= def_losses

            # ── Conquest trigger ─────────────────────────────────────
            if state.armies[atk_to] <= 0:
                state.conquered_this_turn = True
                conquered_defender = state.owner[atk_to]

                # Occupation move: attacker MUST move armies into conquered territory
                min_move = state.last_attack_dice  # minimum = dice rolled
                max_move = state.armies[atk_from] - 1  # must leave at least 1
                min_move = min(min_move, max_move)  # safety clamp

                if hasattr(agent, 'choose_occupation_armies'):
                    move = agent.choose_occupation_armies(
                        encoded, atk_from, atk_to, min_move, max_move)
                else:
                    move = min_move  # backward compatible default

                # Auto-clamp requested move to valid bounds
                move = max(min_move, min(move, max_move))

                # STRICT VALIDATION of occupation move
                if not (min_move <= move <= max_move):
                    raise InvalidActionError(
                        f"Occupation move must be {min_move}-{max_move} armies "
                        f"(requested {move}). Min = dice rolled ({state.last_attack_dice}), "
                        f"max = source armies - 1 ({state.armies[atk_from] - 1}).")

                state.armies[atk_to] = move
                state.armies[atk_from] -= move
                state.owner[atk_to] = player

                # ── Player elimination trigger ──────────────────────
                if np.sum(state.owner == conquered_defender) == 0:
                    state.eliminated[conquered_defender] = True
                    # Loot cards
                    state.cards[player].extend(state.cards[conquered_defender])
                    state.cards[conquered_defender] = []

                # ── Win condition: >= 24 territories ────────────────
                if np.sum(state.owner == player) >= WIN_THRESHOLD:
                    state.game_over = True
                    state.winner = player
                    state.current_phase = GamePhase.GAME_OVER
                    return state

                # Re-encode periodically
                conquests_since_encode += 1
                if conquests_since_encode >= 3:
                    encoded = self.encode_state(state, player)
                    conquests_since_encode = 0

        # ── PHASE 3: FORTIFY (single move, strictly validated) ──────────
        state.current_phase = GamePhase.FORTIFY

        valid_forts = self.get_valid_fortifications(state, player)
        if valid_forts:
            if getattr(agent, 'requires_full_state', False):
                fort_choice = agent.fortify(encoded, valid_forts, state)
            else:
                fort_choice = agent.fortify(encoded, valid_forts)
            if fort_choice is not None:
                f_from, f_to, f_n = fort_choice
                
                # Auto-clamp f_n to valid range since AI only sees normalized armies
                max_move = state.armies[f_from] - 1
                f_n = max(1, min(f_n, max_move))

                # STRICT VALIDATION of fortification
                self.validate_fortify(state, player, f_from, f_to, f_n)
                state.armies[f_from] -= f_n
                state.armies[f_to] += f_n

        # ── PHASE 4: DRAW ───────────────────────────────────────────────
        state.current_phase = GamePhase.DRAW

        if state.conquered_this_turn:
            state.cards[player].append(int(self.rng.integers(0, 3)))

        # ── Advance to next player ──────────────────────────────────────
        self._advance_turn(state)
        return state

    def _advance_turn(self, state: GameState):
        """Move to the next non-eliminated player. Handle MAX_TURNS timeout."""
        for _ in range(NUM_PLAYERS):
            state.current_player = (state.current_player + 1) % NUM_PLAYERS
            if state.current_player == 0:
                state.turn += 1
                if state.turn >= MAX_TURNS:
                    state.game_over = True
                    state.current_phase = GamePhase.GAME_OVER
                    counts = np.array([np.sum(state.owner == p) for p in range(NUM_PLAYERS)])
                    state.winner = int(np.argmax(counts))
                    return
            if not state.eliminated[state.current_player]:
                state.current_phase = GamePhase.REINFORCE
                return

    def play_game(self, agents: list) -> tuple[int, GameState]:
        """Play a full game and return (winner, final_state)."""
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

    return {
        "territories": n_territories,
        "territory_frac": n_territories / NUM_TERRITORIES,
        "continent_ratio": continent_ratio,
        "continent_progress": continent_progress,
        "army_ratio": army_ratio,
    }

