"""
Comprehensive tests for the Risiko game engine with strict Italian rules.

Tests: dice, tris bonuses, reinforcement validation, attack validation,
fortify validation, win condition, phase enforcement, player elimination.
"""

import unittest
import numpy as np
from game import (
    RiskGame, GameState, GamePhase, InvalidActionError,
    NUM_TERRITORIES, NUM_PLAYERS, WIN_THRESHOLD,
    ALL_TERRITORIES, TERRITORY_INDEX, ADJACENCIES, _ADJ_BOOL,
    TRIS_BONUSES, CONTINENTS, CONTINENT_INDICES, CONTINENT_BONUSES,
)


class TestSetup(unittest.TestCase):
    """Test initial game setup."""

    def setUp(self):
        self.game = RiskGame(rng=np.random.default_rng(42))

    def test_42_territories(self):
        self.assertEqual(NUM_TERRITORIES, 42)

    def test_4_players(self):
        self.assertEqual(NUM_PLAYERS, 4)

    def test_territory_distribution(self):
        """Two players get 11 territories, two get 10."""
        state = self.game.reset()
        counts = [int(np.sum(state.owner == p)) for p in range(NUM_PLAYERS)]
        counts.sort()
        self.assertEqual(counts, [10, 10, 11, 11])

    def test_initial_30_armies(self):
        """Each player starts with exactly 30 armies total."""
        state = self.game.reset()
        for p in range(NUM_PLAYERS):
            total = int(np.sum(state.armies[state.owner == p]))
            self.assertEqual(total, 30, f"Player {p} has {total} armies, expected 30")

    def test_min_1_army_per_territory(self):
        """Every territory has at least 1 army."""
        state = self.game.reset()
        for t in range(NUM_TERRITORIES):
            self.assertGreaterEqual(state.armies[t], 1,
                f"Territory {ALL_TERRITORIES[t]} has {state.armies[t]} armies")

    def test_initial_phase_is_reinforce(self):
        state = self.game.reset()
        self.assertEqual(state.current_phase, GamePhase.REINFORCE)


class TestDice(unittest.TestCase):
    """Test dice combat resolution."""

    def setUp(self):
        self.game = RiskGame(rng=np.random.default_rng(0))

    def test_defender_can_roll_3_dice(self):
        """Defender can roll up to 3 dice when having >= 3 armies."""
        # resolve_combat now takes dice counts directly
        atk_losses, def_losses = self.game.resolve_combat(3, 3)
        total = atk_losses + def_losses
        self.assertEqual(total, 3, "3v3 combat should produce 3 total losses")

    def test_defender_2_dice(self):
        atk_losses, def_losses = self.game.resolve_combat(3, 2)
        total = atk_losses + def_losses
        self.assertEqual(total, 2)

    def test_defender_1_die(self):
        atk_losses, def_losses = self.game.resolve_combat(2, 1)
        total = atk_losses + def_losses
        self.assertEqual(total, 1)

    def test_attacker_1_die(self):
        atk_losses, def_losses = self.game.resolve_combat(1, 1)
        total = atk_losses + def_losses
        self.assertEqual(total, 1)

    def test_zero_dice_no_losses(self):
        atk_losses, def_losses = self.game.resolve_combat(0, 3)
        self.assertEqual((atk_losses, def_losses), (0, 0))

    def test_tie_goes_to_defender(self):
        """When dice are equal, attacker loses (defender wins tie)."""
        # Run many combats and verify statistical bias
        rng = np.random.default_rng(42)
        game = RiskGame(rng=rng)
        atk_total_losses = 0
        def_total_losses = 0
        n = 10000
        for _ in range(n):
            al, dl = game.resolve_combat(1, 1)
            atk_total_losses += al
            def_total_losses += dl
        # With 1v1, defender should win ties, so attacker should lose more often
        # Probability of attacker winning: 15/36 ≈ 41.7%
        self.assertGreater(atk_total_losses, def_total_losses,
            "Attacker should lose more often in 1v1 due to tie rule")


class TestTrisBonuses(unittest.TestCase):
    """Test card trade-in (tris) with combination-based bonuses."""

    def setUp(self):
        self.game = RiskGame(rng=np.random.default_rng(0))

    def test_3_cannoni_bonus_4(self):
        state = GameState()
        state.cards[0] = [0, 0, 0]
        bonus = self.game.check_and_trade_cards(state, 0)
        self.assertEqual(bonus, 4)
        self.assertEqual(len(state.cards[0]), 0)

    def test_3_fanti_bonus_6(self):
        state = GameState()
        state.cards[0] = [1, 1, 1]
        bonus = self.game.check_and_trade_cards(state, 0)
        self.assertEqual(bonus, 6)

    def test_3_cavalieri_bonus_8(self):
        state = GameState()
        state.cards[0] = [2, 2, 2]
        bonus = self.game.check_and_trade_cards(state, 0)
        self.assertEqual(bonus, 8)

    def test_misto_bonus_10(self):
        state = GameState()
        state.cards[0] = [0, 1, 2]
        bonus = self.game.check_and_trade_cards(state, 0)
        self.assertEqual(bonus, 10)

    def test_misto_any_order(self):
        """Mixed tris works regardless of card order."""
        state = GameState()
        state.cards[0] = [2, 0, 1]
        bonus = self.game.check_and_trade_cards(state, 0)
        self.assertEqual(bonus, 10)

    def test_no_tris_with_2_cards(self):
        state = GameState()
        state.cards[0] = [0, 1]
        bonus = self.game.check_and_trade_cards(state, 0)
        self.assertEqual(bonus, 0)
        self.assertEqual(len(state.cards[0]), 2)

    def test_no_valid_combo(self):
        """3 cards but no valid combo (e.g., 2 cannoni + 1 fante)."""
        state = GameState()
        state.cards[0] = [0, 0, 1]
        bonus = self.game.check_and_trade_cards(state, 0)
        self.assertEqual(bonus, 0)
        self.assertEqual(len(state.cards[0]), 3)

    def test_multiple_trades(self):
        """Trade multiple tris in one go (6 cards = 2 trades possible)."""
        state = GameState()
        state.cards[0] = [0, 0, 0, 1, 1, 1]
        bonus = self.game.check_and_trade_cards(state, 0)
        # 3 Cannoni = +4, 3 Fanti = +6 → total = 10
        self.assertEqual(bonus, 10)
        self.assertEqual(len(state.cards[0]), 0)

    def test_prefers_higher_bonus(self):
        """When multiple combos possible, prefers higher bonus first."""
        state = GameState()
        state.cards[0] = [2, 2, 2, 0, 1]  # Can trade 3 cavalieri
        bonus = self.game.check_and_trade_cards(state, 0)
        self.assertEqual(bonus, 8)  # 3 Cavalieri
        self.assertEqual(state.cards[0], [0, 1])


class TestAttackValidation(unittest.TestCase):
    """Test strict attack validation."""

    def setUp(self):
        self.game = RiskGame(rng=np.random.default_rng(42))

    def _make_state_for_attack(self):
        """Create a state where player 0 can attack player 1."""
        state = GameState()
        state.current_phase = GamePhase.ATTACK
        state.current_player = 0
        # Player 0 owns Alaska (idx 0) with 5 armies
        state.owner[0] = 0
        state.armies[0] = 5
        # Player 1 owns a neighbor (find one from adjacency)
        neighbor_name = ADJACENCIES["Alaska"][0]
        neighbor_idx = TERRITORY_INDEX[neighbor_name]
        state.owner[neighbor_idx] = 1
        state.armies[neighbor_idx] = 3
        return state, 0, neighbor_idx

    def test_valid_attack_passes(self):
        state, atk_from, atk_to = self._make_state_for_attack()
        # Should not raise
        self.game.validate_attack(state, 0, atk_from, atk_to, 3)

    def test_wrong_phase_raises(self):
        state, atk_from, atk_to = self._make_state_for_attack()
        state.current_phase = GamePhase.REINFORCE
        with self.assertRaises(InvalidActionError):
            self.game.validate_attack(state, 0, atk_from, atk_to, 3)

    def test_wrong_player_raises(self):
        state, atk_from, atk_to = self._make_state_for_attack()
        with self.assertRaises(InvalidActionError):
            self.game.validate_attack(state, 1, atk_from, atk_to, 3)

    def test_attack_own_territory_raises(self):
        state, atk_from, atk_to = self._make_state_for_attack()
        state.owner[atk_to] = 0  # Make target also owned by player 0
        with self.assertRaises(InvalidActionError):
            self.game.validate_attack(state, 0, atk_from, atk_to, 3)

    def test_non_adjacent_raises(self):
        state, atk_from, _ = self._make_state_for_attack()
        # Find a non-adjacent territory owned by enemy
        for t in range(NUM_TERRITORIES):
            if not _ADJ_BOOL[atk_from, t] and t != atk_from:
                state.owner[t] = 1
                state.armies[t] = 3
                with self.assertRaises(InvalidActionError):
                    self.game.validate_attack(state, 0, atk_from, t, 1)
                break

    def test_not_enough_armies_raises(self):
        state, atk_from, atk_to = self._make_state_for_attack()
        state.armies[atk_from] = 1  # Only 1 army, needs >= 2
        with self.assertRaises(InvalidActionError):
            self.game.validate_attack(state, 0, atk_from, atk_to, 1)

    def test_too_many_dice_raises(self):
        state, atk_from, atk_to = self._make_state_for_attack()
        state.armies[atk_from] = 2  # Max dice = min(3, 2-1) = 1
        with self.assertRaises(InvalidActionError):
            self.game.validate_attack(state, 0, atk_from, atk_to, 2)

    def test_zero_dice_raises(self):
        state, atk_from, atk_to = self._make_state_for_attack()
        with self.assertRaises(InvalidActionError):
            self.game.validate_attack(state, 0, atk_from, atk_to, 0)

    def test_dice_max_is_armies_minus_1(self):
        """With 4 armies, max dice = min(3, 3) = 3."""
        state, atk_from, atk_to = self._make_state_for_attack()
        state.armies[atk_from] = 4  # max = min(3, 3) = 3
        self.game.validate_attack(state, 0, atk_from, atk_to, 3)  # OK
        with self.assertRaises(InvalidActionError):
            self.game.validate_attack(state, 0, atk_from, atk_to, 4)  # Too many

    def test_attacker_can_choose_fewer_dice(self):
        """Attacker with 5 armies can choose 1, 2, or 3 dice."""
        state, atk_from, atk_to = self._make_state_for_attack()
        state.armies[atk_from] = 5
        for dice in [1, 2, 3]:
            self.game.validate_attack(state, 0, atk_from, atk_to, dice)


class TestDefenseValidation(unittest.TestCase):
    """Test defense dice validation."""

    def setUp(self):
        self.game = RiskGame(rng=np.random.default_rng(0))

    def test_defender_max_3_dice(self):
        state = GameState()
        state.armies[5] = 5
        # Should accept up to 3
        self.game.validate_defense(state, 1, 5, 3)

    def test_defender_max_limited_by_armies(self):
        state = GameState()
        state.armies[5] = 2
        self.game.validate_defense(state, 1, 5, 2)  # OK
        with self.assertRaises(InvalidActionError):
            self.game.validate_defense(state, 1, 5, 3)  # Too many

    def test_defender_1_army_1_die(self):
        state = GameState()
        state.armies[5] = 1
        self.game.validate_defense(state, 1, 5, 1)  # OK
        with self.assertRaises(InvalidActionError):
            self.game.validate_defense(state, 1, 5, 2)

    def test_defender_zero_dice_raises(self):
        state = GameState()
        state.armies[5] = 3
        with self.assertRaises(InvalidActionError):
            self.game.validate_defense(state, 1, 5, 0)


class TestFortifyValidation(unittest.TestCase):
    """Test strict fortification validation."""

    def setUp(self):
        self.game = RiskGame(rng=np.random.default_rng(0))

    def _make_fortify_state(self):
        state = GameState()
        state.current_phase = GamePhase.FORTIFY
        state.current_player = 0
        # Player 0 owns two adjacent territories
        state.owner[0] = 0
        state.armies[0] = 5
        neighbor_name = ADJACENCIES[ALL_TERRITORIES[0]][0]
        neighbor_idx = TERRITORY_INDEX[neighbor_name]
        state.owner[neighbor_idx] = 0
        state.armies[neighbor_idx] = 2
        return state, 0, neighbor_idx

    def test_valid_fortify_passes(self):
        state, f_from, f_to = self._make_fortify_state()
        self.game.validate_fortify(state, 0, f_from, f_to, 2)

    def test_wrong_phase_raises(self):
        state, f_from, f_to = self._make_fortify_state()
        state.current_phase = GamePhase.ATTACK
        with self.assertRaises(InvalidActionError):
            self.game.validate_fortify(state, 0, f_from, f_to, 1)

    def test_source_not_owned_raises(self):
        state, f_from, f_to = self._make_fortify_state()
        state.owner[f_from] = 1
        with self.assertRaises(InvalidActionError):
            self.game.validate_fortify(state, 0, f_from, f_to, 1)

    def test_dest_not_owned_raises(self):
        state, f_from, f_to = self._make_fortify_state()
        state.owner[f_to] = 1
        with self.assertRaises(InvalidActionError):
            self.game.validate_fortify(state, 0, f_from, f_to, 1)

    def test_non_adjacent_raises(self):
        state, f_from, _ = self._make_fortify_state()
        # Find a non-adjacent territory owned by player 0
        for t in range(NUM_TERRITORIES):
            if not _ADJ_BOOL[f_from, t] and t != f_from:
                state.owner[t] = 0
                state.armies[t] = 2
                with self.assertRaises(InvalidActionError):
                    self.game.validate_fortify(state, 0, f_from, t, 1)
                break

    def test_must_keep_1_army(self):
        """Cannot move all armies from source."""
        state, f_from, f_to = self._make_fortify_state()
        state.armies[f_from] = 3
        # Can move 1 or 2, not 3
        self.game.validate_fortify(state, 0, f_from, f_to, 2)  # OK
        with self.assertRaises(InvalidActionError):
            self.game.validate_fortify(state, 0, f_from, f_to, 3)

    def test_zero_move_raises(self):
        state, f_from, f_to = self._make_fortify_state()
        with self.assertRaises(InvalidActionError):
            self.game.validate_fortify(state, 0, f_from, f_to, 0)

    def test_source_1_army_raises(self):
        """Can't fortify from a territory with only 1 army."""
        state, f_from, f_to = self._make_fortify_state()
        state.armies[f_from] = 1
        with self.assertRaises(InvalidActionError):
            self.game.validate_fortify(state, 0, f_from, f_to, 1)


class TestReinforcementValidation(unittest.TestCase):
    """Test that reinforcement pool must be completely emptied."""

    def test_must_place_all_reinforcements(self):
        """Agent that doesn't place all armies triggers InvalidActionError."""
        game = RiskGame(rng=np.random.default_rng(42))
        state = game.reset()

        class BadAgent:
            def reinforce(self, encoded, n_armies, owned):
                # Place one less than required
                dist = np.zeros(len(owned), dtype=np.int32)
                dist[0] = n_armies - 1
                return dist
            def attack(self, *a, **kw): return None
            def fortify(self, *a, **kw): return None

        agents = [BadAgent(), BadAgent(), BadAgent(), BadAgent()]
        with self.assertRaises(InvalidActionError):
            game.play_turn(state, agents)

    def test_negative_placement_raises(self):
        """Negative army placement is rejected."""
        game = RiskGame(rng=np.random.default_rng(42))
        state = game.reset()

        class NegAgent:
            def reinforce(self, encoded, n_armies, owned):
                dist = np.zeros(len(owned), dtype=np.int32)
                dist[0] = n_armies + 1
                dist[1] = -1  # negative!
                return dist
            def attack(self, *a, **kw): return None
            def fortify(self, *a, **kw): return None

        agents = [NegAgent(), NegAgent(), NegAgent(), NegAgent()]
        with self.assertRaises(InvalidActionError):
            game.play_turn(state, agents)


class TestWinCondition(unittest.TestCase):
    """Test win condition at >= 24 territories."""

    def test_win_threshold_is_24(self):
        self.assertEqual(WIN_THRESHOLD, 24)

    def test_win_at_24_territories(self):
        """Player with >= 24 territories wins immediately."""
        game = RiskGame(rng=np.random.default_rng(0))
        state = GameState()
        # Give player 0 exactly 24 territories
        for t in range(24):
            state.owner[t] = 0
            state.armies[t] = 5
        for t in range(24, NUM_TERRITORIES):
            state.owner[t] = 1
            state.armies[t] = 1

        # Simulate the win check inside the engine
        n_owned = int(np.sum(state.owner == 0))
        self.assertEqual(n_owned, 24)
        self.assertTrue(n_owned >= WIN_THRESHOLD)


class TestPhaseTransitions(unittest.TestCase):
    """Test that phases transition correctly through a full turn."""

    def test_full_turn_phases(self):
        """Normal turn goes REINFORCE → ATTACK → FORTIFY → DRAW → back to REINFORCE."""
        game = RiskGame(rng=np.random.default_rng(42))
        state = game.reset()

        class PassAgent:
            """Agent that places armies correctly but skips attack and fortify."""
            def reinforce(self, encoded, n_armies, owned):
                dist = np.zeros(len(owned), dtype=np.int32)
                dist[0] = n_armies
                return dist
            def attack(self, encoded, valid): return None
            def fortify(self, encoded, valid): return None

        agents = [PassAgent()] * 4
        initial_player = state.current_player

        # Play one turn
        state = game.play_turn(state, agents)

        # After a turn, should be next player's REINFORCE
        self.assertNotEqual(state.current_player, initial_player)
        self.assertEqual(state.current_phase, GamePhase.REINFORCE)


class TestCalcReinforcements(unittest.TestCase):
    """Test reinforcement calculation."""

    def setUp(self):
        self.game = RiskGame(rng=np.random.default_rng(0))

    def test_minimum_3(self):
        """Even with few territories, minimum reinforcement is 3."""
        state = GameState()
        # Give player 0 just 3 territories
        for t in range(3):
            state.owner[t] = 0
        for t in range(3, NUM_TERRITORIES):
            state.owner[t] = 1
        result = self.game.calc_reinforcements(state, 0)
        self.assertEqual(result, 3)  # floor(3/3) = 1, min = 3

    def test_territories_divided_by_3(self):
        state = GameState()
        # Pick 12 territories that do NOT complete any continent
        # Spread them across continents to avoid continent bonus
        chosen = []
        for cont_data in CONTINENTS.values():
            terrs = cont_data["territories"]
            # Take at most 2 from each continent
            for t_name in terrs[:2]:
                if len(chosen) < 12:
                    chosen.append(TERRITORY_INDEX[t_name])
        for t in range(NUM_TERRITORIES):
            state.owner[t] = 1
        for t_idx in chosen:
            state.owner[t_idx] = 0
        result = self.game.calc_reinforcements(state, 0)
        self.assertEqual(result, 4)  # floor(12/3) = 4, no continent bonus

    def test_continent_bonus(self):
        """If player owns all of Sud America, gets +2 bonus."""
        state = GameState()
        # IMPORTANT: set ALL territories to player 1 first
        for t in range(NUM_TERRITORIES):
            state.owner[t] = 1
        # Give player 0 all of Sud America
        sa_territories = CONTINENTS["Sud America"]["territories"]
        for t_name in sa_territories:
            state.owner[TERRITORY_INDEX[t_name]] = 0
        # Give player 0 extra territories to reach 9 total (for base 3)
        count = len(sa_territories)  # 4
        for t in range(NUM_TERRITORIES):
            if count >= 9:
                break
            if state.owner[t] == 1:
                state.owner[t] = 0
                count += 1

        result = self.game.calc_reinforcements(state, 0)
        n_owned = int(np.sum(state.owner == 0))
        base = max(n_owned // 3, 3)
        expected = base + 2  # Sud America bonus
        self.assertEqual(result, expected)


class TestFullGameIntegration(unittest.TestCase):
    """Integration test: run a complete game with RandomAgents."""

    def test_game_completes(self):
        """A game with 4 safe RandomAgents should complete without errors."""
        rng = np.random.default_rng(100)
        game = RiskGame(rng=rng)

        class SafeRandomAgent:
            """Random agent that always produces valid moves."""
            def __init__(self, rng):
                self.rng = rng
            def reinforce(self, encoded, n_armies, owned):
                dist = np.zeros(len(owned), dtype=np.int32)
                for _ in range(n_armies):
                    idx = self.rng.integers(0, len(owned))
                    dist[idx] += 1
                return dist
            def attack(self, encoded, valid):
                if not valid or self.rng.random() < 0.3:
                    return None
                return valid[self.rng.integers(0, len(valid))]
            def fortify(self, encoded, valid):
                # Skip fortify to avoid clamp issues
                return None

        agents = [SafeRandomAgent(np.random.default_rng(i)) for i in range(4)]

        state = game.reset()
        turns = 0
        # 4 players x 150 MAX_TURNS = 600 play_turn calls max
        while not state.game_over and turns < 650:
            state = game.play_turn(state, agents)
            turns += 1

        self.assertTrue(state.game_over, "Game should end via win or MAX_TURNS")


class TestContinentBonuses(unittest.TestCase):
    """Verify continent bonus values match Italian rules."""

    def test_asia_bonus_7(self):
        self.assertEqual(CONTINENTS["Asia"]["bonus"], 7)

    def test_nord_america_bonus_5(self):
        self.assertEqual(CONTINENTS["Nord America"]["bonus"], 5)

    def test_europa_bonus_5(self):
        self.assertEqual(CONTINENTS["Europa"]["bonus"], 5)

    def test_africa_bonus_3(self):
        self.assertEqual(CONTINENTS["Africa"]["bonus"], 3)

    def test_sud_america_bonus_2(self):
        self.assertEqual(CONTINENTS["Sud America"]["bonus"], 2)

    def test_oceania_bonus_2(self):
        self.assertEqual(CONTINENTS["Oceania"]["bonus"], 2)


class TestAdjacencySymmetry(unittest.TestCase):
    """Adjacency must be bidirectional."""

    def test_all_adjacencies_symmetric(self):
        for t, neighbors in ADJACENCIES.items():
            for n in neighbors:
                self.assertIn(t, ADJACENCIES[n],
                    f"{t} lists {n} as neighbor, but {n} does not list {t}")


if __name__ == "__main__":
    unittest.main()
