"""
Run 50 matches: saved DQN agent (dqn_policy_final.pt) vs greedy baseline.
Players: P0 = DQN, P1/P2/P3 = Aggressive Greedy (from greedy_baseline_agent).

- Uses AggressiveGreedyAgent: prioritizes capturing enemy generals/cities to end games.
- Model: We load from checkpoint and use online_net for P0; weight fingerprint
  is printed to confirm we're not using a fresh/untrained network.
- Games: A winner requires 3 eliminations (one player left).
"""

import torch
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from environment import GeneralsEnv
from controller import DQNController
from agents import DQNAgent
from run_game import run_game

from greedy_baseline_agent import AggressiveGreedyAgent, _setup_environment_patches


CHECKPOINT_PATH = os.path.join("checkpoints", "dqn_policy_final.pt")
NUM_MATCHES = 50
EVAL_ENV_MAX_STEPS = 50_000  # allow games to finish (need 3 eliminations for a winner)


def make_greedy_agent_callable(env: GeneralsEnv) -> Callable[[Any], int]:
    """
    Return a callable(controller) -> action_index that uses the aggressive
    greedy agent (chases wins: capture generals/cities). Baseline action space;
    we set env.player_id = env.current_player_index, then map chosen move to
    controller's valid index.
    """
    greedy_agent = AggressiveGreedyAgent()

    def greedy_act(controller: DQNController) -> int:
        # So baseline's get_reduced_action_space uses the current player
        env.player_id = env.current_player_index
        controller.get_valid_actions_for_agent()
        valid_explicit = getattr(controller, "_valid_explicit", [])
        if not valid_explicit:
            return 0
        # Baseline action space (may include moves to invisible cells)
        legal_actions, action_mapping = env.get_reduced_action_space()
        if not legal_actions:
            return 0
        # Best move from baseline scoring
        best_action_id = greedy_agent.select(env)
        if best_action_id is None:
            return 0
        move = action_mapping.get(best_action_id)
        if move is None:
            return 0
        fx, fy, tx, ty = move
        # Map to controller index (only visible moves are in valid_explicit)
        for i, (ex, ey, et, eu) in enumerate(valid_explicit):
            if (ex, ey, et, eu) == (fx, fy, tx, ty):
                return i
        # Chosen move not in valid_explicit (e.g. to invisible cell); pick best among valid
        best_idx = 0
        best_score = float("-inf")
        for i, (ex, ey, et, eu) in enumerate(valid_explicit):
            sc = greedy_agent._evaluate_move(
                env.game.grid[ey][ex], env.game.grid[eu][et], env.current_player_index
            )
            if sc > best_score:
                best_score = sc
                best_idx = i
        return best_idx

    return greedy_act


def main() -> Dict[str, Any]:
    _setup_environment_patches()  # adds get_reduced_action_space to GeneralsEnv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GeneralsEnv(grid_size=(10, 10), training_mode=True, device="cpu")
    env.max_steps = EVAL_ENV_MAX_STEPS  # allow games to finish with a winner

    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. Train first with train_self_play.py"
        )

    dqn_agent = DQNAgent.load(CHECKPOINT_PATH, device)
    dqn_agent.epsilon = 0.0  # greedy evaluation (no exploration)

    # Verify we're using the loaded model, not a fresh one
    def _weight_fingerprint(net: torch.nn.Module) -> float:
        with torch.no_grad():
            p = next(net.parameters())
            return float(p.sum().item())
    fp = _weight_fingerprint(dqn_agent.online_net)
    expected_h, expected_w = dqn_agent.state_shape[0], dqn_agent.state_shape[1]
    if (env.grid_height, env.grid_width) != (expected_h, expected_w):
        raise ValueError(
            f"Env grid {env.grid_height}x{env.grid_width} does not match loaded model "
            f"state_shape {dqn_agent.state_shape}. Retrain or use matching grid_size."
        )
    print(
        f"Loaded DQN from {CHECKPOINT_PATH}\n"
        f"  state_shape={dqn_agent.state_shape} max_actions={dqn_agent.max_actions} "
        f"epsilon={dqn_agent.epsilon}\n"
        f"  (weight fingerprint: {fp:.2f} — if ~0 or tiny, model may be untrained)\n"
    )

    controllers: List[DQNController] = [DQNController(env, device) for _ in range(4)]
    greedy_fn = make_greedy_agent_callable(env)
    agents: List[Callable[[Any], int]] = [
        dqn_agent.act,
        greedy_fn,
        greedy_fn,
        greedy_fn,
    ]

    dqn_wins = 0
    greedy_wins = 0
    other = 0  # None winner or draw
    matches: List[Dict[str, Any]] = []

    print(f"Running {NUM_MATCHES} matches: P0 (DQN) vs P1,P2,P3 (Aggressive Greedy)")
    print(f"Max steps per game: {EVAL_ENV_MAX_STEPS} (game ends when one player remains)\n")

    for game_id in range(NUM_MATCHES):
        result = run_game(env, controllers, agents, max_steps=EVAL_ENV_MAX_STEPS)
        winner = result["winner"]
        per_player = result.get("per_player_metrics", [])
        matches.append({
            "winner": winner,
            "step_count": result["step_count"],
            "game_over": result.get("game_over", False),
            "per_player_metrics": per_player,
        })
        if winner == 0:
            dqn_wins += 1
        elif winner in (1, 2, 3):
            greedy_wins += 1
        else:
            other += 1
        # Same format as training: winner, steps, wins + per-player (reward, land, troops, moves, status)
        w = result["winner"]
        winner_str = f"P{w}" if w is not None else "None"
        print(
            f"Match {game_id + 1}/{NUM_MATCHES} | Winner: {winner_str} | "
            f"Steps: {result['step_count']} | DQN: {dqn_wins} | Greedy: {greedy_wins}"
        )
        if per_player:
            print("  Per-player (reward, land, troops, moves, status):")
            for i, p in enumerate(per_player):
                status = "eliminated" if p.get("eliminated", False) else "alive"
                print(
                    f"    P{i}: reward={p['reward']:.2f} land={p['territory']} troops={p['army']} moves={p['moves']} [{status}]"
                )

    stats = {
        "dqn_wins": dqn_wins,
        "greedy_wins": greedy_wins,
        "other": other,
        "num_matches": NUM_MATCHES,
        "matches": matches,
    }
    print("\n--- Results ---")
    print(f"DQN (P0) wins:    {dqn_wins}/{NUM_MATCHES} ({100 * dqn_wins / NUM_MATCHES:.1f}%)")
    print(f"Greedy (P1–P3):   {greedy_wins}/{NUM_MATCHES} ({100 * greedy_wins / NUM_MATCHES:.1f}%)  [Aggressive Greedy]")
    if other:
        print(f"Other (no winner): {other}")
    return stats


if __name__ == "__main__":
    stats = main()
    # Example: use stats programmatically
    # print(stats["dqn_wins"], stats["greedy_wins"], stats["matches"][-1]["step_count"])
