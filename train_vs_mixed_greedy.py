"""
Train the loaded DQN agent (P0) against 3 Aggressive Greedy opponents (P1, P2, P3).

Loads from checkpoints/dqn_policy_vs_normal_greedy_game_200.pt and continues
training. Only P0's transitions are used for training. Same spec as train_vs_normal_greedy:
saves to checkpoints/dqn_policy_vs_aggressive_greedy_final.pt and intermediate
dqn_policy_vs_aggressive_greedy_game_200.pt etc.
"""

import torch
import os
import random
from typing import Any, Callable, Dict, List

from environment import GeneralsEnv
from controller import DQNController
from agents import DQNAgent
from run_game import run_game

from greedy_baseline_agent import AggressiveGreedyAgent, _setup_environment_patches


LOAD_CHECKPOINT = os.path.join("checkpoints", "dqn_policy_vs_normal_greedy_game_200.pt")
SAVE_CHECKPOINT = os.path.join("checkpoints", "dqn_policy_vs_aggressive_greedy_final.pt")
CHECKPOINT_DIR = "checkpoints"
TRAIN_EVERY_N_MOVES = 40
NUM_GAMES = 500
SAVE_EVERY_N_GAMES = 200
MAX_STEPS_PER_GAME = 10_000
# Set to an int (e.g. 42) for reproducibility.
TRAIN_SEED = None


def make_greedy_callable(env: GeneralsEnv, agent: Any) -> Callable[[Any], int]:
    """Return callable(controller) -> action_index for the given greedy agent (e.g. AggressiveGreedyAgent)."""
    def greedy_act(controller: DQNController) -> int:
        env.player_id = env.current_player_index
        controller.get_valid_actions_for_agent()
        valid_explicit = getattr(controller, "_valid_explicit", [])
        if not valid_explicit:
            return 0
        legal_actions, action_mapping = env.get_reduced_action_space()
        if not legal_actions:
            return 0
        best_action_id = agent.select(env)
        if best_action_id is None:
            return 0
        move = action_mapping.get(best_action_id)
        if move is None:
            return 0
        fx, fy, tx, ty = move
        for i, (ex, ey, et, eu) in enumerate(valid_explicit):
            if (ex, ey, et, eu) == (fx, fy, tx, ty):
                return i
        best_idx = 0
        best_score = float("-inf")
        for i, (ex, ey, et, eu) in enumerate(valid_explicit):
            sc = agent._evaluate_move(
                env.game.grid[ey][ex], env.game.grid[eu][et], env.current_player_index
            )
            if sc > best_score:
                best_score = sc
                best_idx = i
        return best_idx
    return greedy_act


def main() -> None:
    if TRAIN_SEED is not None:
        random.seed(TRAIN_SEED)
        try:
            import numpy as np
            np.random.seed(TRAIN_SEED)
        except Exception:
            pass
        torch.manual_seed(TRAIN_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(TRAIN_SEED)
    _setup_environment_patches()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GeneralsEnv(grid_size=(10, 10), training_mode=True, device="cpu")
    env.max_steps = MAX_STEPS_PER_GAME  # env defaults to 5000; use our limit

    if not os.path.isfile(LOAD_CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {LOAD_CHECKPOINT}. Run train_vs_normal_greedy.py first."
        )

    dqn_agent = DQNAgent.load(LOAD_CHECKPOINT, device)
    dqn_agent.epsilon = max(dqn_agent.epsilon, 0.05)

    controllers: List[DQNController] = [DQNController(env, device) for _ in range(4)]
    aggressive_fn = make_greedy_callable(env, AggressiveGreedyAgent())
    agents: List[Callable[[Any], int]] = [
        dqn_agent.act,
        aggressive_fn,
        aggressive_fn,
        aggressive_fn,
    ]

    def on_after_step(
        step_count: int,
        player_index: int,
        controller: DQNController,
        step_info: Dict[str, Any],
    ) -> None:
        # Only store and train on P0 (DQN) transitions
        if player_index != 0:
            return
        trans = controller.get_last_transition()
        if trans is not None:
            state, action, reward, next_state, done = trans
            reward_deltas = step_info.get("reward_deltas")
            if reward_deltas is not None and 0 in reward_deltas:
                reward = reward_deltas[0]
            dqn_agent.push_transition(state, action, reward, next_state, done)
        if step_count > 0 and step_count % TRAIN_EVERY_N_MOVES == 0:
            dqn_agent.train_step()

    print(
        f"Training DQN (P0) vs 3× Aggressive Greedy (P1,P2,P3)\n"
        f"Loaded from {LOAD_CHECKPOINT}\n"
        f"Save to {SAVE_CHECKPOINT}\n"
        f"Games: {NUM_GAMES}, train every {TRAIN_EVERY_N_MOVES} moves\n"
    )
    wins = [0, 0, 0, 0]
    total_steps = 0

    for game_id in range(NUM_GAMES):
        result = run_game(
            env,
            controllers,
            agents,
            max_steps=MAX_STEPS_PER_GAME,
            on_after_step=on_after_step,
        )
        total_steps += result["step_count"]
        winner = result["winner"]
        if winner is not None and 0 <= winner < 4:
            wins[winner] += 1
        dqn_agent.decay_epsilon()

        if (game_id + 1) % SAVE_EVERY_N_GAMES == 0:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            path = os.path.join(CHECKPOINT_DIR, f"dqn_policy_vs_aggressive_greedy_game_{game_id + 1}.pt")
            dqn_agent.save(path)
            print(f"  Saved to {path}")

        if (game_id + 1) % 10 == 0 or game_id == 0:
            m = result.get("per_player_metrics")
            w = result["winner"]
            winner_str = f"P{w}" if w is not None else "None"
            print(
                f"Game {game_id + 1}/{NUM_GAMES} | Winner: {winner_str} | "
                f"Steps: {result['step_count']} | Wins: P0={wins[0]} P1={wins[1]} P2={wins[2]} P3={wins[3]}"
            )
            if m:
                print("  Per-player (reward, land, troops, moves, status):")
                for i, p in enumerate(m):
                    status = "eliminated" if p.get("eliminated", False) else "alive"
                    print(
                        f"    P{i}: reward={p['reward']:.2f} land={p['territory']} troops={p['army']} moves={p['moves']} [{status}]"
                    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    dqn_agent.save(SAVE_CHECKPOINT)
    print(f"\nDone. Total steps: {total_steps}. Win counts: {wins}")
    print(f"Saved trained agent to {SAVE_CHECKPOINT}")


if __name__ == "__main__":
    main()
