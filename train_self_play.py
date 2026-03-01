"""
Self-play training loop: 1 shared policy (DQN), 4 copies playing each other (1v1v1v1).
All transitions from all agents go into one dataset; we train the same DQN by random
sampling from that shared replay buffer. 1000 games, training every 40 moves.
"""

import torch
import os
from typing import Any, Dict, List

from environment import GeneralsEnv
from controller import DQNController
from agents import DQNAgent
from run_game import run_game


TRAIN_EVERY_N_MOVES = 40
NUM_GAMES = 500
SAVE_EVERY_N_GAMES = 200
CHECKPOINT_DIR = "checkpoints"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GeneralsEnv(grid_size=(10, 10), training_mode=True, device="cpu")
    env.max_steps = 10_000  # env defaults to 5000; match run_game limit

    # State shape and max_actions must match the env's grid (H, W, C) and H*W*4
    state_shape = (env.grid_height, env.grid_width, 6)
    max_actions = env.grid_width * env.grid_height * 4

    # One controller per player (all share the same env)
    controllers: List[DQNController] = [DQNController(env, device) for _ in range(4)]

    # Single shared DQN policy and replay buffer for all 4 players (self-play)
    shared_agent = DQNAgent(
        device=device,
        state_shape=state_shape,
        max_actions=max_actions,
        lr=1e-4,
        buffer_capacity=100_000,
        batch_size=64,
        gamma=0.99,
        min_buffer_size=1000,
        target_update_freq=1000,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=1e-5,
    )

    # All 4 players use the same policy
    agent_callables: List[Any] = [shared_agent.act for _ in range(4)]

    def on_after_step(
        step_count: int,
        player_index: int,
        controller: DQNController,
        step_info: Dict[str, Any],
    ) -> None:
        # Store every transition (from any player) in the shared dataset
        trans = controller.get_last_transition()
        if trans is not None:
            state, action, reward, next_state, done = trans
            # Use full reward for this player (env returns only tile reward; capture/win bonus is in reward_deltas)
            reward_deltas = step_info.get("reward_deltas")
            if reward_deltas is not None and player_index in reward_deltas:
                reward = reward_deltas[player_index]
            shared_agent.push_transition(
                state, action, reward, next_state, done
            )
        # Every 40 moves: one training step on the shared DQN (random sample from shared buffer)
        if step_count > 0 and step_count % TRAIN_EVERY_N_MOVES == 0:
            shared_agent.train_step()

    print(
        f"Starting self-play: {NUM_GAMES} games, 1 shared policy, "
        f"train every {TRAIN_EVERY_N_MOVES} moves"
    )
    wins = [0, 0, 0, 0]
    total_steps = 0

    for game_id in range(NUM_GAMES):
        result = run_game(
            env,
            controllers,
            agent_callables,
            max_steps=10_000,
            on_after_step=on_after_step,
        )
        total_steps += result["step_count"]
        winner = result["winner"]
        if winner is not None and 0 <= winner < 4:
            wins[winner] += 1
        shared_agent.decay_epsilon()

        if (game_id + 1) % SAVE_EVERY_N_GAMES == 0:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            path = os.path.join(CHECKPOINT_DIR, f"dqn_policy_game_{game_id + 1}.pt")
            shared_agent.save(path)
            print(f"  Saved agent to {path}")

        if (game_id + 1) % 10 == 0 or game_id == 0:
            m = result.get("per_player_metrics")
            print(
                f"Game {game_id + 1}/{NUM_GAMES} | Winner: {winner} | "
                f"Steps: {result['step_count']} | Wins: {wins}"
            )
            if m:
                print("  Per-player (reward, land, troops, moves, status):")
                for i, p in enumerate(m):
                    status = "eliminated" if p.get("eliminated", False) else "alive"
                    print(
                        f"    P{i}: reward={p['reward']:.2f} land={p['territory']} troops={p['army']} moves={p['moves']} [{status}]"
                    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    final_path = os.path.join(CHECKPOINT_DIR, "dqn_policy_final.pt")
    shared_agent.save(final_path)
    print(f"\nDone. Total steps: {total_steps}. Win counts: {wins}")
    print(f"Saved final agent to {final_path}")


if __name__ == "__main__":
    main()
