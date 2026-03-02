"""
Self-play training loop: 1 shared policy (DQN), 4 copies playing each other (1v1v1v1).
All transitions from all agents go into one dataset; we train the same DQN by random
sampling from that shared replay buffer. 1000 games, training every 40 moves.

Option: set NUM_PARALLEL_GAMES > 1 to run that many games in parallel using CPU
multiprocessing (e.g. 4 = 4 games at once). When 1, runs sequentially as before.
"""

import torch
import os
import multiprocessing
from typing import Any, Dict, List, Tuple

from environment import GeneralsEnv
from controller import DQNController
from agents import DQNAgent
from run_game import run_game


TRAIN_EVERY_N_MOVES = 40
NUM_GAMES = 500
SAVE_EVERY_N_GAMES = 200
CHECKPOINT_DIR = "checkpoints"
MAX_STEPS_PER_GAME = 10_000
# Run this many games in parallel via multiprocessing (1 = sequential).
NUM_PARALLEL_GAMES = 4


def _dqn_self_play_worker(
    online_state_dict: Dict[str, torch.Tensor],
    epsilon: float,
    agent_config: Dict[str, Any],
    game_index: int,
) -> Tuple[List[Tuple[Any, int, float, Any, bool]], Any, int]:
    """
    Run one self-play game in a worker process. Returns (transitions, winner, step_count).
    Transitions use CPU tensors for pickling.
    """
    torch.set_num_threads(1)
    from environment import GeneralsEnv
    from controller import DQNController
    from agents import DQNAgent
    from run_game import run_game

    device = torch.device("cpu")
    env = GeneralsEnv(grid_size=(10, 10), training_mode=True, device="cpu")
    env.max_steps = MAX_STEPS_PER_GAME

    state_shape = (env.grid_height, env.grid_width, 6)
    max_actions = env.grid_width * env.grid_height * 4
    config = dict(agent_config)
    config["state_shape"] = state_shape
    config["max_actions"] = max_actions
    config["buffer_capacity"] = 1000
    config["min_buffer_size"] = 0

    agent = DQNAgent(device=device, **config)
    agent.online_net.load_state_dict(online_state_dict)
    agent.target_net.load_state_dict(online_state_dict)
    agent.epsilon = epsilon

    controllers: List[DQNController] = [DQNController(env, device) for _ in range(4)]
    agent_callables: List[Any] = [agent.act for _ in range(4)]
    transitions: List[Tuple[Any, int, float, Any, bool]] = []

    def on_after_step(
        step_count: int,
        player_index: int,
        controller: DQNController,
        step_info: Dict[str, Any],
    ) -> None:
        trans = controller.get_last_transition()
        if trans is not None:
            state, action, reward, next_state, done = trans
            reward_deltas = step_info.get("reward_deltas")
            if reward_deltas is not None and player_index in reward_deltas:
                reward = reward_deltas[player_index]
            state_cpu = state.detach().cpu() if state.is_cuda else state.cpu()
            next_cpu = next_state.detach().cpu() if next_state.is_cuda else next_state.cpu()
            transitions.append((state_cpu, action, reward, next_cpu, done))

    result = run_game(
        env,
        controllers,
        agent_callables,
        max_steps=MAX_STEPS_PER_GAME,
        on_after_step=on_after_step,
    )
    return (transitions, result["winner"], result["step_count"])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GeneralsEnv(grid_size=(10, 10), training_mode=True, device="cpu")
    env.max_steps = MAX_STEPS_PER_GAME  # env defaults to 5000; match run_game limit

    # State shape and max_actions must match the env's grid (H, W, C) and H*W*4
    state_shape = (env.grid_height, env.grid_width, 6)
    max_actions = env.grid_width * env.grid_height * 4

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
    agent_config = {
        "lr": 1e-4,
        "buffer_capacity": 100_000,
        "batch_size": 64,
        "gamma": 0.99,
        "min_buffer_size": 1000,
        "target_update_freq": 1000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 1e-5,
    }
    wins = [0, 0, 0, 0]
    total_steps = 0

    if NUM_PARALLEL_GAMES <= 1:
        # Sequential: original loop
        controllers: List[DQNController] = [DQNController(env, device) for _ in range(4)]
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
            f"Starting self-play: {NUM_GAMES} games, 1 shared policy (sequential), "
            f"train every {TRAIN_EVERY_N_MOVES} moves"
        )
        for game_id in range(NUM_GAMES):
            result = run_game(
                env,
                controllers,
                agent_callables,
                max_steps=MAX_STEPS_PER_GAME,
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
    else:
        # Parallel: run NUM_PARALLEL_GAMES at a time via multiprocessing
        print(
            f"Starting self-play: {NUM_GAMES} games, 1 shared policy, "
            f"{NUM_PARALLEL_GAMES} parallel games, train every {TRAIN_EVERY_N_MOVES} moves"
        )
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(NUM_PARALLEL_GAMES) as pool:
            for batch_start in range(0, NUM_GAMES, NUM_PARALLEL_GAMES):
                batch_size = min(NUM_PARALLEL_GAMES, NUM_GAMES - batch_start)
                state_dict = {k: v.cpu().clone() for k, v in shared_agent.online_net.state_dict().items()}
                args_list = [
                    (state_dict, shared_agent.epsilon, agent_config, batch_start + i)
                    for i in range(batch_size)
                ]
                results = pool.starmap(_dqn_self_play_worker, args_list)
                for (transitions, winner, step_count) in results:
                    total_steps += step_count
                    if winner is not None and 0 <= winner < 4:
                        wins[winner] += 1
                    for (s, a, r, s2, d) in transitions:
                        shared_agent.push_transition(s, a, r, s2, d)
                    shared_agent.decay_epsilon()
                num_new = sum(len(t[0]) for t in results)
                for _ in range(num_new // TRAIN_EVERY_N_MOVES):
                    shared_agent.train_step()
                last_game_id = batch_start + batch_size - 1
                last_result = results[-1]
                if (last_game_id + 1) % SAVE_EVERY_N_GAMES == 0:
                    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                    path = os.path.join(CHECKPOINT_DIR, f"dqn_policy_game_{last_game_id + 1}.pt")
                    shared_agent.save(path)
                    print(f"  Saved agent to {path}")
                if (last_game_id + 1) % 10 == 0 or last_game_id == 0:
                    print(
                        f"Game {last_game_id + 1}/{NUM_GAMES} | Winner: {last_result[1]} | "
                        f"Steps: {last_result[2]} | Wins: {wins}"
                    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    final_path = os.path.join(CHECKPOINT_DIR, "dqn_policy_final.pt")
    shared_agent.save(final_path)
    print(f"\nDone. Total steps: {total_steps}. Win counts: {wins}")
    print(f"Saved final agent to {final_path}")


if __name__ == "__main__":
    main()
