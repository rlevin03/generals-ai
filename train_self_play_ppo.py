"""
Self-play training with PPO: 1 shared policy, 4 copies playing each other (1v1v1v1).

Same structure as train_self_play (DQN): one env, 4 controllers, 4 agent callables
(shared_agent.act). All transitions go into one rollout buffer; when buffer reaches
rollout_size we run PPO update (GAE + clipped loss). Uses reward_deltas for full reward.

Option: set NUM_PARALLEL_GAMES > 1 to run that many games in parallel using CPU
multiprocessing (e.g. 4 = 4 games at once). When 1, runs sequentially as before.
"""

import torch
import os
import multiprocessing
from typing import Any, Dict, List, Tuple

from environment import GeneralsEnv
from controller import PPOController
from agents import PPOAgent
from run_game import run_game


TRAIN_EVERY_N_MOVES = 40
NUM_GAMES = 500
SAVE_EVERY_N_GAMES = 200
CHECKPOINT_DIR = "checkpoints"
ROLLOUT_SIZE = 2048
MAX_STEPS_PER_GAME = 10_000
NUM_PARALLEL_GAMES = 4


def _ppo_self_play_worker(
    net_state_dict: Dict[str, torch.Tensor],
    agent_config: Dict[str, Any],
    game_index: int,
) -> Tuple[List[Tuple[Any, int, float, Any, bool, float]], Any, int]:
    """Run one PPO self-play game in a worker. Returns (transitions, winner, step_count)."""
    torch.set_num_threads(1)
    from environment import GeneralsEnv
    from controller import PPOController
    from agents import PPOAgent
    from run_game import run_game

    device = torch.device("cpu")
    env = GeneralsEnv(grid_size=(10, 10), training_mode=True, device="cpu")
    env.max_steps = MAX_STEPS_PER_GAME

    state_shape = (env.grid_height, env.grid_width, 6)
    max_actions = env.grid_width * env.grid_height * 4
    config = dict(agent_config)
    config["state_shape"] = state_shape
    config["max_actions"] = max_actions
    config["rollout_size"] = 999999

    agent = PPOAgent(device=device, **config)
    agent.net.load_state_dict(net_state_dict)

    controllers: List[PPOController] = [PPOController(env, device) for _ in range(4)]
    agent_callables: List[Any] = [agent.act for _ in range(4)]
    transitions: List[Tuple[Any, int, float, Any, bool, float]] = []

    def on_after_step(
        step_count: int,
        player_index: int,
        controller: PPOController,
        step_info: Dict[str, Any],
    ) -> None:
        trans = controller.get_last_transition()
        if trans is not None:
            state, action, reward, next_state, done = trans
            reward_deltas = step_info.get("reward_deltas")
            if reward_deltas is not None and player_index in reward_deltas:
                reward = reward_deltas[player_index]
            log_prob, _ = agent.get_last_log_prob_and_value()
            state_cpu = state.detach().cpu() if state.is_cuda else state.cpu()
            next_cpu = next_state.detach().cpu() if next_state.is_cuda else next_state.cpu()
            transitions.append((state_cpu, action, reward, next_cpu, done, log_prob))

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
    env.max_steps = MAX_STEPS_PER_GAME  # env defaults to 5000; use our limit

    state_shape = (env.grid_height, env.grid_width, 6)
    max_actions = env.grid_width * env.grid_height * 4

    shared_agent = PPOAgent(
        device=device,
        state_shape=state_shape,
        max_actions=max_actions,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        rollout_size=ROLLOUT_SIZE,
        batch_size=64,
        n_epochs=4,
    )
    agent_config = {
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "rollout_size": ROLLOUT_SIZE,
        "batch_size": 64,
        "n_epochs": 4,
    }
    wins = [0, 0, 0, 0]
    total_steps = 0

    if NUM_PARALLEL_GAMES <= 1:
        # Sequential
        controllers: List[PPOController] = [PPOController(env, device) for _ in range(4)]
        agent_callables: List[Any] = [shared_agent.act for _ in range(4)]

        def on_after_step(
            step_count: int,
            player_index: int,
            controller: PPOController,
            step_info: Dict[str, Any],
        ) -> None:
            trans = controller.get_last_transition()
            if trans is not None:
                state, action, reward, next_state, done = trans
                reward_deltas = step_info.get("reward_deltas")
                if reward_deltas is not None and player_index in reward_deltas:
                    reward = reward_deltas[player_index]
                log_prob, _ = shared_agent.get_last_log_prob_and_value()
                shared_agent.push_transition(
                    state, action, reward, next_state, done, log_prob
                )
            if step_count > 0 and step_count % TRAIN_EVERY_N_MOVES == 0:
                loss = shared_agent.train_step()
                if loss is not None and step_count % (TRAIN_EVERY_N_MOVES * 10) == 0:
                    print(f"  PPO loss: {loss:.4f}")

        print(
            f"Starting PPO self-play: {NUM_GAMES} games, 1 shared policy (sequential), "
            f"train every {TRAIN_EVERY_N_MOVES} moves, rollout_size={ROLLOUT_SIZE}"
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

            if (game_id + 1) % SAVE_EVERY_N_GAMES == 0:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                path = os.path.join(CHECKPOINT_DIR, f"ppo_policy_game_{game_id + 1}.pt")
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
        # Parallel
        print(
            f"Starting PPO self-play: {NUM_GAMES} games, 1 shared policy, "
            f"{NUM_PARALLEL_GAMES} parallel games, rollout_size={ROLLOUT_SIZE}"
        )
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(NUM_PARALLEL_GAMES) as pool:
            for batch_start in range(0, NUM_GAMES, NUM_PARALLEL_GAMES):
                batch_size = min(NUM_PARALLEL_GAMES, NUM_GAMES - batch_start)
                state_dict = {k: v.cpu().clone() for k, v in shared_agent.net.state_dict().items()}
                args_list = [
                    (state_dict, agent_config, batch_start + i)
                    for i in range(batch_size)
                ]
                results = pool.starmap(_ppo_self_play_worker, args_list)
                for (transitions, winner, step_count) in results:
                    total_steps += step_count
                    if winner is not None and 0 <= winner < 4:
                        wins[winner] += 1
                    for (s, a, r, s2, d, lp) in transitions:
                        shared_agent.push_transition(s, a, r, s2, d, lp)
                if len(shared_agent.buffer) >= ROLLOUT_SIZE:
                    loss = shared_agent.train_step()
                    if loss is not None and batch_start % 10 == 0:
                        print(f"  PPO loss: {loss:.4f}")
                last_game_id = batch_start + batch_size - 1
                last_result = results[-1]
                if (last_game_id + 1) % SAVE_EVERY_N_GAMES == 0:
                    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                    path = os.path.join(CHECKPOINT_DIR, f"ppo_policy_game_{last_game_id + 1}.pt")
                    shared_agent.save(path)
                    print(f"  Saved agent to {path}")
                if (last_game_id + 1) % 10 == 0 or last_game_id == 0:
                    print(
                        f"Game {last_game_id + 1}/{NUM_GAMES} | Winner: {last_result[1]} | "
                        f"Steps: {last_result[2]} | Wins: {wins}"
                    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    final_path = os.path.join(CHECKPOINT_DIR, "ppo_policy_final.pt")
    shared_agent.save(final_path)
    print(f"\nDone. Total steps: {total_steps}. Win counts: {wins}")
    print(f"Saved final PPO agent to {final_path}")


if __name__ == "__main__":
    main()
