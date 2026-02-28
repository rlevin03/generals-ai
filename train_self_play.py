"""
Self-play training loop: 4 DQNs vs each other (1v1v1v1), 1000 games,
training every 40 moves via random sampling from replay buffers.
"""

import torch
from typing import Any, Dict, List

from environment import GeneralsEnv
from controller import DQNController
from agents import DQNAgent, DEFAULT_STATE_HWC, DEFAULT_MAX_ACTIONS
from run_game import run_game


TRAIN_EVERY_N_MOVES = 40
NUM_GAMES = 1000


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GeneralsEnv(grid_size=(10, 10), training_mode=True, device="cpu")

    # One controller per player (all share the same env)
    controllers: List[DQNController] = [DQNController(env, device) for _ in range(4)]

    # Four independent DQN agents (each with own network and replay buffer)
    agents: List[DQNAgent] = [
        DQNAgent(
            device=device,
            state_shape=DEFAULT_STATE_HWC,
            max_actions=DEFAULT_MAX_ACTIONS,
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
        for _ in range(4)
    ]

    # Callables for run_game: agent i uses agents[i].act(controller)
    agent_callables: List[Any] = [lambda c, i=j: agents[i].act(c) for j in range(4)]

    def on_after_step(
        step_count: int,
        player_index: int,
        controller: DQNController,
        step_info: Dict[str, Any],
    ) -> None:
        # Push last transition to the player who just moved
        trans = controller.get_last_transition()
        if trans is not None:
            state, action, reward, next_state, done = trans
            agents[player_index].push_transition(
                state, action, reward, next_state, done
            )
        # Every 40 moves: train all 4 DQNs with random sampling from their buffers
        if step_count > 0 and step_count % TRAIN_EVERY_N_MOVES == 0:
            for a in agents:
                a.train_step()

    print(f"Starting self-play: {NUM_GAMES} games, train every {TRAIN_EVERY_N_MOVES} moves")
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
        # Decay epsilon for all agents after each game
        for a in agents:
            a.decay_epsilon()

        if (game_id + 1) % 50 == 0 or game_id == 0:
            print(
                f"Game {game_id + 1}/{NUM_GAMES} | Winner: {winner} | "
                f"Steps: {result['step_count']} | Wins: {wins}"
            )

    print(f"\nDone. Total steps: {total_steps}. Win counts: {wins}")


if __name__ == "__main__":
    main()
