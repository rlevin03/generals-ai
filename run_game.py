"""
Run a single Generals game until it ends.

Loop: get current player index from env, tell that player's controller to make a move
(using the env), store results. Repeats until the game ends.

Usage:
  - Pass one env and a list of 4 controllers (one per player).
  - Each controller is bound to the same env; env.current_player_index indicates whose turn.
  - Pass a list of 4 agent callables: agent(controller) -> action_index.
  - The script runs the loop and returns stored results (winner, steps, per-step info).
"""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from environment import GeneralsEnv


def run_game(
    env: GeneralsEnv,
    controllers: List[Any],
    agents: List[Callable[[Any], int]],
    max_steps: int = 10_000,
) -> Dict[str, Any]:
    """
    Run one game until done. Each iteration: get current player index, that player's
    controller makes a move (via env), store results. No class — just a loop.

    Args:
        env: Generals environment (holds game state and current_player_index 0-3).
        controllers: List of 4 controllers; controllers[i] is used when current_player_index == i.
        agents: List of 4 callables; agents[i](controller) returns action index for player i.
        max_steps: Safety limit on steps.

    Returns:
        Dict with keys: winner (int or None), step_count (int), done (bool),
        steps (list of per-step info dicts), game_over (bool).
    """
    env.reset()
    steps: List[Dict[str, Any]] = []
    step_count = 0

    while not env.is_done() and step_count < max_steps:
        current = env.current_player_index
        controller = controllers[current]
        agent_fn = agents[current]

        # Controller provides state/valid actions; agent picks action index
        action_idx = agent_fn(controller)
        next_state, reward, done, info = controller.step(action_idx)

        step_count += 1
        steps.append({
            "player": current,
            "reward": reward,
            "done": done,
            **info,
        })

    winner = getattr(env.game, "winner", None)
    if env.game.game_over and hasattr(env.game, "winner"):
        winner = env.game.winner

    return {
        "winner": winner,
        "step_count": step_count,
        "done": env.is_done(),
        "game_over": env.game.game_over,
        "steps": steps,
    }


def random_agent(controller: Any) -> int:
    """Agent that picks a random valid action index. Use as agents[i] for testing."""
    valid = controller.get_valid_actions_for_agent()
    return random.choice(valid) if valid else 0


if __name__ == "__main__":
    import torch
    from controller import DQNController

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GeneralsEnv(grid_size=(10, 10), training_mode=True, device="cpu")

    # One controller per player (all share the same env)
    controllers = [DQNController(env, device) for _ in range(4)]
    # Simple random agents for demo
    agents = [random_agent for _ in range(4)]

    result = run_game(env, controllers, agents)
    print("Winner:", result["winner"])
    print("Steps:", result["step_count"])
    print("Game over:", result["game_over"])
