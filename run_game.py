"""
Run a single Generals game until it ends.

Loop: get current player index from env, tell that player's controller to make a move
(using the env), store results. Repeats until the game ends.

Usage:
  - Pass one env and a list of 4 controllers (one per player).
  - Each controller is bound to the same env; env.current_player_index indicates whose turn.
  - Pass a list of 4 agent callables: agent(controller) -> action_index.
  - The script runs the loop and returns summary (winner, step_count, game_over, per_player_metrics).
"""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from environment import GeneralsEnv


def _territory_and_army_for_player(game: Any, player_id: int) -> Tuple[int, int]:
    """Count land (cells) and total troops for a given player from game grid."""
    territory, army = 0, 0
    for row in game.grid:
        for cell in row:
            if cell.owner == player_id:
                territory += 1
                army += cell.army
    return territory, army


def run_game(
    env: GeneralsEnv,
    controllers: List[Any],
    agents: List[Callable[[Any], int]],
    max_steps: int = 10_000,
    on_after_step: Optional[Callable[[int, int, Any, Dict[str, Any]], None]] = None,
    on_before_step: Optional[Callable[[int, int, Any], None]] = None,
) -> Dict[str, Any]:
    """
    Run one game until done. Each iteration: get current player index, that player's
    controller makes a move (via env), store results. No class — just a loop.

    Args:
        env: Generals environment (holds game state and current_player_index 0-3).
        controllers: List of 4 controllers; controllers[i] is used when current_player_index == i.
        agents: List of 4 callables; agents[i](controller) returns action index for player i.
        max_steps: Safety limit on steps.
        on_after_step: Optional callback(step_count, player_index, controller, step_info) after each step.
        on_before_step: Optional callback(step_count, player_index, controller) before each step.

    Returns:
        Dict with keys: winner (int or None), step_count (int), done (bool),
        game_over (bool), per_player_metrics (list of dicts). No per-step list (saves memory).
    """
    env.reset()
    step_count = 0
    # Per-player cumulative reward and move count during the game
    per_player_reward: List[float] = [0.0, 0.0, 0.0, 0.0]
    per_player_moves: List[int] = [0, 0, 0, 0]

    while not env.is_done() and step_count < max_steps:
        current = env.current_player_index
        controller = controllers[current]
        agent_fn = agents[current]

        if on_before_step is not None:
            on_before_step(step_count, current, controller)

        # Controller provides state/valid actions; agent picks action index
        action_idx = agent_fn(controller)
        next_state, reward, done, info = controller.step(action_idx)

        step_count += 1
        reward_deltas = info.get("reward_deltas")
        if reward_deltas is not None:
            for pid, delta in reward_deltas.items():
                if 0 <= pid < 4:
                    per_player_reward[pid] += delta
        else:
            per_player_reward[current] += reward
        if info.get("action_taken", True):
            per_player_moves[current] += 1
        step_info = {
            "player": current,
            "reward": reward,
            "done": done,
            **info,
        }
        if on_after_step is not None:
            on_after_step(step_count, current, controller, step_info)

    winner = getattr(env.game, "winner", None)
    if env.game.game_over and hasattr(env.game, "winner"):
        winner = env.game.winner

    # Final land (territory), troops (army), and eliminated status per player
    per_player_metrics: List[Dict[str, Any]] = []
    for i in range(4):
        territory, army = _territory_and_army_for_player(env.game, i)
        is_alive = env.game.players[i].is_alive
        # Treat 0 land as eliminated (general captured / all territory lost)
        eliminated = not is_alive or (territory == 0 and army == 0)
        per_player_metrics.append({
            "reward": per_player_reward[i],
            "territory": territory,
            "army": army,
            "moves": per_player_moves[i],
            "eliminated": eliminated,
        })

    # If no winner from game (e.g. step limit): winner = single non-eliminated player (same rule as display)
    if winner is None or winner == -1:
        not_eliminated = [i for i in range(4) if not per_player_metrics[i]["eliminated"]]
        if len(not_eliminated) == 1:
            winner = not_eliminated[0]

    return {
        "winner": winner,
        "step_count": step_count,
        "done": env.is_done(),
        "game_over": env.game.game_over,
        "per_player_metrics": per_player_metrics,
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
    m = result.get("per_player_metrics", [])
    if m:
        print("Per-player (reward, land, troops, moves, status):")
        for i, p in enumerate(m):
            status = "eliminated" if p.get("eliminated", False) else "alive"
            print(f"  P{i}: reward={p['reward']:.2f} land={p['territory']} troops={p['army']} moves={p['moves']} [{status}]")
