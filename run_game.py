"""
Run a single Generals game until it ends.

Loop: get current player index from env, that player's agent chooses an action index,
env.step_by_index(index) advances the game and returns (next_state_tensor, reward, done, info).
No controllers: agents receive env and get state/valid actions from it.

Usage:
  - Pass one env, a list of 4 agent callables agent(env) -> action_index, and device.
  - env.current_player_index indicates whose turn.
  - Returns summary (winner, step_count, game_over, per_player_metrics).
"""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

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
    agents: List[Callable[[GeneralsEnv], int]],
    device: torch.device,
    max_steps: int = 10_000,
    on_after_step: Optional[Callable[[int, int, GeneralsEnv, Dict[str, Any]], None]] = None,
    on_before_step: Optional[Callable[[int, int, GeneralsEnv], None]] = None,
) -> Dict[str, Any]:
    """
    Run one game until done. Agents interact directly with env (no controllers).

    Args:
        env: Generals environment (holds game state and current_player_index 0-3).
        agents: List of 4 callables; agents[i](env) returns action index for player i.
        device: Device for env state tensors.
        max_steps: Safety limit on steps.
        on_after_step: Optional callback(step_count, player_index, env, step_info) after each step.
        on_before_step: Optional callback(step_count, player_index, env) before each step.

    Returns:
        Dict with keys: winner (int or None), step_count (int), done (bool),
        game_over (bool), per_player_metrics (list of dicts).
    """
    env.reset(device)
    step_count = 0
    per_player_reward: List[float] = [0.0, 0.0, 0.0, 0.0]
    per_player_moves: List[int] = [0, 0, 0, 0]

    while not env.is_done() and step_count < max_steps:
        current = env.current_player_index
        agent_fn = agents[current]

        if on_before_step is not None:
            on_before_step(step_count, current, env)

        action_idx = agent_fn(env)
        next_state, reward, done, info = env.step_by_index(action_idx)

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
            on_after_step(step_count, current, env, step_info)

    winner = getattr(env.game, "winner", None)
    if env.game.game_over and hasattr(env.game, "winner"):
        winner = env.game.winner

    per_player_metrics: List[Dict[str, Any]] = []
    for i in range(4):
        territory, army = _territory_and_army_for_player(env.game, i)
        is_alive = env.game.players[i].is_alive
        eliminated = not is_alive or (territory == 0 and army == 0)
        per_player_metrics.append({
            "reward": per_player_reward[i],
            "territory": territory,
            "army": army,
            "moves": per_player_moves[i],
            "eliminated": eliminated,
        })

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


def random_agent(env: GeneralsEnv) -> int:
    """Agent that picks a random valid action index. Use as agents[i] for testing."""
    valid = env.get_valid_action_indices()
    return random.choice(valid) if valid else 0


def random_agent(env: GeneralsEnv) -> int:
    """Agent that picks a random valid action index. Use as agents[i] for testing."""
    valid = env.get_valid_action_indices()
    return random.choice(valid) if valid else 0


def run_game_with_controllers(
    env: GeneralsEnv,
    controllers: List[Any],
    agents: List[Callable[[Any], int]],
    max_steps: int = 10_000,
    on_after_step: Optional[Callable[[int, int, Any, Dict[str, Any]], None]] = None,
    on_before_step: Optional[Callable[[int, int, Any], None]] = None,
) -> Dict[str, Any]:
    """
    Legacy: run one game with controllers (for train_vs_*_greedy scripts).
    Prefer run_game(env, agents, device, ...) for new code.
    """
    env.reset()
    step_count = 0
    per_player_reward: List[float] = [0.0, 0.0, 0.0, 0.0]
    per_player_moves: List[int] = [0, 0, 0, 0]

    while not env.is_done() and step_count < max_steps:
        current = env.current_player_index
        controller = controllers[current]
        agent_fn = agents[current]
        if on_before_step is not None:
            on_before_step(step_count, current, controller)
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
        step_info = {"player": current, "reward": reward, "done": done, **info}
        if on_after_step is not None:
            on_after_step(step_count, current, controller, step_info)

    winner = getattr(env.game, "winner", None)
    if env.game.game_over and hasattr(env.game, "winner"):
        winner = env.game.winner
    per_player_metrics = []
    for i in range(4):
        territory, army = _territory_and_army_for_player(env.game, i)
        is_alive = env.game.players[i].is_alive
        eliminated = not is_alive or (territory == 0 and army == 0)
        per_player_metrics.append({
            "reward": per_player_reward[i],
            "territory": territory,
            "army": army,
            "moves": per_player_moves[i],
            "eliminated": eliminated,
        })
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GeneralsEnv(grid_size=(10, 10), training_mode=True, device="cpu")
    env.max_steps = 10_000

    agents_list = [random_agent for _ in range(4)]
    result = run_game(env, agents_list, device)
    print("Winner:", result["winner"])
    print("Steps:", result["step_count"])
    print("Game over:", result["game_over"])
    m = result.get("per_player_metrics", [])
    if m:
        print("Per-player (reward, land, troops, moves, status):")
        for i, p in enumerate(m):
            status = "eliminated" if p.get("eliminated", False) else "alive"
            print(f"  P{i}: reward={p['reward']:.2f} land={p['territory']} troops={p['army']} moves={p['moves']} [{status}]")
