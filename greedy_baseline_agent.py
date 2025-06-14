from __future__ import annotations

import argparse, random
from typing import List, Tuple

from generals_rl_env_gpu import GeneralsEnv
from generals import CellType

# ── Add reduced‑action helper at run‑time ────────────────────────────

def _patch_get_reduced_action_space(self: GeneralsEnv) -> Tuple[List[int], dict[int, Tuple[int,int,int,int]]]:
    """Return *(legal_ids, id→(fx,fy,tx,ty))* using env's own validators."""
    legal, mapping, idx = [], {}, 0
    for y in range(self.grid_height):
        for x in range(self.grid_width):
            c = self.game.grid[y][x]
            if c.owner != self.player_id or c.army < 1:
                continue
            for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:  # UDLR
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                    continue
                if not self._is_valid_move(x, y, nx, ny):
                    continue
                mapping[idx] = (x, y, nx, ny)
                legal.append(idx)
                idx += 1
    # store mapping on env for step_with_reduced_action
    self.action_mapping = mapping
    return legal, mapping

# Attach once
if not hasattr(GeneralsEnv, "get_reduced_action_space"):
    GeneralsEnv.get_reduced_action_space = _patch_get_reduced_action_space

# ── Silence verbose debug prints ────────────────────────────────────

import sys, io, contextlib
from generals import Game  # after import so class is defined

# 1) wrap Game.execute_move to suppress its print spam
if not hasattr(Game, "_orig_execute_move"):
    Game._orig_execute_move = Game.execute_move  # backup

    def _quiet_execute_move(self, move):
        with contextlib.redirect_stdout(io.StringIO()):
            return Game._orig_execute_move(self, move)

    Game.execute_move = _quiet_execute_move

# replace the built‑in step_with_reduced_action with a silent version

def _silent_step_with_reduced_action(self: GeneralsEnv, action_idx: int):
    """Same logic as original but without every-100‑step debug spam."""
    if not getattr(self, "action_mapping", None):
        return self.step(0)  # no‑op if nothing legal yet
    if action_idx not in self.action_mapping:
        return self._get_observation(), self.invalid_move_penalty, False, {"invalid_move": True}

    fx, fy, tx, ty = self.action_mapping[action_idx]
    fat = fy * self.grid_width * 4 + fx * 4
    if   ty < fy: fat += 0
    elif ty > fy: fat += 1
    elif tx < fx: fat += 2
    else:         fat += 3
    return self.step(fat)

GeneralsEnv.step_with_reduced_action = _silent_step_with_reduced_action

if hasattr(GeneralsEnv, "get_valid_source_cells"):
    def _quiet_get_valid_source_cells(self):
        return [(x,y) for y in range(self.grid_height) for x in range(self.grid_width)
                if (c:=self.game.grid[y][x]).owner==self.player_id and c.army>=1]
    GeneralsEnv.get_valid_source_cells = _quiet_get_valid_source_cells

# ── Greedy Agent ──────────────────────────────────────────────────── ────────────────────────────────────────────────────

class GreedyAgent:
    """Single‑step heuristic – no memory between turns."""

    def select(self, env: GeneralsEnv):
        legal, mapping = env.get_reduced_action_space()
        if not legal:
            return None  # skip turn if nothing legal

        best_score, best_act = -1e9, random.choice(legal)
        for aid in legal:
            fx, fy, tx, ty = mapping[aid]
            src  = env.game.grid[fy][fx]
            dest = env.game.grid[ty][tx]

            score  = 0.0
            # 1. capture neutral city fast
            if dest.type == CellType.CITY and dest.owner == -1 and src.army > dest.army:
                score += 1_000
            # 2. grab neutral land
            if dest.owner == -1:
                score += 100
            # 3. attack weaker enemy tile
            if dest.owner not in (-1, env.player_id) and src.army > dest.army:
                score += 50 + (dest.army - src.army)
            # 4. prefer moves that mobilise bigger stacks
            score += src.army * 0.1

            if score > best_score:
                best_score, best_act = score, aid
        return best_act

# ── Episode runner ──────────────────────────────────────────────────

def run_episode(agent: GreedyAgent, grid: int):
    env = GeneralsEnv(player_id=0, num_opponents=1, training_mode=True)
    if grid != env.grid_width:
        env.grid_width = env.grid_height = grid  # quick square resize
    env.reset()

    done = False
    while not done:
        act_id = agent.select(env)
        if act_id is None:
            _, _, done, _ = env.step(0)  # dummy – env imposes small penalty
        else:
            _, _, done, _ = env.step_with_reduced_action(act_id)

    return env.game.winner == env.player_id

# ── Main / CLI ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Greedy baseline for Generals.io")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--grid", type=int, default=25, help="board size (square)")
    args = ap.parse_args()

    agent = GreedyAgent()
    wins = 0
    for ep in range(1, args.episodes + 1):
        wins += run_episode(agent, args.grid)
        if ep % 20 == 0 or ep == args.episodes:
            print(f"Episode {ep}/{args.episodes} – win‑rate: {wins/ep:.1%}")

if __name__ == "__main__":
    main()
