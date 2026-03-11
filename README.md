# Generals.io AI - Reinforcement Learning Project

A Python implementation of the popular Generals.io real-time strategy game with multiple AI agents trained using reinforcement learning (DQN and PPO).

## 🎮 About Generals.io

Generals.io is a real-time strategy game where players compete to capture territory and eliminate opponents. Each player starts with a general (crown symbol) and must expand their territory by moving armies between adjacent cells. The goal is to capture the enemy general while protecting your own.

### Game Features

- **Real-time or turn-based gameplay** (turn-based used for RL training)
- **Fog of war** – players only see their territory and adjacent cells
- **Territory control** – captured cells generate armies over time
- **Cities** – special cells that provide additional army generation
- **Mountains** – impassable terrain that blocks movement
- **Multiple players** – 4 players in a single game for self-play training

### Game Mechanics

- **Army generation**: Generals and cities generate 1 army per turn; all owned cells generate 1 army every N turns
- **Movement**: Move armies from owned cells to adjacent cells (4 directions)
- **Combat**: When attacking, the stronger force wins; ties go to defender
- **Victory**: Capture the enemy general to eliminate a player; last player standing wins

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch numpy pygame matplotlib pandas tensorboard gym
```

### Running the Base Game

```bash
python game/generals.py
```

**Controls:**

- **Mouse**: Click to select territory, right-click to move
- **WASD/Arrow Keys**: Chain moves from selected cell
- **Shift**: Move half army instead of full army
- **Tab**: Switch between player views
- **ESC**: Clear selection
- **R**: Restart game (when game over)

## 🤖 Architecture (Post-Refactor)

### Environment → Agent Flow (No Controllers in Self-Play)

For **self-play DQN** and **self-play PPO**, agents interact **directly with the environment**:

- **Environment** (`environment/generals_rl_env_gpu.py`):
  - Owns the `Game` (from `game/generals.py`).
  - Exposes **state as a PyTorch tensor** `(1, H, W, 6)` via `get_state_tensor(device)`.
  - Exposes **valid actions as indices** `[0..n-1]` via `get_valid_action_indices()` (indexed over the current list of valid explicit moves).
  - **`step_by_index(action_index)`**: executes the move for that index and returns `(next_state_tensor, reward, done, info)`. Reward is for the acting player; `info` can include `reward_deltas` for all players.
  - **`get_last_transition()`**: returns `(state_tensor, action_idx, reward, next_state_tensor, done)` after each step for training.
  - **`reset(device)`**: with `device` set, returns `(state_tensor, valid_indices, info)`; with `device=None`, returns numpy observation (legacy).

- **Agents** (`agents/DQN_agent.py`, `agents/PPO_agent.py`):
  - **`act(env)`**: call `env.get_state_tensor(device)` and `env.get_valid_action_indices()`, then return a single **action index** (integer).
  - No controller: the env handles indexing of valid moves and conversion to the game’s explicit moves internally.

- **Run loop** (`run_game.py`):
  - **`run_game(env, agents, device, max_steps=..., on_after_step=..., on_before_step=...)`**: no controllers. Each step: `action_idx = agents[current](env)`, then `next_state, reward, done, info = env.step_by_index(action_idx)`. Callbacks receive `(step_count, player_index, env, step_info)`; use `env.get_last_transition()` to push transitions to the agent.

### Controllers (Legacy / Vs-Greedy Scripts)

Scripts that train or evaluate **vs greedy baselines** still use **controllers**:

- **`run_game_with_controllers(env, controllers, agents, ...)`**: one controller per player; `agents[i](controller)` returns an action index; `controller.step(action_idx)` runs the move and returns the next state tensor, reward, done, info.
- Used by: `train_vs_normal_greedy.py`, `train_vs_mixed_greedy.py`, `run_eval_dqn_vs_greedy.py`.

## 📜 Training Scripts

### 1. DQN Self-Play (Env-Only, No Controllers)

```bash
# Train one shared DQN with 4 copies playing each other (1v1v1v1)
python train_self_play.py
```

- Uses `run_game(env, agents, device, ...)`. All transitions go into a shared replay buffer; training every 40 moves.
- Checkpoints: `checkpoints/dqn_policy_game_*.pt`, `checkpoints/dqn_policy_final.pt`.
- Set `NUM_PARALLEL_GAMES > 1` in the script to run multiple games in parallel (multiprocessing).

### 2. PPO Self-Play (Env-Only, No Controllers)

```bash
# Train one shared PPO policy with 4 copies playing each other
python train_self_play_ppo.py
```

- Same env-only flow. Rollout buffer; when it reaches `ROLLOUT_SIZE`, run GAE + PPO update.
- Checkpoints: `checkpoints/ppo_policy_game_*.pt`, `checkpoints/ppo_policy_final.pt`.
- Set `NUM_PARALLEL_GAMES > 1` for parallel games.

### 3. DQN vs Greedy (Uses Controllers)

```bash
# Train DQN (P0) vs 3 normal greedy opponents
python train_vs_normal_greedy.py

# Train DQN (P0) vs 3 aggressive greedy opponents (requires existing checkpoint)
python train_vs_mixed_greedy.py
```

- Use `run_game_with_controllers`. Controllers translate env state/valid actions and step by index.

### 4. Evaluation: DQN vs Greedy

```bash
# Run N matches: saved DQN vs greedy baseline
python run_eval_dqn_vs_greedy.py
```

- Uses `run_game_with_controllers`; loads checkpoint and reports win rates and metrics.

### 5. Greedy Baseline

```bash
# Run/evaluate greedy baseline agent
python greedy_baseline_agent.py
```

## ⚙️ Configuration

### Game / Env

- **Grid**: e.g. `grid_size=(10, 10)` in `GeneralsEnv` (and passed to `Game`).
- **Observation**: 6 channels `(owner, army, is_city, is_general, is_mountain, is_visible)`; state shape `(H, W, 6)`.
- **Valid actions**: list of explicit moves `(from_x, from_y, to_x, to_y)`; agents see them as indices `0..n-1`.

### Training (Self-Play)

- **train_self_play.py**: `TRAIN_EVERY_N_MOVES`, `NUM_GAMES`, `MAX_STEPS_PER_GAME`, `SAVE_EVERY_N_GAMES`, replay buffer size, epsilon decay, etc.
- **train_self_play_ppo.py**: `ROLLOUT_SIZE`, `TRAIN_EVERY_N_MOVES`, `n_epochs`, `batch_size`, GAE/PPO hyperparameters.

## 📁 Project Structure

```
├── game/
│   └── generals.py           # Core game (Cell, Game, Player, turn-based/real-time)
├── environment/
│   └── generals_rl_env_gpu.py # RL env: state tensor, indexed actions, step_by_index, get_last_transition
├── controller/               # Used by vs-greedy scripts only
│   ├── dqn_controller.py
│   └── ppo_controller.py
├── agents/
│   ├── DQN_agent.py          # DQN: act(env) -> index
│   └── PPO_agent.py         # PPO: act(env) -> index
├── run_game.py               # run_game(env, agents, device) and run_game_with_controllers(...)
├── train_self_play.py        # DQN self-play (env-only)
├── train_self_play_ppo.py    # PPO self-play (env-only)
├── train_vs_normal_greedy.py # DQN vs 3× normal greedy (controllers)
├── train_vs_mixed_greedy.py  # DQN vs 3× aggressive greedy (controllers)
├── run_eval_dqn_vs_greedy.py # Evaluate DQN vs greedy
├── greedy_baseline_agent.py  # Greedy baseline + env patches
├── checkpoints/              # Saved policies
└── README.md
```

### Observation & Action Space

- **Observation**: PyTorch tensor `(1, H, W, 6)` – owner, normalized army, city/general/mountain/visibility flags (fog of war applied for current player).
- **Actions**: Integer index in `[0..n-1]` where `n` is the number of valid explicit moves (from owned cell to visible non-mountain neighbor). The env maps index → `(from_x, from_y, to_x, to_y)` and executes the move.

## 🐛 Troubleshooting

- **CUDA out of memory**: Reduce batch size or use `device="cpu"` in env/training.
- **Slow training**: Use `NUM_PARALLEL_GAMES > 1` in self-play scripts (CPU multiprocessing).
- **Invalid moves**: Ensure you use the env’s `get_valid_action_indices()` and step with `step_by_index(index)`; the env only allows indices into the current valid list.
- **Pygame not found**: Install with `pip install pygame` when running the human-playable game (`game/generals.py`).

## Design Notes

- **Grid size**: The env passes `grid_width`/`grid_height` to `Game` so state and valid actions cover the full board for all four players.
- **Valid moves**: Only real moves (from owned cell to visible, non-mountain neighbor); no pass/idle in the valid list.
- **Reward**: Per-step reward for the acting player (e.g. tile capture); `info["reward_deltas"]` can include capture/win bonuses per player.
- **Elimination**: Handled by game logic (general captured → player dead); reward can add bonuses for kills/wins via `reward_deltas`.

## Acknowledgments

- Original Generals.io game concept (this implementation is a rough copy).
