# Important Features

This document highlights the main design and implementation choices that make the project maintainable, scalable, and effective for training RL agents on the Generals-style game.

---

## 1. System Design: Separation of Concerns

The codebase is split into four clear layers so that each part has a single responsibility and can be changed or tested independently.

| Layer | Role | Location |
|-------|------|----------|
| **Game** | Core rules: grid, cells, movement, combat, fog of war, win/loss. No RL, no tensors. | `game/generals.py` |
| **Environment** | RL interface: wraps the game, exposes state and valid actions, runs steps, computes rewards (tile capture, elimination bonus, win bonus). Agent-agnostic. | `environment/generals_rl_env_gpu.py` |
| **Controller** | Adapter between env and a specific agent type: converts raw state → agent input (e.g. tensors), valid moves → action indices, and agent action index → explicit env action. Stores last transition for training. | `controller/dqn_controller.py`, `controller/ppo_controller.py` |
| **Agent** | Policy + learning: consumes state and valid-action indices from a controller, outputs an action index. No direct dependency on the game or env; depends only on a small controller protocol. | `agents/DQN_agent.py`, `agents/PPO_agent.py` |

**Benefits**

- **Swappable agents:** DQN and PPO use the same controller interface (`get_state_for_agent`, `get_valid_actions_for_agent`, `step`, `get_last_transition`), so you can plug in a new algorithm without touching game or env.
- **Consistent env API:** The env exposes `get_state()`, `get_valid_actions_explicit()`, and `step_explicit()`. Any controller (and thus any agent) can sit on top of it.
- **Focused testing:** You can test game logic without RL, env without a specific agent, and agents with a mock controller.

**Iterative development**

- Change reward shaping (e.g. win bonus, capture bonus) only in the **environment**.
- Change grid size or observation channels in **env** and **game**; controllers and agents stay the same as long as state shape is passed through.
- Add a new agent (e.g. A2C) by implementing the same controller protocol and adding a new controller if needed; no refactor of game or env.

---

## 2. CPU Parallelization

Self-play training can run **multiple games at once** using Python multiprocessing to better use multi-core CPUs.

- **Where:** `train_self_play.py` (DQN) and `train_self_play_ppo.py` (PPO).
- **Option:** `NUM_PARALLEL_GAMES` (default `4`). Set to `1` for the original sequential behavior.
- **Mechanism:** A **spawn** pool of worker processes. Each worker:
  - Receives the current policy weights (e.g. `state_dict`) and config.
  - Builds its own env and agent (CPU), runs one full game, and collects transitions.
  - Returns transitions (and winner, step count) to the main process.
- **Main process:** Aggregates transitions from all workers, pushes them into the shared replay/rollout buffer, runs training steps (DQN: every 40 moves; PPO: when buffer reaches `rollout_size`), and optionally saves/prints. Policy is updated in the main process; the next batch of games uses the updated weights.

Workers use `torch.set_num_threads(1)` to avoid oversubscription. The main process can use GPU for the shared agent; workers stay on CPU so there are no CUDA multiprocessing issues.

---

## 3. Curriculum Learning

Training is staged so the policy first learns from self-play, then from playing against fixed baselines of increasing difficulty.

1. **Self-play (DQN or PPO)**  
   - One shared policy, 4 copies playing each other.  
   - All transitions go into one replay/rollout buffer; training is on mixed experience from all players.  
   - Output: e.g. `dqn_policy_final.pt` or `ppo_policy_final.pt`.

2. **Vs. normal greedy (DQN only)**  
   - Load the self-play checkpoint (`dqn_policy_final.pt`).  
   - P0 = DQN, P1–P3 = normal greedy (expand, take neutral cities, attack weaker).  
   - Only P0’s transitions are used for training.  
   - Output: `dqn_policy_vs_normal_greedy_final.pt` (and intermediates like `dqn_policy_vs_normal_greedy_game_200.pt`).

3. **Vs. aggressive greedy (DQN only)**  
   - Load the vs-normal-greedy checkpoint (e.g. `dqn_policy_vs_normal_greedy_game_200.pt`).  
   - P0 = DQN, P1–P3 = aggressive greedy (prioritize capturing generals/cities).  
   - Same training setup as vs normal greedy.  
   - Output: `dqn_policy_vs_aggressive_greedy_final.pt`.

This order (self-play → vs normal → vs aggressive) gives a curriculum: the policy first learns general play, then adapts to greedy opponents, then to more aggressive ones, without refactoring the rest of the system.

---

## 4. Neural Network Architecture (DQN and PPO)

Both DQN and PPO use the same **spatial backbone** so that state representation is consistent and hyperparameters can be compared fairly.

**Shared backbone**

- **Input:** State tensor of shape `(H, W, 6)` (e.g. 10×10×6): channels are owner, army (normalized), is_city, is_general, is_mountain, is_visible.
- **Conv block:** 3×3 convs with BatchNorm and ReLU: 6 → 32 → 64 → 128 channels.
- **Flatten** then **MLP:** 128×H×W → 512 → 256 → 256 → 256 → 256 (ReLU).

**DQN (Dueling)**

- The backbone’s 256-d vector is fed into two heads:
  - **Value stream:** Linear(256 → 1).
  - **Advantage stream:** Linear(256 → max_actions).
- Output: `Q(s,a) = value + (advantage - mean(advantage))`. Action selection is over **valid action indices** provided by the controller (masking invalid actions).

**PPO (Actor–Critic)**

- Same conv + MLP backbone.
- **Actor:** Linear(256 → max_actions) for policy logits; sampling is over controller-provided valid indices.
- **Critic:** Linear(256 → 1) for state value.
- Single shared `ActorCritic` module; forward returns (logits, value). GAE and PPO loss (clipped surrogate, value loss, entropy) are computed in the agent.

Using the same backbone keeps the representation comparable between DQN and PPO and simplifies reuse (e.g. loading a DQN backbone for a new head).

---

## 5. Model Training Considerations

**DQN**

- **Replay buffer:** Capacity 100k; store (s, a, r, s′, done) with full reward (including capture/win from `reward_deltas`).
- **Training:** Every 40 moves (global or per batch when using parallel workers), sample a minibatch, compute TD target with target network, MSE loss, Adam.
- **Target network:** Updated every 1000 training steps.
- **Exploration:** ε-greedy; ε decayed per game (e.g. toward 0.1). Eval can use the checkpoint’s ε (or fixed 0.05) so evaluation matches the policy’s training behavior.

**PPO**

- **Rollout buffer:** Collect (s, a, r, s′, done, log_prob). When buffer size ≥ `rollout_size` (e.g. 2048), run one PPO update.
- **GAE:** Advantages and returns computed with γ and λ (e.g. 0.99, 0.95); advantages normalized before the update.
- **Update:** Multiple epochs over the buffer in minibatches; clipped surrogate loss + value loss + entropy bonus; gradient clipping.
- **Reward:** Same as DQN—use `reward_deltas` (tile + capture + win) so elimination and winning are reflected in the policy gradient.

**General**

- **Reward design:** Implemented in the env: per-tile rewards, capture bonus, win bonus. Controllers pass through `reward_deltas` so the agent always trains on the full reward.
- **Step limit:** Env and scripts align on `max_steps` (e.g. 10k) so games can terminate by step limit; winner is then inferred from the single non-eliminated player when the game doesn’t set a winner (e.g. on timeout).
- **Checkpointing:** Self-play and vs-greedy scripts save periodically (e.g. every 200 games) and at the end so you can resume or evaluate at different stages of the curriculum.

These choices keep training stable, consistent with the curriculum, and easy to extend (e.g. more reward terms in the env, or a new agent that reuses the same controller and reward pipeline).
