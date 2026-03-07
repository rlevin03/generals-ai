# Neural Network Design: DQN and PPO

This document describes the neural network architectures used for the **Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)** agents in this project. Both algorithms share the same **spatial backbone** so that state representation is consistent and results can be compared fairly.

---

## 1. State Representation (Input)

- **Shape:** `(H, W, C)` with `C = 6`. Typical grid sizes use `H × W` (e.g. 10×10, 20×25). The network accepts batches as `(B, C, H, W)` (channel-first) or `(B, H, W, C)` (channel-last); input is normalized to channel-first internally.
- **Channels (in order):**
  1. **owner** — Cell owner: -1 (neutral), 0–3 (players).
  2. **army** — Normalized army count: `min(cell.army, 999) / 100.0`.
  3. **is_city** — Binary: city tile or not.
  4. **is_general** — Binary: general tile or not.
  5. **is_mountain** — Binary: mountain (impassable) or not.
  6. **is_visible** — Binary: cell visible to the current player or not.

Observations are produced by the environment (`GeneralsEnv`); see `environment/generals_rl_env_gpu.py` for the exact observation construction.

---

## 2. Shared Backbone (Conv + MLP)

Both DQN and PPO use the **same convolutional and fully connected backbone** to process the state tensor.

### 2.1 Convolutional block

- **Input:** `(B, C, H, W)` with `C = 6`.
- Three 2D convolutions, each with kernel size 3×3, padding 1 (spatial size unchanged):
  - Conv1: 6 → 32 channels, then BatchNorm2d(32), ReLU.
  - Conv2: 32 → 64 channels, then BatchNorm2d(64), ReLU.
  - Conv3: 64 → 128 channels, then BatchNorm2d(128), ReLU.
- **Output:** `(B, 128, H, W)`.

### 2.2 Flatten and MLP

- Flatten conv output: `(B, 128, H, W)` → `(B, 128×H×W)`.
- Fully connected stack:
  - Linear(128×H×W → 512), ReLU
  - Linear(512 → 256), ReLU
  - Linear(256 → 256), ReLU
  - Linear(256 → 256), ReLU
  - Linear(256 → 256) — **no activation after this layer** (backbone output is 256-dimensional).

The **backbone output** is a **256-dimensional feature vector** per batch item. All action and value outputs are computed from this vector.

---

## 3. DQN: Dueling Q-Network

**File:** `agents/DQN_agent.py` — class `QNetwork`.

DQN uses a **Dueling** architecture: the 256-d backbone output is fed into two separate heads instead of a single Q head.

### 3.1 Heads

| Head              | Layer                    | Output shape   |
|-------------------|--------------------------|-----------------|
| Value stream      | Linear(256 → 1)          | `(B, 1)`        |
| Advantage stream  | Linear(256 → max_actions) | `(B, max_actions)` |

### 3.2 Q-value combination

- `V(s) = value_stream(backbone(s))`
- `A(s, a) = advantage_stream(backbone(s))`
- Q-values are computed as:
  ```text
  Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
  ```
  So the network outputs one scalar value and one advantage vector per state; Q-values are derived from them. This helps when the value of a state is similar across many actions.

### 3.3 Action selection

- The network outputs Q-values for a **fixed maximum number of actions** `max_actions` (e.g. `25×20×4` for a 25×20 grid).
- The **controller** supplies the list of **valid action indices** for the current state. The agent masks invalid actions (e.g. by setting their Q-values to a large negative value) and selects among valid indices only (e.g. ε-greedy during training, greedy at evaluation).

### 3.4 Training setup (reference)

- **Online network** and **target network**: two copies of `QNetwork`; target is updated periodically from the online network.
- Optimizer: Adam (default learning rate 1e-4).
- Training uses replay buffer, TD target, and standard DQN loss (MSE or similar on TD error).

---

## 4. PPO: Actor–Critic

**File:** `agents/PPO_agent.py` — class `ActorCritic`.

PPO uses a single **shared backbone** (same conv + MLP as above) with two heads: **actor** (policy) and **critic** (value).

### 4.1 Heads

| Head   | Layer                    | Output shape   | Role |
|--------|--------------------------|----------------|------|
| Actor  | Linear(256 → max_actions)| `(B, max_actions)` | Policy logits over actions |
| Critic | Linear(256 → 1)          | `(B, 1)`       | State value V(s) |

### 4.2 Forward pass

- **Input:** State `x` (batch), in `(B, H, W, C)` or `(B, C, H, W)`; converted to `(B, C, H, W)` internally.
- **Backbone:** Same conv + MLP → 256-d vector.
- **Outputs:**
  - `logits = actor(backbone(x))` — one logit per action index.
  - `value = critic(backbone(x)).squeeze(-1)` — scalar value per batch item.

So one forward call returns `(logits, value)`.

### 4.3 Policy and action selection

- Policy: categorical distribution over actions from `logits`. Invalid actions are handled by the controller: only **valid action indices** are considered (e.g. logits for invalid actions are masked or sampling is restricted to valid indices).
- Action is sampled from this distribution (training) or taken as argmax (evaluation). The agent stores `log_prob` and `value` for each step for the PPO update.

### 4.4 Training (PPO loss)

- **Rollout:** Collect trajectories (state, action, reward, next_state, done, log_prob, value).
- **GAE:** Compute advantages and returns using Generalized Advantage Estimation (λ and γ).
- **PPO update:** Multiple epochs over mini-batches of the rollout. For each batch:
  - **Policy loss:** Clipped surrogate (ratio = exp(new_log_prob - old_log_prob), clip with ε, e.g. 0.2).
  - **Value loss:** MSE between predicted value and return.
  - **Entropy bonus:** Mean entropy of the policy (negative coefficient to encourage exploration).
- Combined loss: `policy_loss + value_coef * value_loss - entropy_coef * entropy`.
- Gradient clipping (e.g. max norm 0.5) is applied to the shared `ActorCritic` parameters.

---

## 5. Summary Table

| Component        | DQN (Dueling)                    | PPO (Actor–Critic)        |
|-----------------|-----------------------------------|----------------------------|
| Backbone        | Shared (conv + MLP → 256)         | Same                       |
| Value           | Value stream: Linear(256→1)       | Critic: Linear(256→1)      |
| Actions         | Advantage stream: Linear(256→max_actions) | Actor: Linear(256→max_actions) |
| Output          | Q(s,a) = V + (A - mean(A))        | (logits, value)             |
| Action space    | Valid indices from controller     | Valid indices from controller |
| Training        | Online + target net, replay, TD   | GAE + clipped surrogate + value + entropy |

---

## 6. Design rationale

- **Shared backbone:** Keeps state representation identical for DQN and PPO, so comparisons and transfer (e.g. reusing a trained backbone for a different head) are straightforward.
- **Spatial structure:** 3×3 convs with padding preserve grid spatial structure; the MLP then aggregates into a 256-d vector used for both value and action heads.
- **Dueling (DQN):** Separating V(s) and A(s,a) helps when many actions have similar Q-values; the network can focus on the value of the state and relative advantages of actions.
- **Single ActorCritic (PPO):** One backbone for both policy and value reduces parameters and encourages a shared representation that is useful for both acting and valuing.

Default constants (e.g. `DEFAULT_STATE_HWC = (20, 25, 6)`, `DEFAULT_MAX_ACTIONS = 25*20*4`) are defined in `agents/DQN_agent.py` and `agents/PPO_agent.py`; training scripts may override them based on the environment grid size.
