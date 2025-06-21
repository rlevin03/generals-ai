# Generals.io AI - Reinforcement Learning Project

A Python implementation of the popular Generals.io real-time strategy game with multiple AI agents trained using reinforcement learning techniques.

## 🎮 About Generals.io

Generals.io is a real-time strategy game where players compete to capture territory and eliminate opponents. Each player starts with a general (crown symbol) and must expand their territory by moving armies between adjacent cells. The goal is to capture the enemy general while protecting your own.

### Game Features

- **Real-time gameplay** with 0.5-second turns
- **Fog of war** - players can only see their territory and adjacent cells
- **Territory control** - captured cells generate armies over time
- **Cities** - special cells that provide additional army generation
- **Mountains** - impassable terrain that blocks movement
- **Multiple players** - support for 1-4 players in a single game

### Game Mechanics

- **Army Generation**: Generals and cities generate 1 army per turn (0.5 seconds)
- **Territory Bonus**: All owned cells generate 1 army every 25 turns (12.5 seconds)
- **Movement**: Move armies from owned cells to adjacent cells
- **Combat**: When attacking, armies fight and the stronger force wins
- **Victory**: Capture the enemy general to win

## 📸 Screenshots & Visualizations

_[Add screenshots of the game interface, training curves, and AI gameplay here]_

### Suggested Images to Add:

- Game board with multiple players and fog of war
- Training progress curves showing win rates and rewards
- AI agent gameplay demonstrations
- Tournament results and statistics

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch numpy pygame matplotlib pandas tensorboard gym
```

### Running the Base Game

```bash
python generals.py
```

**Controls:**

- **Mouse**: Click to select territory, right-click to move
- **WASD/Arrow Keys**: Chain moves from selected cell
- **Shift**: Move half army instead of full army
- **Tab**: Switch between player views
- **ESC**: Clear selection
- **R**: Restart game (when game over)

## 🤖 AI Agents & Training

This project includes several AI agents and training scripts:

### 1. DQN Agent Training

```bash
# Train a DQN agent from scratch
python gpu_dqn_training.py

# Train with specific parameters
python gpu_dqn_training.py --episodes 5000 --batch-size 256 --lr 0.0005
```

### 2. REINFORCE Agent Training

```bash
# Train REINFORCE agent against a frozen DQN opponent
python train_reinforce_vs_greedy.py --model_path checkpoints/dqn_model.pth

# Custom training parameters
python train_reinforce_vs_greedy.py \
    --model_path checkpoints/dqn_model.pth \
    --episodes 3000 \
    --lr 0.001 \
    --gamma 0.99
```

### 3. Tournament Evaluation

```bash
# Run tournament between DQN and Greedy agents
python greedy_vs_dqn.py --model checkpoints/dqn_model.pth --games 100

# Run with custom number of games
python greedy_vs_dqn.py --model checkpoints/dqn_model.pth --games 500
```

### 4. AI Agent Demo

```bash
# Play against AI with visualization
python generals_rl_demo.py checkpoints/model.pth

# Play without rendering (faster)
python generals_rl_demo.py checkpoints/model.pth --no-render

# Play multiple games
python generals_rl_demo.py checkpoints/model.pth --games 5

# Analyze agent performance
python generals_rl_demo.py checkpoints/model.pth --analyze --games 20
```

### 5. Greedy Baseline Agent

```bash
# Evaluate greedy baseline agent
python greedy_baseline_agent.py --episodes 200 --grid 25
```

## ⚙️ Configuration & Constants

### Game Constants (`generals.py`)

```python
# Display settings
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
GRID_WIDTH = 25
GRID_HEIGHT = 20
CELL_SIZE = 30

# Game timing
TURN_DURATION = 0.5  # Each turn is 0.5 seconds
ARMY_GENERATION_INTERVAL = 25 * TURN_DURATION  # Every 12.5 seconds

# Map generation
num_mountains = random.randint(15, 25)  # Number of mountains
num_cities = random.randint(8, 12)      # Number of cities
min_distance = 15  # Minimum distance between generals
```

### Training Constants (`gpu_dqn_training.py`)

```python
# Training hyperparameters
lr = 5e-4                    # Learning rate
batch_size = 256            # Batch size for training
gamma = 0.99                # Discount factor
epsilon_start = 1.0         # Initial exploration rate
epsilon_decay = 0.995       # Exploration decay rate
epsilon_min = 0.05          # Minimum exploration rate

# Environment settings
num_parallel_envs = 4       # Number of parallel environments
buffer_capacity = 200000    # Replay buffer size
update_target_every = 1000  # Target network update frequency
```

### REINFORCE Constants (`train_reinforce_vs_greedy.py`)

```python
# Reward shaping constants
REWARD_CONSTANTS = {
    'win_bonus': 1000.0,
    'loss_penalty': -200.0,
    'city_capture_bonus': 50.0,
    'enemy_territory_bonus': 10.0,
    'territory_gain_multiplier': 5.0,
    'step_penalty': -0.002,
    'invalid_move_penalty': -0.1,
    'movement_bonus': 0.005,
    'frontier_bonus': 0.01,
    'frontier_growth_bonus': 0.02,
    'exploration_bonus': 0.005,
    'enemy_city_spotting_bonus': 1.0
}

# Training parameters
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPISODES = 3000
DEFAULT_GAMMA = 0.99
DEFAULT_ENTROPY_BETA = 0.01
DEFAULT_GRADIENT_CLIP = 0.5
```

## 📊 Training & Evaluation

### Training Progress

Training progress is automatically logged and visualized:

- **TensorBoard logs**: `runs/` directory
- **Training statistics**: `training_stats/` directory
- **Model checkpoints**: `checkpoints/` directory
- **Tournament results**: CSV files with timestamps

### Monitoring Training

```bash
# View TensorBoard logs
tensorboard --logdir runs/

# Check training statistics
ls training_stats/
cat training_stats/training_summary.json
```

### Performance Metrics

- **Win Rate**: Percentage of games won
- **Average Reward**: Mean episode reward
- **Territory Control**: Average final territory count
- **Episode Length**: Average game duration
- **Invalid Move Rate**: Percentage of invalid actions

## 🏗️ Project Structure

```
generals-ai/
├── generals.py                    # Main game implementation
├── generals_rl_env_gpu.py         # RL environment wrapper
├── gpu_dqn_training.py           # DQN training script
├── train_reinforce_vs_greedy.py  # REINFORCE training script
├── greedy_vs_dqn.py              # Tournament evaluation
├── generals_rl_demo.py           # AI agent demo
├── greedy_baseline_agent.py      # Greedy baseline agent
├── policy_network.py             # Policy network for REINFORCE
├── DQN_agent.py                  # DQN agent implementation
├── checkpoints/                  # Saved model checkpoints
├── runs/                         # TensorBoard logs
├── training_stats/               # Training statistics and plots
│   ├── episode_stats.csv
│   ├── training_curves.png  # Distribution of action types
│   └── training_stats.csv        # Reward progression during training
└── README.md                     # This file
```

### Network Architecture

The agents use convolutional neural networks with:

- **Input**: 6-channel observation (owner, army, city, general, mountain, visibility)
- **Convolutional layers**: 3 layers with batch normalization
- **Dueling DQN**: Separate value and advantage streams
- **Output**: Q-values for all possible actions

### Action Space

- **Full action space**: 2000 actions (25×20×4 directions)
- **Reduced action space**: Only valid moves (dynamically sized)
- **Movement**: 4 directions (up, down, left, right)

## 🎯 Tips for Training

1. **Start Simple**: Begin with 1 opponent before scaling to multiple opponents
2. **Monitor Progress**: Use TensorBoard to track training metrics
3. **Save Checkpoints**: Regular checkpointing prevents loss of progress
4. **Experiment**: Try different reward functions and hyperparameters
5. **Evaluate**: Run tournaments to compare agent performance

## 🐛 Troubleshooting

### Common Issues

- **CUDA out of memory**: Reduce batch size or number of parallel environments
- **Slow training**: Increase number of parallel environments or use GPU
- **Poor performance**: Adjust reward constants or training hyperparameters
- **Invalid moves**: Check action space configuration and environment setup

## 📈 Results & Benchmarks

_[Add your license information here]_

## 🙏 Acknowledgments

- Original Generals.io game concept
- PyTorch and Gym frameworks
- Reinforcement learning community

---

**Note**: This project is primarily AI-generated and serves as a demonstration of reinforcement learning techniques applied to strategy games.
