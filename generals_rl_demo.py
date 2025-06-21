"""
Generals.io Reinforcement Learning Agent Demo

This module provides a demonstration of a trained Deep Q-Network (DQN) agent
playing the game Generals.io. It includes both interactive gameplay with
visualization and batch analysis capabilities.

Most of this code is AI generated.

"""

import torch
import torch.nn as nn
import numpy as np
import pygame
import time
import os
import argparse
from typing import Optional, Tuple
from generals import GameRenderer
from generals_rl_env_gpu import GeneralsEnv


class ConvDQN(nn.Module):
    """
    Convolutional Deep Q-Network (DQN) for Generals.io.
    
    This network uses a dueling DQN architecture with convolutional layers
    to process the game state and output Q-values for action selection.
    
    Attributes:
        conv: Convolutional layers for feature extraction
        fc: Fully connected layers for processing
        value_stream: Value function stream (V(s))
        advantage_stream: Advantage function stream (A(s,a))
        use_reduced_actions: Whether to use reduced action space
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int, 
                 use_reduced_actions: bool = True) -> None:
        """
        Initialize the ConvDQN network.
        
        Args:
            input_shape: Shape of input observation (height, width, channels)
            n_actions: Number of possible actions
            use_reduced_actions: Whether to use reduced action space
        """
        super(ConvDQN, self).__init__()
        
        h, w, c = input_shape
        self.use_reduced_actions = use_reduced_actions
        
        # Convolutional layers with batch normalization
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate convolutional output dimensions
        conv_h = h // 2
        conv_w = w // 2
        conv_out_size = 128 * conv_h * conv_w
        
        # Fully connected layers with dropout
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Dueling DQN architecture components
        self.value_stream = nn.Linear(256, 1)
        
        if use_reduced_actions:
            self.advantage_stream = nn.Linear(256, h * w * 4)
        else:
            self.advantage_stream = nn.Linear(256, n_actions)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize network weights using Xavier uniform initialization.
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, 
                valid_actions_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, height, width, channels)
            valid_actions_mask: Boolean mask for valid actions
            
        Returns:
            Q-values for all actions
        """
        # Convert from (B, H, W, C) to (B, C, H, W) for convolutions
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        
        # Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        # Apply valid actions mask if provided
        if valid_actions_mask is not None:
            q_values = q_values.masked_fill(~valid_actions_mask, float('-inf'))
        
        return q_values


class GeneralsRLAgent:
    """
    Reinforcement Learning agent for Generals.io using a trained DQN model.
    
    This agent loads a pre-trained model and uses it to make decisions
    during gameplay. It supports epsilon-greedy exploration for training
    and deterministic action selection for evaluation.
    
    Attributes:
        device: Device to run the model on (CPU/GPU)
        model: The trained ConvDQN model
    """
    
    def __init__(self, model_path: str, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        """
        Initialize the RL agent with a trained model.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run the model on
        """
        self.device = device
        
        # Load the trained model
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model architecture
        env = GeneralsEnv(training_mode=False)
        self.model = ConvDQN(env.observation_space.shape, env.action_space.n).to(device)
        
        # Load model weights (handle DataParallel wrapper if present)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if any(key.startswith('module.') for key in state_dict.keys()):
            # Remove 'module.' prefix from DataParallel models
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Display training information if available
        if isinstance(checkpoint, dict):
            print(f"Model trained for {checkpoint.get('episode', 'unknown')} episodes")
            print(f"Training epsilon: {checkpoint.get('epsilon', 'unknown')}")
        
    @torch.no_grad()
    def get_action(self, observation: np.ndarray, 
                   valid_actions_mask: Optional[np.ndarray] = None, 
                   epsilon: float = 0.0) -> int:
        """
        Select an action using the trained model.
        
        Args:
            observation: Current game state observation
            valid_actions_mask: Boolean mask of valid actions
            epsilon: Probability of random action selection
            
        Returns:
            Selected action index
        """
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Random valid action
            if valid_actions_mask is not None:
                valid_indices = np.where(valid_actions_mask)[0]
                if len(valid_indices) > 0:
                    return np.random.choice(valid_indices)
            return np.random.randint(0, len(valid_actions_mask) if valid_actions_mask is not None else 100)
        
        # Get Q-values from model
        q_values = self.model(obs_tensor).squeeze(0).cpu().numpy()
        
        # Apply valid actions mask
        if valid_actions_mask is not None:
            q_values[~valid_actions_mask] = -np.inf
        
        # Return action with highest Q-value
        return np.argmax(q_values)


class StatsRenderer:
    """Handles the visual rendering of game statistics and controls."""
    
    def __init__(self, screen: pygame.Surface):
        """
        Initialize the stats renderer.
        
        Args:
            screen: Pygame screen surface
        """
        self.screen = screen
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 48)
    
    def render_stats(self, step_count: int, reward: float, total_reward: float, 
                    territory: int, army: int) -> None:
        """
        Render game statistics on screen.
        
        Args:
            step_count: Current step number
            reward: Current step reward
            total_reward: Cumulative reward
            territory: Current territory count
            army: Current army count
        """
        stats_text = [
            f"Step: {step_count}",
            f"Reward: {reward:.2f}",
            f"Total Reward: {total_reward:.2f}",
            f"Territory: {territory}",
            f"Army: {army}",
            "",
            "Controls:",
            "SPACE - Toggle speed",
            "ESC - Quit",
            "TAB - Switch player view"
        ]
        
        y_offset = 10
        for text in stats_text:
            text_surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25
    
    def render_game_over(self, game_won: bool) -> None:
        """
        Render game over screen.
        
        Args:
            game_won: Whether the agent won the game
        """
        result_text = "VICTORY!" if game_won else "DEFEAT"
        result_color = (0, 255, 0) if game_won else (255, 0, 0)
        text_surface = self.large_font.render(result_text, True, result_color)
        text_rect = text_surface.get_rect(center=(500, 400))
        self.screen.blit(text_surface, text_rect)


def handle_pygame_events(env: GeneralsEnv, delay: float) -> Tuple[bool, float]:
    """
    Handle pygame events and return updated delay.
    
    Args:
        env: The game environment
        delay: Current delay between moves
        
    Returns:
        Tuple of (should_quit, updated_delay)
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True, delay
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Toggle between fast and normal speed
                delay = 0 if delay > 0 else 0.5
            elif event.key == pygame.K_ESCAPE:
                return True, delay
            elif event.key == pygame.K_TAB:
                # Switch player view for local multiplayer
                alive_players = [p.id for p in env.game.players if p.is_alive]
                if len(alive_players) > 1:
                    current_idx = alive_players.index(env.game.current_player)
                    env.game.current_player = alive_players[(current_idx + 1) % len(alive_players)]
                    print(f"Switched to Player {env.game.current_player + 1}'s view")
    return False, delay


def play_game(model_path: str, render: bool = True, num_opponents: int = 1, 
              games: int = 1, delay: float = 0.5) -> None:
    """
    Play Generals.io using the trained model.
    
    Args:
        model_path: Path to the trained model checkpoint
        render: Whether to render the game visually
        num_opponents: Number of AI opponents (1-3)
        games: Number of games to play
        delay: Delay between moves in seconds
    """
    # Initialize the agent
    agent = GeneralsRLAgent(model_path)
    
    # Game statistics
    wins = 0
    total_games = 0
    
    for game_num in range(games):
        print(f"\n{'='*50}")
        print(f"Starting Game {game_num + 1}/{games}")
        print(f"{'='*50}")
        
        # Create game environment
        env = GeneralsEnv(
            player_id=0,
            num_opponents=num_opponents,
            training_mode=False
        )
        
        # Initialize pygame for rendering if needed
        if render:
            pygame.init()
            screen = pygame.display.set_mode((1000, 800))
            pygame.display.set_caption(f"Generals.io RL Agent - Game {game_num + 1}")
            clock = pygame.time.Clock()
            game_renderer = GameRenderer(screen)
            stats_renderer = StatsRenderer(screen)
        
        # Reset environment
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        print(f"Playing against {num_opponents} opponent(s)")
        print("Agent is Player 1 (Red)")
        
        # Main game loop
        while not done:
            # Handle pygame events if rendering
            if render:
                should_quit, delay = handle_pygame_events(env, delay)
                if should_quit:
                    pygame.quit()
                    return
            
            # Get valid actions and select action
            valid_actions = env.get_valid_actions()
            action = agent.get_action(obs, valid_actions, epsilon=0.0)
            
            # Take action in environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Render if enabled
            if render:
                game_renderer.render(env.game)
                stats_renderer.render_stats(
                    step_count, reward, total_reward,
                    info.get('territory', 0), info.get('army', 0)
                )
                
                pygame.display.flip()
                
                # Control game speed
                if delay > 0:
                    time.sleep(delay)
                clock.tick(60)  # Cap at 60 FPS
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"Step {step_count}: Territory={info.get('territory', 0)}, "
                      f"Army={info.get('army', 0)}, Reward={total_reward:.2f}")
        
        # Game over processing
        total_games += 1
        game_won = env.game.winner == env.player_id
        if game_won:
            wins += 1
            print(f"\n🎉 VICTORY! Agent won the game!")
        else:
            print(f"\n💀 DEFEAT. Agent lost the game.")
        
        print(f"Final stats: Steps={step_count}, Total Reward={total_reward:.2f}")
        print(f"Win rate so far: {wins}/{total_games} ({100*wins/total_games:.1f}%)")
        
        if render:
            # Show game over screen
            stats_renderer.render_game_over(game_won)
            pygame.display.flip()
            time.sleep(3)  # Show result for 3 seconds
    
    # Final statistics
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS: {wins}/{games} wins ({100*wins/games:.1f}% win rate)")
    print(f"{'='*50}")
    
    if render:
        pygame.quit()


def analyze_gameplay(model_path: str, num_games: int = 10) -> None:
    """
    Analyze agent's gameplay performance without rendering.
    
    Args:
        model_path: Path to the trained model checkpoint
        num_games: Number of games to analyze
    """
    agent = GeneralsRLAgent(model_path)
    
    # Statistics tracking
    stats = {
        'wins': 0,
        'total_territory': [],
        'total_army': [],
        'game_lengths': [],
        'final_rewards': []
    }
    
    print(f"Analyzing {num_games} games...")
    
    for i in range(num_games):
        # Create environment for analysis
        env = GeneralsEnv(player_id=0, num_opponents=3, training_mode=True)
        obs = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        # Play game until completion or step limit
        while not done and steps < 2000:  # Cap at 2000 steps
            valid_actions = env.get_valid_actions()
            action = agent.get_action(obs, valid_actions, epsilon=0.0)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        # Record statistics
        if env.game.winner == env.player_id:
            stats['wins'] += 1
        stats['total_territory'].append(info.get('territory', 0))
        stats['total_army'].append(info.get('army', 0))
        stats['game_lengths'].append(steps)
        stats['final_rewards'].append(total_reward)
        
        print(f"Game {i+1}: {'Win' if env.game.winner == env.player_id else 'Loss'}, "
              f"Territory={info.get('territory', 0)}, Steps={steps}")
    
    # Print analysis summary
    print(f"\n{'='*50}")
    print(f"Analysis of {num_games} games:")
    print(f"Win rate: {100*stats['wins']/num_games:.1f}%")
    print(f"Average territory: {np.mean(stats['total_territory']):.1f}")
    print(f"Average army: {np.mean(stats['total_army']):.1f}")
    print(f"Average game length: {np.mean(stats['game_lengths']):.1f} steps")
    print(f"Average final reward: {np.mean(stats['final_rewards']):.2f}")
    print(f"{'='*50}")


def list_available_checkpoints() -> None:
    """List available model checkpoints in the checkpoints directory."""
    print("\nAvailable checkpoints:")
    if os.path.exists("checkpoints"):
        checkpoint_files = [f for f in os.listdir("checkpoints") if f.endswith(".pth")]
        if checkpoint_files:
            for file in checkpoint_files:
                print(f"  - checkpoints/{file}")
        else:
            print("  No checkpoint files found.")
    else:
        print("  Checkpoints directory not found.")


def main() -> None:
    """Main function to handle command line arguments and run the demo."""
    parser = argparse.ArgumentParser(
        description='Play Generals.io with trained RL agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s checkpoints/model.pth                    # Play one game with rendering
  %(prog)s checkpoints/model.pth --no-render        # Play without visualization
  %(prog)s checkpoints/model.pth --games 5          # Play 5 games
  %(prog)s checkpoints/model.pth --analyze --games 20  # Analyze 20 games
        """
    )
    
    parser.add_argument('model_path', type=str, 
                       help='Path to the trained model checkpoint')
    parser.add_argument('--no-render', action='store_true', 
                       help='Disable rendering (faster execution)')
    parser.add_argument('--games', type=int, default=1, 
                       help='Number of games to play (default: 1)')
    parser.add_argument('--opponents', type=int, default=1, 
                       help='Number of opponents (1-3, default: 1)')
    parser.add_argument('--delay', type=float, default=0.5, 
                       help='Delay between moves in seconds (default: 0.5)')
    parser.add_argument('--analyze', action='store_true', 
                       help='Run analysis mode (no rendering, batch processing)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.opponents < 1 or args.opponents > 3:
        print("Error: Number of opponents must be between 1 and 3")
        exit(1)
    
    if args.games < 1:
        print("Error: Number of games must be at least 1")
        exit(1)
    
    if args.delay < 0:
        print("Error: Delay must be non-negative")
        exit(1)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        list_available_checkpoints()
        exit(1)
    
    # Run appropriate mode
    if args.analyze:
        print("Running analysis mode...")
        analyze_gameplay(args.model_path, num_games=args.games)
    else:
        print("Running gameplay mode...")
        play_game(
            args.model_path,
            render=not args.no_render,
            num_opponents=args.opponents,
            games=args.games,
            delay=args.delay
        )


if __name__ == "__main__":
    main()