import torch
import torch.nn as nn
import numpy as np
import pygame
import time
import os
from generals import Game, GRID_WIDTH, GRID_HEIGHT, CellType, GameRenderer
from generals_rl_env_gpu import GeneralsEnv
import argparse

# Import your trained model architecture
class ConvDQN(nn.Module):
    """Your trained DQN architecture"""
    def __init__(self, input_shape, n_actions, use_reduced_actions=True):
        super(ConvDQN, self).__init__()
        
        h, w, c = input_shape
        self.use_reduced_actions = use_reduced_actions
        
        # Convolutional layers with batch norm
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
        
        # Calculate conv output size
        conv_h = h // 2
        conv_w = w // 2
        conv_out_size = 128 * conv_h * conv_w
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Dueling DQN architecture
        self.value_stream = nn.Linear(256, 1)
        
        if use_reduced_actions:
            self.advantage_stream = nn.Linear(256, h * w * 4)
        else:
            self.advantage_stream = nn.Linear(256, n_actions)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x, valid_actions_mask=None):
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        
        # Dueling DQN
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        # Apply valid actions mask if provided
        if valid_actions_mask is not None:
            q_values = q_values.masked_fill(~valid_actions_mask, float('-inf'))
        
        return q_values

class GeneralsRLAgent:
    """Agent that uses the trained model to play Generals.io"""
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load the model
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize the model
        env = GeneralsEnv(training_mode=False)
        self.model = ConvDQN(env.observation_space.shape, env.action_space.n).to(device)
        
        # Load state dict (handle DataParallel wrapper if present)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if any(key.startswith('module.') for key in state_dict.keys()):
            # Remove 'module.' prefix
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Print training info if available
        if isinstance(checkpoint, dict):
            print(f"Model trained for {checkpoint.get('episode', 'unknown')} episodes")
            print(f"Training epsilon: {checkpoint.get('epsilon', 'unknown')}")
        
    @torch.no_grad()
    def get_action(self, observation, valid_actions_mask=None, epsilon=0.0):
        """Get action from the model"""
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Random valid action
            if valid_actions_mask is not None:
                valid_indices = np.where(valid_actions_mask)[0]
                if len(valid_indices) > 0:
                    return np.random.choice(valid_indices)
            return np.random.randint(0, len(valid_actions_mask))
        
        # Get Q-values from model
        q_values = self.model(obs_tensor).squeeze(0).cpu().numpy()
        
        # Apply valid actions mask
        if valid_actions_mask is not None:
            q_values[~valid_actions_mask] = -np.inf
        
        # Return best action
        return np.argmax(q_values)

def play_game(model_path, render=True, num_opponents=1, games=1, delay=0.5):
    """Play Generals.io using the trained model"""
    
    # Initialize the agent
    agent = GeneralsRLAgent(model_path)
    
    # Game statistics
    wins = 0
    total_games = 0
    
    for game_num in range(games):
        print(f"\n{'='*50}")
        print(f"Starting Game {game_num + 1}/{games}")
        print(f"{'='*50}")
        
        # Create environment
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
            renderer = GameRenderer(screen)
        
        # Reset environment
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        print(f"Playing against {num_opponents} opponent(s)")
        print("Agent is Player 1 (Red)")
        
        while not done:
            # Handle pygame events if rendering
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            delay = 0 if delay > 0 else 0.5  # Toggle speed
                        elif event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return
            
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Get action from agent (no exploration during demo)
            action = agent.get_action(obs, valid_actions, epsilon=0.0)
            
            # Take action
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Render if enabled
            if render:
                renderer.render(env.game)
                
                # Display stats
                font = pygame.font.Font(None, 24)
                stats_text = [
                    f"Step: {step_count}",
                    f"Reward: {reward:.2f}",
                    f"Total Reward: {total_reward:.2f}",
                    f"Territory: {info.get('territory', 0)}",
                    f"Army: {info.get('army', 0)}",
                    "",
                    "Controls:",
                    "SPACE - Toggle speed",
                    "ESC - Quit"
                ]
                
                y_offset = 10
                for text in stats_text:
                    text_surface = font.render(text, True, (255, 255, 255))
                    screen.blit(text_surface, (10, y_offset))
                    y_offset += 25
                
                pygame.display.flip()
                
                # Control game speed
                if delay > 0:
                    time.sleep(delay)
                clock.tick(60)  # Cap at 60 FPS
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"Step {step_count}: Territory={info.get('territory', 0)}, "
                      f"Army={info.get('army', 0)}, Reward={total_reward:.2f}")
        
        # Game over
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
            font = pygame.font.Font(None, 48)
            result_text = "VICTORY!" if game_won else "DEFEAT"
            result_color = (0, 255, 0) if game_won else (255, 0, 0)
            text_surface = font.render(result_text, True, result_color)
            text_rect = text_surface.get_rect(center=(500, 400))
            screen.blit(text_surface, text_rect)
            
            pygame.display.flip()
            time.sleep(3)  # Show result for 3 seconds
    
    # Final statistics
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS: {wins}/{games} wins ({100*wins/games:.1f}% win rate)")
    print(f"{'='*50}")
    
    if render:
        pygame.quit()

def analyze_gameplay(model_path, num_games=10):
    """Analyze agent's gameplay without rendering"""
    agent = GeneralsRLAgent(model_path)
    
    stats = {
        'wins': 0,
        'total_territory': [],
        'total_army': [],
        'game_lengths': [],
        'final_rewards': []
    }
    
    for i in range(num_games):
        env = GeneralsEnv(player_id=0, num_opponents=3, training_mode=True)
        obs = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        while not done and steps < 2000:  # Cap at 2000 steps
            valid_actions = env.get_valid_actions()
            action = agent.get_action(obs, valid_actions, epsilon=0.0)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        # Record stats
        if env.game.winner == env.player_id:
            stats['wins'] += 1
        stats['total_territory'].append(info.get('territory', 0))
        stats['total_army'].append(info.get('army', 0))
        stats['game_lengths'].append(steps)
        stats['final_rewards'].append(total_reward)
        
        print(f"Game {i+1}: {'Win' if env.game.winner == env.player_id else 'Loss'}, "
              f"Territory={info.get('territory', 0)}, Steps={steps}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Analysis of {num_games} games:")
    print(f"Win rate: {100*stats['wins']/num_games:.1f}%")
    print(f"Avg territory: {np.mean(stats['total_territory']):.1f}")
    print(f"Avg army: {np.mean(stats['total_army']):.1f}")
    print(f"Avg game length: {np.mean(stats['game_lengths']):.1f} steps")
    print(f"Avg final reward: {np.mean(stats['final_rewards']):.2f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play Generals.io with trained RL agent')
    parser.add_argument('model_path', type=str, help='Path to the trained model checkpoint')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--games', type=int, default=1, help='Number of games to play')
    parser.add_argument('--opponents', type=int, default=1, help='Number of opponents (1-3)')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between moves (seconds)')
    parser.add_argument('--analyze', action='store_true', help='Run analysis mode (no rendering)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        print("\nAvailable checkpoints:")
        if os.path.exists("checkpoints"):
            for file in os.listdir("checkpoints"):
                if file.endswith(".pth"):
                    print(f"  - checkpoints/{file}")
        exit(1)
    
    if args.analyze:
        # Run analysis mode
        analyze_gameplay(args.model_path, num_games=args.games)
    else:
        # Play games
        play_game(
            args.model_path,
            render=not args.no_render,
            num_opponents=args.opponents,
            games=args.games,
            delay=args.delay
        )