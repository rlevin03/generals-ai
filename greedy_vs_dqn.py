"""
Tournament script for comparing DQN agent against Greedy baseline agent.

This module implements a tournament system to evaluate the performance of a trained
DQN agent against a greedy baseline agent in the Generals game environment.
The tournament tracks detailed statistics including win rates, game duration,
and resource distribution.

Most of this code is AI generated.

"""

import argparse
import csv
import time
from datetime import datetime
from typing import Tuple

from generals_rl_env_gpu import GeneralsEnv
from greedy_baseline_agent import GreedyAgent, _setup_environment_patches
from generals_rl_demo import GeneralsRLAgent

_setup_environment_patches()


def count_player_stats(game, player_id: int) -> Tuple[int, int]:
    """
    Count the total armies and territory owned by a specific player.
    
    Args:
        game: The current game state object
        player_id: The ID of the player to count stats for
        
    Returns:
        Tuple containing (total_armies, total_territory) for the player
    """
    armies = sum(cell.army for row in game.grid for cell in row if cell.owner == player_id)
    territory = sum(1 for row in game.grid for cell in row if cell.owner == player_id)
    return armies, territory


def run_tournament(model_path: str, num_games: int = 100) -> None:
    """
    Run a tournament between DQN agent and Greedy agent.
    
    This function orchestrates a series of games between a trained DQN agent
    and a greedy baseline agent. It tracks comprehensive statistics including
    win rates, game duration, and resource distribution. Results are saved
    to a timestamped CSV file for detailed analysis.
    
    Args:
        model_path: Path to the trained DQN model file
        num_games: Number of games to play in the tournament (default: 100)
        
    Returns:
        None. Results are printed to console and saved to CSV file.
    """
    # Initialize environment and agents
    env = GeneralsEnv(player_id=0, num_opponents=1, training_mode=True)
    dqn_agent = GeneralsRLAgent(model_path)
    greedy_agent = GreedyAgent()
    
    # Tournament statistics
    dqn_wins = 0
    greedy_wins = 0
    total_steps = 0
    
    # Create CSV file for detailed logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"tournament_results_{timestamp}.csv"
    
    # Define CSV field names for consistent data structure
    fieldnames = [
        'game', 'winner', 'steps', 'time', 
        'dqn_armies', 'dqn_territory',
        'greedy_armies', 'greedy_territory'
    ]
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        print(f"\nStarting tournament: DQN vs Greedy ({num_games} games)")
        print("=" * 50)
        
        # Main tournament loop
        for game in range(num_games):
            obs = env.reset()
            done = False
            step_count = 0
            game_start_time = time.time()
            
            # Single game loop
            while not done:
                # DQN agent's turn (no exploration, pure exploitation)
                valid_actions = env.get_valid_actions()
                dqn_action = dqn_agent.get_action(obs, valid_actions, epsilon=0.0)
                
                if dqn_action is not None:
                    obs, reward, done, info = env.step(dqn_action)
                else:
                    # Handle case where no valid action is available
                    env.game.update()
                    obs, reward, done, info = env.step(0)
                
                if done:
                    break
                
                # Greedy agent's turn
                greedy_action = greedy_agent.select(env)
                if greedy_action is not None:
                    obs, reward, done, info = env.step_with_reduced_action(greedy_action)
                else:
                    # Handle case where no valid action is available
                    obs, reward, done, info = env.step(0)
                
                step_count += 1
            
            # Calculate game statistics
            game_time = time.time() - game_start_time
            winner = "DQN" if env.game.winner == env.player_id else "Greedy"
            
            # Get final resource statistics for both players
            dqn_armies, dqn_territory = count_player_stats(env.game, env.player_id)
            greedy_armies, greedy_territory = count_player_stats(env.game, 1)
            
            # Update tournament statistics
            if winner == "DQN":
                dqn_wins += 1
            else:
                greedy_wins += 1
            
            total_steps += step_count
            
            # Write detailed game data to CSV
            writer.writerow({
                'game': game + 1,
                'winner': winner,
                'steps': step_count,
                'time': f"{game_time:.1f}",
                'dqn_armies': dqn_armies,
                'dqn_territory': dqn_territory,
                'greedy_armies': greedy_armies,
                'greedy_territory': greedy_territory
            })
            
            # Print game result with detailed statistics
            print(f"\nGame {game + 1}/{num_games}")
            print(f"Winner: {winner}")
            print(f"Steps: {step_count} - Time: {game_time:.1f}s")
            print(f"DQN - Armies: {dqn_armies}, Territory: {dqn_territory}")
            print(f"Greedy - Armies: {greedy_armies}, Territory: {greedy_territory}")
            print("-" * 50)
    
    # Print comprehensive tournament results
    print("\nTournament Results:")
    print("=" * 50)
    print(f"DQN Wins: {dqn_wins} ({dqn_wins/num_games*100:.1f}%)")
    print(f"Greedy Wins: {greedy_wins} ({greedy_wins/num_games*100:.1f}%)")
    print(f"Average Steps per Game: {total_steps/num_games:.1f}")
    print(f"\nDetailed results have been saved to: {csv_filename}")


def main():
    """
    Main entry point for the tournament script.
    
    Parses command line arguments and initiates the tournament between
    DQN and Greedy agents.
    """
    parser = argparse.ArgumentParser(
        description="Run a tournament between DQN and Greedy agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Path to trained DQN model"
    )
    parser.add_argument(
        "--games", 
        type=int, 
        default=100, 
        help="Number of games to play in the tournament"
    )
    
    args = parser.parse_args()
    run_tournament(args.model, args.games)


if __name__ == "__main__":
    main()
