import torch
import numpy as np
from generals_rl_env_gpu import GeneralsEnv
from greedy_baseline_agent import GreedyAgent
from generals_rl_demo import GeneralsRLAgent
import time
import csv
from datetime import datetime

def count_player_stats(game, player_id):
    """Count armies and territory for a player."""
    armies = sum(cell.army for row in game.grid for cell in row if cell.owner == player_id)
    territory = sum(1 for row in game.grid for cell in row if cell.owner == player_id)
    return armies, territory

def run_tournament(model_path, num_games=100):
    """
    Run a tournament between DQN agent and Greedy agent.
    """
    # Initialize environment
    env = GeneralsEnv(player_id=0, num_opponents=1, training_mode=True)
    
    # Initialize agents
    dqn_agent = GeneralsRLAgent(model_path)
    greedy_agent = GreedyAgent()
    
    # Statistics tracking
    dqn_wins = 0
    greedy_wins = 0
    total_steps = 0
    
    # Create CSV file for detailed logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"tournament_results_{timestamp}.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['game', 'winner', 'steps', 'time', 
                     'dqn_armies', 'dqn_territory',
                     'greedy_armies', 'greedy_territory']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        print(f"\nStarting tournament: DQN vs Greedy ({num_games} games)")
        print("=" * 50)
        
        for game in range(num_games):
            obs = env.reset()
            done = False
            step_count = 0
            game_start_time = time.time()
            
            while not done:
                # DQN agent's turn
                valid_actions = env.get_valid_actions()
                dqn_action = dqn_agent.get_action(obs, valid_actions, epsilon=0.0)
                
                if dqn_action is not None:
                    obs, reward, done, info = env.step(dqn_action)
                else:
                    env.game.update()
                    obs, reward, done, info = env.step(0)
                
                if done:
                    break
                
                # Greedy agent's turn
                greedy_action = greedy_agent.select(env)
                if greedy_action is not None:
                    obs, reward, done, info = env.step_with_reduced_action(greedy_action)
                else:
                    obs, reward, done, info = env.step(0)
                
                step_count += 1
            
            # Record game statistics
            game_time = time.time() - game_start_time
            winner = "DQN" if env.game.winner == env.player_id else "Greedy"
            
            # Get final stats for both players
            dqn_armies, dqn_territory = count_player_stats(env.game, env.player_id)
            greedy_armies, greedy_territory = count_player_stats(env.game, 1)
            
            if winner == "DQN":
                dqn_wins += 1
            else:
                greedy_wins += 1
            
            total_steps += step_count
            
            # Write game data to CSV
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
            
            # Print game result with detailed stats
            print(f"\nGame {game + 1}/{num_games}")
            print(f"Winner: {winner}")
            print(f"Steps: {step_count} - Time: {game_time:.1f}s")
            print(f"DQN - Armies: {dqn_armies}, Territory: {dqn_territory}")
            print(f"Greedy - Armies: {greedy_armies}, Territory: {greedy_territory}")
            print("-" * 50)
    
    # Print tournament results
    print("\nTournament Results:")
    print("=" * 50)
    print(f"DQN Wins: {dqn_wins} ({dqn_wins/num_games*100:.1f}%)")
    print(f"Greedy Wins: {greedy_wins} ({greedy_wins/num_games*100:.1f}%)")
    print(f"Average Steps per Game: {total_steps/num_games:.1f}")
    print(f"\nDetailed results have been saved to: {csv_filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a tournament between DQN and Greedy agents")
    parser.add_argument("--model", type=str, required=True, help="Path to trained DQN model")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    args = parser.parse_args()
    
    run_tournament(args.model, args.games)
