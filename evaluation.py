import pickle
import matplotlib.pyplot as plt
from RLAgent import RLAgent
from SchwabRoyaleEnv import SchwabRoyaleEnv
import torch
import os

# Setup
num_teams = 2
agents_per_team = {}
env = SchwabRoyaleEnv(grid_size=(10, 10), num_teams=num_teams, num_obstacles=10, max_turns=100)

# Load agents and their reward history
for team_id in range(num_teams):
    agent = RLAgent(env, team_id=team_id)

    # Load model (optional if you're comparing performance)
    model_path = f"team{team_id}_model.pth"
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"Loaded model for Team {team_id}")

    # Load reward history
    reward_path = f"team{team_id}_rewards.pkl"
    if os.path.exists(reward_path):
        with open(reward_path, "rb") as f:
            agent.rewards_per_episode = pickle.load(f)
        print(f"Loaded rewards for Team {team_id} ({len(agent.rewards_per_episode)} episodes)")
    else:
        print(f"Warning: No reward file found for Team {team_id}")
        agent.rewards_per_episode = []

    agents_per_team[team_id] = agent

# Compute and display results
avg_rewards = {}
for team_id, agent in agents_per_team.items():
    if agent.rewards_per_episode:
        avg = sum(agent.rewards_per_episode) / len(agent.rewards_per_episode)
    else:
        avg = 0
    avg_rewards[team_id] = avg

# Declare winner
winning_team = max(avg_rewards, key=avg_rewards.get)
print("\n🏆 Evaluation Results 🏆")
for team_id, avg in avg_rewards.items():
    print(f"Team {team_id} - Avg Reward: {avg:.2f}")
print(f"\n🏅 Winning Team: Team {winning_team}")

# Plot reward trends
plt.figure(figsize=(10, 6))
for team_id, agent in agents_per_team.items():
    if agent.rewards_per_episode:
        plt.plot(agent.rewards_per_episode, label=f"Team {team_id}")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
