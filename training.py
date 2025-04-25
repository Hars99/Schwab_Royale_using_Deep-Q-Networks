# training.py
import torch
from SchwabRoyaleEnv import SchwabRoyaleEnv
from RLAgent import RLAgent
import numpy as np
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_teams = 2
episodes = 200
target_update_freq = 10
max_turns = 100

# Initialize environment and agents
env = SchwabRoyaleEnv(grid_size=(10, 10), num_teams=num_teams, num_obstacles=10, max_turns=max_turns)
agents_per_team = {}

for team_id in range(num_teams):
    agents_per_team[team_id] = RLAgent(env, team_id=team_id).to(device)

for episode in range(episodes):
    obs = env.reset()
    total_team_rewards = {team_id: 0 for team_id in agents_per_team}
    done = False

    while not done:
        actions = {}
        # Step: Each agent acts
        for team_id, agent in agents_per_team.items():
            actions.update({
                agent_name: agent.act(obs)
                for agent_name in env.teams[team_id]
            })

        next_obs, rewards, done, _ = env.step(actions)

        # Store experience and learn
        for team_id, agent in agents_per_team.items():
            team_reward = 0
            for agent_name in env.teams[team_id]:
                transition = (obs, actions[agent_name], rewards[agent_name], next_obs, float(done))
                error = agent.compute_td_error(*transition)
                agent.remember(transition, error)
                team_reward += rewards[agent_name]
            agent.track_rewards(team_reward)
            total_team_rewards[team_id] += team_reward
            agent.learn()

        obs = next_obs

    # Update target network periodically
    if episode % target_update_freq == 0:
        for agent in agents_per_team.values():
            agent.update_target()

    # 🏆 Determine Winner
    winner_team = max(total_team_rewards.items(), key=lambda x: x[1])[0]
    print(f"Episode {episode + 1}/{episodes} | Rewards: {total_team_rewards} | 🏆 Winner: Team {winner_team}")

# Plot reward curves
for team_id, agent in agents_per_team.items():
    print(f"\nTeam {team_id} Reward Plot:")
    agent.plot_rewards()
# Save models and rewards
for team_id, agent in agents_per_team.items():
    agent.save_model(f"team{team_id}_model.pth")
    with open(f"team{team_id}_rewards.pkl", "wb") as f:
        pickle.dump(agent.rewards_per_episode, f)
