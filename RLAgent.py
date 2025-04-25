import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, transition, error):
        priority = (abs(error) + 1e-5) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, torch.FloatTensor(weights), indices

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha

class RLAgent(nn.Module):
    def __init__(self, env, team_id, hidden_dim=128, lr=1e-3):
        super(RLAgent, self).__init__()
        self.team_id = team_id
        self.observation_dim = np.prod(env.grid_size)
        self.action_dim = env.action_space.n
        self.q_network = QNetwork(self.observation_dim, self.action_dim, hidden_dim)
        self.target_network = QNetwork(self.observation_dim, self.action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(capacity=10000)
        self.gamma = 0.95
        self.batch_size = 32
        self.update_target()

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.beta = 0.4

        self.rewards_per_episode = []

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        obs_tensor = torch.FloatTensor(obs.flatten()).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        return int(torch.argmax(q_values).item())

    def remember(self, transition, error):
        self.memory.push(transition, error)

    def compute_td_error(self, obs, action, reward, next_obs, done):
        obs_tensor = torch.FloatTensor(obs.flatten()).unsqueeze(0)
        next_obs_tensor = torch.FloatTensor(next_obs.flatten()).unsqueeze(0)
        with torch.no_grad():
            q_val = self.q_network(obs_tensor)[0, action]
            next_action = self.q_network(next_obs_tensor).argmax().item()
            next_q_val = self.target_network(next_obs_tensor)[0, next_action]
            target = reward + self.gamma * next_q_val * (1 - done)
        return (q_val - target).item()

    def learn(self):
        if len(self.memory.buffer) < self.batch_size:
            return
        batch, weights, indices = self.memory.sample(self.batch_size, self.beta)
        obs, action, reward, next_obs, done = zip(*batch)
        obs = torch.FloatTensor(np.array([o.flatten() for o in obs]))
        next_obs = torch.FloatTensor(np.array([no.flatten() for no in next_obs]))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        q_vals = self.q_network(obs).gather(1, action.unsqueeze(1)).squeeze()
        next_actions = self.q_network(next_obs).argmax(1)
        next_q_vals = self.target_network(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze()
        target = reward + self.gamma * next_q_vals * (1 - done)

        errors = (q_vals - target.detach()).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, errors)

        weights = weights.to(q_vals.device)
        loss = (F.mse_loss(q_vals, target.detach(), reduction='none') * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def track_rewards(self, reward):
        self.rewards_per_episode.append(reward)

    def plot_rewards(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.rewards_per_episode, label="Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Reward Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())
