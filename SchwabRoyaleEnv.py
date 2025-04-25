import gym
import numpy as np
from gym import spaces

class SchwabRoyaleEnv(gym.Env):
    def __init__(self, grid_size=(10, 10), num_teams=2, num_obstacles=10, max_turns=100):
        super(SchwabRoyaleEnv, self).__init__()
        self.grid_size = grid_size
        self.num_teams = num_teams
        self.num_agents = num_teams * 3
        self.num_obstacles = num_obstacles
        self.max_turns = max_turns
        self.turn_count = 0
        self.agent_roles = ["H", "N", "S"]
        self.team_health = {i: 60 for i in range(self.num_teams)}
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size[0], grid_size[1]), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        self.reset()

    def reset(self):
        self.agents = {}
        self.teams = {i: [] for i in range(self.num_teams)}
        self.obstacles = set()
        self.turn_count = 0
        while len(self.obstacles) < self.num_obstacles:
            x, y = np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])
            self.obstacles.add((x, y))
        for team_id in range(self.num_teams):
            for role in self.agent_roles:
                while True:
                    x, y = np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])
                    if (x, y) not in self.obstacles:
                        break
                agent_name = f"{team_id}-{role}"
                self.agents[agent_name] = {'position': [x, y], 'health': 20, 'team': team_id, 'role': role}
                self.teams[team_id].append(agent_name)
        return self._get_observation()

    def step(self, actions):
        rewards = {agent: -1 for agent in self.agents}
        new_positions = {}
        self.turn_count += 1

        # Movement
        for agent, action in actions.items():
            x, y = self.agents[agent]['position']
            new_x, new_y = x, y
            if action == 0: new_y = max(0, y - 1)
            elif action == 1: new_y = min(self.grid_size[1] - 1, y + 1)
            elif action == 2: new_x = max(0, x - 1)
            elif action == 3: new_x = min(self.grid_size[0] - 1, x + 1)
            if (new_x, new_y) not in self.obstacles:
                new_positions[agent] = [new_x, new_y]

        for agent, pos in new_positions.items():
            self.agents[agent]['position'] = pos

        # Combat
        for agent, action in actions.items():
            if action == 4:
                rewards[agent] += self._attack(agent)
            elif action == 5:
                rewards[agent] += self._heal(agent)

        # Team health check
        for team_id in self.teams:
            self.team_health[team_id] = sum(self.agents[agent]['health'] for agent in self.teams[team_id])
        alive_teams = [team for team in self.team_health if self.team_health[team] > 0]
        done = len(alive_teams) == 1 or self.turn_count >= self.max_turns

        return self._get_observation(), rewards, done, {}

    def _attack(self, agent):
        x, y = self.agents[agent]['position']
        attacker_role = self.agents[agent]['role']
        attacker_team = self.agents[agent]['team']
        damage = 4 if attacker_role == "S" else 8 if attacker_role == "N" else 0
        reward = 0
        for target, info in self.agents.items():
            if info['team'] != attacker_team and abs(info['position'][0] - x) + abs(info['position'][1] - y) <= 1:
                if info['health'] > 0:
                    info['health'] = max(0, info['health'] - damage)
                    reward += 10
        return reward

    def _heal(self, agent):
        if self.agents[agent]['role'] == "H":
            x, y = self.agents[agent]['position']
            for target, info in self.agents.items():
                if info['team'] == self.agents[agent]['team'] and info['position'] == [x, y]:
                    info['health'] = min(20, info['health'] + 4)
                    return 5
        return 0

    def _get_observation(self):
        grid = np.zeros(self.grid_size)
        for x, y in self.obstacles:
            grid[x, y] = -1
        for agent in self.agents.values():
            x, y = agent['position']
            grid[x, y] = 1
        return grid

    def render(self):
        grid = np.full(self.grid_size, '.', dtype=object)
        for x, y in self.obstacles:
            grid[x, y] = '#'
        for agent_name, agent in self.agents.items():
            x, y = agent['position']
            grid[x, y] = f"{agent_name}({agent['health']})"
        for row in grid:
            print(" ".join(f"{cell:>8}" for cell in row))
        print("\nTeam Health:")
        for team, health in self.team_health.items():
            print(f"Team {team}: {health} HP")
if __name__ == "__main__":
    env = SchwabRoyaleEnv()
    obs = env.reset()
    print("Initial Observation (Grid):")
    print(obs)

    print("\n--- Initial Environment Render ---")
    env.render()

    # Example actions (all agents move down (action=1))
    sample_actions = {agent: 1 for agent in env.agents}
    obs, rewards, done, _ = env.step(sample_actions)

    print("\n--- After One Step (All agents moved down) ---")
    env.render()

    print("\nSample Rewards after one step:")
    for agent, reward in rewards.items():
        print(f"{agent}: {reward}")

