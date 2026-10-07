import unittest

import numpy as np

from RLAgent import RLAgent
from SchwabRoyaleEnv import SchwabRoyaleEnv


class SchwabRoyaleEnvironmentTests(unittest.TestCase):
    def make_env(self, max_turns=10):
        return SchwabRoyaleEnv(
            grid_size=(5, 5),
            num_teams=2,
            num_obstacles=0,
            max_turns=max_turns,
        )

    def idle_actions(self, env):
        return {agent_name: 0 for agent_name in env.agents}

    def test_reset_returns_valid_float32_grid(self):
        env = self.make_env()
        observation = env.reset()
        env.obstacles.add((0, 0))
        observation = env._get_observation()

        self.assertEqual(observation.shape, (5, 5))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(np.isin(observation, (-1.0, 0.0, 1.0)).all())
        self.assertEqual(len(env.teams), 2)
        self.assertEqual(sum(map(len, env.teams.values())), 6)

    def test_valid_action_moves_agent(self):
        env = self.make_env()
        agent_name = "0-H"
        env.agents[agent_name]["position"] = [2, 2]
        actions = self.idle_actions(env)

        _, rewards, done, _ = env.step(actions)

        self.assertEqual(env.agents[agent_name]["position"], [2, 1])
        self.assertEqual(rewards[agent_name], -1)
        self.assertFalse(done)

    def test_attack_reduces_adjacent_enemy_health(self):
        env = self.make_env()
        env.agents["0-S"]["position"] = [2, 2]
        env.agents["1-H"]["position"] = [2, 3]
        env.agents["1-N"]["position"] = [4, 4]
        env.agents["1-S"]["position"] = [3, 4]
        actions = self.idle_actions(env)
        actions["0-S"] = 4

        _, rewards, _, _ = env.step(actions)

        self.assertEqual(env.agents["1-H"]["health"], 16)
        self.assertEqual(rewards["0-S"], 9)

    def test_healer_heals_injured_co_located_ally(self):
        env = self.make_env()
        env.agents["0-H"]["position"] = [0, 0]
        env.agents["0-N"]["position"] = [0, 0]
        env.agents["0-N"]["health"] = 10
        actions = self.idle_actions(env)
        actions["0-H"] = 5

        _, rewards, _, _ = env.step(actions)

        self.assertEqual(env.agents["0-N"]["health"], 14)
        self.assertEqual(env.agents["0-H"]["health"], 20)
        self.assertEqual(rewards["0-H"], 4)

    def test_episode_ends_on_last_team_eliminated(self):
        env = self.make_env()
        for agent_name in env.teams[1]:
            env.agents[agent_name]["health"] = 0

        _, _, done, _ = env.step(self.idle_actions(env))

        self.assertTrue(done)
        self.assertEqual(env.get_winner(), 0)

    def test_turn_limit_with_multiple_surviving_teams_is_draw(self):
        env = self.make_env(max_turns=1)

        _, _, done, _ = env.step(self.idle_actions(env))

        self.assertTrue(done)
        self.assertIsNone(env.get_winner())


class DQNAgentTests(unittest.TestCase):
    def make_agent(self, device="cpu"):
        env = SchwabRoyaleEnv(
            grid_size=(5, 5),
            num_teams=2,
            num_obstacles=0,
            max_turns=10,
        )
        return RLAgent(env, team_id=0, device=device)

    def test_action_is_in_valid_range(self):
        agent = self.make_agent()
        agent.epsilon = 0.0
        action = agent.act(np.zeros((5, 5), dtype=np.float32))

        self.assertGreaterEqual(action, 0)
        self.assertLess(action, agent.action_dim)

    def test_replay_memory_insertion(self):
        agent = self.make_agent()
        state = np.zeros((5, 5), dtype=np.float32)
        agent.remember((state, 0, 1.0, state, 0.0), error=0.5)

        self.assertEqual(len(agent.memory.buffer), 1)
        self.assertGreater(agent.memory.priorities[0], 0)

    def test_batch_learning_runs_on_cpu(self):
        agent = self.make_agent()
        agent.batch_size = 2
        state = np.zeros((5, 5), dtype=np.float32)
        transition = (state, 0, 1.0, state, 0.0)
        for _ in range(agent.batch_size):
            error = agent.compute_td_error(*transition)
            agent.remember(transition, error)

        agent.learn()

        self.assertEqual(agent.device.type, "cpu")
        self.assertTrue(
            all(parameter.device.type == "cpu" for parameter in agent.parameters())
        )


if __name__ == "__main__":
    unittest.main()
