import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from RLAgent import RLAgent
from SchwabRoyaleEnv import SchwabRoyaleEnv


def train(
    episodes=200,
    target_update_freq=10,
    max_turns=100,
    output_dir="outputs",
    device=None,
    seed=None,
    plot_rewards=False,
):
    if episodes < 1:
        raise ValueError("episodes must be greater than zero.")
    if target_update_freq < 1:
        raise ValueError("target_update_freq must be greater than zero.")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    env = SchwabRoyaleEnv(
        grid_size=(10, 10),
        num_teams=2,
        num_obstacles=10,
        max_turns=max_turns,
    )
    agents_per_team = {
        team_id: RLAgent(env, team_id=team_id, device=device)
        for team_id in range(env.num_teams)
    }
    results = {team_id: 0 for team_id in agents_per_team}
    draws = 0

    for episode in range(episodes):
        obs = env.reset()
        total_team_rewards = {team_id: 0 for team_id in agents_per_team}
        done = False

        while not done:
            actions = {}
            for team_id, agent in agents_per_team.items():
                actions.update(
                    {
                        agent_name: agent.act(obs)
                        for agent_name in env.teams[team_id]
                    }
                )

            next_obs, rewards, done, _ = env.step(actions)

            for team_id, agent in agents_per_team.items():
                for agent_name in env.teams[team_id]:
                    transition = (
                        obs,
                        actions[agent_name],
                        rewards[agent_name],
                        next_obs,
                        float(done),
                    )
                    error = agent.compute_td_error(*transition)
                    agent.remember(transition, error)
                    total_team_rewards[team_id] += rewards[agent_name]
                agent.learn()

            obs = next_obs

        for team_id, agent in agents_per_team.items():
            agent.track_rewards(total_team_rewards[team_id])

        winner = env.get_winner()
        if winner is None:
            draws += 1
            outcome = "Draw"
        else:
            results[winner] += 1
            outcome = f"Team {winner + 1} wins"
        print(
            f"Episode {episode + 1}/{episodes} | "
            f"Rewards: {total_team_rewards} | Outcome: {outcome}"
        )

        if (episode + 1) % target_update_freq == 0:
            for agent in agents_per_team.values():
                agent.update_target()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for team_id, agent in agents_per_team.items():
        agent.save_model(output_path / f"team{team_id}_model.pth")
        with (output_path / f"team{team_id}_rewards.pkl").open("wb") as file:
            pickle.dump(agent.rewards_per_episode, file)

    if plot_rewards:
        for team_id, agent in agents_per_team.items():
            print(f"\nTeam {team_id + 1} reward history:")
            agent.plot_rewards()

    print("\nTraining outcomes (by surviving team):")
    for team_id, wins in results.items():
        print(f"Team {team_id + 1} wins: {wins}")
    print(f"Draws: {draws}")
    print(f"Saved models and episode rewards to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train team-shared DQN agents in Schwab Royale."
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--target-update-freq", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Compute device; auto selects CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--plot-rewards",
        action="store_true",
        help="Display each team's per-episode training reward history.",
    )
    args = parser.parse_args()
    if args.max_turns < 1:
        parser.error("--max-turns must be greater than zero.")
    if args.episodes < 1:
        parser.error("--episodes must be greater than zero.")
    if args.target_update_freq < 1:
        parser.error("--target-update-freq must be greater than zero.")

    train(
        episodes=args.episodes,
        target_update_freq=args.target_update_freq,
        max_turns=args.max_turns,
        output_dir=args.output_dir,
        device=None if args.device == "auto" else args.device,
        seed=args.seed,
        plot_rewards=args.plot_rewards,
    )


if __name__ == "__main__":
    main()
