import argparse
from pathlib import Path

import numpy as np
import torch

from RLAgent import RLAgent
from SchwabRoyaleEnv import SchwabRoyaleEnv


def evaluate(episodes=100, model_dir="outputs", max_turns=100, device=None, seed=None):
    if episodes < 1:
        raise ValueError("episodes must be greater than zero.")
    if max_turns < 1:
        raise ValueError("max_turns must be greater than zero.")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    env = SchwabRoyaleEnv(
        grid_size=(10, 10),
        num_teams=2,
        num_obstacles=10,
        max_turns=max_turns,
    )
    agents = {
        team_id: RLAgent(env, team_id=team_id, device=device)
        for team_id in range(env.num_teams)
    }
    model_root = Path(model_dir)
    model_paths = {
        team_id: model_root / f"team{team_id}_model.pth"
        for team_id in agents
    }
    missing_models = [path for path in model_paths.values() if not path.is_file()]
    if missing_models:
        missing_list = ", ".join(str(path) for path in missing_models)
        raise FileNotFoundError(
            f"Missing trained model(s): {missing_list}. "
            "Run training.py first or provide the correct --model-dir."
        )

    for team_id, agent in agents.items():
        agent.load_model(model_paths[team_id])
        agent.epsilon = 0.0

    wins = {team_id: 0 for team_id in agents}
    draws = 0
    reward_totals = {team_id: 0.0 for team_id in agents}
    episode_lengths = []

    for episode in range(episodes):
        obs = env.reset()
        episode_rewards = {team_id: 0.0 for team_id in agents}
        done = False
        turns = 0

        while not done:
            actions = {}
            for team_id, agent in agents.items():
                actions.update(
                    {
                        agent_name: agent.act(obs)
                        for agent_name in env.teams[team_id]
                    }
                )
            obs, rewards, done, _ = env.step(actions)
            turns += 1
            for team_id, team_agents in env.teams.items():
                episode_rewards[team_id] += sum(
                    rewards[agent_name] for agent_name in team_agents
                )

        winner = env.get_winner()
        if winner is None:
            draws += 1
            outcome = "Draw"
        else:
            wins[winner] += 1
            outcome = f"Team {winner + 1} wins"
        for team_id in agents:
            reward_totals[team_id] += episode_rewards[team_id]
        episode_lengths.append(turns)
        print(
            f"Evaluation episode {episode + 1}/{episodes} | "
            f"Turns: {turns} | Outcome: {outcome}"
        )

    print(f"\nEvaluation summary over {episodes} episodes:")
    for team_id in agents:
        print(
            f"Team {team_id + 1}: wins={wins[team_id]}, "
            f"average episode reward={reward_totals[team_id] / episodes:.2f}"
        )
    print(f"Draws: {draws}")
    print(f"Average episode length: {sum(episode_lengths) / episodes:.2f} turns")
    return {
        "wins": wins,
        "draws": draws,
        "average_rewards": {
            team_id: reward_totals[team_id] / episodes for team_id in agents
        },
        "average_episode_length": sum(episode_lengths) / episodes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained team policies by playing complete games."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--model-dir", default="outputs")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Compute device; auto selects CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be greater than zero.")
    if args.max_turns < 1:
        parser.error("--max-turns must be greater than zero.")

    try:
        evaluate(
            episodes=args.episodes,
            model_dir=args.model_dir,
            max_turns=args.max_turns,
            device=None if args.device == "auto" else args.device,
            seed=args.seed,
        )
    except FileNotFoundError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
