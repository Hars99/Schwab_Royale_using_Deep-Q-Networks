# Schwab Royale using Deep Q-Networks

## Overview

Schwab Royale is a custom multi-agent reinforcement-learning experiment built around a grid-based team battle environment. Two teams, each with a healer, necromancer, and swordsman, compete to eliminate the opposing team before the turn limit. A team wins when it is the only team with living agents; if the turn limit is reached while multiple teams remain alive, the match is a draw.

Each team uses one shared Deep Q-Network (DQN) policy to choose actions for its three agents. The project is an experimental game and does not model real financial markets.

## Key Features

- Custom grid-based environment with two teams and three roles per team.
- Six discrete actions: four movement directions, attack, and heal.
- Team-shared neural Q-network with experience replay and a target network.
- Double DQN target selection and prioritized experience replay.
- CPU training by default when CUDA is unavailable; optionally uses CUDA when available.
- Training saves a model and one cumulative reward value per team per episode.
- Evaluation plays new episodes with loaded models and reports team wins, draws, average team rewards, and average episode length.

## Environment

The scripts configure a 10 × 10 grid with 10 randomly placed obstacles, two teams, three agents per team, and a 100-turn limit. Each agent starts with 20 health. The grid observation is a shared `float32` array: empty cells are `0`, obstacles are `-1`, and cells occupied by agents are `1`. It does not encode agent identity, team, role, or health.

Example visualization of the game environment:

![Example Schwab Royale game environment](https://github.com/user-attachments/assets/d0052888-43a4-4d4c-8b05-073b0c9b4a69)


Each team has:

- **Healer (H):** action 5 heals an injured, living teammate on the same cell by up to 4 health. If there is no injured teammate on that cell, an injured healer may heal itself. Healing a full-health unit or having no valid target gives no heal reward.
- **Necromancer (N):** action 4 deals 8 damage to each living enemy at Manhattan distance 1 or less. There is no revival mechanic.
- **Swordsman (S):** action 4 deals 4 damage to each living enemy at Manhattan distance 1 or less.

Actions 0–3 move up, down, left, and right respectively. Movement is clamped to the grid and blocked by obstacles. Action 4 attacks; action 5 heals. Each action starts with a reward of -1; damaging an enemy adds 10 per target, and a successful heal adds 5. Team episode reward is the sum of its agents' rewards over the episode.

An episode ends when at most one team has living agents or the turn limit is reached. The remaining team wins only if exactly one team is alive. A turn-limit ending with multiple living teams, or no surviving team, is a draw.

This is a custom multi-agent environment with Gym spaces; `step()` takes a dictionary mapping agent IDs to individual actions and returns one shared observation. It is not a standard single-agent Gym environment API implementation.

## Reinforcement Learning Approach

One `RLAgent` controls each team. Its policy is shared across that team's agents, which all receive the same grid observation. The observation is flattened and passed into a fully connected Q-network with two 128-unit ReLU hidden layers and six action values.

- **Action selection:** epsilon-greedy, with epsilon decaying from 1.0 toward a minimum of 0.05 during learning.
- **Replay memory:** prioritized replay buffer with capacity 10,000, priority exponent `alpha=0.6`, and importance-sampling parameter `beta=0.4`.
- **Double DQN:** the online Q-network selects the next action, and the target network evaluates it.
- **Target network:** copied from the online network every 10 training episodes by default.
- **Optimization:** Adam, learning rate `1e-3`, discount factor `0.95`, batches of 32, and importance-weighted squared TD error.

## Training Workflow

```text
Environment Reset
       ↓
Shared Grid Observation
       ↓
Team DQN Selects Each Agent's Action
       ↓
Environment Step
       ↓
Rewards and Next Observation
       ↓
Prioritized Replay Memory
       ↓
Double DQN Network Update
       ↓
Repeat Until the Match Ends
```

## Evaluation

`evaluation.py` requires a saved model for each team. It loads both models, disables epsilon exploration, and runs complete matches in the same environment rules used by training. It reports wins by surviving team, draws, average episode reward per team, and average number of turns. A team win is determined from the environment's living-agent health state, not from reward totals.

Evaluation uses new randomly initialized matches. Set `--seed` to initialize NumPy and PyTorch random generators for a repeatable starting point. No benchmark result is included because the repository contains no prior trained model or reproducible evaluation output.

## Project Structure

```text
.
├── SchwabRoyaleEnv.py       # Grid environment, roles, actions, rewards, termination
├── RLAgent.py               # Q-network, prioritized replay, action selection and learning
├── training.py              # Train team agents and save model/reward files
├── evaluation.py            # Load models and evaluate complete matches
├── requirements.txt         # Runtime Python dependencies
└── tests/
    └── test_game.py         # Deterministic environment and CPU agent tests
```

The PowerPoint and PDF report are supplementary project materials; current implementation claims in this README are based on the Python source.

## Installation

From the repository root, create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Training

Train with the default 200 episodes and automatically select CUDA when available (otherwise CPU):

```powershell
python training.py
```

For a short CPU run with a fixed seed:

```powershell
python training.py --episodes 3 --max-turns 20 --device cpu --seed 7
```

Training saves `team0_model.pth`, `team1_model.pth`, `team0_rewards.pkl`, and `team1_rewards.pkl` in `outputs/` by default. The reward histories contain one cumulative team reward per completed episode. Use `--output-dir` to choose another location.

## Tests

Run the environment and agent unit tests with Python's built-in test runner:

```powershell
python -m unittest discover -s tests -v
```

## Evaluation

Run 100 evaluation episodes using the models produced by training:

```powershell
python evaluation.py --episodes 100 --model-dir outputs
```

For a short seeded CPU evaluation:

```powershell
python evaluation.py --episodes 3 --max-turns 20 --model-dir outputs --device cpu --seed 7
```

Evaluation stops with a clear error if either team's model file is missing.

## Limitations

- The game mechanics, rewards, and team coordination are deliberately simplified.
- The shared grid observation omits agent identity, role, team, and health information.
- Dead agents are not removed from the action loop, so the current simulation does not fully enforce incapacitation.
- Random map placement and training outcomes depend on random seeds and hyperparameters; GPU execution may not be bit-for-bit deterministic.
- There are no published benchmark results or saved trained models in the repository.
- Gym spaces describe a per-agent action and shared grid observation, but the multi-agent `step()` signature is custom rather than API-compatible.

## Future Improvements

- Improve reward shaping and provide richer, agent-specific observations.
- Add explicit dead-agent handling and stronger team coordination.
- Explore Dueling DQN, recurrent policies, and systematic benchmark evaluation.
- Add visual game rendering and match replay.

## Author

Harshith Manikhanta Sunkara
