# Schwab_Royale_using_Multi_Deep-Q-Networks

# ⚔️ SchwabRoyale

**SchwabRoyale** is a multi-agent, multi-team **reinforcement learning battle royale environment**, designed to explore emergent behavior, cooperation, and competition between agents with different goals and abilities.

---

##  Key Features

-  Multi-agent & multi-team setting (2–3 teams with 3 agents each)
-  Team and individual rewards for balanced cooperation & competition
-  Agent roles: Healer (H), Necromancer (N), Swordsman (S) — each with special rules
-  Deep Q-Learning with experience replay & target networks
-  Expandable for research: selfishness, team spite, communication, and more

---

##  Project Structure

```bash
SchwabRoyale/
│
├── SchwabRoyaleEnv.py       # Game environment and core logic
├── RLAgent.py               # Deep Q-Learning agent with upgrade options
├── training.py              # Training script for all agents
├── evaluation.py            # Agent testing and performance evaluation


```

---

##  How It Works

### 🔹 Environment (`SchwabRoyaleEnv.py`)
- Built from scratch, gym-style interface
- Agents have unique roles and attack/skill behavior:
  - **Healer (H)**: can heal teammates
  - **Necromancer (N)**: deals ranged damage, potential for revival
  - **Swordsman (S)**: melee-focused attacker
- Actions, health, turns, visibility, and rewards are all managed here.
- Image of sample Environment
- ![image](https://github.com/user-attachments/assets/d0052888-43a4-4d4c-8b05-073b0c9b4a69)


### 🔹 Agents (`RLAgent.py`)
- Implements a DQN-based learning agent.
- Supports:
  -  Experience Replay
  -  Target Networks
  -  Prioritized Experience Replay *(optional)*
  -  Double DQN & Dueling DQN *(optional extensions)*

### 🔹 Training (`training.py`)
- Trains agents over multiple episodes.
- Tracks rewards, kill count, and team victory.
- Agents are trained independently but in the same environment loop.

### 🔹 Evaluation (`evaluation.py`)
- Tests trained models.
- Useful for visualizing agent behavior, cooperation, and strategy evolution.

---

##  Research Directions

-  Agent personality tuning (selfish vs. cooperative)
-  Partial observability (fog of war)
-  Emergent communication between team members
-  Meta-controller (Overmind) for adaptive learning
-  Sparse reward solutions: curiosity-driven learning, option frameworks
-  Curriculum learning: gradual environment complexity
-  Encrypted language emergence between agents

---

##  Getting Started

###  Requirements

- Python 3.8+
- PyTorch
- NumPy
- OpenAI Gym (if extended)
- Matplotlib (for evaluation)

Install dependencies:

```bash
pip install -r requirements.txt
```

###  Run Training

```bash
python training.py
```

###  Evaluate Agents

```bash
python evaluation.py
```

---

##  Results (after running the code)

- Win rate graphs
- Reward trends
- Heatmaps for agent movement and attack patterns

![WhatsApp Image 2025-04-18 at 21 32 55_02cc80fd](https://github.com/user-attachments/assets/77e9002b-9c6d-48df-a15e-48c1501c7252)


---

##  Inspiration & Use Cases

- Study multi-agent behavior in mixed cooperative-competitive settings.
- Explore agent communication and encryption.
- Use as a sandbox for RL experiments in sparse reward environments.

---

---

##  License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and share!

---

##  Contributions

Issues, suggestions, and pull requests are welcome!  
Want to add new agent types, game modes, or training tricks? Fork and build your version of SchwabRoyale!

```

---
