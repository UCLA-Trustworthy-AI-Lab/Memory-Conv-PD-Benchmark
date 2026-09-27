# Repeated Prisoner's Dilemma

This project implements a repeated Prisoner's Dilemma experiment with LLM-based agents. Agents are randomly matched into supergames and make sequential decisions while retaining conversational context across rounds and supergames.

## Game Logic

Each experiment consists of multiple **randomized supergames**.

1. **Agent matching**
   - At the beginning of each supergame, all agents are randomly shuffled and paired.
   - Each pair plays one repeated Prisoner's Dilemma.

2. **Repeated interaction**
   - Each supergame uses either:
     - a fixed number of rounds, or
     - an indefinite horizon with a geometric stopping rule.
   - With random stopping enabled, the game ends after each completed round with probability `STOP_PROB`.

3. **Round procedure**

   Each round follows:

   **Message → Belief → Reasoning → Action → End**

   - **Message:** Agents communicate with their opponent when conversation is enabled.
   - **Belief:** Each agent reports the probability that the opponent will cooperate.
   - **Reasoning:** Each agent explains the considerations relevant to its decision without stating its final action.
   - **Action:** Each agent chooses `C` (Cooperate) or `D` (Defect).
   - **End:** Both actions and payoffs are revealed to the agents before the next round.

4. **Memory**
   - Each agent maintains its previous API response through `previous_response_id`.
   - This allows the LLM to retain conversational context from previous rounds and supergames.
   - Agent memory is reset after the experiment.

5. **Payoffs**

   The default payoff matrix is:

   | | C | D |
   |---|---:|---:|
   | **C** | (2, 2) | (-1, 5) |
   | **D** | (5, -1) | (0, 0) |

## Folder Structure

```text
.
├── main.py          # Main experiment script and configuration
├── agent.py         # LLM agent class and conversation/memory handling
├── supergame.py     # Supergame and round logic
├── game_rule.py     # Game-rule prompts given to agents
├── payoff.py        # Payoff matrix and payoff calculations
├── output/          # Experiment results and configuration files
├── .env             # API key configuration (not committed)
└── README.md        # Project documentation