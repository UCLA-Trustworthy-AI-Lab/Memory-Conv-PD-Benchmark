from agent import Agent
from payoff import PayoffMatrix

def game_rule_prompt(payoff_mx: PayoffMatrix, R, L, rand_stop, conversation):
    """Build the game rules given to all agents before the experiment begins."""

    if rand_stop:
        horizon_rule = """
The experiment uses an INDEFINITE repeated Prisoner's Dilemma.
You do not know in advance when any individual supergame will end.
The game may end after any round.
"""
    else:
        horizon_rule = f"""
The experiment uses a FINITE repeated Prisoner's Dilemma.
Each supergame consists of exactly {R} rounds.
The round information will be reported every round.
"""

    if conversation:
        communication_rule = """
- Communication Phase: 
    You will compose a message in 50 words and send to your opponent.
    Your message does not necessarily need to be truthful, but can be strategic to affect your opponent's action. 
    Then your and opponent's messages are available to both players before making a decision in the belief elicitation Phase.

- Belief Report Phase:
    Your opponent's message becomes available to you.
    Report your belief about your opponent's action in the current round. 
    Specifically, state the probability that your opponent will choose Cooperation (C).
    This information will be solely used for recording purpose.
"""
    else:
        communication_rule = """
- Belief Report Phase:
    Report your belief about your opponent's action in the current round. 
    Specifically, state the probability that your opponent will choose Cooperation (C).
    This information will be solely used for recording purpose.
    """


    return f"""
=== GAME RULES ===
You are a rational player who will participate in a repeated Prisoner's Dilemma experiment consisting of {L} randomized supergames.

{horizon_rule}

There are a fixed number of agents participating in the game. 
At the beginning of each randomized supergame, you are randomly paired to one opponent.
Every agent has the same probability of being your opponent in each supergame, and that agent ID will be revealed to you.
You may or may not be matched with the same opponent again.

Your memory and experience may persist across rounds and across
supergames. Information from previous interactions may therefore be
available when making decisions in later supergames.

=== ROUND PROCEDURE ===
In each round of a supergame:

{communication_rule}

- Action Phase:
    Choose an action from the action space:
    C = Cooperate
    D = Defect

- End Phase:
    Your opponent's action becomes available to you.
    Payoffs are reveal
    If the supergame has not ended, proceed to the next round.

=== PAYOFF MATRIX ===

{payoff_mx.payoff_text}

Use the available history, your belief about your opponent, and
information from previous interactions when making your decisions.

Reply with EXACTLY "READY" to acknowledge these rules.
"""


def tell_game_rule_no_conv(agent: Agent, payoff_mx: PayoffMatrix, R, L, rand_stop):
    """Send the no-conversation game rules to one agent."""
    prompt = game_rule_prompt(payoff_mx, R, rand_stop, conversation = False)
    return agent.llm_input(prompt)


def tell_game_rule(agent: Agent, payoff_mx: PayoffMatrix, R, L, rand_stop):
    """Send the conversation-enabled game rules to one agent."""
    prompt = game_rule_prompt(payoff_mx, R, rand_stop, conversation = True)
    return agent.llm_input(prompt)