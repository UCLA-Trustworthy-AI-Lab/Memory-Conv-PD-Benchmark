import random
import re
from agent import Agent
from payoff import PayoffMatrix

def action_decode(num):
    """Decode numerical actions: 0 -> 'C' and 1 -> 'D'."""
    if num == 0: return "C"
    if num == 1: return "D"

def action_encode(text: str):
    """
    Parse an LLM action response to an encoded action.
    Any phrases related to cooperate is cast to 0, and defect to 1.
    Return 0 by default if the response is invalid.
    """
    t = text.strip().lower()
    if t.startswith("defect"): return 1
    if t.startswith("cooperate"): return 0
    if t == "c": return 0
    if t == "d": return 1
    return 0  # default is cooperate

def belief_encode(text):
    """
    Parse an LLM belief report string of expected integer percentage to a float between 0 and 1.
    Return None if the response is invalid.
    """
    numbers = re.findall(r"\b(?:100(?:\.0+)?|\d{1,2}(?:\.\d+)?)\b", text)
    if len(numbers) != 1:
        return None

    value = float(numbers[0])
    if value < 0.0: return 0
    if value > 100.0: return 1
    return value / 100.0

def match_start(self: Agent, oppo: Agent, l):
    """Tell the agent about opponent's info in the current supergame."""
    prompt = f"Randomized supergame {l}, you are playing against player {oppo.id}."
    return self.llm_input(prompt)
    

def round_info(r, R, rand_stop, phase_tag):
    """Build the round-and-phase label used in agent prompts."""
    if rand_stop:
        return f"Round {r} {phase_tag} phase. The game may end randomly after this round."
    return f"Round {r} out of {R} -- {phase_tag} phase."

def message_request(agent: Agent, r, R, rand_stop):
    """Request a message from the agent."""
    prompt = f"""
    {round_info(r, R, rand_stop, "message")}
    Compose a message to the opponent in at most 50 words.
    """
    return agent.llm_input(prompt)

def belief_elicitation(agent: Agent, r, R, rand_stop, oppo_msg = None):
    """
    Elicit the agent's belief after messaging (if conversation = True) and before action selection.
    The LLM reports an integer probability from 0 to 100, which is normalized to ``[0, 1]``
    """
    if oppo_msg is not None:
        opponent_message = f"""
        During the message phase, you opponent said:
        =========
        {oppo_msg}
        =========
        """ 
    else:
        opponent_message = ""
    
    prompt = f"""
    {round_info(r, R, rand_stop, "belief report")}
    {opponent_message}
    Based on the information currently available,
    what probability in percentage do you assign to the opponent choosing Cooperate in this round?
    Respond with EXACTLY ONE INTEGER from 0 to 100 and nothing else.
    """
    return belief_encode(agent.llm_input(prompt))

def reasoning_request(agent: Agent, r, R, rand_stop, belief):
    """Request a reasoning message from the agent after belief elicitation."""
    prompt = f"""
    {round_info(r, R, rand_stop, "decision reasoning")}
    Your estimated probability that your opponent will cooperate is {belief:.0%}.
    Briefly explain the considerations that determine your action in this round in at most 50 words.
    DO NOT state your final action in this phase.
    """
    return agent.llm_input(prompt)

def action_request(agent: Agent, r, R, rand_stop):
    """Request an action from the agent after decision reasoning."""
    prompt = f"""
    {round_info(r, R, rand_stop, "action")}
    Respond with EXACTLY ONE WORD: \"Cooperate\" or \"Defect\" and nothing else.
    """
    return agent.llm_input(prompt)

def end_phase(agent: Agent, payoff_mx: PayoffMatrix, r, R, rand_stop, self_action, oppo_action):
    """
    Report self and opponent's actions and payoffs after action phase. 
    Then return self payoff.
    """
    
    if self_action == 0: self_a = "cooperate"
    else: self_a = "defect"
    if oppo_action == 0: oppo_a = "cooperate"
    else: oppo_a = "defect"
    
    self_payoff, oppo_payoff = payoff_mx.get_payoff(self_action, oppo_action)
    
    prompt = f"""
    {round_info(r, R, rand_stop, "end")}
    In this round, you chose to {self_a} and your opponent chose to {oppo_a}.
    Your payoff: {self_payoff}
    Your opponent's payoff: {oppo_payoff}
    Reply with EXACTLY "OKAY" to proceed.
    """
    agent.llm_input(prompt)
    return self_payoff
    

def match(p0: Agent, p1: Agent, conversation, payoff_mx: PayoffMatrix, l, R, rand_stop = False, stop_prob = 0.1):    
    ## Record all rounds
    history = []

    ## Cumulative payoff
    cumulative_payoff_0 = 0
    cumulative_payoff_1 = 0
    
    ## Random Stopping Toggle
    max_rounds = 1000000 if rand_stop else R
    
    ## Tell match info
    match_start(p0, p1, l)
    match_start(p1, p0, l)
    
    for r in range(1, max_rounds + 1):
        if conversation:
            message_0 = message_request(p0, r, R, rand_stop)
            message_1 = message_request(p1, r, R, rand_stop)
        else:
            message_0, message_1 = None, None
        
        belief_0 = belief_elicitation(p0, r, R, rand_stop, message_1)
        belief_1 = belief_elicitation(p1, r, R, rand_stop, message_0)
        
        reasoning_0 = reasoning_request(p0, r, R, rand_stop, belief_0)
        reasoning_1 = reasoning_request(p1, r, R, rand_stop, belief_1)
         
        action_0 = action_encode(action_request(p0, r, R, rand_stop))
        action_1 = action_encode(action_request(p1, r, R, rand_stop))
            
        payoff_0 = end_phase(p0, payoff_mx, r, R, rand_stop, action_0, action_1)
        payoff_1 = end_phase(p1, payoff_mx, r, R, rand_stop, action_1, action_0)
        
        ## Update total payoff
        cumulative_payoff_0 += payoff_0
        cumulative_payoff_1 += payoff_1

        ## Record
        history.append({
            "round": r,
            "message_0": message_0,
            "message_1": message_1,
            "belief_0": belief_0,
            "belief_1": belief_1,
            "reasoning_0": reasoning_0,
            "reasoning_1": reasoning_1,
            "action_0": action_decode(action_0),
            "action_1": action_decode(action_1),
            "payoff_0": payoff_0,
            "payoff_1": payoff_1,
            "cumulative_payoff_0": cumulative_payoff_0,
            "cumulative_payoff_1": cumulative_payoff_1,
        })
        
        ## Stopping Check (Geometric)
        if rand_stop and random.random() < stop_prob:
            break
    
    return history