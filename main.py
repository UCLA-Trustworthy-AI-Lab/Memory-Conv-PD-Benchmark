import os
import csv
import random
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from agent import Agent
from payoff import PayoffMatrix
from game_rule import tell_game_rule, tell_game_rule_no_conv
from supergame import match


###################
## CONTROL PANEL ##
SEED = 40               # Same randomness setting for every model
CONV = True             # Conversation Regime
RAND_STOP = True        # Random stop
L = 2                   # Number of supergames each agent played
R = 6                   # Number of rounds per match or expected number of rounds under random stopping
STOP_PROB = 1 / R       # Game stopping probability in each round
NUM_EACH_AGENT = 2      # Number of agents created for each model (at least one of this variable and number of types in POOL should be even)

## Very Cheap Test
POOL = [
    "gpt-5-nano",
    "gpt-4.1-nano",
    "gpt-4o-mini"
]

# POOL = [
#     "gpt-4o",
#     "gpt-4.1-mini",
#     "gpt-5.5"
# ]

PAYOFF = PayoffMatrix(R = 2, T = 5, P = 0, S = -1)

###########################
## Step 1: Create agents ##

## Load key from environment
load_dotenv()
key = os.getenv("key")
if key is None: 
    raise ValueError("KEY FAILED: key")

## Create agents
agents: list[Agent] = []
agent_id = 1

for model in POOL:
    for i in range(NUM_EACH_AGENT):
        agents.append(Agent(agent_id, model, key))
        agent_id += 1

if len(agents) % 2 != 0: 
    raise ValueError("Total number of agents must be even.")

## Tell game rule
for a in agents:
    if CONV:    tell_game_rule(a, PAYOFF, R, RAND_STOP)
    else:       tell_game_rule_no_conv(a, PAYOFF, R, RAND_STOP)


########################################
## Step 2: Create directory and files ##

## Make directory
os.makedirs(f"./output", exist_ok = True)

## Create setting file 
setting_text = (
    f"SEED = {SEED}\n"
    f"CONV = {CONV}\n"
    f"RAND_STOP = {RAND_STOP}\n"
    f"L = {L}\n"
    f"R = {R}\n"
    f"STOP_PROB = {STOP_PROB}\n"
    f"NUM_EACH_AGENT = {NUM_EACH_AGENT}\n"
    f"POOL = {POOL}\n"
    f"PAYOFF: R = {PAYOFF.R}, T = {PAYOFF.T}, P = {PAYOFF.P}, S = {PAYOFF.S}\n"
)

setting_path = Path("./output/setting.txt")
setting_i = 1
while setting_path.exists():
    setting_path = Path(f"./output/setting ({setting_i}).txt")
    setting_i += 1
setting_path.write_text(setting_text)

## Create experiment file
exp_path = Path("./output/experiment.tsv")
exp_i = 1
while exp_path.exists():
    exp_path = Path(f"./output/experiment ({exp_i}).tsv")
    exp_i += 1
    
with open(exp_path, "w", newline = "", encoding = "utf-8-sig") as f:
    writer = csv.writer(f, delimiter = "\t")
    writer.writerow([
        "Supergame", "ID_0", "ID_1", "Model_0", "Model_1", "Round",
        "Action_0", "Action_1", "Belief_0", "Belief_1",
        "Payoff_0", "Payoff_1", "Cumulative_Payoff_0",
        "Cumulative_Payoff_1", "Message_0", "Message_1",
        "Reasoning_0", "Reasoning_1"
    ])


############################################
## Step 3: Repeat L randomized supergames ##

def run_match(p0, p1, l, seed):
    rng = random.Random(seed)
    records = match(p0 = p0, p1 = p1, conversation = CONV, payoff_mx = PAYOFF,
                    l = l, R = R, rand_stop = RAND_STOP, stop_prob = STOP_PROB, rng = rng)
    return p0, p1, records

random.seed(SEED)
with ThreadPoolExecutor(max_workers = len(agents) // 2) as executor:  ## Parallel Threading
    for l in range(1, L + 1):
        print(f"Randomized supergame {l} out of {L} started.")
        
        ## Random matching and arrange 
        permute = random.sample(agents, len(agents))
        pairs = [(permute[i], permute[i + 1]) for i in range(0, len(permute), 2)]
        match_seeds = [random.randint(0, 2**32 - 1) for _ in pairs]  ## deterministic seed for each match
        futures = [executor.submit(run_match, p0, p1, l, seed) for (p0, p1), seed in zip(pairs, match_seeds)]
        results = [future.result() for future in futures]
        
        ## Run matches and write tsv
        with open(exp_path, "a", newline = "", encoding = "utf-8-sig") as f:
            writer = csv.writer(f, delimiter = "\t")
            for p0, p1, records in results:
                for record in records:
                    writer.writerow([l, p0.id, p1.id, p0.model, p1.model, record["round"],
                                    record["action_0"], record["action_1"],            
                                    record["belief_0"], record["belief_1"],
                                    record["payoff_0"], record["payoff_1"],
                                    record["cumulative_payoff_0"], record["cumulative_payoff_1"],
                                    record["message_0"], record["message_1"],
                                    record["reasoning_0"], record["reasoning_1"]])
## Close memory
for a in agents:
    a.reset_memory()