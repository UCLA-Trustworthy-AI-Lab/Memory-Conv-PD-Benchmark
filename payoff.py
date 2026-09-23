ACTION_NAME = {0: "C", 1: "D"}

class PayoffMatrix:
    def __init__(self, R, T, P, S):
        self.R = R
        self.T = T
        self.P = P
        self.S = S
        self.rho = T / R
        self.nu = (R - S) / R

        self.matrix = {
            (0,0): (R, R),   ## (C, C)
            (0,1): (S, T),   ## (C, D)
            (1,0): (T, S),   ## (D, C)
            (1,1): (P, P),   ## (D, D)
        }

    def get_payoff(self, action0, action1):
        """Return the payoffs for the given pair of actions."""
        return self.matrix[(action0, action1)]
    
    def payoff_text(self):
        """Format the payoff matrix as text."""
        lines = []
        for (a0, a1), (p0, p1) in self.matrix.items():
            lines.append(f"- ({ACTION_NAME[a0]}, {ACTION_NAME[a1]}) -> ({p0}, {p1})")

        return "\n".join(lines)