import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


"""
# Notes
1. discount (gamma) =0.95
2. policy: probability of moving in any of the four directions = 0.25
3. Any action at blue => +5 reward => jump to the red square
4. Any action at green => +2.5 reward => jump to the red or yellow square with probability 0.5

# Objectives
(section 1): Estimate the value function for each of the states
1. Solve the system of Bellman equations explicitly             => how can i explicitly solve the equations?
2. Iterative policy evaluation                                  => what is iterative policy evaluation?
3. value iteration.                                             => what is value iteration?
Which states have the highest value? Does this surprise you?

(section 2):  Determine the optimal policy for the gridworld problem
1. explicitly solving the Bellman optimality equation
2. using policy iteration with iterative policy evaluation 
3. policy improvement with value iteration.
"""


class Agent:
    def __init__(self):
        self.discount = 0.95
        self.policy_probability = 0.25
        self.expected_rewards = np.zeros((5, 5))

    def explicit(self, g=0.95):
        """
        eq1 = max(up:-1+g*v1, down:g*v6, left:-1+g*v1, right:g*v2)
        eq2 = 5+g*v18
        eq3 = max(-1+g*v3, g*v8, g*v2, g*v4)
        eq4 = max(-1+g*v4, g*v9, g*v3, g*v5)
        eq5 = 0.5*(2.5 + g*v18) + 0.5*(2.5 + g*v25)
        eq6 = max(g*v1, g*v11, -1+g*v6, g*v7)
        eq7 = max(g*v2, g*v12, g*v6, g*v8)
        eq8 = max(g*v3, g*v13, g*v7, g*v9)
        eq9 = max(g*v4, g*v14, g*v8, g*v10)
        eq10 = max(g*v5, g*v15, g*v9, -1+g*v10)
        eq11 = max(g*v6, g*v16, -1+g*v11, g*v12)
        eq12 = max(g*v7, g*v17, g*v11, g*v13)
        eq13 = max(g*v8, g*v18, g*v12, g*v14)
        eq14 = max(g*v9, g*v19, g*v13, g*v15)
        eq15 = max(g*v10, g*v20, g*v14, -1+g*v15)
        eq16 = max(g*v11, g*v21, -1+g*v16, g*v17)
        eq17 = max(g*v12, g*v22, g*v16, g*v18)
        eq18 = max(g*v13, g*v23, g*v17, g*v19)
        eq19 = max(g*v14, g*v24, g*v18, g*v20)
        eq20 = max(g*v15, g*v25, g*v19, -1+g*v20)
        eq21 = max(g*v16, -1+g*v21, -1+g*v21, g*v22)
        eq22 = max(g*v17, -1+g*v22, g*v21, g*v23)
        eq23 = max(g*v18, -1+g*v23, g*v22, g*v24)
        eq24 = max(g*v19, -1+g*v24, g*v23, g*v25)
        eq25 = max(g*v20, -1+g*v25, g*v24, -1+g*v25)
        """
        def equations(p):
            v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25 = p
            eq1 = max(-1 + g * v1, g * v6, -1 + g * v1, g * v2)
            eq2 = 5 + g * v18
            eq3 = max(-1 + g * v3, g * v8, g * v2, g * v4)
            eq4 = max(-1 + g * v4, g * v9, g * v3, g * v5)
            eq5 = 0.5 * (2.5 + g * v18) + 0.5 * (2.5 + g * v25)
            eq6 = max(g * v1, g * v11, -1 + g * v6, g * v7)
            eq7 = max(g * v2, g * v12, g * v6, g * v8)
            eq8 = max(g * v3, g * v13, g * v7, g * v9)
            eq9 = max(g * v4, g * v14, g * v8, g * v10)
            eq10 = max(g * v5, g * v15, g * v9, -1 + g * v10)
            eq11 = max(g * v6, g * v16, -1 + g * v11, g * v12)
            eq12 = max(g * v7, g * v17, g * v11, g * v13)
            eq13 = max(g * v8, g * v18, g * v12, g * v14)
            eq14 = max(g * v9, g * v19, g * v13, g * v15)
            eq15 = max(g * v10, g * v20, g * v14, -1 + g * v15)
            eq16 = max(g * v11, g * v21, -1 + g * v16, g * v17)
            eq17 = max(g * v12, g * v22, g * v16, g * v18)
            eq18 = max(g * v13, g * v23, g * v17, g * v19)
            eq19 = max(g * v14, g * v24, g * v18, g * v20)
            eq20 = max(g * v15, g * v25, g * v19, -1 + g * v20)
            eq21 = max(g * v16, -1 + g * v21, -1 + g * v21, g * v22)
            eq22 = max(g * v17, -1 + g * v22, g * v21, g * v23)
            eq23 = max(g * v18, -1 + g * v23, g * v22, g * v24)
            eq24 = max(g * v19, -1 + g * v24, g * v23, g * v25)
            eq25 = max(g * v20, -1 + g * v25, g * v24, -1 + g * v25)
            return eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14, eq15, eq16, eq17, eq18, eq19, eq20, eq21, eq22, eq23, eq24, eq25

        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25 = fsolve(equations, np.ones(25))
        return v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25


if __name__ == '__main__':
    agent = Agent()
    v = np.array(agent.explicit()).reshape((5, 5))
    print(v) # todo: is this correct?
