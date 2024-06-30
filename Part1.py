import numpy as np
import matplotlib.pyplot as plt


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

        pass

    def run(self):
        pass


if __name__ == '__main__':
    pass
