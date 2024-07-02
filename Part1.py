import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


"""
# Notes
1. discount (gamma) =0.95
2. policy: probability of moving in any of the four directions = 0.25
3. Any action at blue => +5 reward => jump to 17
4. Any action at green => +2.5 reward => jump to 17 or 24 square with probability 0.5

# Objectives
(section 1): Estimate the value function for each of the states (v)
1. Solve the system of Bellman equations explicitly              => needs a fix
2. Iterative policy evaluation                                  
3. value iteration.                                              => needs a fix
Which states have the highest value? Does this surprise you?

(section 2):  Determine the optimal policy for the gridworld problem (pi)
1. explicitly solving the Bellman optimality equation
2. using policy iteration with iterative policy evaluation 
3. policy improvement with value iteration.
"""


class ValueAgent:
    """
    State numbers:
    0   1   2   3   4
    5   6   7   8   9
    10  11  12  13  14
    15  16  17  18  19
    20  21  22  23  24
    """
    def __init__(self, discount=0.95):
        self.discount = discount
        self.value_function = None

    def visualize_results(self):
        plt.figure()
        sns.heatmap(self.value_function.reshape((5,5)), cmap='coolwarm', annot=True, fmt='.2f', square=True)
        plt.title("5x5 Gridworld - State Value Function")
        plt.show()

    def explicit(self, g=0.95):  # todo: this gives wrong outputs. fix
        """
        This method depends on solving a system of non-linear equations whose results depend on a set of initial values.
        As a result, there is a chance that the solver of the equations gets stuck in a local optima while finding the
        solution.
        """
        def equations(p):
            v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25 = p
            eq1 = max(-0.5 + g * v1, g * v6, -0.5 + g * v1, g * v2)
            eq2 = 5 + g * v18
            eq3 = max(-0.5 + g * v3, g * v8, g * v2, g * v4)
            eq4 = max(-0.5 + g * v4, g * v9, g * v3, g * v5)
            eq5 = 0.5 * (2.5 + g * v18) + 0.5 * (2.5 + g * v25)
            eq6 = max(g * v1, g * v11, -0.5 + g * v6, g * v7)
            eq7 = max(g * v2, g * v12, g * v6, g * v8)
            eq8 = max(g * v3, g * v13, g * v7, g * v9)
            eq9 = max(g * v4, g * v14, g * v8, g * v10)
            eq10 = max(g * v5, g * v15, g * v9, -0.5 + g * v10)
            eq11 = max(g * v6, g * v16, -0.5 + g * v11, g * v12)
            eq12 = max(g * v7, g * v17, g * v11, g * v13)
            eq13 = max(g * v8, g * v18, g * v12, g * v14)
            eq14 = max(g * v9, g * v19, g * v13, g * v15)
            eq15 = max(g * v10, g * v20, g * v14, -0.5 + g * v15)
            eq16 = max(g * v11, g * v21, -0.5 + g * v16, g * v17)
            eq17 = max(g * v12, g * v22, g * v16, g * v18)
            eq18 = max(g * v13, g * v23, g * v17, g * v19)
            eq19 = max(g * v14, g * v24, g * v18, g * v20)
            eq20 = max(g * v15, g * v25, g * v19, -0.5 + g * v20)
            eq21 = max(g * v16, -0.5 + g * v21, -0.5 + g * v21, g * v22)
            eq22 = max(g * v17, -0.5 + g * v22, g * v21, g * v23)
            eq23 = max(g * v18, -0.5 + g * v23, g * v22, g * v24)
            eq24 = max(g * v19, -0.5 + g * v24, g * v23, g * v25)
            eq25 = max(g * v20, -0.5 + g * v25, g * v24, -0.5 + g * v25)
            return eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14, eq15, eq16, eq17, eq18, eq19, eq20, eq21, eq22, eq23, eq24, eq25

        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25 = fsolve(equations, np.random.normal(size=25))
        self.value_function = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25])
        return self.value_function

    def mean_update_value_function(self, threshold=0.1):
        """
        1. initialize the state values randomly.
        2. for every state:
                for every possible action in that state
                    for every possible destination state after the said action
                        record the achieved rewards
                average all the recorded rewards and assign the result to the value function of that state
        """
        old_values = self.value_function.copy()
        for state_index in range(25):
            rewards_accumulated = []
            for action_index in range(4):
                if state_index == 1:
                    rewards_accumulated.append(5 + self.discount*self.value_function[17])
                elif state_index == 4:
                    rewards_accumulated.append(0.5*(2.5 + self.discount*self.value_function[17]) +
                                               0.5*(2.5 + self.discount*self.value_function[24]))  # todo: check if this should be summed or should it be appended twice? it affects the denominator when calculating the mean of the rewards.
                elif state_index in [0, 5, 10, 15, 20] and action_index == 2:
                    rewards_accumulated.append(-0.5 + self.discount*self.value_function[state_index])
                elif state_index in [0, 2, 3] and action_index == 0:
                    rewards_accumulated.append(-0.5 + self.discount*self.value_function[state_index])
                elif state_index in [9, 14, 19, 24] and action_index == 3:
                    rewards_accumulated.append(-0.5 + self.discount*self.value_function[state_index])
                elif state_index in [20, 21, 22, 23, 24] and action_index == 1:
                    rewards_accumulated.append(-0.5 + self.discount*self.value_function[state_index])
                else:
                    if action_index == 0:
                        rewards_accumulated.append(self.discount * self.value_function[state_index - 5])
                    elif action_index == 1:
                        rewards_accumulated.append(self.discount * self.value_function[state_index + 5])
                    elif action_index == 2:
                        rewards_accumulated.append(self.discount * self.value_function[state_index - 1])
                    else:
                        rewards_accumulated.append(self.discount * self.value_function[state_index + 1])
            self.value_function[state_index] = np.array(rewards_accumulated).mean()
        stop = (np.abs(self.value_function - old_values) < threshold).any()
        return stop

    def iterative_policy_evaluation(self,  threshold=0.1, patience=1e5):
        self.value_function = np.random.normal(size=25)
        stop = False
        run_count = 0
        while not stop or run_count < patience:
            run_count += 1
            stop = self.mean_update_value_function(threshold=threshold)
        print(f"Iteration count at halt = {run_count}")
        return self.value_function

    def max_update_value_function(self, threshold=0.1):  # todo: is the only difference between value iteration and iterative policy evaluation mean vs max?
        old_values = self.value_function.copy()
        for state_index in range(25):
            rewards_accumulated = []
            for action_index in range(4):
                if state_index == 1:
                    rewards_accumulated.append(5 + self.discount*self.value_function[17])
                elif state_index == 4:
                    rewards_accumulated.append(0.5*(2.5 + self.discount*self.value_function[17]) +
                                               0.5*(2.5 + self.discount*self.value_function[24]))  # todo: check if this should be summed or should it be appended twice? it affects the denominator when calculating the mean of the rewards.
                elif state_index in [0, 5, 10, 15, 20] and action_index == 2:
                    rewards_accumulated.append(-0.5 + self.discount*self.value_function[state_index])
                elif state_index in [0, 2, 3] and action_index == 0:
                    rewards_accumulated.append(-0.5 + self.discount*self.value_function[state_index])
                elif state_index in [9, 14, 19, 24] and action_index == 3:
                    rewards_accumulated.append(-0.5 + self.discount*self.value_function[state_index])
                elif state_index in [20, 21, 22, 23, 24] and action_index == 1:
                    rewards_accumulated.append(-0.5 + self.discount*self.value_function[state_index])
                else:
                    if action_index == 0:
                        rewards_accumulated.append(self.discount * self.value_function[state_index - 5])
                    elif action_index == 1:
                        rewards_accumulated.append(self.discount * self.value_function[state_index + 5])
                    elif action_index == 2:
                        rewards_accumulated.append(self.discount * self.value_function[state_index - 1])
                    else:
                        rewards_accumulated.append(self.discount * self.value_function[state_index + 1])
            self.value_function[state_index] = np.array(rewards_accumulated).max()
        stop = (np.abs(self.value_function - old_values) < threshold).any()
        return stop

    def value_iteration(self, threshold=0.1, patience=1e5):  # todo: this returns values much different from the iterative policy evaluation. fix
        self.value_function = np.random.normal(size=25)
        stop = False
        run_count = 0
        while not stop or run_count < patience:
            run_count += 1
            stop = self.max_update_value_function(threshold=threshold)
        print(f"Iteration count at halt = {run_count}")
        return self.value_function


class PolicyAgent:
    def __init__(self, discount=0.95):
        self.discount = discount
        self.value_function = None

    def visualize_results(self):
        plt.figure()
        sns.heatmap(self.value_function.reshape((5,5)), cmap='coolwarm', annot=True, fmt='.2f', square=True)
        plt.title("5x5 Gridworld - State Value Function")
        plt.show()

    def explicit(self):
        pass

    def iterative_policy_evaluation(self):
        pass

    def value_iteration(self):
        pass

if __name__ == '__main__':
    agent = ValueAgent()

    # agent.explicit()
    # agent.iterative_policy_evaluation()
    agent.value_iteration()

    agent.visualize_results()
