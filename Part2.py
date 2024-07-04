import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


"""
# Notes
1. discount (gamma) = 0.95
2. policy: probability of moving in any of the four directions = 0.25
3. Any action at blue => +5 reward => jump to 17
4. Any action at green => +2.5 reward => jump to 17 or 24 square with probability 0.5
5. States 14 and 20 are terminal states
6. Any action taken between white squares has a award of -0.2

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


    State numbers:
    0   1   2   3   4
    5   6   7   8   9
    10  11  12  13  14
    15  16  17  18  19
    20  21  22  23  24
"""


def visualize_results(vector, title):
    plt.figure()
    sns.heatmap(vector.reshape((5, 5)), cmap='coolwarm', annot=True, fmt='.2f', square=True)
    plt.title(f"5x5 Gridworld - {title}")
    plt.show()


class ExplicitAgent:
    def __init__(self, discount=0.95):
        self.discount = discount
        self.value_function = None

    def explicit(self, g=0.95):  # todo: the outputs are different from the other methods. check if this is fine.
        """
        By manually choosing the best possible action in some of the states, we are left with about 1.5 million
        possible equations due to not knowing which action in each state yields the best values. On those 1.5 million
        equations, one of them satisfies the Bellman equations and is the optimal value function.
        Regardless, I brute-forced it and luckily it doesn't take too much time to run.
        """
        g = g
        def fsolve_function(p, index_list):
            v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25 = p
            # equations with these numbers have to be tested: 4, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25
            eqs1 = g * v2 - v1
            eqs2 = 5 + g * v18 - v2
            eqs3 = g * v2 - v3
            eqs4 = [g * v3 - v4, g * v5 - v4]  ###
            eqs5 = 0.5 * (2.5 + g * v18) + 0.5 * (2.5 + g * v25) - v5
            eqs6 = [g * v1 - v6, g * v7 - v6]  ###
            eqs7 = g * v2 - v7
            eqs8 = [g * v3 - v8, g * v7 - v8]  ###
            eqs9 = [g * v4 - v9, g * v8 - v9, g * v10 - v9]  ###
            eqs10 = [g * v5 - v10, g * v9 - v10]  ###
            eqs11 = [g * v6 - v11, g * v12 - v11]  ###
            eqs12 = g * v7 - v12
            eqs13 = [g * v8 - v13, g * v12 - v13]  ###
            eqs14 = [g * v9 - v14, g * v13 - v14, g * v15 - v14]  ###
            eqs15 = [g * v10 - v15, g * v14 - v15]  ###
            eqs16 = [g * v11 - v16, g * v17 - v16]  ###
            eqs17 = g * v12 - v17
            eqs18 = [g * v13 - v18, g * v17 - v18, g * v19 - v18]  ###
            eqs19 = [g * v14 - v19, g * v18 - v19, g * v20 - v19]  ###
            eqs20 = [g * v15-v20, g * v19-v20]  ###
            eqs21 = [g * v16-v21, g * v22-v21]  ###
            eqs22 = g * v17 - v22
            eqs23 = [g * v18 - v23, g * v22 - v23, g * v24 - v23]  ###
            eqs24 = [g * v19 - v24, g * v23 - v24, g * v25 - v24]  ###
            eqs25 = [g * v20 - v25, g * v24 - v25]  ###
            return (eqs1, eqs2, eqs3, eqs4[index_list[0]], eqs5, eqs6[index_list[1]], eqs7, eqs8[index_list[2]],
                    eqs9[index_list[3]], eqs10[index_list[4]], eqs11[index_list[5]], eqs12, eqs13[index_list[6]],
                    eqs14[index_list[7]], eqs15[index_list[8]], eqs16[index_list[9]], eqs17, eqs18[index_list[10]],
                    eqs19[index_list[11]], eqs20[index_list[12]], eqs21[index_list[13]], eqs22, eqs23[index_list[14]],
                    eqs24[index_list[15]], eqs25[index_list[16]])

        def check_validity(max_indices):
            v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25 = self.value_function
            # Max indices: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            c0 = [g * v3 - v4, g * v5 - v4]
            c1 = [g * v1 - v6, g * v7 - v6]
            c2 = [g * v3 - v8, g * v7 - v8]
            c3 = [g * v4 - v9, g * v8 - v9, g * v10 - v9]
            c4 = [g * v5 - v10, g * v9 - v10]
            c5 = [g * v6 - v11, g * v12 - v11]
            c6 = [g * v8 - v13, g * v12 - v13]
            c7 = [g * v9 - v14, g * v13 - v14, g * v15 - v14]
            c8 = [g * v10 - v15, g * v14 - v15]
            c9 = [g * v11 - v16, g * v17 - v16]
            c10 = [g * v13 - v18, g * v17 - v18, g * v19 - v18]
            c11 = [g * v14 - v19, g * v18 - v19, g * v20 - v19]
            c12 = [g * v15-v20, g * v19-v20]
            c13 = [g * v16-v21, g * v22-v21]
            c14 = [g * v18 - v23, g * v22 - v23, g * v24 - v23]
            c15 = [g * v19 - v24, g * v23 - v24, g * v25 - v24]
            c16 = [g * v20 - v25, g * v24 - v25]
            if (c0[max_indices[0]] < np.array(c0)).any():
                return False
            if (c1[max_indices[1]] < np.array(c1)).any():
                return False
            if (c2[max_indices[2]] < np.array(c2)).any():
                return False
            if (c3[max_indices[3]] < np.array(c3)).any():
                return False
            if (c4[max_indices[4]] < np.array(c4)).any():
                return False
            if (c5[max_indices[5]] < np.array(c5)).any():
                return False
            if (c6[max_indices[6]] < np.array(c6)).any():
                return False
            if (c7[max_indices[7]] < np.array(c7)).any():
                return False
            if (c8[max_indices[8]] < np.array(c8)).any():
                return False
            if (c9[max_indices[9]] < np.array(c9)).any():
                return False
            if (c10[max_indices[10]] < np.array(c10)).any():
                return False
            if (c11[max_indices[11]] < np.array(c11)).any():
                return False
            if (c12[max_indices[12]] < np.array(c12)).any():
                return False
            if (c13[max_indices[13]] < np.array(c13)).any():
                return False
            if (c14[max_indices[14]] < np.array(c14)).any():
                return False
            if (c15[max_indices[15]] < np.array(c15)).any():
                return False
            if (c16[max_indices[16]] < np.array(c16)).any():
                return False
            return True


        def run():
            # equations with these numbers have to be tested: 4, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25
            indices = [2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2]
            for index_eq4 in range(indices[0]):
                for index_eq6 in range(indices[1]):
                    for index_eq8 in range(indices[2]):
                        for index_eq9 in range(indices[3]):
                            for index_eq10 in range(indices[4]):
                                for index_eq11 in range(indices[5]):
                                    for index_eq13 in range(indices[6]):
                                        for index_eq14 in range(indices[7]):
                                            for index_eq15 in range(indices[8]):
                                                for index_eq16 in range(indices[9]):
                                                    for index_eq18 in range(indices[10]):
                                                        for index_eq19 in range(indices[11]):
                                                            for index_eq20 in range(indices[12]):
                                                                for index_eq21 in range(indices[13]):
                                                                    for index_eq23 in range(indices[14]):
                                                                        for index_eq24 in range(indices[15]):
                                                                            for index_eq25 in range(indices[16]):
                                                                                asd = [index_eq4, index_eq6, index_eq8, index_eq9, index_eq10, index_eq11, index_eq13, index_eq14, index_eq15, index_eq16, index_eq18, index_eq19, index_eq20, index_eq21, index_eq23, index_eq24, index_eq25]
                                                                                self.value_function = fsolve(
                                                                                    func=fsolve_function, args=(asd,), x0=np.zeros(25))
                                                                                if check_validity(asd):
                                                                                    # print("The valid result found.")
                                                                                    return asd

        max_indices = run()
        # print(f"Max indices: {max_indices}")
        return self.value_function

    def calculate_policy_estimation(self):
        self.policy_function = []
        for state_index in range(25):
            expected_rewards = []
            for action_index in range(4):
                if state_index == 1:
                    expected_rewards.append(5 + self.discount * self.value_function[17])
                elif state_index == 4:
                    expected_rewards.append(0.5 * (2.5 + self.discount * self.value_function[17]) +
                                            0.5 * (2.5 + self.discount * self.value_function[24]))  # todo: check if this should be summed or should it be appended twice? it affects the denominator when calculating the mean of the rewards.
                elif state_index in [0, 5, 10, 15, 20] and action_index == 2:
                    expected_rewards.append(-0.5 + self.discount * self.value_function[state_index])
                elif state_index in [0, 2, 3] and action_index == 0:
                    expected_rewards.append(-0.5 + self.discount * self.value_function[state_index])
                elif state_index in [9, 14, 19, 24] and action_index == 3:
                    expected_rewards.append(-0.5 + self.discount * self.value_function[state_index])
                elif state_index in [20, 21, 22, 23, 24] and action_index == 1:
                    expected_rewards.append(-0.5 + self.discount * self.value_function[state_index])
                else:
                    if action_index == 0:
                        expected_rewards.append(self.discount * self.value_function[state_index - 5])
                    elif action_index == 1:
                        expected_rewards.append(self.discount * self.value_function[state_index + 5])
                    elif action_index == 2:
                        expected_rewards.append(self.discount * self.value_function[state_index - 1])
                    else:
                        expected_rewards.append(self.discount * self.value_function[state_index + 1])
            self.policy_function.append(np.array(expected_rewards).argmax())
        return self.policy_function
# ================================================================================================================