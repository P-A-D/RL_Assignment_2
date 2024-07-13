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

    State numbers:
    0   1   2   3   4
    5   6   7   8   9
    10  11  12  13  14
    15  16  17  18  19
    20  21  22  23  24
"""
# =====================================================================================================


def visualize_results(vector, title):
    """
    Plots the value functions.
    """
    plt.figure()
    sns.heatmap(vector.reshape((5, 5)), cmap='coolwarm', annot=True, fmt='.2f', square=True)
    plt.title(f"Question 2 - {title}")
    plt.show()


def select_epsilon_action(action, epsilon):
    """
    Returns an action to take in an epsilon-greedy policy.
    """
    return action if np.random.random() < epsilon else np.random.choice(range(4))


def find_next_state(state, action):
    """
    returns a pair of (next state, generated reward) given the current state and action.
    """
    if state == 1:
        return 22, 5
    elif state == 4:
        return np.random.choice([22, 24]), 2.5
    elif state in [14, 20]:
        return state, 0
    elif action == 0:
        if state - 5 == 14:
            return 14, 0
        return (state - 5, -0.2) if state - 5 >= 0 else (state, -0.5)
    elif action == 1:
        if state + 5 in [14, 20]:
            return state+5, 0
        return (state + 5, -0.2) if state + 5 < 25 else (state, -0.5)
    elif action == 2:
        if state - 1 == 20:
            return 20, 0
        return (state - 1, -0.2) if state % 5 - 1 >= 0 else (state, -0.5)
    else:
        if state + 1 == 14:
            return 14, 0
        return (state + 1, -0.2) if (state + 1) % 5 != 0 else (state, -0.5)


# ================================================================================================================
# ============================================== Section 1 =======================================================
# ================================================================================================================
class MonteCarloES:
    """
    The class for finding the optimal policy using Monte Carlo method with exploring starts.
    """
    def __init__(self, discount=0.95):
        self.discount = discount
        self.policy = np.random.randint(low=0, high=4, size=25)
        self.state_action_values = np.random.normal(size=(25, 4))
        self.state_action_visits = np.zeros((25, 4))
        self.returns = []

    def learn(self, discount=0.95, episodes=100000, max_episode_length=2500):
        for _ in range(episodes):
            initial_state = np.random.choice(range(25))
            initial_action = np.random.choice(range(4))
            state_sequence = [initial_state]
            action_sequence = [initial_action]
            reward_sequence = []
            steps = 0
            # generating a sequence
            while True:
                steps += 1
                next_state, reward = find_next_state(state_sequence[-1], action_sequence[-1])
                reward_sequence.append(reward)
                if next_state in [14, 20] or steps > max_episode_length:
                    break
                state_sequence.append(next_state)
                action_sequence.append(self.policy[next_state])

            # making the pairs
            pairs = list(zip(state_sequence, action_sequence))

            # evaluating Q(s,a)
            g = 0
            for i in range(len(reward_sequence)-1, -1, -1):
                g = g*discount + reward_sequence[i]
                if (state_sequence[i], action_sequence[i]) not in pairs[:i]:
                    self.state_action_visits[state_sequence[i], action_sequence[i]] += 1
                    self.state_action_values[state_sequence[i], action_sequence[i]] += 1/self.state_action_visits[state_sequence[i], action_sequence[i]] * (g - self.state_action_values[state_sequence[i], action_sequence[i]])
                    self.policy[state_sequence[i]] = np.argmax(self.state_action_values[state_sequence[i], :])
        print(f"The best policy found is:")
        print(np.array(self.policy).reshape((5, 5)))
# =====================================================================================================


class MonteCarloESoft:
    """
    The class for finding the optimal policy using Monte Carlo method using epsilon greedy, without exploring starts.
    """
    def __init__(self, discount=0.95):
        self.discount = discount
        self.policy = np.random.randint(low=0, high=4, size=25)
        self.state_action_values = np.random.normal(size=(25, 4))
        self.state_action_visits = np.zeros((25, 4))
        self.returns = []

    def learn(self, discount=0.95, episodes=10**6, max_episode_length=10000, epsilon=0.1):
        for _ in range(episodes):
            initial_state = np.random.choice(range(25))
            state_sequence = [initial_state]
            action_sequence = [select_epsilon_action(self.policy[initial_state], epsilon)]
            reward_sequence = []
            steps = 0

            # generating a sequence
            while True:
                steps += 1
                next_state, reward = find_next_state(state_sequence[-1], action_sequence[-1])
                reward_sequence.append(reward)
                if next_state in [14, 20] or steps > max_episode_length:
                    break
                state_sequence.append(next_state)
                action_sequence.append(select_epsilon_action(self.policy[next_state], epsilon))

            # making the pairs
            pairs = list(zip(state_sequence, action_sequence))

            # evaluating Q(s,a)
            g = 0
            for i in range(len(reward_sequence)-1, -1, -1):
                g = g*discount + reward_sequence[i]
                if (state_sequence[i], action_sequence[i]) not in pairs[:i]:
                    self.state_action_visits[state_sequence[i], action_sequence[i]] += 1
                    self.state_action_values[state_sequence[i], action_sequence[i]] += 1/self.state_action_visits[state_sequence[i], action_sequence[i]] * (g - self.state_action_values[state_sequence[i], action_sequence[i]])
                    self.policy[state_sequence[i]] = np.argmax(self.state_action_values[state_sequence[i], :])
        print(f"The best policy found is:")
        print(np.array(self.policy).reshape((5, 5)))


# ================================================================================================================
# ============================================== Section 2 =======================================================
# ================================================================================================================
class Behavior:
    """
        The class for finding the optimal policy using off policy Monte Carlo method (using behavior policy)
    """
    def __init__(self):
        self.behavior_policy = lambda _: np.random.choice(range(4))
        self.state_action_values = np.random.normal(size=(25, 4))
        self.state_action_csums = np.zeros((25, 4))  # C(s, a) matrix
        self.target_policy = np.argmax(self.state_action_values, axis=1)

    def learn(self, discount=0.95, episodes=10**8, max_episode_length=10000):
        for _ in range(episodes):
            # generating a sequence
            initial_state = np.random.choice(range(25))
            state_sequence = [initial_state]
            action_sequence = [self.behavior_policy(None)]
            reward_sequence = []
            steps = 0
            while True:
                steps += 1
                next_state, reward = find_next_state(state_sequence[-1], action_sequence[-1])
                reward_sequence.append(reward)
                if next_state in [14, 20] or steps > max_episode_length:
                    break
                state_sequence.append(next_state)
                action_sequence.append(self.behavior_policy(next_state))

            # evaluating Q(s,a)
            g = 0
            w = 1
            for i in range(len(reward_sequence)-1, -1, -1):
                g = g*discount + reward_sequence[i]
                self.state_action_csums[state_sequence[i], action_sequence[i]] += w
                self.state_action_values[state_sequence[i], action_sequence[i]] += w/self.state_action_csums[state_sequence[i], action_sequence[i]] * (g - self.state_action_values[state_sequence[i], action_sequence[i]])
                self.target_policy[state_sequence[i]] = np.argmax(self.state_action_values[state_sequence[i], :])
                if self.target_policy[state_sequence[i]] != action_sequence[i]:
                    break
                w /= 0.25
        print(f"The best policy found is:")
        print(np.array(self.target_policy).reshape((5, 5)))


# ================================================================================================================
# ============================================== Section 3 =======================================================
# ================================================================================================================

class PolicyIteration:  # todo: think more about the open-ended part of this question
    """
        The class for finding the optimal policy using policy iteration.
    """
    def __init__(self, discount=0.95, permute=True):
        self.discount = discount
        self.value_function = None
        self.policy = None
        self.five_location = 1
        self.half_five_location = 4
        self.permute = permute

    def find_new_v(self, old_state, next_state, reward):
        if old_state == self.half_five_location:
            return 0.5*(2.5 + self.discount * self.value_function[24]) + 0.5*(2.5 + self.discount * self.value_function[22])
        elif old_state == self.five_location:
            return reward + self.discount * self.value_function[22]
        else:
            return reward + self.discount * self.value_function[next_state]

    def evaluate_policy(self, threshold=0.1):
        def eval_states():
            old_values = self.value_function.copy()
            for state_index in range(25):
                if state_index in [14, 20]:
                    self.value_function[state_index] = 0
                    continue
                next_state, reward = find_next_state(state_index, self.policy[state_index])
                self.value_function[state_index] = self.find_new_v(state_index, next_state, reward)
            stop = (np.abs(np.delete(self.value_function, [14, 20]) - np.delete(old_values, [14, 20])) < threshold).any()
            return stop
        stop = False
        run_count = 0
        patience = 1e5
        while not stop or run_count < patience:
            run_count += 1
            stop = eval_states()

    def improve_policy(self):
        for state_index in range(25):
            if state_index == 0:
                self.policy[state_index] = np.argmax([-np.inf, self.value_function[5], -np.inf, self.value_function[1]])
            elif state_index == 4:
                self.policy[state_index] = np.argmax([-np.inf, self.value_function[9], self.value_function[3], -np.inf])
            elif state_index == 20:
                self.policy[state_index] = np.argmax([self.value_function[15], -np.inf, -np.inf, self.value_function[21]])
            elif state_index == 24:
                self.policy[state_index] = np.argmax([self.value_function[19], -np.inf, self.value_function[23], -np.inf])
            elif state_index in [5, 10, 15]:
                self.policy[state_index] = np.argmax([self.value_function[state_index - 5],
                                                      self.value_function[state_index + 5],
                                                      -np.inf,
                                                      self.value_function[state_index + 1]])
            elif state_index in [1, 2, 3]:
                self.policy[state_index] = np.argmax([-np.inf,
                                                      self.value_function[state_index + 5],
                                                      self.value_function[state_index - 1],
                                                      self.value_function[state_index + 1]])
            elif state_index in [21, 22, 23]:
                self.policy[state_index] = np.argmax([self.value_function[state_index - 5],
                                                      -np.inf,
                                                      self.value_function[state_index - 1],
                                                      self.value_function[state_index + 1],
                                                      ])
            elif state_index in [9, 14, 19]:
                self.policy[state_index] = np.argmax([self.value_function[state_index - 5],
                                                      self.value_function[state_index + 5],
                                                      self.value_function[state_index - 1],
                                                      -np.inf])
            else:
                self.policy[state_index] = np.argmax([self.value_function[state_index - 5],
                                                      self.value_function[state_index + 5],
                                                      self.value_function[state_index - 1],
                                                      self.value_function[state_index + 1]])
        # return policy

    def estimate_optimal_policy(self, patience=1000, reach_patience=True):
        self.policy = np.zeros(25)
        self.value_function = np.ones(25)
        counter = 0
        while True and counter < patience:
            if self.permute:
                if np.random.uniform() < 0.1:
                    self.five_location, self.half_five_location = self.half_five_location, self.five_location
            self.evaluate_policy()
            previous_policy = self.policy.copy()
            self.improve_policy()
            if (previous_policy == self.policy).all() and not reach_patience:
                # The "not reach_patience" section is to make sure the algorithm faces the permutation before the
                # policy convergence, so we can observe its effects.
                break
            counter += 1
        print(f"Counter at termination = {counter}")
        return self.value_function


# =====================================================================================================
if __name__ == '__main__':
    # agent = MonteCarloES()
    # agent.learn()

    # agent = MonteCarloESoft()
    # agent.learn()

    # agent = Behavior()
    # agent.learn()

    # agent = PolicyIteration(permute=False)
    # agent.estimate_optimal_policy()
    # visualize_results(agent.value_function, "Policy Iteration")
    # print(f"optimal value function = {agent.value_function}")
    # print(f"optimal policy = {agent.policy}")

    # todo: check the results of monte carlo methods with others and prof.
    a = [1, 2, 3, 10, 4, 6]
    print(a)
    print(np.array(a).reshape((2, 3)))

