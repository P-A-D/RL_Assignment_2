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


def visualize_results(vector, title):
    plt.figure()
    sns.heatmap(vector.reshape((5, 5)), cmap='coolwarm', annot=True, fmt='.2f', square=True)
    plt.title(f"5x5 Gridworld - {title}")
    plt.show()


# =====================================================================================================
def find_next_state(state, action):
    """
    returns a pair of (next state, generated reward)
    """
    if state == 1:
        return 22, 5
    elif state == 4:
        return np.random.choice([22, 24]), 2.5
    elif action == 0:
        return (state - 5, -0.2) if state - 5 >= 0 else (state, -0.5)
    elif action == 1:
        return (state + 5, -0.2) if state + 5 < 25 else (state, -0.5)
    elif action == 2:
        return (state - 1, -0.2) if state % 5 - 1 >= 0 else (state, -0.5)
    else:
        return (state + 1, -0.2) if (state + 1) % 5 != 0 else (state, -0.5)


class MonteCarloES:
    def __init__(self, discount=0.95):
        self.discount = discount
        self.policy = np.random.randint(low=0, high=4, size=25)
        self.state_action_values = np.random.normal(size=(25, 4))
        self.state_action_visits = np.zeros((25, 4))
        self.returns = []

    def learn(self, discount=0.95, episodes=10000, patience=1000):
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
                if next_state in [14, 20] or steps > patience:
                    break
                state_sequence.append(next_state)
                action_sequence.append(self.policy[next_state])
            # evaluating Q(s,a)
            g = 0
            for i in range(len(reward_sequence)-1, -1, -1):
                g = g*discount + reward_sequence[i]
                self.state_action_visits[state_sequence[i], action_sequence[i]] += 1
                self.state_action_values[state_sequence[i], action_sequence[i]] += 1/self.state_action_visits[state_sequence[i], action_sequence[i]] * (g - self.state_action_values[state_sequence[i], action_sequence[i]])
            self.policy = np.argmax(self.state_action_values, axis=1)
        print(self.policy)


# =====================================================================================================
if __name__ == '__main__':
    agent = MonteCarloES()
    agent.learn()
