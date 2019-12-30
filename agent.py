import numpy as np
import random

class Agent:
    def __init__(self, epsilon, learning_rate, discount_factor):
        self.epsilon = epsilon
        self.max_epsilon = 1
        self.min_epsilon = 0
        self.epsilon_decay = 0.01
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.Q_table = np.zeros((20, 2))

    def print_Q_table(self):
        print(self.Q_table)

    def update_Q_table(self, state, action, reward, new_state):
        self.Q_table[state, action] = self.Q_table[state, action] * (1 - self.learning_rate) + self.learning_rate * (reward + self.discount_factor * np.max(self.Q_table[new_state, :]))

    def return_action(self, state):
        if self.epsilon > random.uniform(0, 1):
            return random.choice([0, 1])
        else:
            return np.argmax(self.Q_table[state, :])

    def update_epsilon_decay(self, episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay * episode)
