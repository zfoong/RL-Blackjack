import numpy as np
import random
import pandas as pd


class Agent_Q:
    def __init__(self, epsilon, learning_rate, discount_factor, epsilon_decay):
        self.epsilon = epsilon
        self.max_epsilon = 1
        self.min_epsilon = 0.05
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        possible_state = [i for i in range(2, 22)]
        possible_action = ['h', 's']
        self.Q_table = pd.DataFrame(np.zeros((20, 2)), columns=possible_action, index=possible_state)

    def print_Q_table(self):
        print(self.Q_table)

    def update_Q_table(self, state, action, reward, new_state):
        self.Q_table[action][state] = self.Q_table[action][state] * (1 - self.learning_rate) + self.learning_rate * (reward + self.discount_factor * max(self.Q_table.loc[new_state]))

    def return_action(self, state):
        if self.epsilon > random.uniform(0, 1):
            return random.choice(['h', 's'])
        else:
            return self.Q_table.loc[state, :].idxmax(1)

    def update_epsilon_decay(self, episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay * episode)

    def save_Q_table(self, path):
        self.Q_table.to_pickle(path)

    def load_Q_table(self, path):
        self.Q_table = pd.read_pickle(path)
