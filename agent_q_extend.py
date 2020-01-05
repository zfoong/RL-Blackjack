import numpy as np
import random
import pandas as pd


class Agent_Q_extend:
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
        self.replay_buffer = []

    def print_Q_table(self):
        print(self.Q_table)

    def add_replay_buffer(self, state, action, reward, new_state):
        self.replay_buffer.append((state, action, reward, new_state))

    def flush_replay_buffer(self):
        self.replay_buffer.clear()

    def train_replay_buffer(self):
        last = True
        for i in reversed(self.replay_buffer):
            state, action, reward, new_state = i
            self.update_Q_table(state, action, reward, new_state, last)
            last = False
        self.flush_replay_buffer()

    def update_Q_table(self, state, action, reward, new_state, last=False):
        if last is False:
            self.Q_table[action][state] += self.learning_rate * (reward + self.discount_factor * max(self.Q_table.loc[new_state]) - self.Q_table[action][state])
        else:
            self.Q_table[action][state] += self.learning_rate * reward

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
