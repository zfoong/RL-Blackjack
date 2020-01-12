"""
Author : Tham Yik Foong
Student ID : 20200786
Project title : Playing Stylised Blackjack with Q-learning

Academic integrity statement :
I, Tham Yik Foong, have read and understood the School's Academic Integrity Policy, as well as guidance relating to
this module, and confirm that this submission complies with the policy. The content of this file is my own original
work, with any significant material copied or adapted from other sources clearly indicated and attributed.
"""

import numpy as np
import random


class Agent_Q:
    """
    Agent Q with basic Q-learning capability on Blackjack environment
    """
    def __init__(self, epsilon, learning_rate, discount_factor, epsilon_decay):
        self.epsilon = epsilon
        self.max_epsilon = 1
        self.min_epsilon = 0
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.possible_card_val = [i for i in range(2, 22)]
        self.possible_action = ['h', 's']
        self.possible_card_distribution = ['excessive low cards', 'more low cards', 'same amount',
                                           'more high cards', 'excessive high cards']
        self.Q_table = np.zeros((20, 5, 2))

    def print_Q_table(self):
        print(self.Q_table)

    def unpack_state_to_index(self, state):
        card_val, card_distribution = state
        return self.possible_card_val.index(card_val), self.possible_card_distribution.index(card_distribution)

    def update_Q_table(self, state, action, reward, new_state):
        val_index, dis_index = self.unpack_state_to_index(state)
        new_val_index, new_dis_index = self.unpack_state_to_index(new_state)
        action_id = self.possible_action.index(action)
        self.Q_table[val_index, dis_index, action_id] += self.learning_rate * (
                        reward + self.discount_factor * np.max(self.Q_table[new_val_index, new_dis_index, :]) -
                        self.Q_table[val_index, dis_index, action_id])

    def return_action(self, state):
        val_index, dis_index = self.unpack_state_to_index(state)
        if self.epsilon > random.uniform(0, 1):
            return random.choice(self.possible_action)
        else:
            return self.possible_action[np.argmax(self.Q_table[val_index, dis_index, :])]

    def update_epsilon_decay(self, episode, total_episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay * episode / total_episode)

    def save_Q_table(self, path):
        np.save(path, self.Q_table)

    def load_Q_table(self, path):
        self.Q_table = np.load(path)
