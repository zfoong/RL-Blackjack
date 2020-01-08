import numpy as np
import random


class Agent_Q_Extend:
    """
    Agent Q with Q-learning with extra capability on Blackjack environment
    """
    def __init__(self, epsilon, learning_rate, discount_factor, epsilon_decay):
        self.epsilon = epsilon  # exploration factor
        self.max_epsilon = 1  # maximum exploration factor
        self.min_epsilon = 0 # minimum exploration factor
        self.epsilon_decay = epsilon_decay  # epsilon decay rate
        self.learning_rate = learning_rate  # learning rate
        self.discount_factor = discount_factor  # discount factor of next state
        self.possible_card_val = [i for i in range(2, 22)]
        self.possible_action = ['h', 's']
        self.possible_card_distribution = ['excessive low cards', 'more low cards', 'same amount',
                                           'more high cards', 'excessive high cards']
        self.Q_table = np.zeros((20, 5, 2))
        self.replay_buffer = []
        self.max_replay_buffer_length = 20
        self.num_replay_buffer_sampling = 5

    def print_Q_table(self):
        print(self.Q_table)

    def add_replay_buffer(self, state, action, reward, new_state, last):
        """
        Add current state, action, reward and new state to replay buffer
        """
        if len(self.replay_buffer) > self.max_replay_buffer_length:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, new_state, last))

    def flush_replay_buffer(self):
        """
        Clear replay buffer
        """
        self.replay_buffer.clear()

    def train_replay_buffer(self):
        """
        Random sampling from replay buffer and retrain them
        this help agent to train faster
        """
        if len(self.replay_buffer) > self.num_replay_buffer_sampling:
            replay_buffer = random.sample(self.replay_buffer, self.num_replay_buffer_sampling)
            for i in replay_buffer:
                state, action, reward, new_state, last = i
                self.update_Q_table(state, action, reward, new_state, last)

    def unpack_state_to_index(self, state):
        """
        turning state value to numpy index
        """
        card_val, card_distribution = state
        return self.possible_card_val.index(card_val), self.possible_card_distribution.index(card_distribution)

    def update_Q_table(self, state, action, reward, new_state, last=False):
        """
        Update state-action value pair in Q-table
        Since drawing a new set of card and starting a new round is not related to its previous state
        Therefore, using last flag to control whether to propagate state-action value back to previous state
        This approach help to reduce irrelevant info / noise when updating state-action value
        """
        val_index, dis_index = self.unpack_state_to_index(state)
        new_val_index, new_dis_index = self.unpack_state_to_index(new_state)
        action_id = self.possible_action.index(action)
        if last is False:  # update Q-table state-action pair
            self.Q_table[val_index, dis_index, action_id] += self.learning_rate * (
                        reward + self.discount_factor * np.max(self.Q_table[new_val_index, new_dis_index, :]) -
                        self.Q_table[val_index, dis_index, action_id])
        else:
            self.Q_table[val_index, dis_index, action_id] += self.learning_rate * reward

    def return_action(self, state):
        val_index, dis_index = self.unpack_state_to_index(state)
        if self.epsilon > random.uniform(0, 1):
            return random.choice(self.possible_action)
        else:
            return self.possible_action[np.argmax(self.Q_table[val_index, dis_index, :])]

    def update_epsilon_decay(self, episode):
        """
        Decrease exploration factor each episode
        """
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay * episode)

    def save_Q_table(self, path):
        """
        Save Q-table into numpy array
        """
        np.save(path, self.Q_table)

    def load_Q_table(self, path):
        """
        Load Q-table from numpy array
        """
        self.Q_table = np.load(path)
