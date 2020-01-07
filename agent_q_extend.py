import numpy as np
import random


class Agent_Q_extend:
    def __init__(self, epsilon, learning_rate, discount_factor, epsilon_decay):
        self.epsilon = epsilon  # exploration factor
        self.max_epsilon = 1  # maximum exploration factor
        self.min_epsilon = 0  # minimum exploration factor
        self.epsilon_decay = epsilon_decay  # epsilon decay rate
        self.learning_rate = learning_rate  # learning rate
        self.discount_factor = discount_factor  # discount factor of next state
        self.possible_card_val = [i for i in range(2, 22)]
        self.possible_action = ['h', 's']
        self.possible_card_distribution = ['excessive low cards', 'more low cards', 'same amount',
                                           'more high cards', 'excessive high cards']
        self.Q_table = np.zeros((20, 5, 2))  # Q table that keep track of each state-action pair
        # dividing episode to sub episode (from new set of cards is drawn to user perform 'stick' or cards value over
        # 21) and store them into a buffer. When sub episode ended, agent train on the buffer and propagate the reward
        # from end state to all previous states that lead to the reward.
        self.sub_episode_buffer = []

    def print_Q_table(self):
        print(self.Q_table)

    def add_sub_episode_buffer(self, state, action, reward, new_state):
        """
        Add current state, action, reward and enw state to sub episode buffer
        """
        self.sub_episode_buffer.append((state, action, reward, new_state))

    def flush_sub_episode_buffer(self):
        """
        Clear sub episode buffer
        """
        self.sub_episode_buffer.clear()

    def train_sub_episode_buffer(self):
        """
        when sub episode ended, train the buffer by updating the Q-table in a reversed manner.
        Therefore propagating the reward at end state to the previous state.
        """
        last = True
        for i in reversed(self.sub_episode_buffer):
            state, action, reward, new_state = i
            self.update_Q_table(state, action, reward, new_state, last)
            last = False
        self.flush_sub_episode_buffer()  # clear buffer after training on sub-episode is completed

    def unpack_state_to_index(self, state):
        """
        turning state value to numpy index
        """
        card_val, card_distribution = state
        return self.possible_card_val.index(card_val), self.possible_card_distribution.index(card_distribution)

    def update_Q_table(self, state, action, reward, new_state, last=False):
        val_index, dis_index = self.unpack_state_to_index(state)
        new_val_index, new_dis_index = self.unpack_state_to_index(new_state)
        action_id = self.possible_action.index(action)
        if last is False:  # update Q-table state-action pair
            self.Q_table[val_index, dis_index, action_id] += self.learning_rate * (reward + self.discount_factor * np.max(self.Q_table[new_val_index, new_dis_index, :]) - self.Q_table[val_index, dis_index, action_id])
        else:
            self.Q_table[val_index, dis_index, action_id] += self.learning_rate * reward

    def return_action(self, state):
        val_index, dis_index = self.unpack_state_to_index(state)
        if self.epsilon > random.uniform(0, 1):  # epsilon-greedy policy
            return random.choice(self.possible_action)  # explore by randomly choosing from possible action
        else:
            return self.possible_action[np.argmax(self.Q_table[val_index, dis_index, :])]  # exploit with greedy policy

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
