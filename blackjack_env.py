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


class Blackjack:
    def __init__(self, deck_number=1, difference_threshold=4, seed=0):
        self.episode_id = 1
        self.deck_number = deck_number  # number of deck use in each episode
        self.num_low_cards = 0  # number of low value cards 'ace', 'two', 'three', 'four', 'five', 'six' appeared
        self.num_high_cards = 0  # number of high value cards 'seven', 'eight', 'nine',
                                 # 'ten', 'jack', 'queen', 'king' appeared
        self.difference_threshold = difference_threshold  # threshold used to check distribution of low and high cards
        self.seed = seed  # seed of deck shuffling, providing same seed allowed same permutation of shuffled deck
        self.total_reward = 0
        self.suits = ['club', 'diamond', 'heart', 'spade']
        self.ranks = ['ace', 'two', 'three', 'four', 'five', 'six', 'seven',
                      'eight', 'nine', 'ten', 'jack', 'queen', 'king']
        self.poker_card = self.create_a_deck(False)  # create a deck of poker cards with 52 cards sorted in order
        self.deck = self.create_D_deck(self.poker_card, self.deck_number)  # create D deck of shuffled cards
        self.reward_dict = self.card_value_dict()  # define reward for each card
        self.player_hands = [self.pop_card(), self.pop_card()]  # initiate cards in players hand

    def shuffle_deck(self, deck):
        """
        shuffle D deck
        """
        np.random.seed(self.episode_id + self.seed)
        np.random.shuffle(deck)
        return deck

    def create_a_deck(self, shuffle=True):
        """
        Create a deck of poker cards with 52 cards sorted in order
        """
        deck = []
        for i in self.ranks:
            for j in self.suits:
                deck.append("{} {}".format(i, j))
        if shuffle is True:
            deck = self.shuffle_deck(deck)
        return deck

    def create_D_deck(self, poker_card, d=1):
        """
        # create D deck of shuffled cards
        """
        deck = []
        for i in range(d):
            deck = deck + poker_card
        deck = self.shuffle_deck(deck)
        return deck

    def card_value_dict(self):
        """
        define reward for each card
        """
        cards = {}
        for index, i in enumerate(self.ranks):
            for j in self.suits:
                if index is 0:
                    cards["{} {}".format(i, j)] = 11
                elif index < 9:
                    cards["{} {}".format(i, j)] = index + 1
                else:
                    cards["{} {}".format(i, j)] = 10
        return cards

    def calculate_deck_total_value(self, deck, reward_dict):
        """
        Calculate and return total card value of a deck
        """
        total_val = 0
        aces = [l for l in deck if "ace" in l]  # retrieve ace in deck
        for index, k in enumerate(deck):
            total_val += reward_dict[deck[index]]
        if total_val > 21 and len(aces) > 0:  # if card value is over 21, change all value of aces in deck to 1
            total_val -= len(aces) * 10
        return total_val

    def calculate_reward(self, deck, reward_dict, punishment=0):
        """
        Calculate reward based on cards value
        """
        deck_value = self.calculate_deck_total_value(deck, reward_dict)
        if deck_value > 21:
            return punishment
        else:
            return pow(deck_value, 2)

    def pop_card(self):
        """
        pop a card from deck and record the number of occurrence of low value card or high value card
        based on the card poped either being a low value card ('ace', 'two', 'three', 'four', 'five', 'six')
        or high value card ('seven', 'eight', 'nine', 'ten', 'jack', 'queen', 'king')
        """
        card = self.deck.pop(0)
        if card in self.poker_card[0:26]:
            self.num_low_cards += 1
        else:
            self.num_high_cards += 1
        return card

    def calculate_deck_current_distribution(self):
        """
        Calculate distribution of deck by comparing number of occurrence of low value cards and high value cards.
        Provide distribution of cards as state allowed agent to keep track of cards occurrence without having to
        keep track of every cards' occurrence, which will required enormous amount of state and memory,
        causing model to be harder to train.
        difference_threshold allow us to define a range for certain categories below
        """
        if self.num_low_cards > self.num_high_cards + self.difference_threshold * 2:
            return 'excessive low cards'
        elif self.num_low_cards > self.num_high_cards + self.difference_threshold:
            return 'more low cards'
        elif self.num_low_cards + self.difference_threshold * 2 < self.num_high_cards:
            return 'excessive high cards'
        elif self.num_low_cards + self.difference_threshold < self.num_high_cards:
            return 'more high cards'
        else:
            return 'same amount'

    def step(self, action):
        """
        Taking user action and return new state
        :param action: ['h', 's']
        :return: new state, reward, deck finished, new round (either user performs 'stick' or cards is over 21)
        """
        current_reward = 0
        is_done = False  # True if deck is empty
        new_round = False  # True if user performs 'stick' or cards is over 21 and new set of card is drawn

        if action is 'h':
            if len(self.deck) > 0:  # check if there is card left to draw
                self.player_hands.append(self.pop_card())
                if self.calculate_deck_total_value(self.player_hands, self.reward_dict) > 21:  # cards value over 21
                    new_round = True
        elif action is 's':
            new_round = True

        if new_round is True:
            current_reward = self.calculate_reward(self.player_hands, self.reward_dict)
            if len(self.deck) >= 2:  # check if there is more then 2 cards to draw
                self.player_hands = [self.pop_card(), self.pop_card()]
            else:
                is_done = True
        else:
            if len(self.deck) is 0:  # if deck is empty, end episode
                current_reward = self.calculate_reward(self.player_hands, self.reward_dict)
                is_done = True

        self.total_reward += current_reward
        deck_value = self.calculate_deck_total_value(self.player_hands, self.reward_dict)
        card_distribution_state = self.calculate_deck_current_distribution()
        state = (deck_value, card_distribution_state)  # pack cards value and distribution of cards as state
        return state, current_reward, is_done, new_round

    def reset(self):
        """
        Reset environment and start new round
        """
        self.deck = self.create_D_deck(self.poker_card, self.deck_number)
        self.reward_dict = self.card_value_dict()
        self.player_hands = [self.pop_card(), self.pop_card()]
        self.total_reward = 0
        self.episode_id += 1
        self.num_low_cards = 0
        self.num_high_cards = 0

    def return_current_state(self):
        """
        Return current cards value and distribution of card
        """
        return self.calculate_deck_total_value(self.player_hands, self.reward_dict), self.calculate_deck_current_distribution()


if __name__ == '__main__':
    bj = Blackjack(2, seed=0)
    deck_value = bj.calculate_deck_total_value(bj.player_hands, bj.reward_dict)
    print("Game Start! input 's' to stick, 'h' to hit or 'e' to exit game")
    print("State : {}".format(deck_value))
    while True:
        action = input()
        if action is 'e':
            break
        s, r, i, new_round = bj.step(action)
        print("State : {}, Distribution of cards : {}, Reward : {}, is done : {}, deck left : {}".format(s[0], s[1], r, i, len(bj.deck)))
        if i is True:
            break
    print("Game finished!")
    print("Total Reward : {}".format(bj.total_reward))
