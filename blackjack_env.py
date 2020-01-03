import numpy as np


class Blackjack:
    def __init__(self, deck_number=1):
        self.episode_id = 1
        self.deck_number = deck_number
        self.suits = ['club', 'diamond', 'heart', 'spade']
        self.ranks = ['ace', 'two', 'three', 'four', 'five', 'six', 'seven',
                      'eight', 'nine', 'ten', 'jack', 'queen', 'king']
        self.poker_card = self.create_a_deck()
        self.deck = self.create_D_deck(self.poker_card, self.deck_number)
        self.reward_dict = self.card_value_dict()
        self.player_hands = [self.deck.pop(0), self.deck.pop(0)]
        self.total_reward = 0

    def shuffle_deck(self, deck):
        np.random.seed(self.episode_id)
        np.random.shuffle(deck)
        return deck

    def create_a_deck(self, shuffle=True):
        deck = []
        for i in self.ranks:
            for j in self.suits:
                deck.append("{} {}".format(i, j))
        if shuffle is True:
            deck = self.shuffle_deck(deck)
        return deck

    def create_D_deck(self, poker_card, d=1):
        deck = []
        for i in range(d):
            deck = deck + poker_card
        deck = self.shuffle_deck(deck)
        return deck

    def card_value_dict(self):
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
        total_val = 0
        aces = [l for l in deck if "ace" in l]
        for index, k in enumerate(deck):
            total_val += reward_dict[deck[index]]
        if total_val > 21 and len(aces) > 0:
            total_val -= len(aces) * 10
        return total_val

    def calculate_reward(self, deck, reward_dict, punishment=0):
        deck_value = self.calculate_deck_total_value(deck, reward_dict)
        if deck_value > 21:
            return punishment
        else:
            return pow(deck_value, 2)

    def step(self, action):
        current_reward = 0
        is_done = False
        new_round = False

        if action is 'h':
            if len(self.deck) > 0:
                self.player_hands.append(self.deck.pop(0))
                if self.calculate_deck_total_value(self.player_hands, self.reward_dict) > 21:
                    new_round = True
        elif action is 's':
            new_round = True

        if new_round is True:
            current_reward += self.calculate_reward(self.player_hands, self.reward_dict)
            if len(self.deck) >= 2:
                self.player_hands = [self.deck.pop(0), self.deck.pop(0)]
            else:
                is_done = True
        else:
            if len(self.deck) is 0:
                current_reward += self.calculate_reward(self.player_hands, self.reward_dict)
                is_done = True

        self.total_reward += current_reward
        deck_value = self.calculate_deck_total_value(self.player_hands, self.reward_dict)
        return deck_value, current_reward, is_done

    # def step(self, action):
    #     current_reward = 0
    #     is_done = False
    #     new_round = False
    #
    #     if action is 'h':
    #         if len(self.deck) > 0:
    #             self.player_hands.append(self.deck.pop(0))
    #             if self.calculate_deck_total_value(self.player_hands, self.reward_dict) > 21:
    #                 new_round = True
    #     elif action is 's':
    #         new_round = True
    #
    #     if new_round is True:
    #         current_reward += self.calculate_reward(self.player_hands, self.reward_dict)
    #         if len(self.deck) >= 2:
    #             self.player_hands = [self.deck.pop(0), self.deck.pop(0)]
    #         else:
    #             is_done = True
    #     else:
    #         if len(self.deck) is 0:
    #             current_reward += self.calculate_reward(self.player_hands, self.reward_dict)
    #             is_done = True
    #
    #     self.total_reward += current_reward
    #     return self.player_hands, current_reward, is_done

    def reset(self):
        self.deck = self.create_D_deck(self.poker_card, self.deck_number)
        self.reward_dict = self.card_value_dict()
        self.player_hands = [self.deck.pop(0), self.deck.pop(0)]
        self.total_reward = 0
        self.episode_id += 1

    def return_current_state(self):
        return self.calculate_deck_total_value(self.player_hands, self.reward_dict)


if __name__ == '__main__':
    bj = Blackjack(1)
    print(bj.player_hands)
    while True:
        action = input()
        if action is 'e':
            break
        s, r, i = bj.step(action)
        print("State : {}, Reward : {}, is done : {}, deck left : {}".format(s, r, i, len(bj.deck)))
        if i is True:
            break
    print("Game finish")
    print("Total Reward : {}".format(bj.total_reward))