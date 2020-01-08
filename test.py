import random
import numpy as np
import pandas as pd

possible_state = [i for i in range(2, 22)]
possible_action = ['h', 's']
possible_dis = ['e low cards', 'more low cards', 'same amount', 'more high cards', 'e high cards']
suits = ['club', 'diamond', 'heart', 'spade']
ranks = ['ace', 'two', 'three', 'four', 'five', 'six', 'seven',
              'eight', 'nine', 'ten', 'jack', 'queen', 'king']


test = [99,1,2,3,4,5]
test.append(6)
print(test)
kkk = random.sample(test, 2)
print(kkk)


# def create_a_deck():
#     deck = []
#     for i in ranks:
#         for j in suits:
#             deck.append("{} {}".format(i, j))
#     return deck
#
# poker_card = create_a_deck()
# lowc = poker_card[0:26]
# highc = poker_card[26:52]
#
# print("")

# Q_table = np.load('q_table_q.npy')
# df = pd.DataFrame(np.zeros((20, 5)), columns=possible_dis, index=possible_state)
#
# for val_index, i in enumerate(possible_state):
#     for dis_index, j in enumerate(possible_dis):
#         df[j][i] = possible_action[np.argmax(Q_table[val_index, dis_index, :])]
#
# print(Q_table)
#
# print(df)

# def calculate_deck_current_distribution(num_low_cards, num_high_cards):
#     difference_threshold = 4
#     if num_low_cards > num_high_cards + difference_threshold * 2:
#         return 'excessive low cards'
#     elif num_low_cards > num_high_cards + difference_threshold:
#         return 'more low cards'
#     elif num_low_cards + difference_threshold * 2 < num_high_cards:
#         return 'excessive high cards'
#     elif num_low_cards + difference_threshold < num_high_cards:
#         return 'more high cards'
#     else:
#         return 'same amount'
#
# print(calculate_deck_current_distribution(60, 69))

# q_table = pd.read_pickle('q_table_q.pkl')
# print(q_table)
# up = q_table.idxmax(1)
# print(up)

#
# df = pd.DataFrame(np.zeros((100, 2)), columns=possible_action,
#                   index=pd.MultiIndex.from_product([possible_state, possible_dis]))
# df['h'][21]['=='] = 32
# df['s'][21]['=='] = 33
# print(df['s'][21]['=='])
# print(df.loc[21,'==',:].idxmax(1))
#
# df_2 = pd.DataFrame(np.zeros((20, 5)), columns=possible_dis, index=possible_state)
#
# for i in possible_dis:
#     for j in possible_state:
#         df_2[i][j] = df.loc[j, i].idxmax(1)
#
# print(df_2)

