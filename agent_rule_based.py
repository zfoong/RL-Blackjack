"""
Author : Tham Yik Foong
Student ID : 20200786
Project title : Playing Stylised Blackjack with Q-learning

Academic integrity statement :
I, Tham Yik Foong, have read and understood the School's Academic Integrity Policy, as well as guidance relating to
this module, and confirm that this submission complies with the policy. The content of this file is my own original
work, with any significant material copied or adapted from other sources clearly indicated and attributed.
"""

class Agent_rule_based:
    """
    Rule based Agent that only perform 'hit' when card value is below 12 on Blackjack environment
    and 'stick' otherwise
    """
    def __init__(self):
        self.possible_card_val = [i for i in range(2, 22)]

    def return_action(self, state):
        card_val, _ = state
        if card_val <= 11:
            return 'h'
        else:
            return 's'
