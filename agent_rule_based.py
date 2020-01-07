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
