import numpy as np
import random
import pandas as pd


class Agent_rule_based:
    def __init__(self):
        pass

    def return_action(self, state):
        if state <= 10:
            return 'h'
        else:
            return 's'
