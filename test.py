import random
import numpy as np
import pandas as pd

q_table = pd.read_pickle('q_table_q_extend.pkl')
print(q_table)
up = q_table.idxmax(1)
print(up)
