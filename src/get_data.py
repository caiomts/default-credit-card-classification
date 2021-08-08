###################
#Get and save data#
###################

import pandas as pd

raw = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls", index_col=0)

raw.to_csv("data/default_credit_card_clients.csv", index=False)

