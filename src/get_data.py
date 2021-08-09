#########################
# Get and save the data #
#########################

import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

raw = pd.read_excel(url, index_col=0)

raw.to_csv("data/default_credit_card_clients.csv", index=False)
