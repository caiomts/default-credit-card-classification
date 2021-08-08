#############################
# Split train and test data #
#############################

# Imports
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
import yaml

# Parameters
params = {'test_size': yaml.safe_load(open('params.yaml'))['split_tt']['test_size']}

# Load data
data_path = os.path.join('data', 'default_credit_card_clients.csv')
data_raw = pd.read_csv(data_path, skiprows=1)

# X, y
data_raw.reset_index()
X = data_raw.iloc[:, :-1]
y = data_raw[['default payment next month']]

# split
sss = StratifiedShuffleSplit(n_splits=1, test_size=params['test_size'], random_state=42)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

# save
X_train.to_parquet('data/x_train.parquet')
y_train.to_parquet('data/y_train.parquet')
X_test.to_parquet('data/x_test.parquet')
y_test.to_parquet('data/y_test.parquet')
