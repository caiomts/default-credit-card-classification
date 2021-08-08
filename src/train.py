##################
# Model Training #
##################

# Imports
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
import yaml

# Parameters

params = {
    'n_estimators': yaml.safe_load(open('params.yaml'))['train']['n_estimators'],
    'max_leaf_nodes': yaml.safe_load(open('params.yaml'))['train']['max_leaf_nodes'],
    'max_features': yaml.safe_load(open('params.yaml'))['train']['max_features']
}

# Load train sets
X_train = pd.read_parquet(os.path.join('data', 'x_train.parquet'))
y_train = pd.read_parquet(os.path.join('data', 'y_train.parquet'))

# Train
xtr_clf = ExtraTreesClassifier(random_state=42, n_jobs=-1, n_estimators=params['n_estimators'],
                               max_leaf_nodes=params['max_leaf_nodes'], max_features=params['max_features'])
xtr_clf = xtr_clf.fit(X_train, np.ravel(y_train))

# Pickles
os.makedirs('models/', exist_ok=True)
joblib.dump(xtr_clf, os.path.join('models', 'clf.pkl'))

