####################
# Model Evaluation #
####################

# Imports
import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, precision_recall_curve
import json
import joblib
import math

# Load train sets
X_test = pd.read_parquet(os.path.join('data', 'x_test.parquet'))
y_test = pd.read_parquet(os.path.join('data', 'y_test.parquet'))

# Load model
clf = joblib.load(os.path.join('models', 'clf.pkl'))

# predict
y_pred = clf.predict(X_test)
y_pred_scores = clf.predict_proba(X_test)

# Evaluate
precisions, recalls, pr_thresholds = precision_recall_curve(np.ravel(y_test), y_pred_scores[:, 1])

scoring = f1_score(np.ravel(y_test), y_pred)

# Metrics
with open("scores.json", "w") as fd:
    json.dump({"f1": scoring}, fd, indent=4)

nth_point = math.ceil(len(pr_thresholds) / 1000)
prc_points = list(zip(precisions, recalls, pr_thresholds))[::nth_point]

with open("prc.json", "w") as fd:
    json.dump(
        {
            "prc": [
                {"precision": p, "recall": r, "threshold": t}
                for p, r, t in prc_points
            ]
        },
        fd,
        indent=4,
    )

cmx = pd.DataFrame({'actual': np.ravel(y_test), 'predicted': y_pred})
cmx.to_csv('classes.csv', index=False)

