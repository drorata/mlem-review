import pickle

import pandas as pd
import requests
from sklearn.metrics import classification_report

with open("./data/X_test.pkl", "rb") as f:
    X_test: pd.DataFrame = pickle.load(f)
with open("./data/y_test.pkl", "rb") as f:
    y_test: pd.Series = pickle.load(f)

headers = {
    "accept": "application/json",
}

json_data = {
    "data": {"values": X_test.to_dict(orient="records")},
}

response = requests.post(
    "http://localhost:8080/predict", headers=headers, json=json_data
)

print(classification_report(y_pred=response.json(), y_true=y_test))
