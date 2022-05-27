from sklearn.datasets import load_iris


data, y = load_iris(return_X_y=True, as_frame=True)

data.to_pickle("./data/raw_X.pkl")
data.to_pickle("./data/raw_y.pkl")
