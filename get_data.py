from sklearn.datasets import load_iris


X, y = load_iris(return_X_y=True, as_frame=True)

X.to_pickle("./data/raw_X.pkl")
y.to_pickle("./data/raw_y.pkl")
