from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


X, y = load_iris(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

X_train.to_pickle("./data/X_train.pkl")
y_train.to_pickle("./data/y_train.pkl")

X_test.to_pickle("./data/X_test.pkl")
y_test.to_pickle("./data/y_test.pkl")
