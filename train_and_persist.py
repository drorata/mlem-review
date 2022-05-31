from sklearn.ensemble import RandomForestClassifier
import pickle
from mlem.api import save


with open("./data/X_train.pkl", "rb") as f:
    X = pickle.load(f)
with open("./data/y_train.pkl", "rb") as f:
    y = pickle.load(f)

rf = RandomForestClassifier(n_jobs=2, random_state=42,)
rf.fit(X, y)

save(
    rf, "rf", sample_data=X, description="Random Forest Classifier",
)
