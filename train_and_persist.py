from sklearn.ensemble import RandomForestClassifier
import pickle
from mlem.api import save

with open("./data/raw_X.pkl", "rb") as f:
    X = pickle.load(f)
with open("./data/raw_y.pkl", "rb") as f:
    y = pickle.load(f)

rf = RandomForestClassifier(n_jobs=2, random_state=42,)
rf.fit(X, y)

save(
    rf, "rf", sample_data=X, description="Random Forest Classifier",
)
