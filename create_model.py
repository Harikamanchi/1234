import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np

# Dummy training data
X = np.array([
    [5, 1, 7],
    [8, 3, 4],
    [2, 0, 8],
    [9, 4, 3]
])
y = [0, 1, 0, 2]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

with open("mental_health_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model created successfully")
