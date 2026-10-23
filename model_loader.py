from pathlib import Path
import json
import pickle
import numpy as np

BASE_DIR = Path(__file__).resolve().parent


class ThreeLayerMLP:
    """Тот же класс, что в ноутбуке — без обучения, только forward/predict."""

    def __init__(self, input_size=4, hidden_size=10, output_size=3):
        self.W1 = np.zeros((input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.zeros((hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    def _relu(self, z):
        return np.maximum(0, z)

    def _softmax(self, z):
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        return self._softmax(z2)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


def _load_model():
    with open(BASE_DIR / "iris_params.json") as f:
        params = json.load(f)

    m = ThreeLayerMLP(**params)
    weights = np.load(BASE_DIR / "iris_model.npz")
    m.W1, m.b1 = weights["W1"], weights["b1"]
    m.W2, m.b2 = weights["W2"], weights["b2"]
    return m


def _load_scaler():
    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        return pickle.load(f)


model = _load_model()
scaler = _load_scaler()
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    X = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        dtype=np.float64,
    )
    X_scaled = scaler.transform(X)           # ⚠️ обязательно масштабируем!
    y_pred = model.predict(X_scaled)
    return CLASS_NAMES[int(y_pred[0])]
