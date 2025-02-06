import json
import pandas as pd
import numpy as np
 

class Regression:

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        alpha: float = 1e-4,
        iterations: int = 10_000,
    ):
        self.X_data: np.ndarray = self.__normalize_train(X)
        self.Y_data: np.ndarray = self.__normalize_train(Y)

        self.X_data_max: np.ndarray = X.max(axis=0)
        self.Y_data_max = float(Y.max())

        self.alpha: float = alpha
        self.iterations: int = iterations

        self.W_out: np.ndarray = None
        self.b_out: float = None

    def __normalize_train(self, arr: np.ndarray):
        """Normalize input data."""
        arr_copy = arr.astype(float)
        if arr_copy.ndim == 1:
            arr_copy /= arr_copy.max()
        else:
            _, n = arr_copy.shape
            for j in range(n):
                arr_max = arr_copy[:, j].max()
                arr_copy[:, j] /= arr_max
        return arr_copy

    def __normalize_predict(self, arr: np.ndarray):
        """Normalize input data."""
        arr_copy = arr.astype(float)
        n = arr_copy.shape[0]
        for j in range(n):
            arr_copy[j] /= self.X_data_max[j]
        return arr_copy

    def __de_normalize(self, answer: float):
        """De-normalize the prediction."""
        return answer * self.Y_data_max

    def __function(self, W: np.ndarray, X: np.ndarray, b: float):
        return np.dot(W, X) + b

    def __compute_cost(self, X: np.ndarray, Y: np.ndarray, W: np.ndarray, b: float):
        """Compute the cost function for a linear regression model."""
        cost = 0
        m = X.shape[0]
        for i in range(m):
            cost += (self.__function(W, X[i], b) - Y[i]) ** 2
        return cost / (2 * m)

    def __compute_gradient(self, X: np.ndarray, Y: np.ndarray, W: np.ndarray, b: float):
        """Compute the gradient of the cost function for a linear regression model."""
        m, n = X.shape
        dw = np.zeros(n)
        db = 0
        for i in range(m):
            loss = self.__function(W, X[i], b) - Y[i]
            for j in range(n):
                dw[j] += loss * X[i, j]
            db += loss
        dw /= m
        db /= m
        return dw, db

    def __gradient_descent(self, W: np.ndarray, b: float):
        """Compute the gradient descent for a linear regression model."""
        w_copy: np.ndarray = W
        b_copy: float = b
        for i in range(self.iterations + 1):
            dw, db = self.__compute_gradient(self.X_data, self.Y_data, W, b)
            w_copy -= self.alpha * dw
            b_copy -= self.alpha * db
            if i % 500 == 0:
                cost = self.__compute_cost(self.X_data, self.Y_data, w_copy, b_copy)
                print(f"\033[96mEpoch > {i:5d}\033[0m | \033[93mCost: {cost:.6f}\033[0m")
        return w_copy, b_copy

    def train(self):
        self.W_out, self.b_out = self.__gradient_descent(np.zeros(self.X_data.shape[1]), 0)
        data = {
            "w": self.W_out.tolist(),
            "b": self.b_out,
            "x_max": self.X_data_max.tolist(),
            "y_max": float(self.Y_data_max),
        }
        path = "output.json"
        with open(path, "w") as output:
            json.dump(
                data,
                output,
                indent=4,
            )

    def predict(self, X_new: np.ndarray):
        if self.W_out is None or self.b_out is None:
            raise ValueError(f"Call .train() on your {self.__class__.__name__} instance first.")
        X_new_normalized = self.__normalize_predict(X_new)
        prediction_normalized = self.__function(self.W_out, X_new_normalized, self.b_out)
        return self.__de_normalize(prediction_normalized)


def load_model(path) -> Regression:
    model = Regression(np.array([1]), np.array([1]))
    file = pd.read_json(path)
    w = np.array(file["w"])
    b = np.array(file["b"])[0]
    x_max = np.array(file["x_max"])
    y_max = np.array(file["y_max"])[0]

    model.W_out = w
    model.b_out = b
    model.X_data_max = x_max
    model.Y_data_max = y_max

    return model
