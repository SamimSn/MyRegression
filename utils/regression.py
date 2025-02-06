import json
import math
import colorama as color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Regression:

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        alpha: float = 1e-2,
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
        self.history: dict = {}

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
        previous_cost = None
        for i in range(self.iterations + 1):
            dw, db = self.__compute_gradient(self.X_data, self.Y_data, W, b)
            w_copy -= self.alpha * dw
            b_copy -= self.alpha * db
            cost = self.__compute_cost(self.X_data, self.Y_data, w_copy, b_copy)
            if previous_cost is not None and cost > previous_cost:
                print(
                    f"{color.Fore.CYAN}Epoch > {i:5d}{color.Fore.RESET} | {color.Fore.YELLOW}Cost: {cost:.6f}{color.Fore.RESET}"
                )
                print(f"{color.Fore.GREEN}Gradient Descent stopped due to increasing cost{color.Fore.RESET}")
                break
            previous_cost = cost
            if i % (self.iterations / 10) == 0:
                print(
                    f"{color.Fore.CYAN}Epoch > {i:5d}{color.Fore.RESET} | {color.Fore.YELLOW}Cost: {cost:.6f}{color.Fore.RESET}"
                )

            if i % 10 == 0:
                self.history[i] = cost

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

    def predict(self, X_new: np.ndarray, denormalize: bool = True):
        if self.W_out is None or self.b_out is None:
            raise ValueError(f"Call .train() on your {self.__class__.__name__} instance first.")
        X_new_normalized = self.__normalize_predict(X_new)
        prediction_normalized = self.__function(self.W_out, X_new_normalized, self.b_out)
        if denormalize:
            return self.__de_normalize(prediction_normalized)
        else:
            return prediction_normalized

    def plot_cost(self):
        """plots cost function"""
        _, axs = plt.subplots(1, 1, figsize=(4, 4))
        axs.set_xlabel("Iteration")
        axs.set_ylabel("Cost")
        axs.plot(list(self.history.keys()), list(self.history.values()), color="blue")
        plt.tight_layout()
        plt.show()

    def train_accuracy(self):
        """calculates model's accuracy on normalized training data"""
        err = 0
        m = self.X_data.shape[0]
        for i in range(m):
            model_out = self.predict(self.X_data[i], False)
            actual_out = self.Y_data[i]
            if i % (m // 10) == 0:
                print(
                    f"{color.Fore.CYAN}{i}.Model Output > {model_out}{color.Fore.RESET} | {color.Fore.YELLOW}Actual Output: {actual_out}{color.Fore.RESET}"
                )
            err += np.abs(model_out - actual_out)
        print(
            f"{color.Fore.GREEN} >>> Accuracy on train data is {round(float((1 - (err / m)) * 100), 2)} %{color.Fore.RESET}"
        )


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


def plot_data(data, x_columns, y_column):
    x_data = [np.array(data[col]) for col in x_columns]
    y_data = np.array(data[y_column])

    colours = ["blue", "green", "red", "purple", "orange", "cyan", "magenta", "yellow", "black", "brown"]

    n_plots = len(x_columns)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 4 * n_rows))

    axs = axs.flatten()

    for i, ax in enumerate(axs[:n_plots]):
        ax.set_xlabel(x_columns[i])
        ax.set_ylabel(f"{y_column} (g/km)")
        ax.plot(x_data[i], y_data, "o", color=colours[i % len(colours)])

    for i in range(n_plots, len(axs)):
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()


def prepare_data(data, x_columns, y_column):
    x_train = np.array([[data[col][i] for col in x_columns] for i in range(data.shape[0])])
    y_train = np.array(data[y_column])
    return x_train, y_train
