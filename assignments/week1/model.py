import numpy as np

# from tqdm import tqdm


class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):
        """this is init"""
        self.w = 0
        self.b = 0
        # raise NotImplementedError()

    def fit(self, X, y):
        """this is fit _ 1"""
        temp = np.ones(len(X))
        temp = np.expand_dims(temp, axis=1)
        X = np.hstack((temp, X))
        # print(X.shape)
        w = np.linalg.inv(X.T @ X) @ (X.T @ y)
        self.b = w[0]
        self.w = w[1:]
        self.t = w

        # print(w.shape)
        # print(self.b.shape)
        # print(self.w.shape)
        # print(self.b)
        # print(X.shape)
        # raise NotImplementedError()

    def predict(self, X):
        """this is predict _ 1"""
        temp = np.ones(len(X))
        temp = np.expand_dims(temp, axis=1)
        X = np.hstack((temp, X))
        ans = X @ self.t
        # print(ans)
        return ans
        # b = np.expand_dims(self.b, axis=1)
        # print(self.w.shape)

        # y_pred = X
        # raise NotImplementedError()


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """this is fit _1"""
        self.t = np.random.rand(
            9,
        )

        temp = np.ones(len(X))
        temp = np.expand_dims(temp, axis=1)
        X = np.hstack((temp, X))
        for _ in range(epochs):
            # print((X@self.t).shape)
            dl = np.mean(-X.T @ (y - (X @ self.t)))
            # print(dl.shape)
            self.t = self.t - (lr * np.clip(dl, -2, 2))
            # print(self.t)
            # w = np.linalg.inv(X.T @X)@(X.T@y)
        self.b = self.t[0]
        self.w = self.t[1:]

        # print(self.t)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        temp = np.ones(len(X))
        temp = np.expand_dims(temp, axis=1)
        X = np.hstack((temp, X))
        ans = X @ self.t
        # print(ans)
        return ans
