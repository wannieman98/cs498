"""Logistic regression model."""

import numpy as np
from math import exp

class Logistic:
    def __init__(self, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        return 1 / (1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        dimension = X_train.shape[1]
        self.w = np.random.rand(dimension)
        for epoch in range(self.epochs):
            for i in range(X_train.shape[0]):
                x_i = X_train[i]
                y_i = y_train[i]
                sigmoid = self.sigmoid(np.dot(x_i, self.w.T))
                self.w = self.w - self.lr*(sigmoid-y_i)*x_i

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """

        pred_prob = self.sigmoid(np.dot(self.w, X_test.T))
        pred_value = np.where(pred_prob >= 0.5, 1, 0)
        return pred_value
 