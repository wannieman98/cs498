"""Perceptron model."""

import numpy as np
from operator import add, sub

class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None 
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        dimension = len(X_train[0])
        N = len(X_train)
        self.w = np.random.randn(self.n_class, dimension)

        for epoch in range(self.epochs):
            for i in range(len(X_train)):
                train = X_train[i]
                scores = np.dot(self.w, train.T)
                predict = np.argmax(np.dot(self.w, train.T))
                if predict != y_train[i]:
                    for c in range(self.n_class):
                        if scores[c] > scores[y_train[i]]:
                            self.w[y_train[i]] = np.add(self.w[y_train[i]], self.lr*train.T)
                            self.w[c] = np.subtract(self.w[c], self.lr*train.T)
            self.decay()

                        




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
        pred = []
        for test in X_test:
            predicted = int(np.argmax(np.dot(self.w, test.T)))
            pred.append(predicted)
        return pred

    def decay(self):
        self.lr -= self.lr * 0.8