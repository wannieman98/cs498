"""Support Vector Machine (SVM) model."""

import numpy as np
import math

class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None 
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        # dimension = self.w.shape[0]
        # num_of_data = X_train.shape[0]
        # grad_w = np.zeros((dimension, self.n_class))

        # for i in range(num_of_data):
        #   train = X_train[i]
        #   scores = train.dot(self.w)

        #   gt_position = scores[y_train[i]]

        #   for pos in range(self.n_class):
        #     if pos != y_train[i]:
        #         if scores[pos] - gt_position + 1 > 0:
        #             grad_w[:, pos] += X_train[i]
        #             grad_w[:, y_train[i]] -= X_train[i]

        # grad_w /= num_of_data
        # grad_w += self.reg_const * self.w

        # return grad_w     



    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        dimension = X_train.shape[1]

        self.w = np.random.randn(self.n_class, dimension)

        for epoch in range(self.epochs):
            for i in range(len(X_train)):
                train = X_train[i]
                scores = np.dot(train, self.w.T)

                for c in range(self.n_class):
                    if c != y_train[i] and scores[y_train[i]] - scores[c] < 1:
                        self.w[y_train[i]] = self.w[y_train[i]] + self.lr * train
                        self.w[c] = self.w[c] - self.lr * train
                    self.w[c] = (1 - self.lr*(self.reg_const/len(X_train)))*self.w[c]
                    

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
        # TODO: implement me
        pred = np.argmax(X_test.dot(self.w.T), axis=1)
        return pred

