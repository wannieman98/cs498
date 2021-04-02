"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me
        grad_w = np.zeros((self.w.shape[0], self.w.shape[1]))

        for i in range(len(X_train)):
            train = X_train[i]
            y_pred = np.dot(train, self.w.T)
            for c in range(self.n_class):
                if c == y_train[i]:
                    grad_w[c,:] += self.sigmoid(y_pred, c, train, True)
                else:
                    grad_w[c,:] += self.sigmoid(y_pred, c, train, False)

        grad_w /= len(X_train)
        grad_w += self.reg_const * self.w

        return grad_w


    
    def sigmoid(self, x, c, x_i, isyi):
        x -= np.max(x)
        sum_i = np.sum(np.exp(x))
        numerator = np.exp(x[c])
        p = numerator / sum_i
        if isyi:
            return (p - 1) * x_i
        else:
            return p * x_i
        



    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me

        len_of_data, dimension = X_train.shape
        batch_size = 200
        self.w = np.random.randn(self.n_class, dimension) * 0.1

        for epoch in range(self.epochs):
            indicies = np.random.permutation(len_of_data)
            i = 0
            while i < len_of_data:
                x_batch = X_train[indicies[i:i+batch_size], : ]
                y_batch = y_train[indicies[i:i+batch_size], ]

                grad_w = self.calc_gradient(x_batch, y_batch)

                self.w -= self.lr *  grad_w

                i += batch_size
                

        # dimension = len(X_train[1])
        # self.w = np.zeros((self.n_class, dimension))
        # for epoch in range(self.epochs):
        #     for i in range(len(X_train)):

        #         train = X_train[i]
        #         scores = np.dot(train, self.w.T)

        #         for c in range(self.n_class):
        #             if c == y_train[i]:
        #                 self.w[c] += (self.lr * self.sigmoid(scores[y_train[i]], True)) * train
        #             else:
        #                 self.w[c] -= self.lr * self.sigmoid(scores[c], False) * train
        #         self.w = (1 - self.lr * (self.reg_const / len(X_train))) * self.w

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

        p = np.argmax(np.dot(X_test, self.w.T), axis=1)
        return p

