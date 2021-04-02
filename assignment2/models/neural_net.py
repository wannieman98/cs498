"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a cross-entropy loss function and
    L2 regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are passed through
    a softmax, and become the scores for each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        self.t = 0; self.m = {}; self.v = {}


        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.

        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias

        Returns:
            the output
        """
        return np.dot(X, W) + b

    def linear_grad(self, prev_input, upstream_grad):
        """Gradient of Fully connected (linear) layer.

        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias

        Returns:
            the output
        """
        return np.matmul(prev_input.T, upstream_grad)

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        return np.maximum(X, 0)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output data
        """
        X[X <= 0] = 0
        X[X > 0] = 1
        return X

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data

        Returns:
            the output
        """
        exps = np.exp(X - np.max(X, axis=1).reshape((-1,1)))
        return exps / np.sum(exps, axis=1).reshape((-1,1))

    def delta_cross_entropy(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Gradient of softmax.

        Parameters:
            X: the input data
            y: vector of training labels

        Retuns:
            the output data
        """
        m = y.shape[0]
        grad = self.softmax(X)
        grad[range(m), y] -= 1
        return grad / m
    
    def cross_entorpy(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Gradient of softmax.

        Parameters:
            X: the input data
            y: vector of training labels

        Retuns:
            the output data
        """
        m = y.shape[0]
        p = self.softmax(X)
        if np.all(p) == False:
            print(np.all(p))
        log_like = -np.log(p[range(m),y])
        return np.sum(log_like) / m

    def calc_total_loss(self, reg, loss, m):
        cost = 0

        for key in self.params.keys():
            if key[0] == 'W':
                cost += np.sum(np.square(self.params[key]))
        cost *= reg / (2*m)

        total_loss = loss + cost
        total_loss = np.squeeze(total_loss)

        return total_loss
    
    def initialize_grads(self, layer):
        self.gradients['W'+str(layer)] = np.zeros(self.params['W'+str(layer)].shape)
        self.gradients['b'+str(layer)] = np.zeros(self.params['b'+str(layer)].shape)


    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        self.outputs = {}
        # implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.softmax in here.

        score = np.zeros((X.shape[0], self.output_size))

        for layer in range(1, self.num_layers + 1):

            weight = self.params['W' + str(layer)]
            bias = self.params['b' + str(layer)]

            if layer == 1:
                output = self.linear(weight, X, bias)
                self.outputs['o' + str(layer)] = output
                self.outputs['i' + str(layer)] = X
                input_ = self.relu(output)

            elif (layer > 1) and (layer < self.num_layers):
                output = self.linear(weight, input_, bias)
                self.outputs['o' + str(layer)] = output
                self.outputs['i' + str(layer)] = input_
                input_ = self.relu(output)

            else:
                output = self.linear(weight, input_, bias)
                self.outputs['o' + str(layer)] = output
                self.outputs['i' + str(layer)] = input_
                score = self.softmax(output)

        return score

    def backward(self, y: np.ndarray, reg: float = 0.0) -> float:
        """Perform back-propagation and compute the gradients and losses.

        Note: both gradients and loss should include regularization.

        Parameters:
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            reg: Regularization strength

        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.delta_cross_entropy if it helps organize your code.

        upstream_grad = 0
        
        for layer in reversed(range(1, self.num_layers+1)):

            if 'W'+str(layer) not in self.gradients and 'b'+str(layer) not in self.gradients:
                self.initialize_grads(layer)
            
            output = self.outputs['o'+str(layer)]
            weight = self.params['W'+str(layer)]
            m = output.shape[0]
            prev_input = self.outputs['i'+str(layer)]
            
            if layer == self.num_layers:
                loss = self.cross_entorpy(output, y)
                upstream_grad = self.delta_cross_entropy(output, y)
                linear_grad = self.linear_grad(prev_input, upstream_grad)

                self.gradients['W'+str(layer)] = linear_grad + (reg/m)*weight
                self.gradients['b'+str(layer)] = np.sum(upstream_grad, 0)

                upstream_grad =  np.dot(upstream_grad, weight.T)

            else:
                relu_grad = self.relu_grad(output)
                upstream_grad = upstream_grad*relu_grad
                linear_grad = self.linear_grad(prev_input, upstream_grad)

                self.gradients['W'+str(layer)] = linear_grad + (reg/m)*weight
                self.gradients['b'+str(layer)] = np.sum(upstream_grad, 0)
                
                upstream_grad =  np.dot(upstream_grad, weight.T)

        total_loss = self.calc_total_loss(reg, loss, m)

        return total_loss

    def SGD(self, lr):
        for key in self.params.keys():
            self.params[key] -= lr * self.gradients[key]

    def SGD_momentum(self, lr, b1):
        for key in self.params.keys():
                if key not in self.m and key not in self.v:
                    self.m[key] = np.zeros(self.params[key].shape)
                self.m[key] = b1*self.m[key] - lr*self.gradients[key]
                self.params[key] += self.m[key]

    def Adam(self, lr, b1, b2, eps, epoch):
        #TODO: Implement Adam
        self.t = epoch
        for key in self.params.keys():

            if key not in self.m and key not in self.v:
                self.m[key] = np.zeros(self.params[key].shape)
                self.v[key] = np.zeros(self.params[key].shape)

            gt = self.gradients[key]
            # self.update_Adam(b1, b2, gt, key)
            self.params[key] -= lr * np.divide(self.m[key], ((self.v[key]**(1/2)) + eps))

            self.m[key] = b1*self.m[key] + (1 - b1) * gt
            self.v[key] = b2*self.v[key] + (1 - b2) * np.square(gt)

            lr = lr * (1 - b2**self.t)**(1/2) / (1-b1**self.t)
            self.params[key] = self.params[key] - np.divide((lr * self.m[key]), (self.v[key]**(1/2)+eps))


    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
        epoch: int = 0
    ):
        """Update the parameters of the model using the previously calculated
        gradients.

        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        if opt == 'SGD':
            self.SGD(lr)
        elif opt == 'Adam':
            self.Adam(lr, b1, b2, eps, epoch)
        else:
            self.SGD_momentum(lr, b1)