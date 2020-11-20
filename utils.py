import copy
import numpy as np

from scipy.special import expit as logistic_sigmoid
from scipy.special import xlogy


def identity(X):
    return X


def logistic(X):
    return logistic_sigmoid(X, out=X)


def tanh(X):
    return np.tanh(X, out=X)


def relu(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X


def softmax(X):
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]

    return X


ACTIVATIONS = {'identity': identity, 'tanh': tanh, 'logistic': logistic,
               'relu': relu, 'softmax': softmax}


def inplace_identity_derivative(Z, delta):
    return


def inplace_logistic_derivative(Z, delta):
    delta *= Z
    delta *= (1 - Z)


def inplace_tanh_derivative(Z, delta):
    delta *= (1 - Z ** 2)


def inplace_relu_derivative(Z, delta):
    delta[Z == 0] = 0


DERIVATIVES = {'identity': inplace_identity_derivative,
               'tanh': inplace_tanh_derivative,
               'logistic': inplace_logistic_derivative,
               'relu': inplace_relu_derivative}


def squared_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() / 2


def log_loss(y_true, y_prob):
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return - xlogy(y_true, y_prob).sum() / y_prob.shape[0]


def binary_log_loss(y_true, y_prob):
    return -(xlogy(y_true, y_prob) +
             xlogy(1 - y_true, 1 - y_prob)).sum() / y_prob.shape[0]


LOSS_FUNCTIONS = {'squared_loss': squared_loss, 'log_loss': log_loss,
                  'binary_log_loss': binary_log_loss}


class Individual:

    def __init__(self, chromosome, fitness):
        self._chromosome = copy.deepcopy(chromosome)
        self._fitness = fitness

    def chromosome(self):
        return self._chromosome

    def fitness(self):
        return self._fitness

    def set_chromosome(self, chromosome):
        self._chromosome = copy.deepcopy(chromosome)

    def set_fitness(self, fitness):
        self._fitness = fitness
