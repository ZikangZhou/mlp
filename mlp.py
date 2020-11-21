import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelBinarizer

from optimizers import SGDOptimizer, AdamOptimizer, GAOptimizer
from utils import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS, accuracy_score


class MLP:

    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='ga', pop_size=50, crossover_rate=0.4,
                 mutation_rate=0.05, alpha=0.0001, batch_size=64, learning_rate='constant', learning_rate_init=0.001,
                 power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=1e-4, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10):
        self.activation = activation
        self.solver = solver
        self.optimizer = None
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        if not hasattr(self.hidden_layer_sizes, "__iter__"):
            self.hidden_layer_sizes = [self.hidden_layer_sizes]
        self.hidden_layer_sizes = list(self.hidden_layer_sizes)
        self.shuffle = shuffle
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.tol = tol
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.label_binarizer = LabelBinarizer()
        self.out_activation = ""
        self.n_outputs = 0
        self.n_layers = len(hidden_layer_sizes) + 2
        self.weights = []
        self.biases = []
        self.loss_curve = []
        self.no_improvement_count = 0
        if self.early_stopping:
            self.validation_scores = []
            self.best_validation_score = -np.inf
            self.best_weights = []
            self.best_biases = []
        else:
            self.best_loss = np.inf

    def get_params(self):
        return self.weights, self.biases

    def set_params(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def fit(self, X, y):
        self.label_binarizer.fit(y)
        y = self.label_binarizer.transform(y)
        if self.label_binarizer.y_type_ == 'multiclass':
            self.out_activation = 'softmax'
        else:
            self.out_activation = 'logistic'

        n_samples, n_features = X.shape
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        self.n_outputs = y.shape[1]
        layer_units = ([n_features] + self.hidden_layer_sizes + [self.n_outputs])

        self.weights = []
        self.biases = []
        for i in range(self.n_layers - 1):
            if self.solver == 'ga':
                init_bound = 1.0
            else:
                if self.activation == 'logistic':
                    factor = 2
                else:
                    factor = 6
                init_bound = np.sqrt(factor / (layer_units[i] + layer_units[i + 1]))
            self.weights.append(
                np.random.uniform(-init_bound, init_bound, (layer_units[i], layer_units[i + 1])))
            self.biases.append(np.random.uniform(-init_bound, init_bound, layer_units[i + 1]))

        activations = [X]
        for _ in range(len(layer_units) - 1):
            activations.append(None)
        deltas = [None] * (len(activations) - 1)
        weight_grads = [np.empty((n_fan_in, n_fan_out)) for n_fan_in, n_fan_out in
                        zip(layer_units[: -1], layer_units[1:])]
        bias_grads = [np.empty(n_fan_out) for n_fan_out in layer_units[1:]]
        params = self.weights + self.biases
        if self.solver == 'sgd':
            self.optimizer = SGDOptimizer(params, self.learning_rate_init, self.learning_rate, self.momentum,
                                          self.nesterovs_momentum, self.power_t)
        elif self.solver == 'adam':
            self.optimizer = AdamOptimizer(params, self.learning_rate_init, self.beta_1, self.beta_2, self.epsilon)
        else:
            self.optimizer = GAOptimizer(model=self, X=X, y=self.label_binarizer.inverse_transform(y), params=params,
                                         pop_size=self.pop_size, crossover_rate=self.crossover_rate,
                                         mutation_rate=self.mutation_rate)

        if self.early_stopping:
            stratify = y if self.n_outputs == 1 else None
            X, X_val, y, y_val = train_test_split(X, y, random_state=self.random_state,
                                                  test_size=self.validation_fraction, stratify=stratify)
            y_val = self.label_binarizer.inverse_transform(y_val)
        else:
            X_val = None
            y_val = None

        n_samples = X.shape[0]
        sample_idx = np.arange(n_samples, dtype=int)
        time_step = 0
        for it in range(self.max_iter):
            if self.solver == 'ga':
                self.optimizer.update_params()
                loss = 1.0 / accuracy_score(y.ravel(), self.predict(X))
            else:
                accumulated_loss = 0.0
                if self.shuffle:
                    np.random.shuffle(sample_idx)
                batch_slices = [sample_idx[start: start + self.batch_size] for start in
                                range(0, n_samples, self.batch_size)]
                if n_samples % self.batch_size != 0:
                    batch_slices.append(sample_idx[(n_samples - n_samples % self.batch_size):])
                for batch_slice in batch_slices:
                    X_batch = X[batch_slice]
                    y_batch = y[batch_slice]
                    activations[0] = X_batch
                    batch_loss, weight_grads, bias_grads = self.backprop(X_batch, y_batch, activations, deltas,
                                                                         weight_grads, bias_grads)
                    accumulated_loss += batch_loss * (len(batch_slice))
                    grads = weight_grads + bias_grads
                    self.optimizer.update_params(grads)
                loss = accumulated_loss / n_samples

            time_step += n_samples
            self.loss_curve.append(loss)
            print(f"Iteration {it + 1}, accuracy = {accuracy_score(y.ravel(), self.predict(X))}")

            self.update_no_improvement_count(X_val, y_val)
            self.optimizer.iteration_ends(time_step)
            if self.no_improvement_count > self.n_iter_no_change:
                is_stopping = self.optimizer.trigger_stopping()
                if is_stopping:
                    break
                else:
                    self.no_improvement_count = 0
        if self.early_stopping:
            self.weights = self.best_weights
            self.biases = self.best_biases
        return self

    def forward_pass(self, activations):
        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(self.n_layers - 1):
            activations[i + 1] = np.dot(activations[i], self.weights[i])
            activations[i + 1] += self.biases[i]
            if i + 1 != self.n_layers - 1:
                activations[i + 1] = hidden_activation(activations[i + 1])
        output_activation = ACTIVATIONS[self.out_activation]
        activations[self.n_layers - 1] = output_activation(activations[self.n_layers - 1])
        return activations

    def backprop(self, X, y, activations, deltas, weight_grads, bias_grads):
        n_samples = X.shape[0]
        activations = self.forward_pass(activations)
        if self.out_activation == 'logistic':
            loss_func_name = 'binary_log_loss'
        else:
            loss_func_name = 'log_loss'
        loss = LOSS_FUNCTIONS[loss_func_name](y, activations[-1])
        values = np.sum(np.array([np.dot(s.ravel(), s.ravel()) for s in self.weights]))
        loss += (0.5 * self.alpha) * values / n_samples
        last = self.n_layers - 2

        deltas[last] = activations[-1] - y

        weight_grads, bias_grads = self.compute_loss_grad(last, n_samples, activations, deltas, weight_grads,
                                                          bias_grads)
        for i in range(self.n_layers - 2, 0, -1):
            deltas[i - 1] = np.dot(deltas[i], self.weights[i].T)
            inplace_derivative = DERIVATIVES[self.activation]
            inplace_derivative(activations[i], deltas[i - 1])
            weight_grads, bias_grads = self.compute_loss_grad(i - 1, n_samples, activations, deltas, weight_grads,
                                                              bias_grads)
        return loss, weight_grads, bias_grads

    def compute_loss_grad(self, layer, n_samples, activations, deltas, weight_grads, bias_grads):
        weight_grads[layer] = np.dot(activations[layer].T, deltas[layer])
        weight_grads[layer] += (self.alpha * self.weights[layer])
        weight_grads[layer] /= n_samples
        bias_grads[layer] = np.mean(deltas[layer], 0)
        return weight_grads, bias_grads

    def update_no_improvement_count(self, X_val, y_val):
        if self.early_stopping:
            self.validation_scores.append(accuracy_score(y_val, self.predict(X_val)))
            last_valid_score = self.validation_scores[-1]

            if last_valid_score < self.best_validation_score + self.tol:
                self.no_improvement_count += 1
            else:
                self.no_improvement_count = 0
            if last_valid_score > self.best_validation_score:
                self.best_validation_score = last_valid_score
                self.best_weights = [w.copy() for w in self.weights]
                self.best_biases = [b.copy() for b in self.biases]
        else:
            if self.loss_curve[-1] > self.best_loss - self.tol:
                self.no_improvement_count += 1
            else:
                self.no_improvement_count = 0
            if self.loss_curve[-1] < self.best_loss:
                self.best_loss = self.loss_curve[-1]

    def predict(self, X):
        layer_units = [X.shape[1]] + self.hidden_layer_sizes + [self.n_outputs]
        activations = [X]
        for i in range(self.n_layers - 1):
            activations.append(np.empty((X.shape[0], layer_units[i + 1])))
        self.forward_pass(activations)
        y_pred = activations[-1]
        if self.n_outputs == 1:
            y_pred = y_pred.ravel()
        return self.label_binarizer.inverse_transform(y_pred)


def cross_val_score(estimator, X, y, cv=5):
    scores = np.zeros(cv)
    folds = StratifiedKFold(n_splits=cv, random_state=estimator.random_state)
    for (train_indices, test_indices), idx in zip(folds.split(X, y), range(cv)):
        clone_estimator = copy.deepcopy(estimator)
        X_train_folds = X[train_indices]
        y_train_folds = y[train_indices]
        X_test_fold = X[test_indices]
        y_test_fold = y[test_indices]
        clone_estimator.fit(X_train_folds, y_train_folds)
        y_pred = clone_estimator.predict(X_test_fold)
        scores[idx] = accuracy_score(y_test_fold, y_pred)
    return scores


def main():
    data = np.loadtxt("./diabetes.txt")
    X = data[:, 1: 9]
    y = data[:, 9].ravel()
    mlp = MLP(hidden_layer_sizes=(10, 10), solver='ga', crossover_rate=0.8, mutation_rate=0.05, max_iter=1000,
              batch_size=64, learning_rate_init=0.001, pop_size=50, n_iter_no_change=1000)
    scores = cross_val_score(mlp, X, y, 20)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    main()
