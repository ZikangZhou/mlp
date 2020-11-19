import numpy as np


class BaseOptimizer:

    def __init__(self, params):
        self.params = [param for param in params]

    def update_params(self, grads=None):
        updates = self._get_updates(grads)
        for param, update in zip(self.params, updates):
            param += update

    def iteration_ends(self, time_step):
        pass

    def trigger_stopping(self):
        return True


class SGDOptimizer(BaseOptimizer):

    def __init__(self, params, learning_rate_init=0.1, lr_schedule='constant',
                 momentum=0.9, nesterov=True, power_t=0.5):
        super().__init__(params)
        self.learning_rate_init = learning_rate_init
        self.learning_rate = float(learning_rate_init)
        self.lr_schedule = lr_schedule
        self.momentum = momentum
        self.nesterov = nesterov
        self.power_t = power_t
        self.velocities = [np.zeros_like(param) for param in params]

    def iteration_ends(self, time_step):
        if self.lr_schedule == 'invscaling':
            self.learning_rate = (float(self.learning_rate_init) /
                                  (time_step + 1) ** self.power_t)

    def trigger_stopping(self):
        if self.lr_schedule != 'adaptive':
            return True
        if self.learning_rate <= 1e-6:
            return True
        self.learning_rate /= 5.
        return False

    def _get_updates(self, grads):
        updates = [self.momentum * velocity - self.learning_rate * grad
                   for velocity, grad in zip(self.velocities, grads)]
        self.velocities = updates

        if self.nesterov:
            updates = [self.momentum * velocity - self.learning_rate * grad
                       for velocity, grad in zip(self.velocities, grads)]

        return updates


class AdamOptimizer(BaseOptimizer):

    def __init__(self, params, learning_rate_init=0.001, beta_1=0.9,
                 beta_2=0.999, epsilon=1e-8):
        super().__init__(params)
        self.learning_rate_init = learning_rate_init
        self.learning_rate = float(learning_rate_init)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]

    def _get_updates(self, grads):
        self.t += 1
        self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad
                   for m, grad in zip(self.ms, grads)]
        self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
                   for v, grad in zip(self.vs, grads)]
        self.learning_rate = (self.learning_rate_init *
                              np.sqrt(1 - self.beta_2 ** self.t) /
                              (1 - self.beta_1 ** self.t))
        updates = [-self.learning_rate * m / (np.sqrt(v) + self.epsilon)
                   for m, v in zip(self.ms, self.vs)]
        return updates


class GAOptimizer(BaseOptimizer):

    def __init__(self, params):
        super().__init__(params)

    def _get_updates(self, grads):
        return grads
