import numpy as np

class SGD:
    def __init__(self, eps=0.01):
        self.eps = eps

    def update(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.eps * g

class Adam:
    def __init__(self, eps=0.001, beta1=0.9, beta2=0.999):
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [],[]
            for p in params:
                self.m.append(np.zeros_like(p))
                self.v.append(np.zeros_like(p))

        self.iter += 1
        eps_t  = self.eps * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        i = 0

        for p, g in zip(params, grads):
            self.m[i] += (1 - self.beta1) * (g - self.m[i])
            self.v[i] += (1 - self.beta2) * (g**2 - self.v[i])

            p -= eps_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
            i += 1

opt = {'SGD':SGD, 'Adam':Adam}
