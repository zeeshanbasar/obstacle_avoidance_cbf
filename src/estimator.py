import numpy as np
from src import limits

class Sensor():

    def __init__(self, isNoise=False, seed=42, rate_hz=50.0, dropout_seed=0, isDrop=False, p_drop=0):

        self.rng = np.random.default_rng(seed)
        self.rng_drop = np.random.default_rng(dropout_seed)
        self.isDrop = isDrop
        self.isNoise = isNoise
        self.rate_hz = rate_hz
        self.t_last_update = -np.inf
        self.x_hat = None
        self.age = 0.0
        self.p_drop = p_drop
        self.t_next = 0.0
        self.max_age = -np.inf

    def measure(self, x, t):

        if t >= self.t_next - 1e-9:
            xhat = x.copy()
            if self.isNoise:
                xhat[0] += self.rng.normal(0,limits.SIGMA_PX)
                xhat[1] += self.rng.normal(0,limits.SIGMA_PY)
                xhat[2] += self.rng.normal(0,limits.SIGMA_PSI)
                xhat[3] += self.rng.normal(0,limits.SIGMA_V)
            if self.x_hat is None or not self.isDrop or self.rng_drop.random() >= self.p_drop:
                self.x_hat = xhat
                self.t_last_update = t

            self.t_next = t + 1.0/self.rate_hz

        self.age = t - self.t_last_update
        if self.age > self.max_age:
            self.max_age = self.age
        return self.x_hat.copy()

    def reseed(self, seed=42, dropout_seed=0):

        self.rng = np.random.default_rng(seed)
        self.rng_drop = np.random.default_rng(dropout_seed)
        self.t_last_update = -np.inf
        self.x_hat = None
        self.age = 0.0
        self.t_next = 0.0
        self.max_age = -np.inf