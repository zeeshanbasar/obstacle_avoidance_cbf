import numpy as np
from src import limits

class Sensor():

    def __init__(self, isNoise=False, seed=42, rate_hz=50.0):

        self.rng = np.random.default_rng(seed)
        self.isNoise = isNoise
        self.rate_hz = rate_hz
        self.t_last_update = -np.inf
        self.x_hat = None
        self.age = 0.0

    def measure(self, x, t):

        self.age = t - self.t_last_update

        if self.age >= 1./self.rate_hz - 1e-9:
            xhat = x.copy()
            if self.isNoise:
                xhat[0] += self.rng.normal(0,limits.SIGMA_PX)
                xhat[1] += self.rng.normal(0,limits.SIGMA_PY)
                xhat[2] += self.rng.normal(0,limits.SIGMA_PSI)
                xhat[3] += self.rng.normal(0,limits.SIGMA_V)

            self.x_hat = xhat
            self.t_last_update = t

        self.age = t - self.t_last_update
        return self.x_hat.copy()

    def reseed(self, seed):

        self.rng = np.random.default_rng(seed)
        self.t_last_update = -np.inf
        self.x_hat = None
        self.age = 0.0
