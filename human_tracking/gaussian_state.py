from numpy import zeros
from .config import G_MEAN_DIM, G_COVARIANCE_DIM


class GaussianState:
    def __init__(self):
        self.mean = zeros(G_MEAN_DIM)
        self.covariance = zeros(G_COVARIANCE_DIM)
