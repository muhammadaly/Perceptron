from numpy import zeros
from .gaussian_state import GaussianState
from .config import OBJECT_DIMENSIONALITY


class TrackerObject:
    def __init__(self):
        self.state = GaussianState()
        self.dimension = zeros(OBJECT_DIMENSIONALITY)
        self.existence_probability = 0.
