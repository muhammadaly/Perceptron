from numpy import zeros
from .config import G_MEAN_DIM


class PredictionModel:
    @staticmethod
    def predict(self, track, dt):
        pass


class MeasurementModel:
    @staticmethod
    def update(self, track, measurements):
        pass


class UKFConfig:
    def __init__(self):
        self.alpha = 0.5
        self.beta = 0.1
        self.lamda = 0.4


class UnscentedTransform:
    def transform_state(self, m, P):
        pass

    def transform_point(self, pnt_m, pnt_P):
        pass

    def compute_sigma_points(self, m , P):
        pass


class UKF:
    def __init__(self):
        self.config = UKFConfig()

    def predict(self, track, dt):
        pass

    def update(self, track, measurement):
        pass


class Tracker:
    def __init__(self):
        self.ukf = UKF()

    def predict(self, tracks_list, dt):
        for track in tracks_list:
            self.ukf.predict(track, dt)
        return tracks_list

    def correct(self, track_list, measurements_list):
        for measurement in measurements_list:
            track = track_list.get_by_id(measurement.associated_track_id)
            self.ukf.update(track, dt)
        return tracks_list
