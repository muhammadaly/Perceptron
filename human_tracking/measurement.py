from .tracker_object import TrackerObject
from enum import Enum


class ClassificationTypes(Enum):
    Unknown = 0
    Human = 1
    Bag = 2
    Animal = 3


class Measurement(TrackerObject):
    def __init__(self):
        self.classification = ClassificationTypes.Unknown
        super().__init__()

