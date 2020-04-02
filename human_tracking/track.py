from .tracker_object import TrackerObject
from common.data_structures.List import List, ListItem


class Track(TrackerObject, ListItem):
    def __init__(self):
        self.id = 0
        super().__init__()
