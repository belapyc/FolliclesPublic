import numpy as np
from enum import Enum


class Bin:
    def __init__(self, follicles, bin_type):
        self.follicles = follicles
        self.size = len(follicles)
        self.median = np.median(follicles)
        self.mean = np.mean(follicles)
        self.max = max(follicles)
        self.min = min(follicles)
        self.bin_type = bin_type

    def __str__(self):
        return str(self.follicles)


class BinType(Enum):
    BOTTOM_QUARTILE = 1
    LOWER_MID_QUARTILE = 2
    UPPER_MID_QUARTILE = 3
    TOP_QUARTILE = 4
