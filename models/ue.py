from numba.experimental import jitclass
from numba import int32, float64


import settings

spec = [
    ("id", int32),
    ("x", float64),
    ("y", float64),
    ("height", float64),
    ("rate_requirement", float64),
    ("mno", int32),
]


# @jitclass(spec)
class UserEquipment:
    def __init__(self, id: int, x: float, y: float, rate_requirement: float, mno: int, province=None, municipality=None):
        self.id = id
        self.x = x
        self.y = y
        self.height = settings.UE_HEIGHT
        self.rate_requirement = rate_requirement
        self.mno = mno
        self.channel = None
        self.power = 0
        self.province = province
        self.municipality = municipality

    def __str__(self):
        return "UE[{}], requested capacity: {}, x: {}, y: {}".format(self.id, self.rate_requirement, self.x, self.y)