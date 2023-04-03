import base.util as util
from models.channel import Channel
from numba.experimental import jitclass
from numba import int32, float64, boolean

spec = [
    ("id", int32),
    ("radio", int32),
    ("x", float64),
    ("y", float64),
    ("provider", int32),
    ("small_cell", boolean),
    ("channels", int32[:]),
    ("interferers", float64[:]),
    ("area_type", int32),
    ("frequencies", float64[:]),
]


# @jitclass(spec)
class BaseStation:
    def __init__(self, id, radio, x, y, provider=-1, small_cell=False, province=None, municipality=None):
        self.id = id
        self.radio = radio
        self.x = float(x)
        self.y = float(y)
        self.provider = provider
        self.small_cell = small_cell

        self.channels = list()
        self.interferers = list()
        self.area_type = -1

        self.frequencies = set()
        self.province = province
        self.municipality = municipality

    def __str__(self):
        y = self.y
        x = self.x
        radio = str(self.radio)
        startmsg = f"Base station[{self.id}], {x=}, {y=}, {radio=}"
        for channel in self.channels:
            startmsg += "\n\t{}".format(str(channel))
        return startmsg

    def __repr__(self):
        return

    def add_channel(self, id, BS_id, height, frequency, power, angle, mno, bandwidth):
        channel = Channel(id, BS_id, height, frequency, power, angle, mno, bandwidth, beamwidth=360)
        self.channels.append(channel)

# import base.base as base
# from models.channel import Channel
#
#
# class BaseStation:
#     def __init__(self, id, radio, x, y, provider=-1, small_cell=False):
#         self.id = id
#         self.radio = radio
#         self.x = float(x)
#         self.y = float(y)
#         self.provider = provider
#         self.small_cell = small_cell
#
#         self.channels = list()
#         self.interferers = list()
#         self.area_type = base.AreaType
#         self.frequencies = set()
#
#     def __str__(self):
#         y = self.y
#         x = self.x
#         radio = str(self.radio)
#         startmsg = f"Base station[{self.id}], {x=}, {y=}, {radio=}"
#         for channel in self.channels:
#             startmsg += "\n\t{}".format(str(channel))
#         return startmsg
#
#     def __repr__(self):
#         return f"BS[{self.id}]: {self.x=},{self.y=},{self.radio=},#Channels={len(self.channels)}"
#
#     def add_channel(self, id, BS_id, height, frequency, power, angle, bandwidth):
#         channel = Channel(id, BS_id, height, frequency, power, angle, bandwidth, beamwidth=360)
#         self.channels.append(channel)
#
