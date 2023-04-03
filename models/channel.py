from numba import int32, float64
from numba.experimental import jitclass



spec = [
    ("id", int32),
    ("BS_id", int32),
    ("height", float64),
    ("frequency", float64),
    ("power", float64),
    ("main_direction", float64),
    ("bandwidth", float64),
    ("beamwidth", float64),
    ("users", int32[:]),
    ("bs_interferers", int32[:]),
]


# @jitclass(spec)
class Channel:
    def __init__(self, id, BS_id, height, frequency, power, main_direction, mno, bandwidth, beamwidth=360):
        self.id = id
        self.BS_id = BS_id
        self.height = height
        self.frequency = frequency
        self.power = power
        self.main_direction = main_direction
        self.mno = mno
        self.bandwidth = bandwidth
        self.beamwidth = beamwidth

        self.users = list()
        self.bs_interferers = list()

    @property
    def connected_users(self):
        return len(self.users)

    def add_user(self, user):
        self.users.append(user)
