import settings
from itertools import groupby
import math


def macro_bs_const(n_sector, sleep):
    if not sleep:
        return (n_sector * settings.RECTIFIER_POWER) \
               + settings.MICROWAVE_LINK_POWER + settings.MACRO_AIR_CONDITIONING_POWER
    return n_sector * settings.P_SLEEP_MACRO


def micro_bs_const(n_sector, sleep):
    if not sleep:
        return (n_sector * settings.RECTIFIER_POWER) + settings.MICRO_AIR_CONDITIONING_POWER
    return n_sector * settings.P_SLEEP_MICRO


def d2d(x1, y1, x2, y2):
    return math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))


def bs_load(antenna_groups):
    p_load = 0
    sector_count = 0
    p_amp = 0
    for _, antenna_channels in antenna_groups:
        p_out = sum([ue.power for c in antenna_channels for ue in c.users])
        power_amplifier = p_out / settings.EFFICIENCY_POWER_AMP_MIN
        p_amp += power_amplifier
        p_load += (power_amplifier + settings.TRANSCEIVER_POWER)
        sector_count += 1

    p_load += (sector_count * settings.DIGITAL_SIGNAL_PROCESSING_POWER)
    return p_load, sector_count, p_amp
        

def bs_power(bs, category=None, sleep=False):
    key = lambda c: (c.main_direction, c.height)
    antenna_groups = groupby(sorted(bs.channels.copy(), key=key), key=key)
    p_load, n_sector, p_amp = bs_load(antenna_groups)

    p_load = p_load * get_fi()

    if sleep:
        p_load = 0

    if category == 'amp':
        return p_amp
    elif category == 'load':
        return p_load

    if n_sector >= 3:
        p_const = macro_bs_const(n_sector, sleep)
    else:
        p_const = micro_bs_const(n_sector, sleep)

    if category == 'const':
        return p_const

    if category == 'all':
        return p_const, p_load - p_amp * get_fi(), p_amp * get_fi()

    return p_const + p_load * get_fi()


def get_fi():
    pop_percentage = 100 * (settings.PERCENTAGE_PER_ZIP/100) * (settings.PERCENTAGE_ACTIVE/100)
    return settings.LOAD_FACTOR_LOW + ((settings.LOAD_FACTOR_HIGH - settings.LOAD_FACTOR_LOW) *
                                       (pop_percentage / settings.MAX_USER_PERCENTAGE))
