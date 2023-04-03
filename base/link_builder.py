import math
from numba import cuda, int32, njit
from numba.cuda import random
import numpy as np
import base.energy_calc as ec
import settings
import base.util as util


@njit
def d2d(x1, y1, x2, y2):
    return math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))


@njit
def d3d(h1, h2, d__2d):
    d_h = abs(h1 - h2)
    return math.sqrt(d__2d ** 2 + d_h ** 2)


@njit
def hor_gain(bore, geo):
    hor_angle = bore - geo
    if hor_angle > 180:
        hor_angle = hor_angle - 360
    return - min(12 * (hor_angle / settings.HORIZONTAL_BEAMWIDTH3DB) ** 2, 20)


@njit
def find_antenna_gain(bore, geo):
    a_ver = 0
    a_hor = hor_gain(bore, geo)
    return 20 + a_hor + a_ver


@njit
def degrees(radians):
    return 360 * (radians / (2 * math.pi))


@njit
def angle(c1_x, c1_y, c2_x, c2_y):
    dy = c2_y - c1_y
    dx = c2_x - c1_x
    radians = math.atan2(dy, dx)
    return degrees(radians)


@njit
def los_probability(d__2d, area):
    if area == settings.AREA_UMA:
        if d__2d <= 18:
            return 1
        else:
            c = 0 if settings.UE_HEIGHT <= 13 else ((settings.UE_HEIGHT - 13) / 10) ** 1.5
            return (18 / d__2d + math.exp(-d__2d / 63) * (1 - 18 / d__2d)) * (
                    1 + c * (5 / 4) * (d__2d / 100) ** 3 * math.exp(-d__2d / 150))
    elif area == settings.AREA_RMA:
        if d__2d <= 10:
            return 1
        else:
            return math.exp(-((d__2d - 10) / 1000))
    elif area == settings.AREA_UMI:
        if d__2d <= 18:
            return 1
        else:
            return 18 / d__2d + math.exp(-d__2d / 36) * (1 - 18 / d__2d)


@njit
def breakpoint_distance(c_freq, c_height, e_type):
    c_ = 3.0 * 10 ** 8
    if e_type == 'urban':
        effective_height = 1
        return max(1, 4 * (c_height - effective_height) * (settings.UE_HEIGHT - effective_height) * c_freq / c_)
    else:
        return max(1, 2 * math.pi * c_height * settings.UE_HEIGHT * c_freq / c_)


@njit
def path_loss_urban_los(d__2d, d__3d, c_freq, c_height, a_, b_, c_):
    bp = breakpoint_distance(c_freq, c_height, 'urban')

    if d__2d < 10:
        return a_ + b_ * math.log10(d3d(c_height, settings.UE_HEIGHT, 10)) + 20 * math.log10(c_freq / 1e9)  #
    elif d__2d <= bp:
        return a_ + b_ * math.log10(d__3d) + 20 * math.log10(c_freq / 1e9)
    elif d__2d <= 5000:
        return a_ + 40 * math.log10(d__3d) + 20 * math.log10(c_freq / 1e9) \
               - c_ * math.log10(bp ** 2 + (c_height - settings.UE_HEIGHT) ** 2)


@njit
def pathloss_urban_nlos(d__3d, c_freq, a, b, c, d):
    return a + b * math.log10(d__3d) + c * math.log10(c_freq / 1e9) - d * (settings.UE_HEIGHT - 1.5)


@njit
def pathloss_rma_los_pl1(d__3d, build_height, c_freq):
    freq = c_freq / 1e9
    a = 40 * math.pi * d__3d * freq / 3
    hp = build_height ** 1.72
    return 20 * math.log10(a) + min(0.03 * hp, 10) * math.log10(d__3d) - min(0.044 * hp, 14.77) + \
           0.002 * math.log10(build_height) * d__3d


@njit
def get_path_loss(distance_2d, distance_3d, area_type, c_freq, c_height, rng_states_chance, rng_states_fade, pos):
    avg_building_height = settings.AVG_BUILDING_HEIGHT
    avg_street_width = settings.AVG_STREET_WIDTH

    chance = random.xoroshiro128p_uniform_float64(rng_states_chance, pos)

    fading4 = 4 * random.xoroshiro128p_normal_float64(rng_states_fade, pos)
    fading6 = 6 * random.xoroshiro128p_normal_float64(rng_states_fade, pos)
    fading78 = 7.8 * random.xoroshiro128p_normal_float64(rng_states_fade, pos)
    fading8 = 8 * random.xoroshiro128p_normal_float64(rng_states_fade, pos)

    rain = 0
    los = chance <= los_probability(distance_2d, area_type)

    pl = math.inf

    if area_type == settings.AREA_UMA and distance_2d <= 5000:
        pl_los = path_loss_urban_los(distance_2d, distance_3d, c_freq, c_height, 28, 22, 9)
        if los:
            pl = pl_los + rain + fading4
        else:
            pl_nlos = pathloss_urban_nlos(distance_3d, c_freq, 13.54, 39.08, 20, 0.6)
            pl = max(pl_los, pl_nlos) + rain + fading6

    elif area_type == settings.AREA_UMI and distance_2d <= 5000:
        pl_los = path_loss_urban_los(distance_2d, distance_3d, c_freq, c_height, 32.4, 21, 9.5)
        if los:
            pl = pl_los + rain + fading4
        else:
            pl_nlos = pathloss_urban_nlos(distance_3d, c_freq, 22.4, 35.3, 21.3, 0.3)
            pl = max(pl_los, pl_nlos) + rain + fading78

    elif area_type == settings.AREA_RMA and distance_2d <= 10000:
        if los:
            bp = breakpoint_distance(c_freq, c_height, '')
            if distance_2d < 10:
                pl = pathloss_rma_los_pl1(d3d(c_height, settings.UE_HEIGHT, 10), avg_building_height, c_freq)
            elif distance_2d <= bp:
                pl = pathloss_rma_los_pl1(distance_3d, avg_building_height, c_freq) + rain + fading4
            elif distance_2d <= 10000:
                pl1 = pathloss_rma_los_pl1(bp, avg_building_height, c_freq)
                pl = pl1 + 40 + math.log10(distance_3d / bp) + rain + fading6
        else:
            if distance_2d < 10:
                distance_3d = d3d(c_height, settings.UE_HEIGHT, 10)
                nlos_pl = 161.04 - 7.1 * math.log10(avg_street_width) + 7.5 * math.log10(avg_building_height) \
                          - (24.37 - 3.7 * (avg_building_height / max(1, c_height)) ** 2) * math.log10(
                    c_height) + (43.42 - 3.1 * math.log10(c_height)) * (math.log10(distance_3d) - 3) \
                          + 20 * math.log10(c_freq / 1e9) - (
                                  3.2 * math.log10(11.75 * settings.UE_HEIGHT) - 4.97)
                los_pl = path_loss_urban_los(10, distance_3d, c_freq, c_height, 28, 22, 9)
                pl = max(los_pl, nlos_pl) + rain + fading8
            elif distance_2d <= 5000:
                nlos_pl = 161.04 - 7.1 * math.log10(avg_street_width) + 7.5 * math.log10(avg_building_height) \
                          - (24.37 - 3.7 * (avg_building_height / max(1, c_height)) ** 2) * math.log10(
                    c_height) + (43.42 - 3.1 * math.log10(c_height)) * (math.log10(distance_3d) - 3) \
                          + 20 * math.log10(c_freq / 1e9) - (
                                  3.2 * math.log10(11.75 * settings.UE_HEIGHT) - 4.97)
                los_pl = path_loss_urban_los(distance_2d, distance_3d, c_freq, c_height, 28, 22, 9)
                pl = max(los_pl, nlos_pl) + rain + fading8
    return pl


@njit
def thermal_noise(bandwidth):
    tn = settings.BOLTZMANN * settings.TEMPERATURE * bandwidth
    return (10 * math.log10(tn)) + 30  # 30 is to go from dBW to dBm


@njit
def find_noise(bandwidth, radio):
    if radio == settings.RADIO_5G_NR:
        noise_figure = 7.8
    else:
        noise_figure = 5
    return thermal_noise(bandwidth) + noise_figure


@njit
def to_pwr(db):
    return 10 ** (db / 10)


@njit
def to_db(pwr):
    return 10 * math.log10(pwr)


@njit
def is_in(item, l):
    for e in l:
        if item == e:
            return True
    return False


@njit
def get_wavelength(frequency):
    return settings.LIGHT / frequency


# @njit
def friis_transmission_power(p_r, g_t, L):
    p_t = p_r - g_t + L
    return p_t  # - 20 * math.log10(get_wavelength(f)/(3*math.pi*d))


@njit
def shannon_power(C, B, N):
    return N * (2 ** ((C/B)- 1))


"""
Since shared arrays are not feasible, we must create an ordered list of best channel
options for each user. Then we can check (outside of cuda) what division is best.
"""


@cuda.jit
def kernel_snr_calculation(u_data, c_data, bs_data, rng_states_chance, rng_states_fade, BS_ID, BS_DISTS, SNRS, LOSS, NOISE):
    pos = cuda.grid(1)
    if pos < u_data[0].size:

        # Get user data from matrix
        u_ids, u_xs, u_ys, u_mnos = u_data[0], u_data[1], u_data[2], u_data[3]
        bs_xs, bs_ys, bs_mnos, bs_ats, bs_rads \
            = bs_data[1], bs_data[2], bs_data[3], bs_data[4], bs_data[5]
        u_id, u_x, u_y, u_mno = u_ids[pos], u_xs[pos], u_ys[pos], u_mnos[pos]

        for i in range(len(BS_ID[pos])):
            bs_id = BS_ID[pos][i]
            d_2d = BS_DISTS[pos][i]

            channels = c_data[bs_id]
            for j in range(len(channels)):
                c = channels[j]
                # Extract channel data
                c_id, c_bs_id, c_h, c_p, c_md, c_bw, c_f = int(c[0]), int(c[1]), c[2], c[3], c[4], c[5], c[6]

                # Ignore filler channel rows in matrix
                if c_f != 0.0:
                    # Extract base station data
                    bs_x, bs_y, bs_at, bs_rad = bs_xs[c_bs_id], bs_ys[c_bs_id], bs_ats[c_bs_id], bs_rads[c_bs_id]

                    d_3d = d3d(settings.UE_HEIGHT, c_h, d_2d)

                    # Calculation to conform to old antenna gain model
                    antenna_gain = -(20 - find_antenna_gain(c_md, angle(bs_x, bs_y, u_x, u_y)))
                    path_loss = get_path_loss(d_2d, d_3d, bs_at, c_f, c_h, rng_states_chance, rng_states_fade, pos)
                    bandwidth = c_bw
                    radio = bs_rad
                    noise = find_noise(bandwidth, radio)
                    # snr_value = (min(c_p, settings.MAX_CHANNEL_POWER) + antenna_gain) - (path_loss + noise)
                    snr_value = (c_p + antenna_gain) - (path_loss + noise)

                    SNRS[pos][i][j] = snr_value
                    LOSS[pos][i][j] = path_loss
                    NOISE[pos][i][j] = noise - 30


"""
Kernel that finds the closest BS to all users
@param bs_is: indices (NOT IDs) in RES that correspond to lowest 10 distanced base stations
"""

@cuda.jit
def kernel_find_bs_close(u_data, bs_data, BS_IDS, BS_DISTS):
    pos = cuda.grid(1)

    # Get user data from matrix
    u_ids, u_xs, u_ys, u_mnos = u_data[0], u_data[1], u_data[2], u_data[3]
    # Get base station data from matrix
    bs_ids, bs_xs, bs_ys, bs_mnos = bs_data[0], bs_data[1], bs_data[2], bs_data[3]

    if pos < u_ids.size:
        u_id, u_x, u_y, u_mno = u_ids[pos], u_xs[pos], u_ys[pos], u_mnos[pos]

        # Sort the 30 closest base stations and store the result in BS_IDS[pos]
        # WARNING: This calculation is extremely expensive and can be heavily optimized
        for i in range(BS_DISTS.shape[1]):
            m_i = 0
            m = np.inf
            for bs_index in range(len(bs_mnos)):
                # bs_id = int(bs_id)  # Get int from float
                # Find the closest base stations of supported operator
                # 1. Compare user operator to bs operator (U-mno == -1 means all operators are supported)
                # 2. Check if bs was not already found
                # if (u_mno == -1 or bs_mnos[bs_index] == 1 or bs_mnos[bs_index] == -1) \
                if (u_mno == bs_mnos[bs_index] or settings.SHARING) \
                        and not is_in(bs_index, BS_IDS[pos][:i]):
                    dist = d2d(u_x, u_y, bs_xs[bs_index], bs_ys[bs_index])
                    if dist < m:
                        m = dist
                        m_i = bs_index
            BS_IDS[pos][i] = bs_ids[m_i]
            BS_DISTS[pos][i] = m


"""
Kernel that finds the required bandwidths for the best n
channels for each user
"""


@cuda.jit
def kernel_find_bandwidths(u_rr, CLOSE_IDS, CHAN_PREFS, BEST_CHANNELS, BANDWIDTH_REQ, SNRS):
    pos = cuda.grid(1)
    if pos < BANDWIDTH_REQ.shape[0]:
        RATE = u_rr[pos]

        def shannon(rate, snr):
            return rate / math.log2(1 + to_pwr(snr))

        for i in range(CLOSE_IDS.shape[1]):
            # Index from 0..29 in array of size 30
            bs_id = CLOSE_IDS[pos][i]
            bs_count = 0
            for j in range(CHAN_PREFS.shape[2]):
                if bs_count > 2:
                    break
                snr = CHAN_PREFS[pos][i][j]  # Index of 0..34 in array of size 35, value is a channel SNR
                if snr < settings.MINIMUM_SNR:
                    # Ignore bad snr values
                    continue

                c_id = j

                start = -1
                for index in range(BANDWIDTH_REQ[pos].shape[0]):  # Loop through indices of array
                    snr_b = BANDWIDTH_REQ[pos][index]   # At this point, BANDWIDTH_REQ is filled with snrs
                    if snr > snr_b:  # Sorted array, so if value > RES[pos][i], we have insert index
                        start = index  # Set start to index
                        bs_count += 1
                        break  # Break out of loop

                ind = BANDWIDTH_REQ[pos].shape[0] - 1
                if start > ind or start < 0:  # Impossible shift
                    continue
                while True:
                    if ind == start:
                        BANDWIDTH_REQ[pos][ind] = snr  # Insert value
                        BEST_CHANNELS[pos][ind][0] = bs_id  # Insert base station id
                        BEST_CHANNELS[pos][ind][1] = c_id  # Insert channel id
                        break
                    BANDWIDTH_REQ[pos][ind] = BANDWIDTH_REQ[pos][ind - 1]  # Shift values right
                    BEST_CHANNELS[pos][ind][0] = BEST_CHANNELS[pos][ind - 1][0]  # Shift base station ids right
                    BEST_CHANNELS[pos][ind][1] = BEST_CHANNELS[pos][ind - 1][1]  # Shift channel ids right
                    ind = ind - 1  # Reduce index
        for z in range(BANDWIDTH_REQ[pos].shape[0]):
            SNRS[pos][z] = BANDWIDTH_REQ[pos][z]
            BANDWIDTH_REQ[pos][z] = shannon(RATE, BANDWIDTH_REQ[pos][z])  # Replace SNR values with bw requirement


def create_links(users, user_xs, user_ys, base_station_xs, base_station_ys, base_station_mnos, bss, all_chans):
    cuda.current_context().deallocations.clear()
    CLOSE_DISTS_gpu, CLOSE_IDS_gpu, bs_data_gpu, bs_ids, bs_mnos, bs_xs, bs_ys, u_data_gpu, u_ids, u_mnos = \
        format_and_transfer_distance_data(base_station_mnos, base_station_xs, base_station_ys, bss
                                          , user_xs, user_ys, users)

    bs_data_at_gpu, c_data_gpu, m_channels, CHANNEL_PREFS_gpu, LOSS_gpu, NOISE_gpu = \
        format_and_transfer_channel_data(
            all_chans, bs_ids, bs_mnos, bs_xs, bs_ys, bss, len(user_xs))

    u_rr_gpu, CHANNEL_BANDWIDTHS_gpu, BEST_CHANNELS_gpu, SNRS_gpu = format_and_transfer_bandwidth_data(users)

    threadsperblock = 32
    blockspergrid = u_ids.size + (threadsperblock - 1)
    rng_states_chance = cuda.to_device(random.create_xoroshiro128p_states(threadsperblock * blockspergrid
                                                                          , seed=np.random.randint(0, 100)))
    rng_states_fade = cuda.to_device(random.create_xoroshiro128p_states(threadsperblock * blockspergrid
                                                                          , seed=1))

    # Calculate 10 closest base stations to each user and put their ids and respective distances in the matrices
    # Calculate 10 closest base stations to each30, user and put their ids and respective distances in the matrices
    # CLOSE_IDS_gpu and CLOSE_DISTS_gpu with formats:
    # CLOSE_IDS_gpu[u_id]   = [bs_id1, bs_id2, ..., bs_id_10]
    # CLOSE_DISTS_gpu[u_id] = [1000.3121, 1200,2321, ..., 4034.9832]
    kernel_find_bs_close[blockspergrid, threadsperblock](u_data_gpu, bs_data_gpu, CLOSE_IDS_gpu, CLOSE_DISTS_gpu)

    # Calculate the Signal to Noise ratio for the channels of the previously determined n closest base stations.
    # Results end up in CHANNEL_PREFS_gpu with format
    # CHANNEL_PREFS_gpu[u_id][0] = [5.1, -1.2, ... 18.2]
    # ...
    # CHANNEL_PREFS_gpu[u_id][n] = [-2.4, 10.2, ... -8.2]
    kernel_snr_calculation[blockspergrid, threadsperblock](u_data_gpu, c_data_gpu, bs_data_at_gpu, rng_states_chance
                                                           , rng_states_fade, CLOSE_IDS_gpu, CLOSE_DISTS_gpu
                                                           , CHANNEL_PREFS_gpu, LOSS_gpu, NOISE_gpu)

    # Reduce the CHANNEL_PREFS_gpu matrix that has size len(u_ids) * n * m_channels to a smaller size that only
    # includes data for the 5 best channel options.
    # Results end up in matrices BEST_CHANNELS_gpu and BEST_CHANNELS_SNR_gpu with formats:
    # BEST_CHANNELS_gpu[u_id]      = [[bs_id1, c_id1], [bs_id2, c_id3], ... [bs_id5, c_id5]]
    # BEST_CHANNELS_SNR_gpu[u_id]  = [13.2, 12.1, 10.9, 9.6, 1.0]
    kernel_find_bandwidths[blockspergrid, threadsperblock](u_rr_gpu, CLOSE_IDS_gpu
                                                           , CHANNEL_PREFS_gpu, BEST_CHANNELS_gpu
                                                           , CHANNEL_BANDWIDTHS_gpu, SNRS_gpu)
    # print("result sample...")
    # for i in range(10):
    #     print(f'u_id = {i}')
    #     print(CLOSE_DISTS_gpu.copy_to_host()[i])
    #     # print(CHANNEL_PREFS_gpu.copy_to_host()[i])
    #     print(BEST_CHANNELS_gpu.copy_to_host()[i])
    #     print(CHANNEL_BANDWIDTHS_gpu.copy_to_host()[i])

    sleep_prefs = store_bs_sleep_prefs(BEST_CHANNELS_gpu.copy_to_host(), len(bss))

    build_links(CHANNEL_BANDWIDTHS_gpu.copy_to_host(), BEST_CHANNELS_gpu.copy_to_host(), bss, users
                , LOSS_gpu.copy_to_host()
                , CLOSE_IDS_gpu.copy_to_host(), NOISE_gpu.copy_to_host(), sleep_prefs)


def store_bs_sleep_prefs(chan_prefs, n_bs):
    counter = np.zeros(n_bs, dtype=np.int32)
    for pairs in chan_prefs:
        bs_set = set([pair[0] for pair in pairs])
        for bs_id in bs_set:
            counter[bs_id] += 1
    return counter


def get_index_from_bs_id(u_id, bs_id, close_ids):
    user_close_bs_ids = close_ids[u_id]
    for i in range(len(user_close_bs_ids)):
        if user_close_bs_ids[i] == bs_id:
            return i
    return -1


def get_path_loss_channel(u_id, bs_id, c_id, close_ids, path_losses):
    return path_losses[u_id][get_index_from_bs_id(u_id, bs_id, close_ids)][c_id]


def get_noise_channel(u_id, bs_id, c_id, close_ids, noises):
    return noises[u_id][get_index_from_bs_id(u_id, bs_id, close_ids)][c_id]


def get_n_channels_per_antenna(bs, c_0):
    return len([c for c in bs.channels if c.main_direction == c_0.main_direction and c.height == c_0.height])


def build_links(bw_s_per_user, best_pairs_per_user, bss, users, path_loss, close_ids, noises, sleep_prefs):
    print('Commence single-threading...')

    func = lambda z: len(set([int(pair[0]) for pair in best_pairs_per_user[z]]))
    for i in sorted(range(len(list(best_pairs_per_user))), key=func, reverse=True):
        user = users[int(i)]
        user.power_limit = link_user(best_pairs_per_user
                                     , bss, bw_s_per_user
                                     , close_ids
                                     , noises
                                     , path_loss
                                     , user
                                     , sleep_prefs)

    dissatisfied = 0
    disconnected = 0
    infs = 0
    zeros = 0
    # per_province = {}
    # for province in settings.provinces:
    #     per_province[province] = [0, 0, 0, 0, 0, 0]

    for user in users:
        if user.power_limit is True:
            # UE could not be connected with desired bitrate
            dissatisfied += 1
            if not user.power_limit:
                # The base station power limit is not the problem -> the UE cannot be connected at all
                disconnected += 1
                # per_province[user.province][5] += 1
                # debug_disconnected(best_pairs_per_user, bss, bw_s_per_user, close_ids, noises, path_loss, user)
        if user.power == 0:
            zeros += 1
        if user.power == math.inf:
            infs += 1

    p_const = 0
    p_amp = 0
    p_load = 0
    total_power = 0
    power_sleeping = 0
    bs_zeros = 0

    # Artificially increase disconnected rate in case we have sharing...
    l1 = lambda b: sum([len(c.users) for c in b.channels]) != 0
    l2 = lambda b: sum([len(c.users) for c in b.channels])
    bs_user_ordered = sorted(filter(l1, bss), key=l2, reverse=False)
    for bs in bs_user_ordered:
        if len(users) == 0:
            continue
        if round(disconnected/len(users), 4)*100 >= settings.ARTIFICAL_DISCONNECTED_PERCENTAGE:
            break
        for c in bs.channels:
            for u in c.users:
                disconnected += 1
                u.channel = None
                u.power = 0
            c.users.clear()

    for bs in bss:
        u_count = sum([len(c.users) for c in bs.channels])
        no_users = u_count == 0
        const, load, amp = ec.bs_power(bs, category='all', sleep=no_users)
        p_const += const
        p_load += load
        p_amp += amp
        total_power += const + load + amp

        if no_users:
            bs_zeros += 1
            power_sleeping += (const + load + amp)
        #     per_province[bs.province][2] += 1
        #     per_province[bs.province][4] += (const + load + amp) / 1e6
        #
        # per_province[bs.province][0] += u_count
        # per_province[bs.province][1] += 1
        # per_province[bs.province][3] += (const + load + amp) / 1e6

    print(f'Disconnected users: {disconnected}')
    print(f'BS zeros: {bs_zeros}')
    # for k, v in per_province.items():
    #     print(f'{k}\t{v[3]},{v[4]}')

    print(f'Active: {len(users)}, or {round(100*len(users)/17_813_121, 4)}% '
          f'({100 * (settings.PERCENTAGE_PER_ZIP/100) * (settings.PERCENTAGE_ACTIVE/100)}%)')
    if len(users) != 0:
        print(f'Dissatisfied users: {round(dissatisfied/len(users), 4)*100}%')
        print(f'Disconnected users: {round(disconnected/len(users), 4)*100}%')
    else:
        print(f'Dissatisfied users: {0}%')
        print(f'Disconnected users: {0}%')
    print(f'{p_const/1e6}\t{p_load/1e6}\t{p_amp/1e6}')
    print(f'Sleep: {power_sleeping/1e6}')
    print(f'Total: {total_power/1e6}')
    print('')
    print(f'-----------------------------{100 * (settings.PERCENTAGE_PER_ZIP/100) * (settings.PERCENTAGE_ACTIVE/100)}%'
          f'------------------------------------')
    print('')


def cursed_sort(pairs, bss, bw_s, sleep_prefs):
    tups = np.asarray([np.asarray([pairs[i][0], pairs[i][1], bw_s[i]]) for i in range(len(pairs)) if pairs[i][0] != -1])

    any_users = lambda tup: any(len(c.users) > 0 for c in bss[int(tup[0])].channels)
    no_users = lambda tup: all(len(c.users) == 0 for c in bss[int(tup[0])].channels)
    sleep_order = lambda tup: sleep_prefs[int(tup[0])]

    group_0 = np.asarray([tup for tup in tups if any_users(tup)])
    group_1 = np.asarray(sorted([tup for tup in tups if no_users(tup)], key=sleep_order, reverse=True))

    if len(group_0) != 0 and len(group_1) != 0:
        res = np.concatenate((group_0, group_1))
    elif len(group_1) != 0:
        res = group_1
    else:
        res = group_0

    # res = np.asarray(sorted([tup for tup in tups], key=sleep_order, reverse=True))

    for i in range(0, len(res)):
        pairs[i] = [res[i][0], res[i][1]]
        bw_s[i] = res[i][2]
    for i in range(len(res), len(pairs)):
        pairs[i] = [-1, -1]
        bw_s[i] = -1

    return pairs, bw_s


def link_user(best_pairs_per_user, bss, bw_s_per_user, close_ids, noises, path_loss, user, sleep_prefs):
    power_limit = False
    # print(len(set(best_pairs_per_user[i][f][0] for f in range(30))))
    if settings.E_ORDER:
        # Sort base stations to optimize sleep pattern
        # key = lambda pair: any(len(c.users) > 0 for c in bss[pair[0]].channels)
        # best_pairs_per_user[i] = sorted(best_pairs_per_user[i].copy(), key=key, reverse=True)
        best_pairs_per_user[user.id], bw_s_per_user[user.id] \
            = cursed_sort(best_pairs_per_user[user.id].copy()
                          , bss, bw_s_per_user[user.id].copy()
                          , sleep_prefs)

    for j in range(len(best_pairs_per_user[user.id])):
        bs_id = int(best_pairs_per_user[user.id][j][0])
        c_id = int(best_pairs_per_user[user.id][j][1])

        if bs_id == -1 or c_id == -1:
            continue

        try:
            base_station = bss[bs_id]
            channel = base_station.channels[c_id]  # Index, not actual channel id!
        except IndexError:
            print(bs_id, c_id, user.id)
            continue
        user_bandwidth = bw_s_per_user[user.id][j] * 4

        user.user_bandwidth = user_bandwidth
        user.power = 0
        user.channel = None

        if not hasattr(channel, 'power_budget'):
            channel.power_budget = channel.power

        channel.add_user(user)
        user.channel = channel

        BWs = [u.user_bandwidth for u in channel.users]
        # Reset power budget before recalculation
        channel.power_budget = channel.power
        if sum(BWs) > channel.bandwidth:
            BWs = np.multiply((min(len(channel.users), 1) * channel.bandwidth) / sum(BWs), BWs)

        PuW = [to_pwr(get_u_power(bs_id, bss, c_id, channel, close_ids,
                                  noises, path_loss, channel.users[i], channel.users[i].user_bandwidth))
               for i in range(len(channel.users))]
        if to_db(sum(PuW)) > channel.power - 30:
            power_limit = True
            continue
        else:
            power_limit = False
            channel.power_budget = to_db(to_pwr(channel.power_budget - 30) - sum(PuW)) + 30
            for z in range(len(channel.users)):
                u = channel.users[z]
                u.user_bandwidth = BWs[z]
                u.power = PuW[z]

        break
    if user.power < 0:
        print(f'user: {user.id} and {user.power}')
    return power_limit


def get_u_power(bs_id, bss, c_id, channel, close_ids, noises, path_loss, user, user_bandwidth):
    bs_x, bs_y = bss[bs_id].x, bss[bs_id].y
    ang = angle(bs_x, bs_y, user.x, user.y)
    gain = find_antenna_gain(channel.main_direction, ang)
    N = get_noise_channel(user.id, bs_id, c_id, close_ids, noises)
    L = get_path_loss_channel(user.id, bs_id, c_id, close_ids, path_loss)
    C = user.rate_requirement
    P = to_db(shannon_power(C, user_bandwidth, to_pwr(N)))
    Pt = friis_transmission_power(P, gain, L)
    return Pt


def format_and_transfer_power_data(users):
    u_c_f = np.asarray([user.channel.frequency for user in users], dtype=np.float32)
    u_c_f_gpu = cuda.to_device(u_c_f)
    u_c_id = np.asarray([user.channel.id for user in users], dtype=np.int32)
    u_c_id_gpu = cuda.to_device(u_c_id)
    u_bs_id = np.asarray([user.channel.BS_id for user in users], dtype=np.int32)
    u_bs_id_gpu = cuda.to_device(u_bs_id)

    POWERS = np.asarray(np.zeros(shape=len(users)), dtype=np.float32)
    POWERS_gpu = cuda.to_device(POWERS)

    return u_c_f_gpu, u_c_id_gpu, u_bs_id_gpu, POWERS_gpu


# Change length to 10 for experiment
def format_and_transfer_bandwidth_data(users):
    u_rr = np.asarray([u.rate_requirement for u in users], dtype=np.float64)
    u_rr_gpu = cuda.to_device(u_rr)
    BEST_CHANNELS = np.asarray(np.zeros(shape=(len(users), 20, 2)), np.int32)
    BEST_CHANNELS.fill(-1) # Fill with -1 since 0 is also a bs_id/c_id
    BEST_CHANNELS_gpu = cuda.to_device(BEST_CHANNELS)
    CHANNEL_BANDWIDTHS = np.asarray(np.zeros(shape=(len(users), 20)), np.float32)
    CHANNEL_BANDWIDTHS_gpu = cuda.to_device(CHANNEL_BANDWIDTHS)
    SNRS = np.asarray(np.zeros(shape=(len(users), 20)), np.float32)
    SNRS_gpu = cuda.to_device(SNRS)
    return u_rr_gpu, CHANNEL_BANDWIDTHS_gpu, BEST_CHANNELS_gpu, SNRS_gpu


def format_and_transfer_distance_data(base_station_mnos, base_station_xs, base_station_ys, bss, user_xs, user_ys,
                                      users):
    # User data
    u_ids = np.asarray([u.id for u in users], dtype=np.int32)
    u_xs = np.asarray(user_xs, dtype=np.float32)
    u_ys = np.asarray(user_ys, dtype=np.float32)
    u_mnos = np.asarray([u.mno for u in users], dtype=np.int32)
    u_data = np.row_stack((u_ids, u_xs, u_ys, u_mnos))
    # Base Station data
    bs_ids = np.asarray([bs.id for bs in bss])
    bs_xs = np.asarray(base_station_xs)
    bs_ys = np.asarray(base_station_ys)
    bs_mnos = np.asarray(base_station_mnos)
    bs_data = np.row_stack((bs_ids, bs_xs, bs_ys, bs_mnos))
    # Result data
    CLOSE_IDS = np.asarray(np.zeros(shape=(len(u_ids), 10), dtype=np.int32))
    # CLOSE_DISTS = np.asarray(np.zeros(shape=(len(u_ids), len(bs_ids)), dtype=np.float32), dtype=np.float32)
    CLOSE_DISTS = np.asarray(np.zeros(shape=(len(u_ids), 10), dtype=np.float32))
    # Move and allocate on gpu
    u_data_gpu = cuda.to_device(u_data)
    bs_data_gpu = cuda.to_device(bs_data)
    CLOSE_IDS_gpu = cuda.to_device(CLOSE_IDS)
    CLOSE_DISTS_gpu = cuda.to_device(CLOSE_DISTS)
    return CLOSE_DISTS_gpu, CLOSE_IDS_gpu, bs_data_gpu, bs_ids, bs_mnos, bs_xs, bs_ys, u_data_gpu, u_ids, u_mnos


def format_and_transfer_channel_data(
        all_chans, bs_ids, bs_mnos, bs_xs, bs_ys, bss, user_count):
    m_channels = max([len(bs.channels) for bs in bss])
    feature_count = 7

    # Set up a 3 dimensional matrix that contains the data of all channels per base station
    chan_3d_matrix = []
    for chan_list in all_chans:
        bs_chans_data = np.zeros(shape=(m_channels, feature_count), dtype=np.float32)
        for i in range(len(chan_list)):
            c = chan_list[i]
            bs_chans_data[i] = np.asarray(
                [float(c.id), c.BS_id, c.height, c.power, c.main_direction, c.bandwidth, c.frequency], dtype=np.float32)
        chan_3d_matrix.append(bs_chans_data)
    c_ids_matrix = np.asarray(chan_3d_matrix, dtype=np.float32)
    c_data_gpu = cuda.to_device(c_ids_matrix)
    bs_ats = np.asarray([bs.area_type for bs in bss], dtype=np.int32)
    bs_rads = np.asarray([bs.radio for bs in bss], dtype=np.int32)
    bs_data_at = np.row_stack((bs_ids, bs_xs, bs_ys, bs_mnos, bs_ats, bs_rads))
    bs_data_at_gpu = cuda.to_device(bs_data_at)

    # For each user, we want the 10 best sinr values and the corresponding base station
    # e.g.
    # channel_id, base_station =  SINR_CHANNELS[0][0][0], SINR_CHANNELS[0][0][1]
    CHANNEL_PREFS = np.asarray(np.zeros(shape=(user_count, 20, m_channels), dtype=np.float32))
    CHANNEL_PREFS.fill(-1)  # 0 is a bs_id, so fill with -1
    CHANNEL_PREFS_gpu = cuda.to_device(CHANNEL_PREFS)

    LOSS = np.asarray(np.zeros(shape=(user_count, 20, m_channels), dtype=np.float32))
    LOSS_gpu = cuda.to_device(LOSS)

    NOISE = np.asarray(np.zeros(shape=(user_count, 20, m_channels), dtype=np.float32))
    NOISE_gpu = cuda.to_device(NOISE)
    return bs_data_at_gpu, c_data_gpu, m_channels, CHANNEL_PREFS_gpu, LOSS_gpu, NOISE_gpu


def debug_disconnected(best_pairs_per_user, bss, bw_s_per_user, close_ids, noises, path_loss, user):
    print(f'Rate: {user.rate_requirement}')
    for i in range(len(best_pairs_per_user[user.id])):
        [bs_id, c_id] = best_pairs_per_user[user.id][i]
        bs = bss[bs_id]
        c = bs.channels[c_id]
        N = get_noise_channel(i, bs_id, c_id, close_ids, noises)
        L = get_path_loss_channel(i, bs_id, c_id, close_ids, path_loss)
        C = user.rate_requirement
        P = to_db(shannon_power(C, bw_s_per_user[user.id][i] / 20, to_pwr(N)))
        bs_x, bs_y = bss[bs_id].x, bss[bs_id].y
        ang = angle(bs_x, bs_y, user.x, user.y)
        gain = find_antenna_gain(c.main_direction, ang)
        Pt = friis_transmission_power(P, gain, L)

        print(f'BS: {bs.id}, Dist: {d2d(bs.x, bs.y, user.x, user.y)}')
        print(
            f'C1: {c.bandwidth >= bw_s_per_user[user.id][i]}, C2: {bs.power <= 20}, C3: {bs.power + to_pwr(Pt) <= 20}')

