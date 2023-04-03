import json
import sys

import geopandas as gpd
import matplotlib.pylab as pylab
import numpy
import progressbar
from shapely.geometry import Point
from shapely.ops import unary_union

import models.base_station as bso
import base.util as util
import settings
from settings import *

params_a = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'lines.markersize': 8,
          'figure.autolayout': True}
pylab.rcParams.update(params_a)

sys.setrecursionlimit(100000)


# code adapted from Bart Meyers

def find_zip_code_region(zip_codes, cities):
    zip_code_region_data = util.from_data(f'data/BSs/{FILENAME_NOMNO}_zip_code_region_data.p')
    region = util.from_data(f'data/BSs/{FILENAME_NOMNO}_region.p')
    if zip_code_region_data is None:
        if isinstance(cities[0], int):
            zip_code_region_data = zip_codes[zip_codes['postcode'].isin(cities)]
        else:
            zip_code_region_data = zip_codes[zip_codes['municipali'].isin(cities)]

        region = gpd.GeoSeries(unary_union(zip_code_region_data['geometry']))

        util.to_data(zip_code_region_data, f'data/BSs/{FILENAME_NOMNO}zip_code_region_data.p')
        util.to_data(region, f'data/BSs/{FILENAME_NOMNO}region.p')

    return region, zip_code_region_data, region.centroid[0]


def find_region(regions, point):
    for region, municipality, province in regions:
        if region.contains(point).any():
            return True



def load_bs(zip_codes):
    UMA = util.from_data('data/UMA.p')
    RMA = util.from_data('data/RMA.p')
    if UMA is None:
        UMA = unary_union(zip_codes[zip_codes['scenario'] == 'UMA'].geometry)
        RMA = unary_union(zip_codes[zip_codes['scenario'] == 'RMA'].geometry)

        util.to_data(UMA, 'data/UMA.p')
        util.to_data(RMA, 'data/RMA.p')

    all_basestations = util.from_data(f'data/BSs/{FILENAME_GEN_BASE_STATIONS}_seed={settings.BS_SEED}_all_basestations.p')
    xs = util.from_data(f'data/BSs/{FILENAME_GEN_BASE_STATIONS}_seed={settings.BS_SEED}_xs.p')
    ys = util.from_data(f'data/BSs/{FILENAME_GEN_BASE_STATIONS}_seed={settings.BS_SEED}_ys.p')
    channel_count = util.from_data(f'data/BSs/{FILENAME_GEN_BASE_STATIONS}_seed={settings.BS_SEED}_channels.p')
    all_freqs = util.from_data(f'data/BSs/{FILENAME_GEN_BASE_STATIONS}_seed={settings.BS_SEED}_all_freqs.p')
    radios = util.from_data(f'data/BSs/{FILENAME_GEN_BASE_STATIONS}_seed={settings.BS_SEED}_radios.p')

    # bar = progressbar.ProgressBar(maxval=len(all_basestations),
    #                               widgets=[progressbar.Bar('=',
    #                                                        f'Bruh [',
    #                                                        ']'), ' ',
    #                                        progressbar.Percentage(), ' ', progressbar.ETA()])
    # bar.start()
    # if all_basestations is not None:
    #     for bs in all_basestations:
    #         bs.provider = bs.channels[0].mno
    #         bs.province = util.get_province(Point(bs.x, bs.y))
    #         bar.update(bs.id)
    #     util.to_data(all_basestations, f'data/BSs/{FILENAME_GEN_BASE_STATIONS}_seed={12}_all_basestations.p')

    if radios is None:
        print('BSs are not stored in memory')
        all_basestations = list()
        id = 0
        xs, ys = [], []
        radios = []
        all_freqs = set()
        omnidirection = 0

        regions = []

        for province in provinces:
            cities = util.find_cities(province)
            region, zip_data, _ = find_zip_code_region(zip_codes, cities)
            region = region.buffer(settings.BUFFER_SIZE, join_style=3)
            print(zip_data['municipali'].iloc[1])
            regions.append((region, zip_data['municipali'].iloc[1], province))

        with open(BS_PATH) as f:
            bss = json.load(f)

            bar = progressbar.ProgressBar(maxval=len(bss),
                                          widgets=[progressbar.Bar('=',
                                                                   f'Finding BSs {FILENAME_GEN_BASE_STATIONS} [',
                                                                   ']'), ' ',
                                                   progressbar.Percentage(), ' ', progressbar.ETA()])
            bar.start()

            # Loop over base stations
            for key, index in zip(bss.keys(), range(len(bss))):

                bar.update(index)

                bs = bss[key]
                x = float(bs.get('x'))
                y = float(bs.get('y'))
                # if new_region.contains(Point(x, y)).bool() and Point(x, y).distance(
                #         center_disaster) > radius_disaster:

                if find_region(regions, Point(x, y)):
                    create = True
                    if bs.get("type") == "LTE":
                        radio = settings.RADIO_LTE
                    elif bs.get("type") == "5G NR":
                        radio = settings.RADIO_5G_NR
                    elif bs.get("type") == "UMTS":
                        radio = settings.RADIO_UMTS
                    elif bs.get("type") == "GSM":
                        radio = settings.RADIO_GSM
                        create = False
                    else:
                        print(bs.get("HOOFDSOORT"))  # there are no other kinds of BSs in this data set

                    if create:
                        new_bs = bso.BaseStation(id, radio, x, y, province, bs_municipality)
                        channel = 0
                        for key in bs.get("antennas").keys():
                            antenna = bs.get("antennas").get(key)
                            frequency = antenna.get("frequency")
                            provider, bandwidth = util.find_provider(frequency / 1e6)
                            if provider in mnos:
                                all_freqs.add(frequency)
                                main_direction = antenna.get('angle')
                                if main_direction != 'Omnidirectional':  # we only consider three-sectorized antenna's
                                    power = POWER_PERCENTAGE * (
                                            antenna.get("power") + 30)   # We convert ERP power in dBW to dBm
                                    height = bs.get('antennas')[str(0)].get("height")

                                    new_bs.add_channel(key, new_bs.id, height, frequency, power, main_direction,
                                                       util.mno_to_id(provider), bandwidth)
                                    new_bs.frequencies.add(frequency)
                                    channel += 1
                                else:
                                    omnidirection += 1

                        if channel > 0:
                            if UMA.contains(Point(x, y)):
                                area_type = 0  # UMA IS 0
                            elif RMA.contains(Point(x, y)):
                                area_type = 1  # RMA IS 1
                            else:
                                area_type = 2  # OTHER IS RMA?
                                # TODO: there are still some BSs that have no zip code? I assume this is RMA
                                # print('No type', x, y)
                            new_bs.area_type = area_type

                            # p = np.random.uniform(0, 1)
                            # if p >= RANDOM_FAILURE:
                            all_basestations.append(new_bs)
                            xs.append(x)
                            ys.append(y)
                            id += 1
                            radios.append(bs.get("type"))

            bar.finish()

        freq_channels = dict() # Dict from channel frequencies to a list of base stations
        channel_count = 0
        for bs in all_basestations:
            for channel in bs.channels:
                channel_count += 1
                if channel.frequency in freq_channels.keys():
                    freq_channels[channel.frequency].add(channel.BS_id)
                else:
                    freq_channels[channel.frequency] = {channel.BS_id}
        for bs in all_basestations:
            for channel in bs.channels:
                interferers = [all_basestations[i] for i in freq_channels[channel.frequency] if
                                  i != channel.BS_id and 1 < util.distance_2d(all_basestations[i].x,
                                                                              all_basestations[i].y, bs.x,
                                                                              bs.y) <= 5_000]
                distances = np.array([util.distance_2d(i.x, i.y, bs.x, bs.y) for i in
                             interferers])
                indices = np.array(distances).argsort()
                channel.bs_interferers = [interferers[i] for i in indices[CUTOFF_VALUE_INTERFERENCE:]]

        util.to_data(all_basestations, f'data/BSs/{FILENAME_GEN_BASE_STATIONS}_seed={settings.BS_SEED}_all_basestations.p')
        util.to_data(xs, f'data/BSs/{FILENAME_GEN_BASE_STATIONS}_seed={settings.BS_SEED}_xs.p')
        util.to_data(ys, f'data/BSs/{FILENAME_GEN_BASE_STATIONS}_seed={settings.BS_SEED}_ys.p')
        util.to_data(channel_count, f'data/BSs/{FILENAME_GEN_BASE_STATIONS}_seed={settings.BS_SEED}_channels.p')
        util.to_data(all_freqs, f'data/BSs/{FILENAME_GEN_BASE_STATIONS}_seed={settings.BS_SEED}_all_freqs.p')
        util.to_data(radios, f'data/BSs/{FILENAME_GEN_BASE_STATIONS}_seed={settings.BS_SEED}_radios.p')

    return xs, ys, all_basestations, channel_count, list(all_freqs)


# if __name__ == '__main__':
#     percentage = 2  # Three user density levels - still tbd
#     seed = 1
#     power3G, power4G, power5G = dict(), dict(), dict()
#     freq3G, freq4G, freq5G = dict(), dict(), dict()
#     for mno in ['KPN', 'T-Mobile', 'Vodafone']:
#         freq3G[mno], freq4G[mno], freq5G[mno] = set(), set(), set()
#         power3G[mno], power4G[mno], power5G[mno] = [], [], []
#
#     for province in ['Drenthe', 'Flevoland', 'Friesland', 'Groningen', 'Limburg', 'Overijssel', 'Utrecht', 'Zeeland',
#                      'Zuid-Holland', 'Gelderland', 'Noord-Brabant', 'Noord-Holland']:
#         # for city in ['Almere', 'Amsterdam', 'Enschede', "'s-Gravenhage", 'Elburg', 'Emmen', 'Groningen', 'Maastricht', 'Eindhoven', 'Middelburg']:
#         for mno in [['KPN'], ['T-Mobile'], ['Vodafone']]:
#             zip_codes = gpd.read_file('data/zip_codes_with_scenarios.shp')
#             cities = base.find_cities(province)
#             # province = None
#             # cities = [city]
#             print(province)
#
#             # params = p.Parameters(seed, zip_codes, mno, percentage, buffer_size=None, city_list=cities,
#             #                       province=province)
#
#             # params = find_zip_code_region(params)
#             zip_code_region = find_zip_code_region(zip_codes, cities)
#             # params = load_bs(params)
#             mno = mno[0]
#
#             for bs in params.BaseStations:
#                 for channel in bs.channels:
#                     if bs.radio == base.BaseStationRadioType.UMTS:
#                         power3G[mno].append(channel.power)
#                         freq3G[mno].add(channel.frequency / 1e6)
#                     elif bs.radio == base.BaseStationRadioType.LTE:
#                         power4G[mno].append(channel.power)
#                         freq4G[mno].add(channel.frequency / 1e6)
#                     elif bs.radio == base.BaseStationRadioType.NR:
#                         power5G[mno].append(channel.power)
#                         freq5G[mno].add(channel.frequency / 1e6)
#
#     x = np.array([1, 2, 3])
#
#     fig, ax = plt.subplots()
#     data = [power3G, power4G, power5G]
#
#     # print(power3G, power4G, power5G)
#     with open("power3G.txt", "w") as output:
#         output.write(str(power3G))
#     with open("power4G.txt", "w") as output:
#         output.write(str(power4G))
#     with open("power5G.txt", "w") as output:
#         output.write(str(power5G))
#
#     plt.hist(data, density=True, histtype='bar', color=colors[:3], label=['3G', '4G', '5G'])
#
#     plt.xlabel('Power (dBW)')
#     plt.legend()
#
#     plt.savefig('power.png', dpi=1000)
#     # plt.show()
#
#     # print('3G', freq3G)
#     # print('4G', freq4G)
#     # print('5G', freq5G)
