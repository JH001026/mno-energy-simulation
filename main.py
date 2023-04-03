import time
import warnings
from copy import deepcopy
from itertools import groupby

import geopandas as gpd
import numpy

import base.util as util
from settings import *
import base.generate_users as user_gen
import base.find_base_stations as antenna
from base import link_builder
import settings

# Regions
provinces = ['Drenthe', 'Flevoland', 'Friesland', 'Groningen', 'Limburg', 'Overijssel', 'Utrecht', 'Zeeland',
             'Zuid-Holland', 'Gelderland', 'Noord-Brabant', 'Noord-Holland']
municipalities = ['Almere', 'Amsterdam', 'Enschede', "'s-Gravenhage", 'Elburg', 'Emmen', 'Groningen', 'Maastricht',
                  'Eindhoven', 'Middelburg']

# MNO's
mnos = ['KPN', 'T-Mobile', 'Vodafone']

result = []

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for p in range(1, 5):
        settings.PERCENTAGE_ACTIVE = 2.5 * p
        # Retrieve inhabitant counts and shapes of all zip-code areas
        zip_codes = gpd.read_file('data/square_statistics.shp')

        # Retrieve all user data, generate if necessary. Region specified in settings.py
        users, x_user, y_user = user_gen.generate_users(zip_codes, PERCENTAGE_PER_ZIP)

        settings.regions = []
        for province in settings.provinces:
            cities = util.find_cities(province)
            region = antenna.find_zip_code_region(zip_codes, cities, province)[1].buffer(settings.BUFFER_SIZE, join_style=3)
            settings.regions.append(region)

        # Retrieve all base station data, generate if necessary
        xs, ys, all_basestations, channel_count, all_frequencies = antenna.load_bs(zip_codes)

        # Filter out all the sub-4G base stations
        xs_n, ys_n, all_basestations_n = [], [], []
        for i in range(len(xs)):
            bs = all_basestations[i]
            if bs.radio != settings.RADIO_UMTS and bs.radio != settings.RADIO_GSM:
                xs_n.append(xs[i])
                ys_n.append(ys[i])
                all_basestations_n.append(all_basestations[i])
        xs = xs_n
        ys = ys_n
        all_basestations = all_basestations_n


        print(f'Generated {len(all_basestations)} base stations')
        print(f'Generated {len(users)} users')

        all_channels = [bs.channels for bs in all_basestations]
        # We need bandwidths and MNOs
        # bws = [[bs_channel.mno for bs_channel in bs_channels] for bs_channels in all_channels]
        mnos = [[bs_channel.mno for bs_channel in bs_channels][0] for bs_channels in all_channels]

        # Ensure index corresponds to ID
        for i in range(len(users)):
            users[i].id = i
        for i in range(len(all_basestations)):
            all_basestations[i].id = i

        link_builder.create_links(users, x_user, y_user, xs, ys, mnos, all_basestations, all_channels)

    # print(len(all_basestations))


# TODO
# GENERAL SIM
# [x] Create list of users
#
# LINK BUILDER
# -- should
# [x] Find n closest Base Stations for each user
# [x] Ensure the n base stations are selected based on MNO
# [X] Rewrite distance finder to only include 10 closest distances
# [x] Use shannon formula to find received power
# [x] Use friis to find transmitted power
# [x] Determine which channel to choose for each user
# [x] INF PATH LOSS STILL IN THE RESULT SET
# [ ]
# [ ] Calculate the total energy cost of the system

# -- could
# [ ] Rewrite channel picker to return only the snr's of the best few channels
# [ ] Calculate the bs_interferes in the kernel
# [ ] Process interference loss
# [ ]
# [ ]
# [ ]
