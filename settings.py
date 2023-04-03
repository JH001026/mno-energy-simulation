from pathlib import Path
import os
import pandas as pd
import numpy as np
import seaborn as sns
import math

# code from Bart Meyers

colors = sns.color_palette("Paired", n_colors=100)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

ROOT_DIR = Path(__file__).parent

BS_PATH = os.path.join(ROOT_DIR, "data", "antennas.json")

# Average height of buildings in an area (used for RMa 5G NR only)
AVG_BUILDING_HEIGHT = 15  # current number based on average two-story building
AVG_STREET_WIDTH = 10

# BASE STATION PROPERTIES
BS_RANGE = 5000  # maximum range of base stations based on the fact that UMa and UMi models cannot exceed 5km

# User equipment properties
UE_HEIGHT = 1.5  # height in meters

# to calculate the noise power
BOLTZMANN = 1.38e-23
TEMPERATURE = 300
LIGHT = 299792458

VERTICAL_BORE = np.radians(8)  # degrees

VERTICAL_BEAMWIDTH3DB = np.radians(65)
HORIZONTAL_BEAMWIDTH3DB = 65

MINIMUM_SNR = 20   #- math.inf # 5 # dB
# PREF_SNR = 20

CUTOFF_VALUE_INTERFERENCE = 3  # the x highest signal BSs will not interfere.
POWER_PERCENTAGE = 0.9

# ASSUMPTION maybe change the power percentage!


#####################################################################
#                                                                   #
#                            PARAMETERS                             #
#                                                                   #
######################################################################

# Regions
provinces = ['Drenthe', 'Flevoland', 'Friesland', 'Groningen', 'Limburg', 'Overijssel', 'Utrecht', 'Zeeland',
             'Zuid-Holland', 'Gelderland', 'Noord-Brabant', 'Noord-Holland']
# provinces = ['Gelderland']
municipalities = ['Almere', 'Amsterdam', 'Enschede', "'s-Gravenhage", 'Elburg', 'Emmen', 'Groningen', 'Maastricht',
                  'Eindhoven', 'Middelburg']
# MNO's
mnos = ['KPN', 'T-Mobile', 'Vodafone']

regions = []

# SEED = 69
BS_SEED = 3
UE_SEED = 69

COUNT_BASE_STATION = 100
COUNT_USERS = 100
COUNT_CHANNELS = 100

FILENAME_GEN_USERS = 'generated_users'
FILENAME_GEN_BASE_STATIONS = 'generated_base_stations'
FILENAME_NOMNO = 'no_mno'

CAPACITY_DISTRIBUTION = False

PERCENTAGE_PER_ZIP = 10
PERCENTAGE_ACTIVE = 2
SHARING = False
E_ORDER = True

ARTIFICAL_DISCONNECTED_PERCENTAGE = 2

BUFFER_SIZE = 0

RANDOM_FAILURE = 0

AREA_UMA = 0
AREA_RMA = 1
AREA_UMI = 2

RADIO_LTE = 0
RADIO_5G_NR = 1
RADIO_UMTS = 2
RADIO_GSM = 3

RESOURCE_BLOCK = 1 * 1e6

MAX_USER_PERCENTAGE = 3

GEO_BUFFER = 2000
MIN_CHANNEL_POWER = 3  # W
MAX_CHANNEL_POWER = 40  # W

# Base station power settings
DIGITAL_SIGNAL_PROCESSING_POWER = 100  # W
# EFFICIENCY_POWER_AMP_MIN = 0.128       # %
EFFICIENCY_POWER_AMP_MIN = 0.4         # %
TRANSCEIVER_POWER = 100                # W
RECTIFIER_POWER = 100                  # W
MACRO_AIR_CONDITIONING_POWER = 225     # W
MICRO_AIR_CONDITIONING_POWER = 60      # W
MICROWAVE_LINK_POWER = 80              # W
LOAD_FACTOR_HIGH = 1                   # ?
LOAD_FACTOR_LOW = 0.93                 # ?
P_SLEEP_MACRO = 75                     # W
P_SLEEP_MICRO = 39                     # W

