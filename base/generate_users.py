import random
from shapely.geometry import Point
import models.ue as ue
import settings
from settings import *
import base.util as util
import numpy as np
import geopandas as gpd
from shapely.ops import unary_union
import base.find_base_stations as antenna


def generate_random(number, polygon):  # to generate users per zip code
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points


# find all users in the specific zip codes
def get_population(zip_codes_region, percentage):
    users = []
    municipalities = []
    division_parameter = percentage / 100
    for index, row in zip_codes_region.iterrows():
        polygon = row['geometry']
        number_of_users = row['aantal_inw']
        points = generate_random(np.ceil(number_of_users * division_parameter), polygon)
        users += points
        municipalities += [row['municipali']]*len(points)  # Add municipality to list
    xs = [point.x for point in users]
    ys = [point.y for point in users]
    return xs, ys, municipalities


def generate_users(zip_codes, percentage):
    all_users = util.from_data(f'data/users/{FILENAME_GEN_USERS}_seed={settings.UE_SEED}_all_users.p')
    x_user = util.from_data(f'data/users/{FILENAME_GEN_USERS}_seed={settings.UE_SEED}_xs.p')
    y_user = util.from_data(f'data/users/{FILENAME_GEN_USERS}_seed={settings.UE_SEED}_ys.p')

    if all_users is None:
        print('Users are not stored in memory')
        all_users = list()
        x_user = list()
        y_user = list()
        for province in provinces:
            print(f'Generating users for {province}')
            cities = util.find_cities(province)
            _, region_data, _ = antenna.find_zip_code_region(zip_codes, cities)
            # print(zip_codes['municipali'])
            # print(zip_codes['municipali'].shape)

            np.random.seed(UE_SEED)
            xs, ys, muns = get_population(region_data, percentage)
            for i in range(len(xs)):
                rate = np.random.uniform(8, 20)

                mno = np.random.randint(0, 3)  # Even chances for each MNO, check if reasonable

                new_user = ue.UserEquipment(i, xs[i], ys[i], rate_requirement=rate * 10 ** 6, mno=mno
                                            , province=province, municipality=muns[i])
                all_users.append(new_user)
                x_user.append(xs[i])
                y_user.append(ys[i])

        util.to_data(all_users, f'data/users/{FILENAME_GEN_USERS}_seed={settings.UE_SEED}_all_users.p')
        util.to_data(x_user, f'data/users/{FILENAME_GEN_USERS}_seed={settings.UE_SEED}_xs.p')
        util.to_data(y_user, f'data/users/{FILENAME_GEN_USERS}_seed={settings.UE_SEED}_ys.p')

    users = []
    xs = []
    ys = []
    np.random.seed(UE_SEED)
    for i in range(len(all_users)):
        if np.random.uniform(0, 100) < settings.PERCENTAGE_ACTIVE:
            all_users[i].id = len(users) - 1
            users.append(all_users[i])
            xs.append(x_user[i])
            ys.append(y_user[i])

    return users, xs, ys


def generate_users_grid(zip_code_region, delta):
    all_users = util.from_data(f'data/users/{FILENAME_GEN_USERS}_seed={settings.UE_SEED}_all_users_grid.p')
    x_user = util.from_data(f'data/users/{FILENAME_GEN_USERS}_seed={settings.UE_SEED}_xs_grid.p')
    y_user = util.from_data(f'data/users/{FILENAME_GEN_USERS}_seed={settings.UE_SEED}_ys_grid.p')
    if all_users is None:
        print('Users are not stored in memory')
        all_users = list()
        [xmin, ymin, xmax, ymax] = gpd.GeoSeries(zip_code_region['geometry']).total_bounds
        xmin, xmax = np.floor(xmin), np.ceil(xmax)
        ymin, ymax = np.floor(ymin), np.ceil(ymax)
        xdelta, ydelta = int(xmax - xmin), int(ymax - ymin)

        xL = np.linspace(xmin, xmax, num=int(xdelta / delta))
        yL = np.linspace(ymin, ymax, num=int(ydelta / delta))

        xs, ys = np.meshgrid(xL, yL)
        xs = xs.flatten()
        ys = ys.flatten()
        x_user = []
        y_user = []

        polygon = gpd.GeoSeries(unary_union(zip_code_region['geometry']))

        i = 0
        for j in range(len(xs)):
            pnt = Point(xs[j], ys[j])
            if polygon.contains(pnt).bool():
                if CAPACITY_DISTRIBUTION:
                    p = 8
                else:
                    p = np.random.uniform(8, 20)
                new_user = ue.UserEquipment(i, xs[j], ys[j], rate_requirement=p * 10 ** 6)
                all_users.append(new_user)
                i += 1
                x_user.append(xs[j])
                y_user.append(ys[j])

        util.to_data(all_users, f'data/users/{FILENAME_GEN_USERS}_seed={settings.UE_SEED}_all_users_grid.p')
        util.to_data(x_user, f'data/users/{FILENAME_GEN_USERS}_seed={settings.UE_SEED}_xs_grid.p')
        util.to_data(y_user, f'data/users/{FILENAME_GEN_USERS}_seed={settings.UE_SEED}_ys_grid.p')

    users = all_users
    x_user = x_user
    y_user = y_user
    return users, x_user, y_user
