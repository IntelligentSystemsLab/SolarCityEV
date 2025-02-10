# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023/10/10 21:59
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023/10/10 21:59
import datetime
import math
import pandas as pd
import numpy as np
import torch
import os
import random
from shapely import Polygon, Point
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error, \
    median_absolute_error, explained_variance_score
from torch.utils.data import Dataset
import geopandas as gpd
import matplotlib.pyplot as plt


def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(False)}"

    return seed


class CreateDataset(Dataset):
    def __init__(self, sample, label, features, date, weekday):
        self.sample = sample
        self.label = label
        self.features = features
        self.date = date
        self.weekday = weekday

    def __len__(self):
        return int(self.sample.shape[0])

    def __getitem__(self, item):
        sample = self.sample[item, :]
        label = self.label[item]
        features = self.features[item, :]
        date = self.date[item]
        weekday = self.weekday[item]

        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        features = torch.tensor(features, dtype=torch.float32)
        date = torch.tensor(date, dtype=torch.float32)
        weekday = torch.tensor(weekday, dtype=torch.float32)

        return sample, label, features, date, weekday


def divide_dataset(sample, label, feature, date, weekday,divide_mode, divide_rate=0.8, devide=True):
    if devide:
        length = len(label)
        train_id=[]
        test_id=[]
        first_day=datetime.datetime(2023,1,1)
        temp_date=datetime.datetime(2022,12,31)
        if divide_mode=='by_day':
            for i in range(length):
                temp_date+=datetime.timedelta(days=1)
                temp_month=temp_date.timetuple().tm_mon
                temp_day=temp_date.timetuple().tm_mday
                if temp_day<=20:
                    train_id.append(i)
                else:
                    test_id.append(i)
        elif divide_mode == 'by_month':
            for i in range(length):
                temp_date+=datetime.timedelta(days=1)
                temp_month=temp_date.timetuple().tm_mon
                temp_day=temp_date.timetuple().tm_mday
                if temp_month in [3,6,9,12]:
                    test_id.append(i)
                else:
                    train_id.append(i)

        # train data
        train_sample = sample[train_id, :]
        train_label = label[train_id]
        train_feature = feature[train_id, :]
        train_date = date[train_id]
        train_weekday = weekday[train_id]
        # test data
        test_sample = sample[test_id, :]
        test_label = label[test_id]
        test_feature = feature[test_id, :]
        test_date = date[test_id]
        test_weekday = weekday[test_id]
    else:
        train_sample = sample[:, :]
        train_label = label[:]
        train_feature = feature[:, :]
        train_date = date[:]
        train_weekday = weekday[:]

        test_sample = sample[:, :]
        test_label = label[:]
        test_feature = feature[:, :]
        test_date = date[:]
        test_weekday = weekday[:]
    return train_sample, train_label, train_feature, train_date, train_weekday, test_sample, test_label, test_feature, test_date, test_weekday


def calculate_metrics(label, output):
    RMSE = np.sqrt(mean_squared_error(label, output))
    MAPE = mean_absolute_percentage_error(label, output)
    MAE = mean_absolute_error(label, output)
    MedAE = median_absolute_error(label, output)
    EVS = explained_variance_score(label, output)
    R2 = r2_score(label, output)
    return RMSE, MAE, MAPE, MedAE, R2, EVS


def select_station_in_city(path, city, data):
    return_data = pd.DataFrame(columns=['station_id', 'latitude', 'longitude'])
    file_path = get_file(path + 'city_boundary/' + city + '/', '.shp')
    city_gdf = gpd.read_file(path + 'city_boundary/' + city + '/' + file_path[0])
    for index, row in data.iterrows():
        latitude, longitude = row['latitude'], row['longitude']
        p = Point(longitude, latitude)
        for idx, row1 in city_gdf.iterrows():
            if p.intersects(row1["geometry"]):
                return_data.loc[len(return_data.index)] = [row['station_id'], row['latitude'], row['longitude']]
    return return_data


def cluster_station(path, city, data, distance, plot=False, save_station=False):
    dbscan = DBSCAN(eps=distance / 111 / 1000, min_samples=1)
    X = data[['latitude', 'longitude']]
    data['cluster'] = dbscan.fit_predict(X)
    return_dict = dict()
    for i in set(data['cluster']):
        return_dict[i] = [[], 0, 0]
    for index, row in data.iterrows():
        return_dict[int(row['cluster'])][0].append(row['station_id'])
        return_dict[int(row['cluster'])][1] += row['longitude']
        return_dict[int(row['cluster'])][2] += row['latitude']
    for i in set(data['cluster']):
        return_dict[i][1] = return_dict[i][1] / len(return_dict[i][0])
        return_dict[i][2] = return_dict[i][2] / len(return_dict[i][0])
    with open("./exp_240217.txt", "a", encoding='utf-8') as f:
        f.writelines(
            '\n' + 'city:' + str(city) + '\n' +
            '半径:' + str(distance) + '\n' +
            '簇数:' + str(len(return_dict.keys())) + '\n'
        )
    return return_dict


def get_file(path, suffix):
    input_template_All = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            input_template_All.append(i)
    return input_template_All


def cluster_hexagon(
        path: str,
        city: str,
        data: pd.DataFrame,
        r: float,
        plot: bool = False
):
    file_path = get_file(path + 'city_boundary/' + city + '/', '.shp')
    city_gdf = gpd.read_file(path + 'city_boundary/' + city + '/' + file_path[0])

    def hexagon(radius, center):
        angle = 360 / 6
        hex_coords = []
        for i in range(6):
            x = center.x + radius * math.cos(math.radians(i * angle))
            y = center.y + radius * math.sin(math.radians(i * angle))
            hex_coords.append((x, y))
        hex_coords.append(hex_coords[0])
        return Polygon(hex_coords)

    hexagons = gpd.GeoDataFrame(columns=["geometry"])
    station_gdf = gpd.GeoDataFrame(columns=["geometry"])

    for index, row in data.iterrows():
        latitude, longitude = row['latitude'], row['longitude']
        p = Point(longitude, latitude)
        station_gdf = station_gdf.append({"geometry": p}, ignore_index=True)

    cluster_dict = dict()
    id = 0
    for idx, row in city_gdf.iterrows():
        city_polygon = row["geometry"]

        bounds = city_polygon.bounds
        min_x, min_y, max_x, max_y = bounds

        radius = r / 111

        x = min_x
        y = min_y
        flag = True
        while x < max_x + radius:
            while y < max_y + math.sqrt(3) * radius / 2:
                center = Point(x, y)
                hexagon_poly = hexagon(radius, center)
                temp_list = []
                for index, row in data.iterrows():
                    latitude, longitude = row['latitude'], row['longitude']
                    p = Point(longitude, latitude)
                    if p.intersects(hexagon_poly):
                        temp_list.append(row['station_id'])
                if len(temp_list) > 0:
                    cluster_dict[id] = [temp_list, x, y]
                    id += 1
                if hexagon_poly.intersects(city_polygon):
                    hexagons = hexagons.append({"geometry": hexagon_poly}, ignore_index=True)
                y += math.sqrt(3) * radius
            if flag:
                y = min_y + math.sqrt(3) * radius / 2
                flag = False
            else:
                y = min_y
                flag = True
            x += 3 * radius / 2
    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        city_gdf.plot(ax=ax)
        hexagons.plot(ax=ax, color='none', edgecolor='red')
        station_gdf.plot(ax=ax, color='yellow')
        plt.show()

    return cluster_dict
