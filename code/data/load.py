# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023/9/25 15:48
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023/9/25 15:48

import datetime
import os
import warnings
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from pandas.core.common import SettingWithCopyWarning

from functions_smooth import oneclassSVM, Moving_Average_Smooth
from utils import cluster_station, cluster_hexagon, select_station_in_city

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

city_dict = {
    '伦敦': ['London', 'UnitedKingdom',],
    '华沙': ['Warsaw', 'Poland',],
    '华盛顿': ['Washington', 'UnitedStates',],
    '哥本哈根': ['Copenhagen', 'Denmark',],
    '圣保罗': ['SaoPaulo', 'Brazil',],
    '圣胡安': ['SanJuan', 'PuertoRico',],
    '墨尔本': ['Melbourne', 'Australia',],
    '多伦多': ['Toronto', 'Canada',],
    '奥斯陆': ['Oslo', 'Norway',],
    '巴黎': ['Paris', 'France',],
    '布拉格': ['Prague', 'CzechRepublic',],
    '布达佩斯': ['Budapest', 'Hungary',],
    '布鲁塞尔': ['Brussels', 'Belgium',],
    '悉尼': ['Sydney', 'Australia',],
    '慕尼黑': ['Munich', 'Germany',],
    '斯德哥尔摩': ['Stockholm', 'Sweden',],
    '旧金山': ['SanFrancisco', 'UnitedStates',],
    '洛杉矶': ['LosAngeles', 'UnitedStates',],
    '渥太华': ['Ottawa', 'Canada',],
    '火奴鲁鲁': ['Honolulu', 'UnitedStates',],
    '特拉维夫': ['TelAviv', 'Israel',],
    '米兰': ['Milan', 'Italy',],
    '约翰内斯堡': ['Johannesburg', 'SouthAfrica',],
    '纽约': ['NewYork', 'UnitedStates',],
    '维也纳': ['Vienna', 'Austria',],
    '罗马': ['Rome', 'Italy',],
    '苏黎世': ['Zurich', 'Switzerland',],
    '蒙特利尔': ['Montreal', 'Canada',],
    '西雅图': ['Seattle', 'UnitedStates',],
    '赫尔辛基': ['Helsinki', 'Finland',],
    '迈阿密': ['Miami', 'UnitedStates',],
    '迪拜': ['Dubai', 'UnitedArabEmirates',],
    '都柏林': ['Dublin', 'Ireland',],
    '阿姆斯特丹': ['Amsterdam', 'Netherlands',],
    '雅典': ['Athens', 'Greece',],
    '雷克雅未克': ['Reykjavik', 'Iceland',],
    '马德里': ['Madrid', 'Spain',],
    '波士顿': ['Boston', 'UnitedStates',],
    '柏林': ['Berlin', 'Germany',],
    '深圳': ['Shenzhen', 'China',],
}

def load_data(city_name,start_date,end_date,path='./data/',rate=0.7,by_station=True):
    day_num=(end_date-start_date).days
    city_name_eng=city_dict[city_name][0]
    charging_data=pd.read_csv('./data_new/ori/'+city_name_eng+'_to_'+str(datetime.date(2023,12,31))+'.csv')
    if os.path.exists('./data_new/station_lanlon/'+city_name_eng+'.csv'):
        station_lanlon=pd.read_csv('./data_new/station_lanlon/'+city_name_eng+'.csv')
    elif os.path.exists('./data_new/station_lanlon/'+city_name_eng+'.xls'):
        station_lanlon=pd.read_excel('./data_new/station_lanlon/'+city_name_eng+'.xls')
    elif os.path.exists('./data/station_lanlon_new/'+city_name_eng+'.xls'):
        station_lanlon=pd.read_excel('./data/station_lanlon_new/'+city_name_eng+'.xls')
    else:
        station_lanlon=select_station_in_city('./data/',city_name_eng,pd.read_csv('./data/station_lanlon/'+city_name+'.csv'))
    for i in range(charging_data.shape[0]):
        i_data=charging_data.iloc[i,3:]
        i_data_list=np.array(list(i_data)).reshape(-1, 1)
        i_data_list=np.nan_to_num(i_data_list)
        i_data_list = oneclassSVM(i_data_list)
        i_data_list = Moving_Average_Smooth(i_data_list)
        i_data_list=list(i_data_list.reshape(-1))
        charging_data.iloc[i, 3:]=i_data_list
    if by_station:
        data = station_lanlon
        distance_list = []
        num = data.shape[0]
        for i in range(num):
            for j in range(i + 1, num):
                xy_dis = geodesic((data.loc[i, 'latitude'], data.loc[i, 'longitude']),
                                  (data.loc[j, 'latitude'], data.loc[j, 'longitude'])).m
                distance_list.append(xy_dis)
        distance_list.sort()
        sx_m=100
        distance_list_ = []
        for i in distance_list:
            if i <= sx_m:
                distance_list_.append(i)
            else:
                break
        distances = np.array(distance_list_)
        differences = np.diff(distances)
        division_point_index_list = np.argsort(-differences)
        division_point=30
        for division_point_index_id in range(division_point_index_list.shape[0]):
            division_point_index = division_point_index_list[division_point_index_id]
            if distances[division_point_index] >= 30:
                division_point = distances[division_point_index]
                break
        if city_name_eng=='Athens':
            division_point=0.000000000000001
        r = division_point
        station_cluster=cluster_station(path,city_name_eng,station_lanlon,r,save_station=False)
    else:
        r=2
        station_cluster = cluster_hexagon(path,city_name_eng,station_lanlon, r)
    all_data=dict()
    time_list = []
    temp_date = start_date
    for i in range(day_num + 1):
        time_list.append(temp_date)
        temp_date += datetime.timedelta(days=1)
    for cluster_id in range(len(station_cluster)):
        s_sample_all = []
        s_label_all = []
        s_feature_all = []
        s_date_all = []
        s_weekday_all = []
        for d_id in time_list[:-6]:
            s_sample=[]
            temp_date=d_id
            for _ in range(6):
                temp_value=0
                for s_id in station_cluster[cluster_id][0]:
                    temp_value+=charging_data.loc[charging_data['station_id'] == s_id, str(temp_date)].values[0]/1
                s_sample.append(temp_value)
                temp_date+=datetime.timedelta(days=1)
            temp_value = 0
            for s_id in station_cluster[cluster_id][0]:
                a = station_lanlon.loc[station_lanlon['station_id'] == s_id]
                temp_value +=charging_data.loc[charging_data['station_id'] == s_id, str(temp_date)].values[0]/1
            s_label=temp_value
            s_feature=[]
            s_feature.append(station_cluster[cluster_id][1])
            s_feature.append(station_cluster[cluster_id][2])
            s_date=temp_date.timetuple().tm_yday
            s_weekday=temp_date.weekday()
            s_sample_all.append(s_sample)
            s_label_all.append(s_label)
            s_feature_all.append(s_feature)
            s_date_all.append(s_date)
            s_weekday_all.append(s_weekday/6)
        all_data[cluster_id]=[np.array(s_sample_all),np.array(s_label_all),np.array(s_feature_all),np.array(s_date_all),np.array(s_weekday_all)]
    train_data = dict()
    test_data = dict()
    train_num=min(int(rate*len(station_cluster)),100)
    all_id=[i for i in range(len(station_cluster))]
    train_id=np.random.choice((all_id),train_num, replace=False)
    if city_name_eng=='Sydney':
        train_id=[3,5,7,4,2,1]
    for cluster_id in range(len(station_cluster)):
        if cluster_id in train_id:
            train_data[cluster_id]=all_data[cluster_id]
        else:
            test_data[cluster_id] = all_data[cluster_id]
    if by_station:
        folder_path='by_station'
    else:
        folder_path='by_hexagon'
    folder = os.path.exists(path+folder_path+'/'+city_name_eng)
    if not folder:
        os.makedirs(path+folder_path+'/'+city_name_eng)
    np.save(path+folder_path+'/'+city_name_eng+'/train_data.npy',train_data)
    np.save(path+folder_path+'/'+city_name_eng+'/test_data.npy',test_data)
    return train_data, test_data




