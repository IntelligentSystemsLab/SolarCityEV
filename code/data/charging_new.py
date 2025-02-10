# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2024/1/31 22:17
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2024/1/31 22:17

import datetime
import tqdm
import warnings
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

city_dict = {
    '华沙': ['Warsaw', 'Poland',],
    '华盛顿': ['Washington', 'UnitedStates',],
    '哥本哈根': ['Copenhagen', 'Denmark',],
    '圣保罗': ['SaoPaulo', 'Brazil',],
    '圣胡安': ['SanJuan', 'PuertoRico',],
    '墨尔本': ['Melbourne', 'Australia',],
    '伦敦': ['London', 'UnitedKingdom',],
    '深圳': ['Shenzhen', 'China',],
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
    '柏林': ['Berlin', 'Germany',],
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
}

if __name__ == '__main__':
    data_column=['station_id','city','country']
    start_date=datetime.date(2022,12,24)
    end_date=datetime.date(2023,12,31)
    temp_date=start_date
    for i in range((end_date-start_date).days+1):
        data_column.append(str(temp_date))
        temp_date+=datetime.timedelta(days=1)
    for city in city_dict.keys():
        charging_dataframe = pd.DataFrame(columns=data_column)
        reader = pd.read_csv('./row/'+city+'.csv', iterator=True, chunksize=1000)
        for chunk in tqdm.tqdm(reader):
            for i in range(chunk.shape[0]):
                chunk['time_new'].iloc[i]=chunk['time_new'].iloc[i][:10]
            station_id_list=set(list(chunk['station_id']))
            time_list=set(list(chunk['time_new']))
            for station_id in station_id_list:
                if station_id not in charging_dataframe['station_id'].values:
                    row_data_column = [station_id, city_dict[city][0], city_dict[city][1]]
                    for i in range((end_date - start_date).days + 1):
                        row_data_column.append(0.0)
                    charging_dataframe.loc[len(charging_dataframe.index)] = row_data_column
                for t in time_list:
                    temp_data1=chunk.loc[chunk['station_id']==station_id,:]
                    temp_data2=temp_data1.loc[temp_data1['time_new']==t,:]
                    kwh_sum=sum(temp_data2['kwh'].values)
                    charging_dataframe.loc[charging_dataframe['station_id'] == station_id, t] +=kwh_sum
        charging_dataframe.to_csv('./ori/' + city_dict[city][0] +'_to_'+str(end_date)+ '.csv', index=False)

