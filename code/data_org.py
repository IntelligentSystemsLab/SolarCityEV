# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2024/4/16 13:30
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2024/4/16 13:30

import datetime
import pandas as pd

city_dict = {
    '阿姆斯特丹': ['Amsterdam', 'Netherlands','AMS' ],
    '雅典': ['Athens', 'Greece', 'ATH'],
    '柏林': ['Berlin', 'Germany','BER'],
    '波士顿': ['Boston', 'UnitedStates', 'BOS'],
    '布鲁塞尔': ['Brussels', 'Belgium','BRU'],
    '布达佩斯': ['Budapest', 'Hungary','BUD'],
    '哥本哈根': ['Copenhagen', 'Denmark', 'CPH'],
    '都柏林': ['Dublin', 'Ireland', 'DUB'],
    '迪拜': ['Dubai', 'UnitedArabEmirates', 'DXB'],
    '赫尔辛基': ['Helsinki', 'Finland', 'HEL'],
    '火奴鲁鲁': ['Honolulu', 'UnitedStates', 'HNL'],
    '约翰内斯堡': ['Johannesburg', 'SouthAfrica', 'JNB'],
    '伦敦': ['London', 'UnitedKingdom','LDN'],
    '洛杉矶': ['LosAngeles', 'UnitedStates', 'LOA'],
    '马德里': ['Madrid', 'Spain', 'MAD'],
    '墨尔本': ['Melbourne', 'Australia','MEL'],
    '迈阿密': ['Miami', 'UnitedStates','MIA'],
    '米兰': ['Milan', 'Italy', 'MIL'],
    '蒙特利尔': ['Montreal', 'Canada', 'MTL'],
    '慕尼黑': ['Munich', 'Germany','MUC'],
    '纽约': ['NewYork', 'UnitedStates', 'NYC'],
    '奥斯陆': ['Oslo', 'Norway','OSL'],
    '渥太华': ['Ottawa', 'Canada', 'OTW'],
    '巴黎': ['Paris', 'France','PAR'],
    '布拉格': ['Prague', 'CzechRepublic','PRG'],
    '雷克雅未克': ['Reykjavik', 'Iceland', 'RKV'],
    '罗马': ['Rome', 'Italy', 'ROM'],
    '西雅图': ['Seattle', 'UnitedStates', 'SEA'],
    '旧金山': ['SanFrancisco', 'UnitedStates', 'SFO'],
    '圣胡安': ['SanJuan', 'PuertoRico','SJU'],
    '圣保罗': ['SaoPaulo', 'Brazil','SPO'],
    '斯德哥尔摩': ['Stockholm', 'Sweden','STO'],
    '悉尼': ['Sydney', 'Australia','SYD'],
    '深圳': ['Shenzhen', 'China','SZX'],
    '特拉维夫': ['TelAviv', 'Israel', 'TLV'],
    '多伦多': ['Toronto', 'Canada','TOR'],
    '维也纳': ['Vienna', 'Austria', 'VIE'],
    '华盛顿': ['Washington', 'UnitedStates','WAS'],
    '华沙': ['Warsaw', 'Poland','WAW'],
    '苏黎世': ['Zurich', 'Switzerland', 'ZRH'],
}

if __name__ == '__main__':
    start_date = datetime.date(2022, 12, 26)
    end_date = datetime.date(2023, 12, 31)
    day_num = (end_date - start_date).days
    time_list = []
    temp_date = start_date
    for i in range(day_num + 1):
        time_list.append(temp_date)
        temp_date += datetime.timedelta(days=1)

    for city in city_dict.keys():
        city_data = pd.DataFrame(columns=['station_id', 'latitude', 'longitude', 'city', 'country', 'date', 'kwh'])
        city_name_eng = city_dict[city][0]
        data = pd.read_csv('./final/pre_result_final/' + city_name_eng+ '/' + city_name_eng + '.csv')
        station_lanlon=pd.read_csv('./final/pre_result_final' + '/' + city_name_eng+'/' + city_name_eng +'_longitude-latitude'+ '.csv')
        data_size=int((data.iloc[0].values.shape[0]-1)/2)


