# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2024/10/12 13:36
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2024/10/12 13:36
import numpy as np
from model.train import baseline_arima_monthly
from utils import seed_everything

seed=2023

city_dict = {
    '渥太华': ['Ottawa', 'Canada', ],
    '伦敦': ['London', 'UnitedKingdom',],
    '华沙': ['Warsaw', 'Poland',],
    '华盛顿': ['Washington', 'UnitedStates',],
    '圣保罗': ['SaoPaulo', 'Brazil',],
    '圣胡安': ['SanJuan', 'PuertoRico',],
    '墨尔本': ['Melbourne', 'Australia',],
    '多伦多': ['Toronto', 'Canada',],
    '奥斯陆': ['Oslo', 'Norway',],
    '巴黎': ['Paris', 'France',],
    '悉尼': ['Sydney', 'Australia',],
    '慕尼黑': ['Munich', 'Germany',],
    '斯德哥尔摩': ['Stockholm', 'Sweden',],
    '旧金山': ['SanFrancisco', 'UnitedStates', ],
    '洛杉矶': ['LosAngeles', 'UnitedStates', ],
    '火奴鲁鲁': ['Honolulu', 'UnitedStates', ],
    '特拉维夫': ['TelAviv', 'Israel', ],
    '米兰': ['Milan', 'Italy', ],
    '约翰内斯堡': ['Johannesburg', 'SouthAfrica', ],
    '纽约': ['NewYork', 'UnitedStates', ],
    '维也纳': ['Vienna', 'Austria', ],
    '罗马': ['Rome', 'Italy', ],
    '苏黎世': ['Zurich', 'Switzerland', ],
    '蒙特利尔': ['Montreal', 'Canada', ],
    '西雅图': ['Seattle', 'UnitedStates', ],
    '赫尔辛基': ['Helsinki', 'Finland', ],
    '迪拜': ['Dubai', 'UnitedArabEmirates', ],
    '都柏林': ['Dublin', 'Ireland', ],
    '阿姆斯特丹': ['Amsterdam', 'Netherlands', ],
    '雅典': ['Athens', 'Greece', ],
    '雷克雅未克': ['Reykjavik', 'Iceland', ],
    '马德里': ['Madrid', 'Spain', ],
    '波士顿': ['Boston', 'UnitedStates', ],
    '布拉格': ['Prague', 'CzechRepublic',],
    '布达佩斯': ['Budapest', 'Hungary',],
    '布鲁塞尔': ['Brussels', 'Belgium',],
    '哥本哈根': ['Copenhagen', 'Denmark', ],
    '迈阿密': ['Miami', 'UnitedStates',],
    '柏林': ['Berlin', 'Germany',],
    '深圳': ['Shenzhen', 'China',],
}

r_folder=['by_station']
baselines=['ARIMA_month']

if __name__ == '__main__':
    for baseline in baselines:
        with open(f"/results/log_{baseline}.txt", "a", encoding='utf-8') as f:
            seed_everything(seed=seed)
            folder_path=r_folder[0]
            for city in city_dict.keys():
                if baseline=='FCNN' and city!='深圳':
                    continue
                for divide_mode in ['by_month']:
                    epochs = 300
                    support_epochs = 5
                    custom_epochs = 0
                    lr = 0.005
                    city_name_eng = city_dict[city][0]
                    train_data=np.load('./data/'+folder_path+'/'+city_name_eng+'/train_data.npy',allow_pickle=True).item()
                    test_data=np.load('./data/'+folder_path+'/'+city_name_eng+'/test_data.npy',allow_pickle=True).item()
                    f.writelines(
                        '\n' +'city:' + str(city) + '\n' +
                        'divide_mode:' + str(divide_mode) + '\n' +
                        'folder_path:' + str(folder_path) + '\n'
                    )
                    f.flush()
                    baseline_arima_monthly(
                        data=train_data,
                        evaluation_data=test_data,
                        log_file=f,
                        city_name=city_name_eng,
                        baseline_name=baseline,
                    )
        f.close()