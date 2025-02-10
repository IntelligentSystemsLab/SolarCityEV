# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023/9/25 14:35
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023/9/25 14:35
import numpy as np
from model.train import meta_train
from utils import seed_everything

seed=2023

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
    '柏林': ['Berlin', 'Germany',],
    '洛杉矶': ['LosAngeles', 'UnitedStates',],
    '深圳': ['Shenzhen', 'China',],
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


r_folder=['by_station']

if __name__ == '__main__':
    with open("/results/log_desktop.txt", "a", encoding='utf-8') as f:
        folder_path=r_folder[0]
        seed_everything(seed=seed)
        # for city in city_dict.keys():
        for city in ['伦敦']:
            for divide_mode in ['by_month']:
                # epochs = 300
                epochs = 1
                support_epochs = 5
                custom_epochs = 5
                lr = 0.005
                city_name_eng = city_dict[city][0]
                train_data=np.load('/data/'+folder_path+'/'+city_name_eng+'/train_data.npy',allow_pickle=True).item()
                test_data=np.load('/data/'+folder_path+'/'+city_name_eng+'/test_data.npy',allow_pickle=True).item()
                f.writelines(
                    '\n' +'city:' + str(city) + '\n' +
                    'divide_mode:' + str(divide_mode) + '\n' +
                    'folder_path:' + str(folder_path) + '\n' +
                    'epochs:' + str(epochs) + '\n' +
                    'support_epochs:' + str(support_epochs) + '\n' +
                    'custom_epochs:' + str(custom_epochs) + '\n' +
                    'lr:' + str(lr) + '\n'
                )
                f.flush()
                total_matrix=meta_train(
                    data=train_data,
                    evaluation_data=test_data,
                    batch_size=None,
                    epochs=epochs,
                    support_epochs=support_epochs,
                    custom_epochs=custom_epochs,
                    lr=lr,
                    print_details=False,
                    log_file=f,
                    mode=folder_path,
                    divide_mode=divide_mode,
                    city_name=city_name_eng
                )
        f.close()