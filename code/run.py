# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023/9/25 14:35
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023/9/25 14:35
import numpy as np
import os
import argparse
from model.train import meta_train
from utils import seed_everything

# Get the project root directory (parent of code directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

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

# Create reverse mapping from English city name to Chinese name
city_eng_to_chn = {city_dict[chn][0]: chn for chn in city_dict.keys()}
available_cities_eng = sorted(city_eng_to_chn.keys())


def parse_args():
    parser = argparse.ArgumentParser(description='Train meta-learning model for EV charging demand prediction')
    
    parser.add_argument('--city', type=str, nargs='+', default=['London'], 
                        help='City name(s) in English (default: London). Can specify multiple cities. Available cities: ' + ', '.join(available_cities_eng))
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs (default: 300)')
    parser.add_argument('--support_epochs', type=int, default=5,
                        help='Number of support epochs (default: 5)')
    parser.add_argument('--custom_epochs', type=int, default=5,
                        help='Number of custom epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate (default: 0.005)')
    parser.add_argument('--divide_mode', type=str, nargs='+', default=['by_month'], choices=['by_month', 'by_day'],
                        help='Data division mode(s) (default: by_month). Can specify multiple modes: by_month and/or by_day')
    parser.add_argument('--folder_path', type=str, default='by_station',
                        help='Data folder path (default: by_station)')
    parser.add_argument('--seed', type=int, default=2023,
                        help='Random seed (default: 2023)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: None)')
    parser.add_argument('--print_details', action='store_true',
                        help='Print detailed training information')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Ensure city and divide_mode are lists
    cities_eng = args.city if isinstance(args.city, list) else [args.city]
    divide_modes = args.divide_mode if isinstance(args.divide_mode, list) else [args.divide_mode]
    
    # Validate city names (English) and convert to Chinese names
    cities_chn = []
    for city_eng in cities_eng:
        if city_eng not in city_eng_to_chn:
            print(f"Error: City '{city_eng}' not found.")
            print(f"Available cities (English): {', '.join(available_cities_eng)}")
            exit(1)
        cities_chn.append(city_eng_to_chn[city_eng])
    
    # Validate divide_modes
    valid_modes = ['by_month', 'by_day']
    for mode in divide_modes:
        if mode not in valid_modes:
            print(f"Error: Invalid divide_mode '{mode}'. Must be one of: {', '.join(valid_modes)}")
            exit(1)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "log_desktop.txt"), "a", encoding='utf-8') as f:
        seed_everything(seed=args.seed)
        
        # Iterate over cities and divide_modes
        for city_eng, city_chn in zip(cities_eng, cities_chn):
            for divide_mode in divide_modes:
                train_data = np.load(os.path.join(data_dir, args.folder_path, city_eng, 'train_data.npy'), allow_pickle=True).item()
                test_data = np.load(os.path.join(data_dir, args.folder_path, city_eng, 'test_data.npy'), allow_pickle=True).item()
                
                f.writelines(
                    '\n' + 'city:' + str(city_eng) + '\n' +
                    'divide_mode:' + str(divide_mode) + '\n' +
                    'folder_path:' + str(args.folder_path) + '\n' +
                    'epochs:' + str(args.epochs) + '\n' +
                    'support_epochs:' + str(args.support_epochs) + '\n' +
                    'custom_epochs:' + str(args.custom_epochs) + '\n' +
                    'lr:' + str(args.lr) + '\n' +
                    'seed:' + str(args.seed) + '\n'
                )
                f.flush()
                
                total_matrix = meta_train(
                    data=train_data,
                    evaluation_data=test_data,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    support_epochs=args.support_epochs,
                    custom_epochs=args.custom_epochs,
                    lr=args.lr,
                    print_details=args.print_details,
                    log_file=f,
                    mode=args.folder_path,
                    divide_mode=divide_mode,
                    city_name=city_eng
                )
        f.close()