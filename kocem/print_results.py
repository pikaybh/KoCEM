# Beautiful table to print results of all categories

import os
import logging
from typing import Dict, Optional

import fire
import json
import numpy as np
from tabulate import tabulate

from utils.data import CAT_SHORT2LONG, DOMAIN_CAT2SUB_CAT
from utils.eval import calculate_ins_level_acc


# Root 
logger_name = 'print_results'
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)
# File Handler
file_handler = logging.FileHandler(f'logs/{logger_name}.log', encoding='utf-8-sig')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(r'%(asctime)s [%(name)s, line %(lineno)d] %(levelname)s: %(message)s'))
logger.addHandler(file_handler)
# Stream Handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter(r'%(message)s'))
logger.addHandler(stream_handler)


def main(path: Optional[str] = "output/blip2_flant5xxl"):
    # load all results
    all_results = {}
    for cat_folder_name in os.listdir(path):
        if cat_folder_name in CAT_SHORT2LONG.values():
            cat_folder_path = os.path.join(path, cat_folder_name)
            result_path = os.path.join(cat_folder_path, 'result.json')
            if os.path.exists(result_path):
                cat_results = json.load(open(result_path))
                all_results[cat_folder_name] = cat_results

    # print results
    headers = [f'Subject ({path.split("/")[-1]})', 'Data Num', 'Acc', 'Std', 'Var']
    table = []

    # add domain Subject
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats: # use the order in DOMAIN_CAT2SUB_CAT
            if cat_name in all_results.keys():
                in_domain_cat_results[cat_name] = all_results[cat_name]
            else:
                pass
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = np.sum([cat_results['num_example'] for cat_results in in_domain_cat_results.values()])
        # table.append(['Overall-' + domain, int(in_domain_data_num), round(in_domain_ins_acc, 3)])
        # 전체 도메인 표준편차 및 분산 계산
        domain_std_devs = [cat_results['std_dev'] for cat_results in in_domain_cat_results.values()]
        domain_variances = [std_dev ** 2 for std_dev in domain_std_devs]  # 분산 계산
        domain_std_dev = np.mean(domain_std_devs) if domain_std_devs else 0
        domain_variance = np.mean(domain_variances) if domain_variances else 0

        table.append([f'Overall-{domain}', int(in_domain_data_num), round(in_domain_ins_acc, 3), round(domain_std_dev, 3), round(domain_variance, 3)])

        # add sub category
        for cat_name, cat_results in in_domain_cat_results.items():
            variance = cat_results['std_dev'] ** 2  # 분산은 표준편차의 제곱
            table.append([
                cat_name, 
                int(cat_results['num_example']), 
                round(cat_results['acc'], 3), 
                round(cat_results['std_dev'], 3), 
                round(variance, 3)
            ])

    # 전체 결과에 대한 표준편차와 분산 계산
    all_ins_acc = calculate_ins_level_acc(all_results)
    all_std_devs = [cat_results['std_dev'] for cat_results in all_results.values()]
    all_variances = [std_dev ** 2 for std_dev in all_std_devs]
    overall_std_dev = np.mean(all_std_devs) if all_std_devs else 0
    overall_variance = np.mean(all_variances) if all_variances else 0

    table.append(['Overall', np.sum([cat_results['num_example'] for cat_results in all_results.values()]), round(all_ins_acc, 3), round(overall_std_dev, 3), round(overall_variance, 3)])

    logger.info(tabulate(table, headers=headers, tablefmt='orgtbl'))


if __name__ == '__main__':
    fire.Fire(main)
