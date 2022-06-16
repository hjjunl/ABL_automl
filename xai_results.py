import warnings
import os

import pandas as pd
import argparse
import pickle

from utils.xai import *
from datetime import datetime

today = datetime.today().strftime("%Y%m%d")

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser("parser")
    parser.add_argument('-test', dest ='test', default='test.csv') # 데이터명
    parser.add_argument('-target', dest= "target", default = 'target') # 타겟 (이름 직접 지정해줘야함)
    parser.add_argument('-model', dest= "model", default = 'XGB') # 타겟 (이름 직접 지정해줘야함)

    parser.add_argument('-large_data', dest= "large_data", default = 'False')

    parser.add_argument('-gpu', dest = 'gpu', default = 'False')
    parser.add_argument('-gpu_id', dest = 'gpu_id', default = '0')
    parser.add_argument('-cpu_cnt', dest = 'cpu_cnt', default = '1')
    
    
    args = parser.parse_args()
    test = args.test
    target = args.target
    model = args.model

    large_data = args.large_data

    gpu = args.gpu
    gpu_id = args.gpu_id 
    cpu_cnt = args.cpu_cnt

    #### data load ####
    print(f'***************  Load dataset ******************')
    if test.split('.')[1] == "csv":
        data = pd.read_csv(f'{os.getcwd()}/data/preprocessed/{test}')

    elif test.split('.')[1] == "pkl":
        data = pd.read_pickle(f'{os.getcwd()}/data/preprocessed/{test}')


    #### Best parameter search ####
    X_test = data.drop(columns=[target])
    y_test = data[[target]].values
    

    #### XAI ####
    if model == 'XGB':
        print(f'***************  XGB XAI ******************')
        xgb_model = pickle.load(open(f'{os.getcwd()}/outputs/{today}/xgboost/models/XGB.pkl', 'rb'))
        xai(xgb_model, X_test, y_test, eval(large_data))

    elif model == 'LGBM':
        print(f'***************  LGBM XAI ******************')
        lgbm_model = pickle.load(open(f'{os.getcwd()}/outputs/{today}/lightgbm/models/LGBM.pkl', 'rb'))
        xai(lgbm_model, X_test, y_test, eval(large_data))

    elif model == 'CBC':
        print(f'***************  CatBoost XAI ******************')
        cbc_model = pickle.load(open(f'{os.getcwd()}/outputs/{today}/catboost/models/CBC.pkl', 'rb'))
        xai(cbc_model, X_test, y_test, eval(large_data))
    else:
        print("Model type ERROR!!! (select model: XGB, LGBM, CBC)")
