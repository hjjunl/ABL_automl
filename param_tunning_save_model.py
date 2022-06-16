import warnings
import os

import pandas as pd
import argparse
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split

from models.xgbmodel import XGBoostModel
from models.lgbmmodel import LGBMModel
from models.catboostmodel import CatBoostModel

today = datetime.today().strftime("%Y%m%d")

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser("parser")
    parser.add_argument('-train', dest ='train', default='train.csv') # 데이터명
    parser.add_argument('-target', dest= "target", default = 'target') # 타겟 (이름 직접 지정해줘야함)
    parser.add_argument('-model', dest= "model", default = 'XGB') # 타겟 (이름 직접 지정해줘야함)
    
    parser.add_argument('-valid_size', dest = 'valid_size', default = '0.2')
    parser.add_argument('-model_save', dest = 'model_save', default = 'True')
    parser.add_argument('-trials', dest = 'trials', default = '1')


    parser.add_argument('-gpu', dest = 'gpu', default = 'False')
    parser.add_argument('-gpu_id', dest = 'gpu_id', default = '0')
    parser.add_argument('-cpu_cnt', dest = 'cpu_cnt', default = '1')
    
    
    args = parser.parse_args()
    train = args.train
    target = args.target
    model = args.model
    
    valid_size = args.valid_size
    model_save = args.model_save
    trials = args.trials

    gpu = args.gpu
    gpu_id = args.gpu_id 
    cpu_cnt = args.cpu_cnt
    
    #### data load ####
    print(f'***************  Load dataset ******************')
    if train.split('.')[1] == "csv":
        data = pd.read_csv(f'{os.getcwd()}/data/preprocessed/{train}')

    elif train.split('.')[1] == "pkl":
        data = pd.read_pickle(f'{os.getcwd()}/data/preprocessed/{train}')

    #### Best parameter search ####
    X = data.drop(columns=[target])
    y = data[[target]].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size = float(valid_size), random_state = 42)

    if model == 'XGB':
        print(f'***************  XGB Best Parameter Search (Optuna) ******************')
        xgb = XGBoostModel(X_train, y_train, X_valid, y_valid)
        xgb.train(Optuna=True, trials=int(trials), gpu=eval(gpu), gpu_id=int(gpu_id), cpu_cnt=int(cpu_cnt), save_model=eval(model_save), file_name='XGB')

    elif model == 'LGBM':
        print(f'***************  LGBM Best Parameter Search (Optuna) ******************')
        lgb = LGBMModel(X_train, y_train, X_valid, y_valid)
        lgb.train(Optuna=True, trials=int(trials), gpu=eval(gpu), gpu_id=int(gpu_id), cpu_cnt=int(cpu_cnt), save_model=eval(model_save), file_name='LGBM')

    elif model == 'CBC':
        print(f'***************  CatBoost Best Parameter Search (Optuna) ******************')
        cbc = CatBoostModel(X_train, y_train, X_valid, y_valid)
        cbc.train(Optuna=True, trials=int(trials), gpu=eval(gpu), gpu_id=int(gpu_id), cpu_cnt=int(cpu_cnt), save_model=eval(model_save), file_name='CBC')

    else:
        print("Model type ERROR!!! (select model: XGB, LGBM, CBC)")




