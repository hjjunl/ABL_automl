import imp
import warnings
import os

import pandas as pd
import argparse
import pickle
import json

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold, cross_val_score

from models.xgbmodel import XGBoostModel
from models.lgbmmodel import LGBMModel
from models.catboostmodel import CatBoostModel

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser("parser")
    parser.add_argument('-train', dest ='train', default='train.csv') # 데이터명
    parser.add_argument('-target', dest= "target", default = 'target') # 타겟 (이름 직접 지정해줘야함)
    
    parser.add_argument('-cv', dest = 'cv', default = '5')

    parser.add_argument('-gpu', dest = 'gpu', default = 'False')
    parser.add_argument('-gpu_id', dest = 'gpu_id', default = '0')
    parser.add_argument('-cpu_cnt', dest = 'cpu_cnt', default = '1')
    
    
    args = parser.parse_args()
    train = args.train
    target = args.target
    
    cv = args.cv

    gpu = args.gpu
    gpu_id = args.gpu_id 
    cpu_cnt = args.cpu_cnt
    
    #### data load ####
    print(f'***************  Load dataset ******************')
    if train.split('.')[1] == "csv":
        data = pd.read_csv(f'{os.getcwd()}/data/raw/{train}')

    elif train.split('.')[1] == "pkl":
        data = pd.read_pickle(f'{os.getcwd()}/data/raw/{train}')


    ### Best parameter search ####
    X = data.drop(columns=[target])
    y = data[[target]].values

    print(f'***************  XGB Best Parameter Search (Random Search) ******************')
    xgb = XGBoostModel(X, y)
    xgb.train(RandomSearch=True, gpu=eval(gpu), gpu_id=int(gpu_id), cpu_cnt=int(cpu_cnt), cv = int(cv))

    print(f'***************  LGBM Best Parameter Search (Random Search) ******************')
    lgb = LGBMModel(X, y)
    lgb.train(RandomSearch=True, cpu_cnt=int(cpu_cnt), cv = int(cv))

    print(f'***************  CatBoost Best Parameter Search (Random Search) ******************')
    cat = CatBoostModel(X, y)
    cat.train(RandomSearch=True, cpu_cnt=int(cpu_cnt), cv = int(cv))



    ### Model Selection ###
    with open('./outputs/xgboost/params/XGB_RandomSearch.json','r') as fp:
        xgb_params = json.load(fp)

    with open('./outputs/lightgbm/params/LGBM_RandomSearch.json','r') as fp:
        lgb_params = json.load(fp)

    with open('./outputs/catboost/params/CBC_RandomSearch.json','r') as fp:
        cat_params = json.load(fp)

    models = {'XGB': XGBClassifier(** xgb_params, n_jobs = int(cpu_cnt), gpu=eval(gpu), gpu_id=gpu_id), \
            'LGBM': LGBMClassifier(** lgb_params, n_jobs = int(cpu_cnt)), 'CBC': CatBoostClassifier(** cat_params, logging_level='Silent', thread_count = int(cpu_cnt))}

        
    kfold = KFold(n_splits=int(cv), shuffle = True, random_state=42)
    answer = []

    for model in models.keys():
        print(model)
        scores = cross_val_score(models[model] , X, y, cv=kfold, scoring='neg_log_loss', n_jobs = int(cpu_cnt))
        answer.append(scores)

    cross_val_result = pd.DataFrame(answer)
    cross_val_result['model'] = models.keys()
    cross_val_result['mean'] = cross_val_result.mean(axis=1)
        
    cross_val_result = cross_val_result.sort_values('mean', ascending = False)
    cross_val_result = cross_val_result.reset_index(drop=True)
    best_model = cross_val_result['model'][0]
            
    print('*********************  Vanilla Model Cross Val *********************')
    print('                                           ')
    print(cross_val_result)
    print('                                           ')
    print(f'***************** Best Model from Vanilla CV : {best_model} ******************')