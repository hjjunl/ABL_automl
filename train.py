from model.train_model import *

import warnings
import os

import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import pickle

warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser("parser")
    parser.add_argument('-data', dest ='data', default='train.csv') # 데이터명
    parser.add_argument('-target', dest= "target", default = 'target') # 타겟 (이름 직접 지정해줘야함)
    parser.add_argument('-model_type', dest='model_type', default = "XGB_clf") # XGB, LGB, CBC 등 모델명
    parser.add_argument('-memo', dest = 'memo', default = 'NaN') # 여러 Trial 들에 대한 메모(옵션) 
    
    parser.add_argument('-test_size', dest = 'test_size', default = '0.2')
    parser.add_argument('-gpu', dest = 'gpu', default = 'False')
    parser.add_argument('-gpu_id', dest = 'gpu_id', default = '0')
    parser.add_argument('-cpu_cnt', dest = 'cpu_cnt', default = '1')
    
    
    args = parser.parse_args()
    file_name = args.data
    target_name = args.target
    model_type = args.model_type
    memo = args.memo
    
    test_size = args.test_size
    gpu = args.gpu
    gpu_id = args.gpu_id 
    cpu_cnt = args.cpu_cnt
    
    #### data load ####
    print(f'***************  {model_type} Final model run ******************')
    if file_name.split('.')[1] == "csv":
        data = pd.read_csv(f'{os.getcwd()}/preprocess_data/{file_name}')

    elif file_name.split('.')[1] == "pkl":
        data = pd.read_pickle(f'{os.getcwd()}/preprocess_data/{file_name}')

            
    X_train, X_valid, y_train, y_valid = train_test_split(data, data[target_name], test_size = float(test_size), random_state = 42)
    
    print('****************  Final Model Training Started  ******************')
    final_model = model_train(X_train, y_train, X_valid, y_valid, model_type, gpu=bool(gpu), gpu_id=int(gpu_id), cpu_cnt=int(cpu_cnt))
    pickle.dump(final_model, open(f'{os.getcwd()}/model_result/{model_type}/{model_type}', 'wb'))
    
    get_results(model_type, final_model, X_train, X_valid, y_train, y_valid, memo)