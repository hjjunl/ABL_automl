import warnings
import os
import pandas as pd

# model import
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


from sklearn.model_selection import KFold, train_test_split, cross_val_score
import argparse

warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("parser")
    parser.add_argument('-data', dest ='data', default= 'train.csv')
    parser.add_argument('-purpose', dest='purpose', default = "clf")
    parser.add_argument('-target', dest = 'target', default = 'target')

    parser.add_argument('-cpu_cnt', dest='cpu_cnt', default = '1')
    parser.add_argument('-test_size', dest = 'test_size', default = '0.2')

    args = parser.parse_args()
    file_name = args.data
    purpose = args.purpose
    cpu_cnt = args.cpu_cnt
    test_size = args.test_size
    target_name = args.target 

    
    if file_name.split('.')[1] == "csv":
        data = pd.read_csv(f'{os.getcwd()}/origin_data/{file_name}')

    elif file_name.split('.')[1] == "pkl":
        data = pd.read_pickle(f'{os.getcwd()}/origin_data/{file_name}')


    data2 = data.drop(columns = [target_name])    
    X_train, X_valid, y_train, y_valid = train_test_split(data2, data[target_name], test_size = float(test_size), random_state = 42)

    if purpose == 'clf':
        models = {'RF' : RandomForestClassifier(n_jobs = int(cpu_cnt)), 'XGB' : XGBClassifier(n_jobs = int(cpu_cnt), verbosity = 0), 'LGB' : LGBMClassifier(n_jobs = int(cpu_cnt)), 
          'CBC' : CatBoostClassifier(logging_level='Silent', thread_count = int(cpu_cnt))}

        kfold = KFold(n_splits=5, shuffle = True, random_state=42)
        answer = []
        
        for model in models.keys():
            print(model)
            scores = cross_val_score(models[model] , X_train, y_train, cv=kfold, scoring='neg_log_loss', n_jobs = int(cpu_cnt))
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

    elif purpose == 'reg':
        models = {'RF_Regressor' : RandomForestRegressor(n_jobs = int(cpu_cnt)), 'XGB_Regressor' : XGBRegressor(n_jobs = int(cpu_cnt), verbosity = 0), 'LGB_Regressor' : LGBMRegressor(n_jobs = int(cpu_cnt)), 'CBC_Regressor' : CatBoostRegressor(logging_level='Silent', thread_count = int(cpu_cnt))}
        
        kfold = KFold(n_splits=5, shuffle = True, random_state=42)
        answer = []
        
        for model in models.keys():
            print(model)
            scores = cross_val_score(models[model] , X_train, y_train, cv=kfold, scoring='neg_mean_squared_error', n_jobs = int(cpu_cnt))
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

    else:
        print("정확한 purpose를 입력하세요!!! (eg. clf / reg)")