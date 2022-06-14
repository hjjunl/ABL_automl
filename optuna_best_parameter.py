import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import warnings
import os


from sklearn.metrics import log_loss, mean_squared_error

from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor

import pandas as pd


from sklearn.model_selection import train_test_split

import argparse

warnings.filterwarnings(action='ignore')
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser("parser")
    parser.add_argument('-data', dest ='data', default= 'train.csv')
    parser.add_argument('-model_type', dest = 'model_type', default = 'XGB_clf') 
    parser.add_argument('-cpu_cnt', dest = 'cpu_cnt', default = 1) 
    parser.add_argument('-trials', dest='trials', default = 1) 

    parser.add_argument('-target', dest= "target", default = 'target')
    parser.add_argument('-test_size', dest = 'test_size', default = 0.2)
    parser.add_argument('-gpu', dest = 'gpu', default = False)
    parser.add_argument('-gpu_id', dest = 'gpu_id', default = 0)
   
    

    args = parser.parse_args()
    file_name = args.data
    model_type = args.model_type 
    cpu_cnt = args.cpu_cnt 
    trials = args.trials 

    target_name = args.target
    test_size = args.test_size
    gpu = args.gpu
    gpu_id = args.gpu_id
    
        
    print(f'***************  {model_type} Optuna run ******************')
    if file_name.split('.')[1] == "csv":
        data = pd.read_csv(f'{os.getcwd()}/preprocess_data/{file_name}')

    elif file_name.split('.')[1] == "pkl":
        data = pd.read_pickle(f'{os.getcwd()}/preprocess_data/{file_name}')
        
    

    data2 = data.drop(columns = [target_name])

    X_train, X_valid, y_train, y_valid = train_test_split(data2, data[target_name], test_size = float(test_size), random_state = 42)


    for model_type in ['XGB_clf', 'LGB_clf','CBC_clf']:# 
        print(model_type)
        def objective(X_train, X_valid, y_train, y_valid, model_type, trial : Trial) -> float :
            if model_type == 'XGB_clf': 
                # binary: logloss
                try:
                    params_xgb = {
                        'random_state' : 420,
                        'learning_rate' : trial.suggest_float('learning_rate', 0.003, 0.1),
                        'n_estimators' : 5000,
                        'max_depth' : trial.suggest_int('max_depth', 3, 16),
                        'reg_alpha' : trial.suggest_float('reg_alpha', 1e-8, 3e-3),
                        'reg_lambda' : trial.suggest_float('reg_lambda', 1e-8, 9e-2),
                        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
                        'max_bin' : trial.suggest_int('max_bin',2,100),
                        'eval_metric' : 'logloss',
                        }


                    if gpu == True:
                        model = XGBClassifier( ** params_xgb)
                        model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds = 100, verbose = 500, tree_method = 'gpu_hist', gpu_id = int(gpu_id))
                    else:
                        model = XGBClassifier(** params_xgb, n_jobs = int(cpu_cnt))
                        model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds = 100, verbose = 500)

                # multi-classification: mlogloss    
                except Exception:
                    print('multi-classification')
                    params_xgb = {
                        'random_state' : 420,
                        'learning_rate' : trial.suggest_float('learning_rate', 0.003, 0.1),
                        'n_estimators' : 5000,
                        'max_depth' : trial.suggest_int('max_depth', 3, 16),
                        'reg_alpha' : trial.suggest_float('reg_alpha', 1e-8, 3e-3),
                        'reg_lambda' : trial.suggest_float('reg_lambda', 1e-8, 9e-2),
                        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
                        'max_bin' : trial.suggest_int('max_bin',2,100),
                        'eval_metric' : 'mlogloss',
                        }     
                    if gpu == True:
                        model = XGBClassifier( ** params_xgb)
                        model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds = 100, verbose = 500, tree_method = 'gpu_hist', gpu_id = int(gpu_id))
                    else:
                        model = XGBClassifier(** params_xgb, n_jobs = int(cpu_cnt))
                        model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds = 100, verbose = 500)
                xgb_pred = model.predict_proba(X_valid)
                # multi-classification or binary
                # binary
                try: 
                    logloss =log_loss(y_valid, xgb_pred[:,1])
                # multi    
                except Exception:
                    logloss = log_loss(y_valid, xgb_pred)
                return logloss

            if model_type == 'LGB_clf': 
                try:
                    params_lgbm = {
                        'random_state' : 420,
                        'learning_rate' : trial.suggest_float('learning_rate', 0.003, 0.1),
                        'n_estimators' : 5000,
                        'metric' : 'logloss',
                        'max_depth' : trial.suggest_int('max_depth', 3, 16),
                        'reg_alpha' : trial.suggest_float('reg_alpha', 1e-8, 3e-3),
                        'reg_lambda' : trial.suggest_float('reg_lambda', 1e-8, 9e-2),
                        'num_leaves' : trial.suggest_int('num_leaves', 2, 256),
                        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
                        'min_child_samples' : trial.suggest_int('min_child_samples', 5, 100),
                        'max_bin' : trial.suggest_int('max_bin',2,100),
                        'scale_pos_weight' : trial.suggest_float('scale_pos_weight', 0, 1),
                        'n_jobs' : int(cpu_cnt)
                        }
                    model = LGBMClassifier( ** params_lgbm )
                    model.fit(
                        X_train,
                        y_train,
                        eval_set = [(X_train, y_train), (X_valid, y_valid)],
                        early_stopping_rounds = 100,
                        eval_metric = 'logloss',
                        verbose = 500
                    )
                    lgb_pred = model.predict_proba(X_valid)
                    logloss =log_loss(y_valid, lgb_pred[:,1])
                    return logloss
                # multi-classification
                except Exception:
                    params_lgbm = {
                        'random_state' : 420,
                        'learning_rate' : trial.suggest_float('learning_rate', 0.003, 0.1),
                        'n_estimators' : 5000,
                        'metric' : 'multi_logloss',
                        'max_depth' : trial.suggest_int('max_depth', 3, 16),
                        'reg_alpha' : trial.suggest_float('reg_alpha', 1e-8, 3e-3),
                        'reg_lambda' : trial.suggest_float('reg_lambda', 1e-8, 9e-2),
                        'num_leaves' : trial.suggest_int('num_leaves', 2, 256),
                        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
                        'min_child_samples' : trial.suggest_int('min_child_samples', 5, 100),
                        'max_bin' : trial.suggest_int('max_bin',2,100),
                        'scale_pos_weight' : trial.suggest_float('scale_pos_weight', 0, 1),
                        'n_jobs' : int(cpu_cnt)
                        }
                    print("fit lgbmclf")
                    model = LGBMClassifier( ** params_lgbm )
                    print('lgbmclf model:', model)
                    model.fit(
                        X_train,
                        y_train,
                        eval_set = [(X_train, y_train), (X_valid, y_valid)],
                        early_stopping_rounds = 100,
                        # eval_metric = 'mlogloss',
                        verbose = 500
                    )
                    lgb_pred = model.predict_proba(X_valid)
                    logloss =log_loss(y_valid, lgb_pred)
                    return logloss
                
            if model_type == 'CBC_clf': 
                try:
                    params_cat = {
                        'random_state' : 420,
                        'learning_rate' : trial.suggest_float('learning_rate', 0.003, 0.1),
                        'n_estimators' : 5000,
                        'eval_metric' : 'Logloss',
                        'depth' : trial.suggest_int('depth', 3, 16),
                        'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
                        'min_child_samples' : trial.suggest_int('min_child_samples', 5, 100),
                        'max_bin' : trial.suggest_int('max_bin',2,100)
                        }

                    if gpu == True:
                        model = CatBoostClassifier( ** params_cat, bootstrap_type='Poisson')
                        model.fit(
                        X_train,
                        y_train,
                        eval_set = [(X_train, y_train), (X_valid, y_valid)],
                        early_stopping_rounds = 100,
                        verbose = 500,
                        task_type = 'GPU',
                        devices = str(gpu_id))

                    else:
                        model = CatBoostClassifier(** params_cat, thread_count = int(cpu_cnt))
                        model.fit(
                            X_train,
                            y_train,
                            eval_set = [(X_train, y_train), (X_valid, y_valid)],
                            early_stopping_rounds = 100,
                            verbose = 500)

                    cat_pred = model.predict_proba(X_valid)

                    logloss =log_loss(y_valid, cat_pred[:,1])
                    return logloss
                # multi-classification
                except Exception:
                    print('mulit class')
                    params_cat = {
                        'random_state' : 420,
                        'learning_rate' : trial.suggest_float('learning_rate', 0.003, 0.1),
                        'n_estimators' : 5000,
                        # 'eval_metric' : 'Logloss',
                        'depth' : trial.suggest_int('depth', 3, 16),
                        # 'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
                        'min_child_samples' : trial.suggest_int('min_child_samples', 5, 100),
                        'max_bin' : trial.suggest_int('max_bin',2,100)
                        }

                    if gpu == True:
                        model = CatBoostClassifier( ** params_cat, bootstrap_type='Poisson')
                        model.fit(
                        X_train,
                        y_train,
                        eval_set = [(X_train, y_train), (X_valid, y_valid)],
                        early_stopping_rounds = 100,
                        verbose = 500,
                        task_type = 'GPU',
                        devices = str(gpu_id))

                    else:
                        print(params_cat)
                        model = CatBoostClassifier(** params_cat, thread_count = int(cpu_cnt))
                        print('model', model)
                        model.fit(
                            X_train,
                            y_train,
                            eval_set = [(X_train, y_train), (X_valid, y_valid)],
                            early_stopping_rounds = 100,
                            verbose = 500)

                    cat_pred = model.predict_proba(X_valid)
                    # df = pd.DataFrame(columns=['real', 'predict'])
                    # df['real'] = y_valid
                    # df['pred'] = cat_pred
                    # print(df)
                    logloss =log_loss(y_valid, cat_pred)
                    return logloss                    


            if model_type == 'XGB_reg':
                params_xgb = {
                    'random_state' : 420,
                    'learning_rate' : trial.suggest_float('learning_rate', 0.003, 0.1),
                    'n_estimators' : 5000,
                    'max_depth' : trial.suggest_int('max_depth', 3, 16),
                    'reg_alpha' : trial.suggest_float('reg_alpha', 1e-8, 3e-3),
                    'reg_lambda' : trial.suggest_float('reg_lambda', 1e-8, 9e-2),
                    'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
                    'max_bin' : trial.suggest_int('max_bin',2,100),
                    'eval_metric' : 'rmse' 
                    }

                if gpu == True:
                    model = XGBRegressor( ** params_xgb)
                    model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds = 100, verbose = 500, tree_method = 'gpu_hist', gpu_id = int(gpu_id))
                else:
                    model = XGBRegressor(** params_xgb, n_jobs = int(cpu_cnt))
                    model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds = 100, verbose = 500)

                xgb_pred = model.predict(X_valid)
                rmse = mean_squared_error(y_valid, xgb_pred, squared=False)
                return rmse

            if model_type == 'LGB_reg':
                params_lgbm = {
                    'random_state' : 420,
                    'learning_rate' : trial.suggest_float('learning_rate', 0.003, 0.1),
                    'n_estimators' : 5000,
                    'max_depth' : trial.suggest_int('max_depth', 3, 16),
                    'reg_alpha' : trial.suggest_float('reg_alpha', 1e-8, 3e-3),
                    'reg_lambda' : trial.suggest_float('reg_lambda', 1e-8, 9e-2),
                    'num_leaves' : trial.suggest_int('num_leaves', 2, 256),
                    'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
                    'min_child_samples' : trial.suggest_int('min_child_samples', 5, 100),
                    'max_bin' : trial.suggest_int('max_bin',2,100),
                    'scale_pos_weight' : trial.suggest_float('scale_pos_weight', 0, 1),
                    'n_jobs' : int(cpu_cnt)
                    }
                model = LGBMRegressor( ** params_lgbm )
                model.fit(
                    X_train,
                    y_train,
                    eval_set = [(X_train, y_train), (X_valid, y_valid)],
                    early_stopping_rounds = 100,
                    eval_metric = 'rmse',
                    verbose = 500
                )
                lgb_pred = model.predict(X_valid)
                rmse = mean_squared_error(y_valid, lgb_pred, squared=False)
                return rmse

            if model_type == 'CBC_reg':
                params_cat = {
                    'random_state' : 420,
                    'learning_rate' : trial.suggest_float('learning_rate', 0.003, 0.1),
                    'n_estimators' : 5000,
                    'eval_metric' : 'RMSE',
                    'depth' : trial.suggest_int('depth', 3, 16),
                    'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
                    'min_child_samples' : trial.suggest_int('min_child_samples', 5, 100),
                    'max_bin' : trial.suggest_int('max_bin',2,100),
                    }

                if gpu == True:
                    model = CatBoostRegressor( ** params_cat, bootstrap_type='Poisson' )
                    model.fit(
                    X_train,
                    y_train,
                    eval_set = [(X_train, y_train), (X_valid, y_valid)],
                    early_stopping_rounds = 100,
                    verbose = 500,
                    task_type = 'GPU',
                    devices = gpu_id)

                else:
                    model = CatBoostRegressor(** params_cat, thread_count = int(cpu_cnt))
                    model.fit(
                        X_train,
                        y_train,
                        eval_set = [(X_train, y_train), (X_valid, y_valid)],
                        early_stopping_rounds = 100,
                        verbose = 500)

                cat_pred = model.predict(X_valid)
                rmse = mean_squared_error(y_valid, cat_pred, squared=False)

                return rmse

        try:
            sampler = TPESampler()
            study = optuna.create_study(
            study_name = 'parameter_opt',
            direction = 'minimize', 
            sampler = sampler
            )

            study.optimize(lambda trial : objective(X_train, X_valid, y_train, y_valid, model_type, trial), n_trials = int(trials), n_jobs = int(cpu_cnt))

            pd.DataFrame(list(study.best_trial.params.items())).set_index(0).to_csv(f'params/Best_Params_{model_type}.csv')

        except Exception as e:
            print("Optuna ERROR !!!")
            print("ERROR Code: ", e)
        else:
            print(f'Best Score : {study.best_value}')
            print(f'Best Trial : {study.best_trial.params}')
            print("FINISH Optuna !!!")

