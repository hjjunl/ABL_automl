import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pandas as pd
import numpy as np
from time import time, ctime
from os.path import exists
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss, mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error, precision_score, recall_score, roc_curve
import warnings

warnings.filterwarnings(action='ignore')

time = time()
timestamp = ctime(time)

# 모델 학습 - 새로운 모델 추가 가능
def model_train(X_train, y_train, X_valid, y_valid, model_type, gpu = False, gpu_id = 0, cpu_cnt = 1):
    try:
        Params = pd.read_csv(f'params/Best_Params_{model_type}.csv')
        Params = Params.set_index('0').to_dict()['1']

    except:
        if model_type == 'LGB_clf':
            model = LGBMClassifier( n_jobs=int(cpu_cnt),
                                    n_estimators = 5000,
                                    random_state = 420)
            model.fit(X_train, y_train, verbose=50, eval_set = [(X_valid, y_valid)], eval_metric = 'logloss', early_stopping_rounds=100)
        

        elif model_type == 'XGB_clf':
            if gpu == True:
                model = XGBClassifier(n_estimators = 5000, random_state = 420, tree_method = 'gpu_hist', gpu_id = int(gpu_id), verbosity = 0)
                model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds = 100, verbose = 500)
            else:
                model = XGBClassifier(n_estimators = 5000, random_state = 420, n_jobs = int(cpu_cnt), verbosity = 0)
                model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds = 100, verbose = 500)


        elif model_type == 'CBC_clf':
            if gpu == True:
                model = CatBoostClassifier(
                                    n_estimators = 5000,
                                    random_state = 420, task_type = 'GPU', devices = str(gpu_id), bootstrap_type='Poisson')
                model.fit(X_train, y_train, verbose=200, eval_set = [(X_valid, y_valid)], early_stopping_rounds=500)
            else:
                model = CatBoostClassifier(
                                    n_estimators = 5000,
                                    random_state = 420, thread_count=int(cpu_cnt))
                model.fit(X_train, y_train, verbose=200, eval_set = [(X_valid, y_valid)], early_stopping_rounds=500)


        elif model_type == 'LGB_reg':
            model = LGBMRegressor(
                                    n_jobs=int(cpu_cnt),
                                    n_estimators = 5000,
                                    random_state = 420)
            model.fit(X_train, y_train, verbose=50, eval_set = [(X_valid, y_valid)], eval_metric = 'rmse', early_stopping_rounds=100)


        elif model_type == 'XGB_reg':
            if gpu == True:
                model = XGBRegressor( n_estimators = 5000, random_state = 420, tree_method = 'gpu_hist', gpu_id = int(gpu_id), verbosity = 0)
                model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds = 100, verbose = 500)
            else:
                model = XGBRegressor( n_estimators = 5000, random_state = 420, n_jobs = int(cpu_cnt), verbosity = 0)
                model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds = 100, verbose = 500)

                
        elif model_type == 'CBC_reg':
            if gpu == True:
                model = CatBoostRegressor(
                                    n_estimators = 5000,
                                    bootstrap_type='Poisson',
                                    random_state = 420, task_type = 'GPU', devices = str(gpu_id))
                model.fit(X_train, y_train, verbose=200, eval_set = [(X_valid, y_valid)], early_stopping_rounds=500)
            else:
                model = CatBoostRegressor(
                                    n_estimators = 5000,
                                    random_state = 420, thread_count=int(cpu_cnt))
                model.fit(X_train, y_train, verbose=200, eval_set = [(X_valid, y_valid)], early_stopping_rounds=500)


        else:
            print("정확한 model type을 입력하세요!!! (eg. XGB_clf or XGB_reg)")

    else:
        if model_type == 'LGB_clf':
            Params['learning_rate'] = float(Params['learning_rate'])
            Params['max_depth'] = int(Params['max_depth'])
            Params['reg_alpha'] = float(Params['reg_alpha'])
            Params['reg_lambda'] = float(Params['reg_lambda'])
            Params['num_leaves'] = int(Params['num_leaves'])
            Params['colsample_bytree'] = float(Params['colsample_bytree'])
            Params['subsample'] = float(Params['subsample'])
            Params['min_child_samples'] = int(Params['min_child_samples'])
            Params['max_bin'] = int(Params['max_bin'])
            Params['scale_pos_weight'] = float(Params['scale_pos_weight'])
            
            model = LGBMClassifier( ** Params,
                                    n_jobs=int(cpu_cnt),
                                    n_estimators = 5000,
                                    random_state = 420)
            model.fit(X_train, y_train, verbose=50, eval_set = [(X_valid, y_valid)], eval_metric = 'logloss', early_stopping_rounds=100)
            
        elif model_type == 'XGB_clf':
            Params['learning_rate'] = float(Params['learning_rate'])
            Params['max_depth'] = int(Params['max_depth'])
            Params['reg_alpha'] = float(Params['reg_alpha'])
            Params['reg_lambda'] = float(Params['reg_lambda'])
            Params['colsample_bytree'] = float(Params['colsample_bytree'])
            Params['subsample'] = float(Params['subsample'])
            Params['max_bin'] = int(Params['max_bin'])
            
            if gpu == True:
                model = XGBClassifier( ** Params, n_estimators = 5000, random_state = 420, tree_method = 'gpu_hist', gpu_id = int(gpu_id), verbosity = 0)
                model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds = 100, verbose = 500)
            else:
                model = XGBClassifier( ** Params, n_estimators = 5000, random_state = 420, n_jobs = int(cpu_cnt), verbosity = 0)
                model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds = 100, verbose = 500)

        elif model_type == 'CBC_clf':
            Params['learning_rate'] = float(Params['learning_rate'])
            Params['depth'] = int(Params['depth'])
            Params['subsample'] = float(Params['subsample'])
            Params['min_child_samples'] = int(Params['min_child_samples'])
            Params['max_bin'] = int(Params['max_bin'])

            if gpu == True:
                model = CatBoostClassifier(** Params,
                                    n_estimators = 5000,
                                    random_state = 420, task_type = 'GPU', devices = str(gpu_id), bootstrap_type='Poisson')
                model.fit(X_train, y_train, verbose=200, eval_set = [(X_valid, y_valid)], early_stopping_rounds=500)
            else:
                model = CatBoostClassifier(** Params,
                                    n_estimators = 5000,
                                    random_state = 420, thread_count=int(cpu_cnt))
                model.fit(X_train, y_train, verbose=200, eval_set = [(X_valid, y_valid)], early_stopping_rounds=500)


        elif model_type == 'LGB_reg':
            Params['learning_rate'] = float(Params['learning_rate'])       
            Params['max_depth'] = int(Params['max_depth'])
            Params['reg_alpha'] = float(Params['reg_alpha'])
            Params['reg_lambda'] = float(Params['reg_lambda'])
            Params['num_leaves'] = int(Params['num_leaves'])
            Params['colsample_bytree'] = float(Params['colsample_bytree'])
            Params['subsample'] = float(Params['subsample'])
            Params['min_child_samples'] = int(Params['min_child_samples'])
            Params['max_bin'] = int(Params['max_bin'])
            Params['scale_pos_weight'] = float(Params['scale_pos_weight'])
            
            model = LGBMRegressor( ** Params,
                                    n_jobs=int(cpu_cnt),
                                    n_estimators = 5000,
                                    random_state = 420)
            model.fit(X_train, y_train, verbose=50, eval_set = [(X_valid, y_valid)], eval_metric = 'rmse', early_stopping_rounds=100)
            
        elif model_type == 'XGB_reg':
            Params['learning_rate'] = float(Params['learning_rate'])
            Params['max_depth'] = int(Params['max_depth'])
            Params['reg_alpha'] = float(Params['reg_alpha'])
            Params['reg_lambda'] = float(Params['reg_lambda'])
            Params['colsample_bytree'] = float(Params['colsample_bytree'])
            Params['subsample'] = float(Params['subsample'])
            Params['max_bin'] = int(Params['max_bin'])
            
            if gpu == True:
                model = XGBRegressor( ** Params, n_estimators = 5000, random_state = 420, verbose = 500, tree_method = 'gpu_hist', gpu_id = int(gpu_id), verbosity = 0)
                model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds = 100)
            else:
                model = XGBRegressor( ** Params, n_estimators = 5000, random_state = 420, n_jobs = int(cpu_cnt), verbosity = 0)
                model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds = 100, verbose = 500)
        
        
        elif model_type == 'CBC_reg':
            Params['learning_rate'] = float(Params['learning_rate'])
            Params['depth'] = int(Params['depth'])
            Params['subsample'] = float(Params['subsample'])
            Params['min_child_samples'] = int(Params['min_child_samples'])
            Params['max_bin'] = int(Params['max_bin'])

            if gpu == True:
                model = CatBoostRegressor( ** Params,
                                    bootstrap_type='Poisson',
                                    n_estimators = 5000,
                                    random_state = 420, task_type = 'GPU', devices = str(gpu_id))
                model.fit(X_train, y_train, verbose=200, eval_set = [(X_valid, y_valid)], early_stopping_rounds=500)
            else:
                model = CatBoostRegressor( ** Params,
                                    bootstrap_type='Poisson',
                                    n_estimators = 5000,
                                    random_state = 420, thread_count=int(cpu_cnt))
                model.fit(X_train, y_train, verbose=200, eval_set = [(X_valid, y_valid)], early_stopping_rounds=500)
    
        else:
            print("정확한 model type을 입력하세요!!! (eg. XGB_clf or XGB_reg)")

    return model



def model_visualization(X_valid2, y_valid2, final_model, auc_val, auc_train, model_type):
    ## ROC Curve 시각화 및 png 파일 output 폴더에 저장
    
    ns_probs = np.zeros((len(y_valid2), 1)) 
    lr_probs = final_model.predict_proba(X_valid2)
    
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_valid2, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_valid2, lr_probs)
    # plot the roc curve for the model
    auc_curve = plt.figure()
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, color ='darkorange', linestyle='--', label=f'{model_type} ROC Curve')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    auc_curve.savefig(f'model_result/{model_type}/ROC_{model_type}.png')
    # show the plot
    plt.close()
    
    print(f' ROC AUC  (Validation) : {auc_val} / train : {auc_train}')


def get_results(model_type, final_model, X_train, X_valid, y_train, y_valid, memo):
    if model_type.split('_')[1] == 'reg':
        val_pred = final_model.predict(X_valid)
        train_pred = final_model.predict(X_train)

        MAE_val =  mean_absolute_error(y_valid, val_pred)
        MAE_train = mean_absolute_error(y_train, train_pred)

        MSE = mean_squared_error(y_valid, val_pred)

        RMSE_val = np.sqrt(MAE_val)
        RMSE_train = np.sqrt(MAE_train)

        R2 = r2_score(y_valid, val_pred)

        print(f" RMSE (Validation) : {RMSE_val} / train : {RMSE_train}")
        print(f" MAE (Validation) : {MAE_val} / train : {MAE_train}")
        
        

        if exists(f'{os.getcwd()}/model_result/{model_type}/{model_type}_score.csv'):
            score_df = pd.read_csv(f'{os.getcwd()}/model_result/{model_type}/{model_type}_score.csv')
            new_score_df = pd.DataFrame({'TimeStamp': timestamp,
                                        'Memo' : memo,
                                        'RMSE' : np.round(RMSE_val,4),
                                        'MAE'  : np.round(MAE_val,4),
                                        'MSE'  : np.round(MSE, 4),
                                        'R2' : np.round(R2,4)
                                        }, index = [1])
            
            score_df2 = pd.concat([score_df, new_score_df], axis = 0)
            score_df2.to_csv(f'{os.getcwd()}/model_result/{model_type}/{model_type}_score.csv', index = 0)
            
        else:
            # 최초 score_df 생성
            score_df = pd.DataFrame({'TimeStamp': timestamp,
                                        'Memo' : memo,
                                        'RMSE' : np.round(RMSE_val,4),
                                        'MAE'  : np.round(MAE_val,4),
                                        'MSE'  : np.round(MSE, 4),
                                        'R2' : np.round(R2,4)
                                        }, index = [0])
            
            score_df.to_csv(f'{os.getcwd()}/model_result/{model_type}/{model_type}_score.csv', index = 0)

    elif model_type.split('_')[1] == 'clf':
        val_pred = final_model.predict(X_valid)
        val_pred_proba = final_model.predict_proba(X_valid)
        
        auc_train = roc_auc_score(y_train, final_model.predict_proba(X_train)[:,1])
        auc_val = roc_auc_score(y_valid, val_pred_proba[:,1])

        
        print(f" Accuracy (Validation) : {accuracy_score(y_valid, val_pred)} / train : {accuracy_score(y_train, final_model.predict(X_train))}")
        print(f" F1 Score (Validation) : {f1_score(y_valid, val_pred)} / train : {f1_score(y_train, final_model.predict(X_train))}")
        
        
        
        if exists(f'{os.getcwd()}/model_result/{model_type}/{model_type}_score.csv'):
            score_df = pd.read_csv(f'{os.getcwd()}/model_result/{model_type}/{model_type}_score.csv')
            new_score_df = pd.DataFrame({'TimeStamp': timestamp,
                                        'Memo' : memo,
                                        'Accuracy' : np.round(accuracy_score(y_valid, val_pred),4),
                                        'LogLoss'  : np.round(log_loss(y_valid, val_pred),4),
                                        'ROC-AUC'  : np.round(roc_auc_score(y_valid, val_pred), 4),
                                        'Precision' : np.round(precision_score(y_valid, val_pred), 4),
                                        'Recall' : np.round(recall_score(y_valid, val_pred),4),
                                        'F-1 Score' : np.round(f1_score(y_valid, val_pred),4)
                                        }, index = [1])
            
            score_df2 = pd.concat([score_df, new_score_df], axis = 0)
            score_df2.to_csv(f'{os.getcwd()}/model_result/{model_type}/{model_type}_score.csv', index = 0)
            
        else:
            # 최초 score_df 생성
            score_df = pd.DataFrame({'TimeStamp': timestamp,
                                    'Memo' : memo,
                                    'Accuracy' : np.round(accuracy_score(y_valid, val_pred),4),
                                    'LogLoss'  : np.round(log_loss(y_valid, val_pred),4),
                                    'ROC-AUC'  : np.round(roc_auc_score(y_valid, val_pred), 4),
                                    'Precision' : np.round(precision_score(y_valid, val_pred), 4),
                                    'Recall' : np.round(recall_score(y_valid, val_pred),4),
                                    'F-1 Score' : np.round(f1_score(y_valid, val_pred),4)
                                    }, index = [0])
            
            score_df.to_csv(f'{os.getcwd()}/model_result/{model_type}/{model_type}_score.csv', index = 0)
        
        model_visualization(X_valid, y_valid, final_model, auc_val, auc_train, model_type)

    else:
        print("정확한 model type을 입력하세요!!! (eg. XGB_reg or XGB_clf)")