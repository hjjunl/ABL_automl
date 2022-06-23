import eli5
from pandas_profiling import ProfileReport
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from models.xgbmodel import *
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import argparse
import warnings
import time
from datetime import datetime
import os

warnings.filterwarnings(action='ignore')
today = datetime.today().strftime("%Y%m%d")



def eda(model, data, target, model_type):
    print("========== Pandas Profiling Report ==========")
    profile = ProfileReport(data, title="Pandas Profiling Report")
    profile.to_file(f'./outputs/{today}/feature_output/{model_type}_EDA.html')

    print("========== SHAP ==========")
    shap.initjs()
    X_train = data.drop(columns=[target])
    X_sampled = X_train.sample(1000, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sampled)

    plt.figure()

    plt.title(f'{model_type} Top SHAP values')

    if model_type == 'LGB_clf':
        shap.summary_plot(shap_values[1], X_sampled, plot_type="bar", show=False, plot_size=[8, 6])
    else:
        shap.summary_plot(shap_values, X_sampled, plot_type="bar", show=False, plot_size=[8, 6])

    plt.savefig(f'./outputs/{today}/feature_output/{model_type}_SHAP.png', dpi=50, bbox_inches='tight')


# RFE, RFECV, Permutation importance
class feature_selections:
    def __init__(self, data, X_train, y_train, model_type):
        self.data = data
        self.X_train = X_train
        self.y_train = y_train
        self.model_type = model_type

    def feature_importance(self, model, X_train, model_type):
        importances = model.feature_importances_
        sorted_idx = importances.argsort()[::-1]
        proportion = (importances / sum(importances))

        importances = pd.DataFrame({'columns': X_train.columns[sorted_idx]
                                       , 'importances': importances[sorted_idx]
                                       , 'proportion': proportion[sorted_idx]

                                    })

        plt.figure(figsize=(15, 6))
        plt.title(f'{model_type} Feature importances')
        plt.barh(importances['columns'][:30], importances['importances'][:30])
        plt.savefig(f'./outputs/{today}/feature_output/{model_type}_importance.png')

        importances.to_pickle(f'./outputs/{today}/feature_output/{model_type}_feature_importances.pkl')

        return importances

    def feature_selection_rfe(self, train, test, importances_df, model, X_train, y_train, target):
        print('****************  Feature Selection Started ******************')
        start = time.time()

        selector = RFE(model, n_features_to_select=int(10), step=2, verbose=500) # 60
        selector = selector.fit(X_train, y_train)
        temp = pd.DataFrame(selector.support_, columns=['feature_bool'])
        importances = pd.concat([importances_df, temp], axis=1)
        final_table = importances[importances['feature_bool'] == True]
        final_table = final_table.reset_index(drop=True).drop(columns='feature_bool')
        features = list(final_table['columns'])

        newdata_train = train[features]
        newdata_train[target] = train[target]
        
        newdata_test = test[features]
        newdata_test[target] = test[target]
        print(final_table)
#         if train.split('.')[1] == "csv":
#             newdata_train.to_csv(f'./data/preprocessed/{today}/{train}.csv, index=False)

#         elif train.split('.')[1] == "pkl":
#             newdata_train.to_pickle(f'./data/preprocessed/{today}/{train}')
#         if test.split('.')[1] == "csv":
#             newdata_train.to_csv(f'./data/preprocessed/{today}/{test}.csv, index=False)

#         elif train.split('.')[1] == "pkl":
#             newdata_train.to_pickle(f'./data/preprocessed/{today}/{test}')
                                 
        print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        feature_selection_rfecv(self, newdata_train, newdata_test, model, train[features], train[target], target)                         

    def feature_selection_rfecv(self, train, test, model, X_train, y_train, target):
        print('****************  Feature Selection RFECV Started ******************')
        print(train)
        start = time.time()

        min_features_to_select = 5
        rfecv = RFECV(model, cv=StratifiedKFold(5), step=1, scoring='f1', importance_getter='auto',
                      min_features_to_select=5)
        # step 제거하는 feature 수
        # min_features_to_select 최소 남기는 feature 수
        selector = rfecv.fit(X_train, y_train)
        temp = pd.DataFrame(selector.support_, columns=['feature_bool'])
        print(temp)
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (f1)")
        plt.plot(
            range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
            rfecv.grid_scores_,
        )
        # plt.show()
        plt.savefig(f'./outputs/{today}/feature_output/rfecv.png')
        print(selector.support_)

        using_col = []
        not_using_col = []
        for idx, i in enumerate(list(selector.support_)):
            if i == True:
                using_col.append(train.columns[idx])
            else:
                not_using_col.append(idx)
        temp['feature_bool'] = temp['feature_bool'].drop(not_using_col)
        temp = temp.dropna()
        # temp['feature_bool'] = temp['feature_bool'][temp['feature_bool']==True]
        print('changed temp', temp)
        print('using_col:', using_col)
        try:
            temp['feature'] = using_col
            print(temp)
        except Exception:
            print('fail')
            print('len temp: ', len(temp))
            print('len using col: ', len(using_col))
        # 중요한 feature 순서 출력
        print(selector.ranking_)
        for i in range(X_train.shape[1]):
            print('column: %d, Selected %s, Rank: %.3f' % (i, rfecv.support_[i], rfecv.ranking_[i]))

        print("Optimal number of features : {0} for model: {1}".format(rfecv.n_features_, model))
        # regression
        scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
        # classification
        scoring = ['f1', 'accuracy', 'recall']

        print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        final_col = using_col + [target]

        new_train = train[final_col]
        new_test = test[final_col]
        if train.split('.')[1] == "csv":
            new_train.to_csv(f'./data/preprocess_data/{today}/{train}.csv', index=False)

        elif train.split('.')[1] == "pkl":
            new_train.to_pickle(f'./data/preprocess_data/{today}/{train}')

        if test.split('.')[1] == "csv":
            new_test.to_csv(f'./data/preprocess_data/{today}/{test}.csv', index=False)

        elif test.split('.')[1] == "pkl":
            new_test.to_pickle(f'./data/preprocess_data/{today}/{test}')
        # 현재시각 - 시작시간 = 실행 시간
        print("time :", time.time() - start)  

    # def permutation_feature_selection(self, model, X_train, y_train, file_name):
    #     print('****************  Feature Selection permutation importance Started ******************')
    #     start = time.time()
    #     from eli5.sklearn import PermutationImportance
    #     from sklearn.feature_selection import SelectFromModel
    #
    #     perm = PermutationImportance(model, scoring="f1", random_state=42).fit(X_train, y_train)
    #     # .fit(X_train, y_train)
    #     sel = SelectFromModel(perm, threshold=0.05, prefit=True)
    #     # X_trans = sel.transform(X_train)
    #
    #     print(eli5.explain_weights(perm, top=10, feature_names=X_train.columns.tolist()))
    #     print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    #     # print(X_trans)


def main(model_type, train, test, target):
    print(f'{os.getcwd()}/data/raw')
    train_d = 'train.csv'
    test_d = 'test.csv'
    
    target = 'fraud_reported'
    if train.split('.')[1] == "csv":
        train_data = pd.read_csv(f'{os.getcwd()}/data/raw/{today}/{train}')
    
    elif train.split('.')[1] == "pkl":
        train_data = pd.read_pickle(f'{os.getcwd()}/data/raw/{today}/{train}')
    
    if test.split('.')[1] == "csv":
        test_data = pd.read_csv(f'{os.getcwd()}/data/raw/{today}/{test}')
    
    elif test.split('.')[1] == "pkl":
        test_data = pd.read_pickle(f'{os.getcwd()}/data/raw/{today}/{test}')
        

    print("========== Feature Importance ==========")
    data2 = train_data.drop(columns=[target])
    
    X_train, X_valid, y_train, y_valid = train_test_split(data2, train_data[target], test_size=float(0.2),
                                                          random_state=42)
    
    print("==========  Model Training For Feature Importance  ===========")
    print(f'                  <model_type : {model_type} >')
    # XGBoostModel.train()
    xgboost_model = XGBoostModel(X_train, y_train, X_valid, y_valid)
    model = xgboost_model.train(Optuna=None, RandomSearch=None, trials=1, params=None, gpu=False, gpu_id=0, cpu_cnt=1, save_model=False, file_name='XGBoostClassifier', random_state=42, cv=5)
    fs_class = feature_selections(train, X_train, y_train, model_type)
    # eda(model, data, target, model_type)
    # rfe -> 100-> 60
    importances_df = fs_class.feature_importance(model, X_train, model_type)
    # 60 -> rest
    fs_class.feature_selection_rfe(train, test, importances_df, model, X_train, y_train, file_name, target)
    fs_class.feature_selection_rfecv(train, test, model, X_train, y_train, file_name, target)
    # fs_class.permutation_feature_selection(train, test, model, X_train, y_train, file_name)


def parse_args():
    parser = argparse.ArgumentParser("parser")
    parser.add_argument('-model_type', '--model_type', type=str, help='EDA model type', required=False,
                        default='XGB_clf')

    parser.add_argument('-train', '--train', type=str, help='Data set (defult: train.csv)', default='train.csv',
                        required=False)
    parser.add_argument('-test', '--test', type=str, help='Data set (defult: test.csv)', default='test.csv',
                        required=False)
    parser.add_argument('-target', '--target', type=str, help='Target column (defult: target)', default='target',
                        required=False)


    args = parser.parse_args()
    return args.model_type, args.train, args.test, args.target
