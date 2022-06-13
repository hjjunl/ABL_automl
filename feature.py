from pandas_profiling import ProfileReport
import shap
import matplotlib.pyplot as plt
import pandas as pd
from model.train_model import *
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import argparse
import warnings

# from datetime import datetime


# date = datetime.today()
# dir_year = date.strftime("%Y-%m-%d").split('-')[0]
# dir_month = date.strftime("%Y-%m-%d").split('-')[1]
# day = date.strftime("%Y-%m-%d").split('-')[2]
warnings.filterwarnings(action='ignore')

def eda(model, data, target, model_type):
    print("========== Pandas Profiling Report ==========")
    profile = ProfileReport(data, title="Pandas Profiling Report")
    profile.to_file(f'feature_output/{model_type}_EDA.html')


    print("========== SHAP ==========")
    shap.initjs()
    X_train = data.drop(columns=[target])
    X_sampled = X_train.sample(1000, random_state = 42)


    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sampled)

    plt.figure()

    plt.title(f'{model_type} Top SHAP values')

    if model_type == 'LGB_clf':
        shap.summary_plot(shap_values[1], X_sampled, plot_type="bar",show=False, plot_size=[8,6])
    else:
        shap.summary_plot(shap_values, X_sampled, plot_type="bar",show=False, plot_size=[8,6])

    plt.savefig(f'feature_output/{model_type}_SHAP.png',dpi = 50,bbox_inches = 'tight')

    

def feature_importance(model, X_train, model_type):
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    proportion = (importances / sum(importances)) 

    importances = pd.DataFrame({'columns' : X_train.columns[sorted_idx]
                    ,'importances' : importances[sorted_idx]
                    , 'proportion' : proportion[sorted_idx]
                    
                    })


    plt.figure(figsize=(15,6))
    plt.title(f'{model_type} Feature importances')
    plt.barh(importances['columns'][:30], importances['importances'][:30])
    plt.savefig(f'feature_output/{model_type}_importance.png') 

    importances.to_pickle(f'feature_output/{model_type}_feature_importances.pkl')

    return importances


def feature_selection_rfe(data, num_feature, importances_df, model, X_train, y_train, file_name, target):
    print('****************  Feature Selection Started ******************')
    print(f'****************  Number of Feature : {num_feature}  ******************')
    
    selector = RFE(model, n_features_to_select = int(num_feature), step = 7, verbose = 500)
    selector = selector.fit(X_train, y_train)
    temp = pd.DataFrame(selector.support_, columns = ['feature_bool'])
    importances = pd.concat([importances_df,temp], axis = 1)
    final_table = importances[importances['feature_bool'] == True]
    final_table = final_table.reset_index(drop = True).drop(columns = 'feature_bool')
    features = list(final_table['columns'])
    
    newdata = data[features]
    newdata[target] = data[target]

    if file_name.split('.')[1] == "csv":
        newdata.to_csv(f'./preprocess_data/{file_name}', index = False)

    elif file_name.split('.')[1] == "pkl":
        newdata.to_pickle(f'./preprocess_data/{file_name}')


def main(model_type, file_name, target, gpu, gpu_id, cpu_cnt, test_size, feature):
    if file_name.split('.')[1] == "csv":
        data = pd.read_csv(f'{os.getcwd()}/origin_data/{file_name}')

    elif file_name.split('.')[1] == "pkl":
        data = pd.read_pickle(f'{os.getcwd()}/origin_data/{file_name}')
        
    print("========== Feature Importance ==========")
    data2 = data.drop(columns = [target])

    X_train, X_valid, y_train, y_valid = train_test_split(data2, data[target], test_size = float(test_size), random_state = 42)

    print("==========  Model Training For Feature Importance  ===========")
    print(f'                  <model_type : {model_type} >')


    
    model = model_train(X_train, y_train, X_valid, y_valid, model_type, gpu=bool(gpu), gpu_id=int(gpu_id), cpu_cnt=int(cpu_cnt))

    eda(model, data, target, model_type)
    importances_df = feature_importance(model, X_train, model_type)
    feature_selection_rfe(data, feature, importances_df, model, X_train, y_train, file_name, target)



def parse_args():
    parser = argparse.ArgumentParser("parser")
    parser.add_argument('-model_type', '--model_type', type=str, help='EDA model type', required=False, default = 'XGB_clf')
    parser.add_argument('-data', '--data', type=str, help='EDA data', required=False, default = 'train.csv')
    parser.add_argument('-target', '--target', type=str, help='Target', required=False, default = 'target') # 타겟 (이름 직접 지정해줘야함)
    parser.add_argument('-gpu', '--gpu', type=str, help='GPU setting (default: False)', default = 'False', required=False)
    parser.add_argument('-gpu_id', '--gpu_id', type=str, help='GPU id (default: 0)', default = '0', required=False)
    parser.add_argument('-cpu_cnt', '--cpu_cnt', type=str, help='The number of cpu count for train model (defult: 1)', default = '1', required=False)

    parser.add_argument('-test_size', '--test_size', type=str, help='Test size for train/test split (default: 0.2)', default = '0.2', required=False)

    parser.add_argument('-feature', '--feature', type=str, help='Top N features (default: 20)', default = '20', required=False)

    args = parser.parse_args()
    return args.model_type, args.data, args.target, args.gpu, args.gpu_id, args.cpu_cnt, args.test_size, args.feature

if __name__ == '__main__':
    main(*parse_args())