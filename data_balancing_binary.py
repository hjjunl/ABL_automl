from lib2to3.pgen2.token import SLASHEQUAL
from utils.sampling import SamplingMethods

import warnings
import os

import pandas as pd
import argparse
import pickle

warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("parser")
    parser.add_argument('-train', dest ='train', default='train.csv') # 데이터명
    parser.add_argument('-target', dest= "target", default = 'target') # 타겟 (이름 직접 지정해줘야함)

    parser.add_argument('-sampling', dest= "sampling", default = 'RandomUnderSample')
    parser.add_argument('-sampling_ratio', dest= "sampling_ratio", default = '0.5')
    parser.add_argument('-sampling_k_neighbors', dest= "sampling_k_neighbors", default = '5')
    parser.add_argument('-sampling_m_neighbors', dest= "sampling_m_neighbors", default = '10')
    parser.add_argument('-smotesvm_stepsize', dest= "smotesvm_stepsize", default = '0.5')
    parser.add_argument('-categorical_features_index', dest= "categorical_features_index", default = 'None')

    parser.add_argument('-cpu_cnt', dest = 'cpu_cnt', default = '1')

    args = parser.parse_args()
    train = args.train
    target_name = args.target

    sampling_method = args.sampling
    sampling_ratio = args.sampling_ratio
    sampling_k_neighbors = args.sampling_k_neighbors
    sampling_m_neighbors = args.sampling_m_neighbors
    smotesvm_stepsize = args.smotesvm_stepsize
    categorical_features_index = args.categorical_features_index

    if categorical_features_index == 'None':
        categorical_features_index = None

    cpu_cnt = args.cpu_cnt

    #### data load ####
    print(f'***************  Load dataset ******************')
    if train.split('.')[1] == "csv":
        data = pd.read_csv(f'{os.getcwd()}/data/raw/{train}')

    elif train.split('.')[1] == "pkl":
        data = pd.read_pickle(f'{os.getcwd()}/data/raw/{train}')

    #### Sampling ####
    print(f'***************  Sampling dataset ******************')
    X = data.drop(columns=[target_name])
    y = data[[target_name]].values

    sampling = SamplingMethods(X, y)
    X_sampling, y_sampling = sampling.sampling_technique(sampling_method=sampling_method, sampling_strategy=sampling_ratio, \
                                                    sampling_k_neighbors = int(sampling_k_neighbors), sampling_m_neighbors = int(sampling_m_neighbors), \
                                                    smotesvm_stepsize = float(smotesvm_stepsize), categorical_features_index = categorical_features_index, cpu_cnt = int(cpu_cnt))
    
    y_df = pd.DataFrame({'target':y_sampling})
    df_sampling = pd.concat([X_sampling, y_df], axis=1)
    df_sampling.to_pickle(f'{os.getcwd()}/data/preprocessed/{train}')