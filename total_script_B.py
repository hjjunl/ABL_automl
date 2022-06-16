import re, os
import subprocess
import argparse


def main(train, test, target, cv, sampling, sampling_ratio, sampling_k_neighbors, sampling_m_neighbors, smotesvm_stepsize, categorical_features_index, \
        model_type, large_data, valid_size, model_save, trials, gpu, gpu_id, cpu_cnt):
    # try:
    #     print("============================= Start get_dataset.py =============================")
    #     subprocess.run(['python', 'get_dataset.py', '-data', data])
    # except Exception as e:
    #     print("Get data ERROR!!!")
    #     print("ERROR Code: ", e)
    # else:
    #     print("============================= Finish get_dataset.py =============================")
    

    # try:
    #     print("============================= Start Data Balancing (binary) =============================")
    #     subprocess.run(['python', 'data_balancing_binary.py', '-train', train, '-target', target, '-sampling', sampling, \
    #                         '-sampling_ratio', sampling_ratio, '-sampling_k_neighbors', sampling_k_neighbors, '-sampling_m_neighbors', sampling_m_neighbors, \
    #                             '-smotesvm_stepsize', smotesvm_stepsize, '-categorical_features_index', categorical_features_index, '-cpu_cnt', cpu_cnt])
    # except Exception as e:
    #     print("Data Balancing ERROR!!!")
    #     print("ERROR Code: ", e)
    # else:
    #     print("============================= Finish Data Balancing =============================")


    try:
        print("============================= Start Data Balancing (multi) =============================")
        subprocess.run(['python', 'data_balancing_multi.py', '-train', train, '-target', target, '-sampling', sampling, \
                            '-sampling_ratio', sampling_ratio, '-sampling_k_neighbors', sampling_k_neighbors, '-sampling_m_neighbors', sampling_m_neighbors, \
                                '-smotesvm_stepsize', smotesvm_stepsize, '-categorical_features_index', categorical_features_index, '-cpu_cnt', cpu_cnt])
    except Exception as e:
        print("Data Balancing ERROR!!!")
        print("ERROR Code: ", e)
    else:
        print("============================= Finish Data Balancing =============================")



    try:
        print("============================= Start Parameter tunning & Model Selection =============================")
        subprocess.run(['python', 'model_selection_B.py', '-train', train, '-target', target, '-valid_size', valid_size,'-model_save', model_save,'-cv', cv, '-gpu', gpu, '-gpu_id', gpu_id,\
                        '-cpu_cnt', cpu_cnt])
    except Exception as e:
        print("Model Selection ERROR!!!")
        print("ERROR Code: ", e)
    else:
        print("============================= Finish Parameter tunning & Model Selection =============================")

    try:
        print("============================= Start XAI =============================")
        subprocess.run(['python', 'xai_results.py', '-test', test, '-target', target, '-model', model_type, \
                            '-large_data', large_data, '-gpu', gpu, '-gpu_id', gpu_id, '-cpu_cnt', cpu_cnt])
    except Exception as e:
        print("XAI ERROR!!!")
        print("ERROR Code: ", e)
    else:
        print("============================= Finish XAI =============================")


def parse_args():
    parser = argparse.ArgumentParser(description="Run total AutoML script")
    parser.add_argument('-train', '--train', type=str, help='Data set (defult: train.csv)', default = 'train.csv', required=False)
    parser.add_argument('-test', '--test', type=str, help='Data set (defult: test.csv)', default = 'test.csv', required=False)
    parser.add_argument('-target', '--target', type=str, help='Target column (defult: target)', default = 'target', required=False)
    
    parser.add_argument('-cv', '--cv', type=str, help='The number of cross Validation', default = '5', required=False)

    parser.add_argument('-sampling', "--sampling", type=str, help='Sampling method (Options: RandomUnderSample(default), RandomOverSample, Smote, SmoteNC, SmoteSVM, ADASYN)', default = 'RandomUnderSample', required=False)
    parser.add_argument('-sampling_ratio', "--sampling_ratio", type=str, help='Sampling ratio', default = '0.5', required=False)
    parser.add_argument('-sampling_k_neighbors', "--sampling_k_neighbors", type=str, help='K neighbors', default = '5', required=False)
    parser.add_argument('-sampling_m_neighbors', "--sampling_m_neighbors", type=str, help='M neighbors', default = '10', required=False)
    parser.add_argument('-smotesvm_stepsize', "--smotesvm_stepsize", type=str, help='smotesvm stepsize', default = '0.5', required=False)
    parser.add_argument('-categorical_features_index', "--categorical_features_index", type=str, help='Categorical features index', default = 'None', required=False)


    parser.add_argument('-model_type', '--model_type', type=str, help='Select Model Type (options: XGB(default), LGBM, CBC)', default = 'XGB', required=False)

    parser.add_argument('-large_data', '--large_data', type=str, help='Large data or not (Options: True, False(default))', default = 'False', required=False)
    parser.add_argument('-valid_size', '--valid_size', type=str, help='Valid size for train/valid split (default: 0.2)', default = '0.2', required=False)
    parser.add_argument('-model_save', '--model_save', type=str, help='Model save or not (Options: True(default), False)', default = 'True', required=False)
    parser.add_argument('-trials', '--trials', type=str, help='Trials for optuna (default: 1)', default = '5', required=False)

    parser.add_argument('-gpu', '--gpu', type=str, help='GPU setting (default: False)', default = 'False', required=False)
    parser.add_argument('-gpu_id', '--gpu_id', type=str, help='GPU id (default: 0)', default = '0', required=False)
    parser.add_argument('-cpu_cnt', '--cpu_cnt', type=str, help='The number of cpu count for train model (defult: 1)', default = '1', required=False)


    args = parser.parse_args()

    return args.train, args.test, args.target, args.cv, args.sampling, args.sampling_ratio, \
        args.sampling_k_neighbors, args.sampling_m_neighbors, args.smotesvm_stepsize, args.categorical_features_index,\
        args.model_type, args.large_data, args.valid_size, args.model_save, args.trials, args.gpu, args.gpu_id, args.cpu_cnt

if __name__ == '__main__':
    main(*parse_args())