import re, os
import subprocess
import argparse


def main(model_type, data, target, cpu_cnt, trials, test_size, gpu, gpu_id, memo, n_feature):
    # try:
    #     print("============================= Start get_dataset.py =============================")
    #     subprocess.run(['python', 'get_dataset.py', '-data', data])
    # except Exception as e:
    #     print("Get data ERROR!!!")
    #     print("ERROR Code: ", e)
    # else:
    #     print("============================= Finish get_dataset.py =============================")



    try:
        print("============================= Start vanilla_cv.py =============================")
        purpose = model_type.split('_')[1]
        subprocess.run(['python', 'vanilla_cv.py', '-data', data, '-purpose', purpose, '-target', target, '-cpu_cnt', cpu_cnt, '-test_size', test_size])
    except Exception as e:
        print("Vanilla cv ERROR!!!")
        print("ERROR Code: ", e)
    else:
        print("============================= Finish vanilla_cv.py =============================")

    

    try:
        print("============================= Start feature.py =============================")
        subprocess.run(['python', 'feature.py', '-model_type', model_type, '-data', data, '-target', target, '-gpu', gpu, '-gpu_id', gpu_id,\
                        '-cpu_cnt', cpu_cnt, '-test_size', test_size, '-feature', n_feature])
    except Exception as e:
        print("Preprocess ERROR!!!")
        print("ERROR Code: ", e)
    else:
        print("============================= Finish feature.py =============================")



    try:
        print("============================= Start optuna_best_parameter.py =============================")
        subprocess.run(['python', 'optuna_best_parameter.py', '-data', data, '-model_type', model_type, '-cpu_cnt', cpu_cnt, \
                            '-trials', trials, '-target', target, '-test_size', test_size, '-gpu', gpu, '-gpu_id', gpu_id])
    except Exception as e:
        print("Optuna ERROR!!!")
        print("ERROR Code: ", e)
    else:
        print("============================= Finish optuna_best_parameter.py =============================")



    try:
        print("============================= Start train.py =============================")
        subprocess.run(['python', 'train.py', '-data', data, '-target', target, '-model_type', model_type, \
                            '-memo', memo, '-test_size', test_size, '-gpu', gpu, '-gpu_id', gpu_id, '-cpu_cnt', cpu_cnt])
    except Exception as e:
        print("Model test ERROR!!!")
        print("ERROR Code: ", e)
    else:
        print("============================= Finish train.py =============================")


def parse_args():
    parser = argparse.ArgumentParser(description="Run total AutoML script")
    parser.add_argument('-model_type', '--model_type', type=str, help='Select Model Type (defult: XGB_clf)', default = 'XGB_clf', required=False)
    parser.add_argument('-data', '--data', type=str, help='Data set (defult: train.csv)', default = 'train.csv', required=False)

    parser.add_argument('-target', '--target', type=str, help='Target column (defult: target)', default = 'target', required=False)

    parser.add_argument('-cpu_cnt', '--cpu_cnt', type=str, help='The number of cpu count for train model (defult: 1)', default = '1', required=False)
    parser.add_argument('-trials', '--trials', type=str, help='Trials for optuna (default: 1)', default = '1', required=False)
    parser.add_argument('-test_size', '--test_size', type=str, help='Test size for train/test split (default: 0.2)', default = '0.2', required=False)
    parser.add_argument('-gpu', '--gpu', type=str, help='GPU setting (default: False)', default = 'False', required=False)
    parser.add_argument('-gpu_id', '--gpu_id', type=str, help='GPU id (default: 0)', default = '0', required=False)

    parser.add_argument('-feature', '--feature', type=str, help='Top N features for feature selection (default: 20)', default = '20', required=False)

    parser.add_argument('-memo', '--memo', type=str, help='Memo (default: NaN)', default = 'NaN', required=False)

    args = parser.parse_args()

    return args.model_type, args.data, args.target, args.cpu_cnt, args.trials, args.test_size, args.gpu, args.gpu_id, args.memo, args.feature

if __name__ == '__main__':
    main(*parse_args())