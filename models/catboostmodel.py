import pandas as pd
import numpy as np
import os
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss
import json
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
from utils import log
import pickle
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime

logger = log.get_logger(__name__)

import pathlib
import os

today = datetime.today().strftime("%Y%m%d")

model_path = f'./outputs/{today}/catboost/models'
param_path = f'./outputs/{today}/catboost/params'
visual_path = f'./outputs/{today}/catboost/visualization'

_OUTPUT_DIR = pathlib.Path(os.path.join('./outputs'))
_DATE_DIR = pathlib.Path(os.path.join(f'./outputs/{today}'))
_CBC_DIR = pathlib.Path(os.path.join(f'./outputs/{today}/catboost'))
_MODEL_DIR = pathlib.Path(os.path.join(model_path))
_PARAM_DIR = pathlib.Path(os.path.join(param_path))
_VISUAL_DIR = pathlib.Path(os.path.join(visual_path))

if not _OUTPUT_DIR.exists():
    _OUTPUT_DIR.mkdir()

if not _DATE_DIR.exists():
    _DATE_DIR.mkdir()

if not _CBC_DIR.exists():
    _CBC_DIR.mkdir()

if not _MODEL_DIR.exists():
    _MODEL_DIR.mkdir()

if not _PARAM_DIR.exists():
    _PARAM_DIR.mkdir()

if not _VISUAL_DIR.exists():
    _VISUAL_DIR.mkdir()

class CatBoostModel:
    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        The function initalise data for training
        Args:
            X_train (pd.DataFrame): Training data for modelling
            y_train (pd.Series): Target variable  for training
            X_test (pd.DataFrame): Testing data for modelling
            y_test (pd.Series): Target variable for testing
            class_weights (int or None): The ratio of positive label and total count.
                                         This ratio can be used as balancing parameter in the model.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train(self, Optuna=None, RandomSearch=None, trials=1, params=None, gpu=False, gpu_id=0, cpu_cnt=1, save_model=False, file_name='CatBoostClassifier', random_state=42, cv=5):
        """
        The function trains CatBoost model with the specified parameter and saves the model in pickle format in the path
        Args:
            Optuna (Bool or None): If True, then performs Optuna on the parameters and trains the model
                                        on best parameter
            RandomSearch (Bool or None): If True, then performs RandomSearch on the parameters and trains the model
                                        on best parameter
            trials (int): Parameter search trials
            params (path or list):If Optuna is implied then pass dict of parameters else for single training pass JSON
                         path consisting of training parameters.
            gpu (Bool or None): If True, then trains the model on gpu.
            gpu_id (int): gpu id.
            cpu_cnt (int): cpu count
            save_model (Bool): If True, the save the model after training is completed.
            file_name (str): file name for the model
            path (str): path to save the model after training
            random_state (int): seed for reproducbility
        Returns:
            model (): Trained CatBoost model
        """

        if not Optuna:
            if not RandomSearch:
                if not gpu:
                    cbc_model = CatBoostClassifier(n_estimators = 5000, thread_count = int(cpu_cnt), random_state = random_state)
                    self.model = cbc_model.fit(self.X_train, self.y_train, eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)], early_stopping_rounds = 100, verbose = 500)
                else:
                    cbc_model = CatBoostClassifier(n_estimators = 5000, bootstrap_type='Poisson', random_state = random_state, task_type = 'GPU', devices = str(gpu_id))
                    self.model = cbc_model.fit(self.X_train, self.y_train, eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)], early_stopping_rounds = 100, verbose = 500)

                if save_model:
                    file_name = str(file_name) + '.pkl'
                    self.save_model(filename = file_name, path = model_path)

                return self.model

            else:
                params_rs = {
                    'learning_rate': [0.003,0.03,0.1],
                    'depth': [3,5,7],
                    'min_child_samples': [5,50,100],
                    'max_bin': [2,50,100],
                }

                if not gpu:
                    cbc_model = CatBoostClassifier(n_estimators = 5000, thread_count = int(cpu_cnt), random_state = random_state)
                    search = RandomizedSearchCV(cbc_model, params_rs, n_iter=trials, scoring='f1_macro', n_jobs=cpu_cnt, cv=cv, random_state=42)
                    rs_result = search.fit(self.X_train, self.y_train)
                    self.model = rs_result
                
                else:
                    cbc_model = CatBoostClassifier(n_estimators = 5000, bootstrap_type='Poisson', task_type = 'GPU', devices = str(gpu_id), random_state = random_state)
                    search = RandomizedSearchCV(cbc_model, params_rs, n_iter=trials, scoring='f1_macro', n_jobs=cpu_cnt, cv=cv, random_state=42)
                    rs_result = search.fit(self.X_train, self.y_train)
                    self.model = rs_result

                with open(param_path+'/CBC_RandomSearch.json', 'w') as fp:
                    json.dump(rs_result.best_params_, fp)

                print('RandomSearch Best Hyperparameters: %s' % rs_result.best_params_)
                
                if save_model:
                    file_name = str(file_name) + '.pkl'
                    self.save_model(filename = file_name, path = model_path)

                return self.model

        else:
            def objective(X_train, X_test, y_train, y_test, gpu, gpu_id, cpu_cnt, trial : Trial) -> float:
                if len(np.unique(y_train)) == 2:
                    ### Binary classification
                    params_cbc = {
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
                        model = CatBoostClassifier(** params_cbc, bootstrap_type='Poisson', task_type = 'GPU', devices = str(gpu_id))
                        model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)], early_stopping_rounds = 100, verbose = 500)
                    else:
                        model = CatBoostClassifier(** params_cbc, thread_count = int(cpu_cnt))
                        model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)], early_stopping_rounds = 100, verbose = 500)
                
                else:
                    ### Multi classification
                    params_cbc = {
                        'random_state' : 420,
                        'learning_rate' : trial.suggest_float('learning_rate', 0.003, 0.1),
                        'n_estimators' : 5000,
                        # 'eval_metric' : 'Logloss',
                        'depth' : trial.suggest_int('depth', 3, 16),
                        # 'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
                        'min_child_samples' : trial.suggest_int('min_child_samples', 5, 100),
                        'max_bin' : trial.suggest_int('max_bin',2,100)
                        }

                    model = CatBoostClassifier(** params_cbc, thread_count = int(cpu_cnt))
                    model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)], early_stopping_rounds = 100, verbose = 500)

                cbc_pred = model.predict_proba(X_test)

                try: 
                    logloss =log_loss(y_test, cbc_pred[:,1])
                except Exception:
                    logloss = log_loss(y_test, cbc_pred)

                return logloss
                
            sampler = TPESampler()
            study = optuna.create_study(
                study_name = 'parameter_opt',
                direction = 'minimize',
                sampler = sampler
            )
            study.optimize(lambda trial : objective(self.X_train, self.X_test, self.y_train, self.y_test, gpu, gpu_id, cpu_cnt, trial), n_trials = int(trials), n_jobs = int(cpu_cnt))

            with open(param_path+'/CBC_Optuna.json', 'w') as fp:
                json.dump(study.best_trial.params, fp)

            print(f'Best Score : {study.best_value}')
            print(f'Best Trial : {study.best_trial.params}')
            print("FINISH Optuna !!!")

            if len(np.unique(self.y_train)) == 2:
                if not gpu:
                    cbc_model = CatBoostClassifier(** study.best_trial.params, thread_count = int(cpu_cnt))
                    self.model = cbc_model.fit(self.X_train, self.y_train, eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)], early_stopping_rounds = 100, verbose = 500)

                    if save_model:
                        file_name = str(file_name) + '.pkl'
                        self.save_model(filename = file_name, path = model_path)

                    return self.model

                else:
                    cbc_model = CatBoostClassifier(** study.best_trial.params, bootstrap_type='Poisson', task_type = 'GPU', devices = str(gpu_id))
                    self.model = cbc_model.fit(self.X_train, self.y_train, eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)], early_stopping_rounds = 100, verbose = 500)
                    
                    if save_model:
                        file_name = str(file_name) + '.pkl'
                        self.save_model(filename = file_name, path = model_path)

                    return self.model

            else:
                cbc_model = CatBoostClassifier(** study.best_trial.params, thread_count = int(cpu_cnt))
                self.model = cbc_model.fit(self.X_train, self.y_train, eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)], early_stopping_rounds = 100, verbose = 500)

                if save_model:
                    file_name = str(file_name) + '.pkl'
                    self.save_model(filename = file_name, path = model_path)

                return self.model


    def predict(self):
        """
        Predict function
        Returns:
            self.model.predict(dtest): The probability for true positive
        """
        return self.model.predict(self.X_test)


    def save_model(self, filename, path = model_path):
        """
        The function saves the trained model to the location in the pickle format
        Args:
            filename (str): The file name to save the model in the pickle format
            path (str): The path to save the model
        """
        logger.info('Saving model as pickle file to the path: {}'.format(path))
        model_path = os.path.join(path, filename)
        pickle.dump(self.model, open(model_path, 'wb'))
        logger.info('Model saved')


    def load_model(self, filename, path = model_path):
        """
        The function loads model from location passed. The function only considered  model with ".pkl" extension
        Args:
            filename (str): the model file name to load
            path (str): the location where model is kept
        """
        logger.info('Loading Model from the location'.format(path))
        model_path = os.path.join(path, filename)
        try:
            self.model = pickle.load(open(model_path, 'rb'))
            logger.info('Model loaded successfully')
        except Exception as error:
            logger.info('Unable to load model from the path: {}'.format(model_path) + repr(error))


    def feature_importance(self, path = visual_path):
        """
        The function returns most important features/variables that is used in building CatBoost model
        The importance is determined by gain scores
        Returns:
            data (pd.DataFrame): The data frame consists of feature name and gain scores
        """
        feature_important = self.model.get_score(importance_type='gain')
        keys = list(feature_important.keys())
        values = list(feature_important.values())
        data = pd.DataFrame(data=values, index=keys, columns=['score']).sort_values(by='score', ascending=False)
        data = data.reset_index()
        data.to_csv(os.path.join(path, 'cbc_feature_importance.csv'), index=False)
        return data


    def shap_explanation(self, path = visual_path, large_data: bool = False):
        if not large_data:
            data = self.X_train
        else:
            data = self.X_train.sample(frac=0.1)
        explainer = shap.TreeExplainer(self.model, data=data, feature_perturbation='interventional',
                                       model_output="probability")
        shap_values = explainer(data)
        create_and_save_shap_plots(shap_values=shap_values, dirName=path)


def create_and_save_shap_plots(shap_values, dirName):
    name = dirName.split("\\")[-1]
    logger.info(dirName)
    save_a_plot(shap_plot(plot=shap.plots.bar, shap_values=shap_values, title=name + "_shap_bar_plot", max_display=15),
                full_dirName=os.path.join(dirName, 'shap_bar_plot.png'))
    save_a_plot(shap_plot(plot=shap.plots.beeswarm, shap_values=shap_values, title=name + "_shap_beeswarm_plot"),
                full_dirName=os.path.join(dirName, 'shap_beeswarm_plot.png'))

    mean_abs_shap_values = np.abs(shap_values.values).mean(0)
    position_sorted_abs_values = np.argsort(mean_abs_shap_values).tolist()[::-1]
    top_x_features = [(shap_values.feature_names[i], mean_abs_shap_values[i]) for i in
                      position_sorted_abs_values[:10]]
    logger.info(top_x_features)


def shap_plot(plot, shap_values, title, **kwargs):
    plot(shap_values, show=False, **kwargs)
    plt.title(title)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def save_a_plot(plot, full_dirName):
    plot.savefig(full_dirName)
    logger.info("Saved {}".format(full_dirName.split("\\")[-1]))
    plt.close()
