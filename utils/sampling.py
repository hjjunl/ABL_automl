from collections import Counter

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

from .log import get_logger

logger = get_logger(__name__)


class SamplingMethods:
    def __init__(self, X, y):
        """
        A custom class to perform various sampling methods
        Args:
            X (pd.DataFrame): Predictor variable
            y (pd.Series ot index): target variable
        """
        self.X = X
        self.y = y

    def sampling_technique(self, sampling_method=None, random_state=42, sampling_strategy='auto',
                           sampling_k_neighbors=5, sampling_m_neighbors=10, smotesvm_stepsize=0.5,
                           categorical_features_index=None,
                           cpu_cnt=1):
        """
        The function performs various sampling techniques
        \\ Please refer for link for detailed documentation: https://imbalanced-learn.readthedocs.io/en/stable/api.html
        The sampling techniques implemented are:
        1. Random under sampling
        2. SMOTE-NC
        3. ADASYN
        4. SMOTE-UnderSample
        5. SVMSMOTE
        6. SMOTE-OverSample
        Only method 1 and method 2 are applicable for categorical data
        Args:
            sampling_method (str or None): Enter sampling method. Currently implemented ("RandomUnderSample",
                                            "SmoteUnderSample","SmoteNC", "SmoteSVM", "ADASYN")
            random_state (int): random seed for reproducibility
            sampling_strategy (str): 'majority': resample only the majority class;
                                     'not minority': resample all classes but the minority class;
                                     'not majority': resample all classes but the majority class;
                                     'all': resample all classes;
                                     'auto': equivalent to 'not minority'.

            sampling_k_neighbors (int): number of nearest neighbours to used to construct synthetic samples.
            categorical_features_index (int or list): index for categorical features
            cpu_cnt (int): cpu count
        Returns:
            X_res (pd.DataFrame): sampled predictor data
            y_res (series) sampled target
        """
        if sampling_method is None or sampling_method not in ["RandomUnderSample", "SmoteUnderSample", "SmoteOverSample",
                                                              "SmoteNC", "SmoteSVM", "ADASYN"]:
            raise NotImplementedError("Please enter sampling methods")

        elif sampling_method == "RandomUnderSample":
            X_res, y_res = self.random_under_sample(random_state, sampling_strategy)

        elif sampling_method == "SmoteUnderSample":
            X_res, y_res = self.smote_under_sample(random_state, sampling_strategy, sampling_k_neighbors, cpu_cnt)

        elif sampling_method == "SmoteOverSample":
            X_res, y_res = self.smote_over_sample(random_state, sampling_strategy, sampling_k_neighbors, cpu_cnt)

        elif sampling_method == "SmoteSVM":
            X_res, y_res = self.smote_svm(random_state, sampling_strategy, sampling_k_neighbors,
                                          sampling_m_neighbors, smotesvm_stepsize, cpu_cnt)
        elif sampling_method == "SmoteNC":
            X_res, y_res = self.smote_nc(categorical_features_index, random_state, sampling_strategy, sampling_k_neighbors, cpu_cnt)

        elif sampling_method == "ADASYN":
            X_res, y_res = self.adasyn(random_state, sampling_strategy, sampling_k_neighbors, cpu_cnt)

        return X_res, y_res

    def random_under_sample(self, random_state, sampling_strategy):
        """
        The function performs random under sampling and returns sampled data
        Args:
            random_state (int): random seed for reproducibility
            sampling_strategy (float): The number of samples in the different classes will be equalized.
        Returns:
            X_res (pd.DataFrame): sampled predictor data
            y_res (series) sampled target
        """
        rus = RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy)
        X_res, y_res = rus.fit_resample(self.X, self.y)
        logger.info('Resampled dataset using random undersampling shape %s' % Counter(y_res))
        return X_res, y_res

    def smote_under_sample(self, random_state, sampling_strategy, sampling_k_neighbors, cpu_cnt):
        """
        The function performs random under sampling and returns sampled data
        Args:
            random_state (int): random seed for reproducibility
            sampling_strategy (float): The number of samples in the different classes will be equalized.
            sampling_k_neighbors (int): If int, number of nearest neighbours to used to construct synthetic samples.
            cpu_cnt (int): The number of cpu counts.
        Returns:
            X_res (pd.DataFrame): sampled predictor data
            y_res (series) sampled target
        """
        over = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=sampling_k_neighbors, random_state=random_state, n_jobs=cpu_cnt)
        under = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        X_res, y_res = pipeline.fit_resample(self.X, self.y)
        logger.info('Resampled dataset using SMOTE undersampling shape %s' % Counter(y_res))
        return X_res, y_res

    def smote_over_sample(self, random_state, sampling_strategy, sampling_k_neighbors, cpu_cnt):
        """
        The function performs random under sampling and returns sampled data
        Args:
            random_state (int): random seed for reproducibility
            sampling_strategy (float): The number of samples in the different classes will be equalized.
            sampling_k_neighbors (int): If int, number of nearest neighbours to used to construct synthetic samples.
            cpu_cnt (int): The number of cpu counts.
        Returns:
            X_res (pd.DataFrame): sampled predictor data
            y_res (series) sampled target
        """
        over = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=sampling_k_neighbors, random_state=random_state, n_jobs=cpu_cnt)
        X_res, y_res = over.fit_resample(self.X, self.y)
        logger.info('Resampled dataset using SMOTE oversampling shape %s' % Counter(y_res))
        return X_res, y_res

    def smote_nc(self, categorical_features_index, random_state, sampling_strategy, sampling_k_neighbors, cpu_cnt):
        """
        The function performs random under sampling and returns sampled data
        Args:
            random_state (int): random seed for reproducibility
            categorical_features_index (int or list): array of indices specifying the categorical features
            sampling_strategy (float): The number of samples in the different classes will be equalized.
            sampling_k_neighbors (int): If int, number of nearest neighbours to used to construct synthetic samples.
            cpu_cnt (int): The number of cpu counts.
        Returns:
            X_res (pd.DataFrame): sampled predictor data
            y_res (series) sampled target
        """
        smote_oversample = SMOTENC(random_state=random_state,
                                         categorical_features_index=categorical_features_index,
                                         k_neighbors=sampling_k_neighbors,
                                         sampling_strategy=sampling_strategy,
                                         n_jobs=cpu_cnt)
        X_res, y_res = smote_oversample.fit_resample(self.X, self.y)
        logger.info('Resampled dataset using SMOTE NC shape %s' % Counter(y_res))
        return X_res, y_res

    def smote_svm(self, random_state, sampling_strategy, sampling_k_neighbors,
                                          sampling_m_neighbors, smotesvm_stepsize, cpu_cnt):
        """
        The function performs random under sampling and returns sampled data
        Args:
            random_state (int): random seed for reproducibility
            sampling_strategy (float): The number of samples in the different classes will be equalized.
            sampling_k_neighbors (int): number of nearest neighbours to used to construct synthetic samples.
            sampling_m_neighbors (int): number of nearest neighbours to use to determine if a minority sample is in
                                        danger
            smotesvm_stepsize (float): Step size when extrapolating
        Returns:
            X_res (pd.DataFrame): sampled predictor data
            y_res (series) sampled target
        """
        oversample = SVMSMOTE(sampling_strategy=sampling_strategy,
                              random_state=random_state,
                              k_neighbors=sampling_k_neighbors,
                              m_neighbors=sampling_m_neighbors,
                              out_step=smotesvm_stepsize,
                              n_jobs=cpu_cnt)

        X_res, y_res = oversample.fit_resample(self.X, self.y)
        logger.info('Resampled dataset using SMOTE SVM shape %s' % Counter(y_res))
        return X_res, y_res

    def adasyn(self, random_state, sampling_strategy, sampling_k_neighbors, cpu_cnt):
        """
        The function performs random under sampling and returns sampled data
        Args:
            random_state (int): random seed for reproducibility
            adasyn_sampling_ratio (float):  It corresponds to the desired ratio of the number of samples in the minority
                                        class over the number of samples in the majority class after resampling
            adasyn_k_neighbors (int): number of nearest neighbours to used to construct synthetic samples.
        Returns:
            X_res (pd.DataFrame): sampled predictor data
            y_res (series) sampled target
        """
        ada = ADASYN(sampling_strategy=sampling_strategy,
                     random_state=random_state,
                     n_neighbors=sampling_k_neighbors,
                     n_jobs=cpu_cnt)
        X_res, y_res = ada.fit_resample(self.X, self.y)
        logger.info('Resampled dataset using ADASYN shape %s' % Counter(y_res))
        return X_res, y_res