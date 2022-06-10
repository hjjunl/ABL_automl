# ABL_automl

## AutoML Process
### 1. 1차 filtering
  - PCA
  - 논문
  - 현업 종사자
  - default or Random search model을 통한 변수 선택
  - 간단하게
### 2. Feature Selection
  - RFE
  - LASSO
  - Permutation Importance
  - RFECV*
### 3.  Data Balancing
  - Oversampling: SMOTE, ADASYN...etc
  - Undersampling
### 4. Hyper Parameter Tuning
  - Optuna*
  - Random Search
### 5. Model Selection
  - Choose the best champion model
  - XGBoost
  - LightGBM
### 6. Deploy + XAI
  - Use the best model as the main model
  - Send model through the RestAPI
  - XAI: Explainable Dashboard, Arena or SHAP
![image](https://user-images.githubusercontent.com/50603209/172993514-f5e9086c-0f43-4fa1-bbe2-139e5eba5689.png)





Sampling
https://github.com/Balacoumarane/casestudy/blob/31839566f1daecf7faab0f5aa80e512020fd9f81/ChurnModule/utils/sampling.py
