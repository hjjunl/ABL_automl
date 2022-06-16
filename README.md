# 목차
[1. 개발 동기](#motivation)

[2. 기본 기능](#basic-function)

[3. 폴더 목록](#folder-list)


[4. 파일 목록](#file-list)


[5. 실행 메뉴얼](#script-manual)


[6. 부록](#etc)


<hr>

### 1. 개발 동기 <span id="motivation"><span>
완벽한 AutoML 은 아니지만 모델 학습에 필요한 기능들 (**전처리 제외**) 을 스크립트형식으로 만든 파일들입니다. 사이트에 가셔서 조금씩만 수정해 주셔도 사용이 가능하기 때문에 모델 개발을 조금이나마 수월하게 하고자 하는 취지로 개발하였습니다.
<hr>
    
### 2. 기본 기능 <span id="basic-function"><span>
* db로부터 data 추출
* Feature Elimination
* Data Balancing
* 모델 선정
* EDA
* 하이퍼파라미터 튜닝
* 모델 학습 및 output 산출

    
<hr>
    
### 3. 폴더 목록 <span id="folder-list"><span>
- query: get_dataset.py 실행 시 필요한 SQL파일이 저장된 폴더
- images: README.md images 폴더
- output
    - {model_name}/models: 모델이 저장된 폴더
    - {model_name}/params: best parameter가 저장된 폴더
    - {model_name}/visualization: 시각화 결과가 저장된 폴더
- log: 로그 파일이 저장된 폴더

<hr>
    
### 4. 실행 스크립트 목록 - Plan A <span id="file-list"><span>
#### model_selection.py
- XGB, LGBM, CatBoost 모델 성능을 비교해 좋은 성능을 내는 모델을 찾기 위한 스크립트
- 하이퍼파라미터의 경우 RandomSearch를 통해 최적의 파라미터를 찾은 결과

#### data_balancing_binary.py / data_balancing_multi.py
- target class별 데이터 분포가 다를 시 데이터를 balancing 해주기 위한 스크립트
- Sampling 옵션: RandomUnderSample, RandomOverSample, Smote, SmoteNC, SmoteSVM, ADASYN
    
#### param_tunning_save_model.py
- Optuna를 통해 최적의 파라미터를 찾은 후 모델을 저장하기 위한 스크립트
    
#### xai_results.py
- XAI 대시보드를 생성하기 위한 스크립트

#### total_script.py
- 위 스크립트들을 한번에 실행시키는 스크립트
    
### 4. 실행 스크립트 목록 - Plan B <span id="file-list"><span>
#### data_balancing_binary.py / data_balancing_multi.py
- target class별 데이터 분포가 다를 시 데이터를 balancing 해주기 위한 스크립트
- Sampling 옵션: RandomUnderSample, RandomOverSample, Smote, SmoteNC, SmoteSVM, ADASYN
    
#### model_selection_B.py
- XGB, LGBM, CatBoost 모델 성능을 비교해 좋은 성능을 내는 모델을 찾기 위한 스크립트
- 하이퍼파라미터의 경우 Optuna를 통해 최적의 파라미터를 찾은 결과
    
#### xai_results.py
- XAI 대시보드를 생성하기 위한 스크립트
    
#### total_script_B.py
- 위 스크립트들을 한번에 실행시키는 스크립트
<hr>
    
    
### 5. 실행 메뉴얼 <span id="script-manual"><span>    
#### 5-1. 실행전 체크리스트
#### Version 확인 필수
- python == 3.8.6
- pip == 20.3.3

**0) 데이터 추출을 위한 준비**
- [ ]  total_script.py / total_script_B.py에서 주석 해제
- [ ]  db_config.env 파일에 목록에 맞는 내용 작성
- [ ]  QUERY의 경우 QUERY문이 별도로 저장되어 있는 SQL파일 경로 작성

    e.g) QUERY = ‘/query/test_query.sql’

**1) 데이터 준비**
- [ ] NaN값 처리되었는지 확인
- [ ] 결측치 처리되었는지 확인
- [ ] object타입이 있는지 확인 (있을 경우 int 혹은 float타입으로 변경)
- [ ] 위 과정을 마친 데이터 형식이 csv혹은 pkl 형식인지 확인 (csv, pkl 형식만 사용가능)
    
**2) 데이터 경로 확인**
- [ ] data/raw 폴더에 원본 데이터가 들어있는지 확인
- [ ] data/preprocessed 폴더에 preprocess 과정이 끝난 데이터가 들어있는지 확인

**3) 기타 사항 확인**
- [ ] total_script.py / total_script_B.py 실행을 위한 target(y값) column명 확인  
- [ ] 사용가능한 gpu 자원 확인 (cli 명령어: nvidia-smi)
    

#### 5-2. requirements 설치
```
pip install -r requirements.txt
```

#### 5-3. total_script.py / total_script_B.py 실행을 위한 arguments 파악
```
python total_script.py --help
```
    
```
python total_script_B.py --help
```
    
##### option 설명
- -train: 학습을 위한 train data (csv, pkl 형식만 가능)
    - default = train.csv
- -test : 평가를 위한 test data(csv, pkl 형식만 가능)
    - default = test.csv
- -target: data의 target column name
    - default = target
- -cv: Cross validation 횟수
    - default = 5

- -sampling: Data balancing을 위한 sampling기법 선택
    - default = RandomUnderSample
    - 사용가능한 옵션: RandomUnderSample, RandomOverSample, Smote, SmoteNC, SmoteSVM, ADASYN
- -sampling_ratio: Data balancing 하기위한 비율
    - default = 0.5
    - Binary classification: float형식으로 비율 설정
    - Multi classificatio: dictionary형식으로 class에 따른 데이터 수 설정 (ex. {0:100, 1:200, 2:150})
- -sampling_k_neighbors: Data balancing을 위한 k neighbor수
    - default = 5
- -sampling_m_neighbors: Data balancing을 위한 m neighbor수
    - default = 10
- -smotesvm_stepsize: Data balancing에서 Smote SVM 실행시 step size
    - default = 0.5
- -smotesvm_stepsize: Data balancing에서 SmoteNC 실행시 categorical features index 설정
    - default = None

- -model_type: 모델 타입 설정
    - default = XGB
    - 사용가능한 옵션: XGB, LGBM, CBC
- -large_data: 큰 데이터 인지 여부 설정
    - default = False
    - 사용가능한 옵션: True, False
- -valid_size: validation 비율 설정
    - default = 0.2
- -model_save: 모델을 저장할지 여부 설정
    - default = True
    - 사용가능한 옵션: True, False
- -trials: optuna 진행 시 trial 횟수
    - default = 1

- -gpu: gpu 설정(XGB, CBC만 가능)
    - default = False
- -gpu_id: gpu_id 설정
    - default = 0
- -cpu_cnt: 사용할 cpu 개수 선택
    - default = 1

#### 5-4. total_script.py / total_script_B.py 실행
**실행 예시**
```
python total_script.py -model_type CBC -data train.csv -target label 
```
    
```
python total_script_B.py -model_type CBC -data train.csv -target label 
```

**일부 기능만 실행 시 주석 처리 후 진행 [6. 부록](#etc) 참고**

**MySQL db 데이터 변환은 주석처리 돼있으므로 사용 시 주석 해제**
    
<hr>
    
    
### 6. 부록 <span id="etc"><span>

#### 6-1. parameter list
<!-- <center> -->
<img src="./images/parameter_list.png" alt="drawing" width="60%"/>
<!-- </center> -->
    
<!-- | clf                                                       |                     |                     | reg                                                    |                     |                     |
|-----------------------------------------------------------|---------------------|---------------------|--------------------------------------------------------|---------------------|---------------------|
| - random_state: 420 default                               |                     |                     | - random_state: 420 default                            |                     |                     |            
| - eval_metric: logloss default                            |                     |                     | - eval_metric: rmse default                            |                     |                     |
| XGB_clf                                                   | LGB_clf             | CBC_clf             | XGB_reg                                                | LGB_reg             | CBC_reg             |
| - learning_rate                                           | - learning_rate     | - learning_rate     | - learning_rate                                        | - learning_rate     | - learning_rate     |
| - n_estimators                                            | - n_estimators      | - n_estimators      | - n_estimators                                         | - n_estimators      | - n_estimators      |
| - max_depth                                               | - max_depth         | - depth             | - max_depth                                            | - max_depth         | - depth             |
| - reg_alpha                                               | - reg_alpha         | - subsample         | - reg_alpha                                            | - reg_alpha         | - subsample         |
| - reg_lambda                                              | - reg_lambda        | - min_child_samples | - reg_lambda                                           | - reg_lambda        | - min_child_samples |
| - colsample_bytree                                        | - num_leaves        | - max_bin           | - colsample_bytree                                     | - num_leaves        | - max_bin           |
| - subsample                                               | - colsample_bytree  | - scale_pos_weight  | - subsample                                            | - colsample_bytree  | - scale_pos_weight  |
| - max_bin                                                 | - subsample         |                     | - max_bin                                              | - subsample         |                     |
|                                                           | - min_child_samples |                     |                                                        | - min_child_samples |                     |
|                                                           | - max_bin           |                     |                                                        | - max_bin           |                     |
|                                                           | - scale_pos_weight  |                     |                                                        | - scale_pos_weight  |                     |
|                                                           | - n_jobs            |                     |                                                        | - n_jobs            |                     | -->
    
    
#### 6-2. model save and load
    
e.g. model save code(train.py에 존재)
```
pickle.dump(final_model, open(f'{os.getcwd()}/model_result/CBC_clf/CBC_clf', 'wb'))
```

e.g. model load code (model load는 별도 진행)
```
model = pickle.load(open(f'{os.getcwd()}/model_result/CBC_clf/CBC_clf', 'rb'))
```
    
    
#### 6-3. 일부 기능 실행
- total_script.py의 main 함수 중 subprocess 주석 처리 후 실행
    

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




Data
1. Multi: https://www.kaggle.com/competitions/prudential-life-insurance-assessment/data
2. Binary: https://github.com/mwitiderrick/insurancedata/blob/master/insurance_claims.csv
