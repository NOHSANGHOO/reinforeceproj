#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
import gym
from gym import spaces

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import os
import joblib

# 데이터가 불균형한 상태에서, 빠르게 분류 성능을 끌어올리는 케이스
# 유사한 데이터 환경(ex. 매월 혹은 분기별 데이터가 추가됨)에서 정기적으로 모델을 최적화 해야할때 
# 사람이 손대지 않고, 빠른 시간내에 준수한 모델을 생성할 수 있다.


# In[2]:


# Data File List Check
file_path = "C:\ReinforceProject\dataset"
file_check = os.listdir(file_path)
print("number of file : ",len(file_check))


# In[3]:


#read csv data
#상권코드 col datatype 통합
csv_data1 = pd.read_csv(file_path+'\\'+'서울시_우리마을가게_상권분석서비스(상권-점포)_2020년.csv',encoding='EUC-KR', header=0)
csv_data2 = pd.read_csv(file_path+'\\'+'서울시_우리마을가게_상권분석서비스(상권-점포)_2021년.csv',encoding='EUC-KR', header=0)
csv_data1['상권_코드'] = csv_data1['상권_코드'].astype(str)
csv_data2['상권_코드'] = csv_data2['상권_코드'].astype(str)

csv_data3 = pd.read_csv(file_path+'\\'+'서울시 우리마을가게 상권분석서비스(상권-추정매출)_2020년.csv',encoding='EUC-KR', header=0)
csv_data4 = pd.read_csv(file_path+'\\'+'서울시 우리마을가게 상권분석서비스(상권-추정매출)_2021년.csv',encoding='EUC-KR', header=0)
csv_data3['상권_코드'] = csv_data3['상권_코드'].astype(str)
csv_data4['상권_코드'] = csv_data4['상권_코드'].astype(str)

csv_data5 = pd.read_csv(file_path+'\\'+'서울시_우리마을가게_상권분석서비스(구_상권_생활인구)_2020년.csv',encoding='EUC-KR', header=0)
csv_data6 = pd.read_csv(file_path+'\\'+'서울시_우리마을가게_상권분석서비스(구_상권_생활인구)_2021년.csv',encoding='EUC-KR', header=0)
csv_data5['상권_코드'] = csv_data5['상권_코드'].astype(str)
csv_data6['상권_코드'] = csv_data6['상권_코드'].astype(str)

csv_data7 = pd.read_csv(file_path+'\\'+'서울시_우리마을가게_상권분석서비스(구_상권_직장인구).csv',encoding='EUC-KR', header=0)
csv_data8 = pd.read_csv(file_path+'\\'+'서울시_우리마을가게_상권분석서비스(구_상권_집객시설).csv',encoding='EUC-KR', header=0)
csv_data7['상권_코드'] = csv_data7['상권_코드'].astype(str)
csv_data8['상권_코드'] = csv_data8['상권_코드'].astype(str)

csv_data9 = pd.read_csv(file_path+'\\'+'서울시_우리마을가게_상권분석서비스(구_상권배후지_소득소비).csv',encoding='EUC-KR', header=0)
csv_data9['상권_코드'] = csv_data9['상권_코드'].astype(str)

csv_data10 = pd.read_csv(file_path+'\\'+'서울시_우리마을가게_상권분석서비스(구_상권_상주인구).csv',encoding='EUC-KR', header=0)
csv_data11 = pd.read_csv(file_path+'\\'+'서울시_우리마을가게_상권분석서비스(구_상권배후지_상주인구).csv',encoding='EUC-KR', header=0)
csv_data10.rename(columns = {'상권 코드': '상권_코드'}, inplace = True) # 컬럼명 다름 주의
csv_data10['상권_코드'] = csv_data10['상권_코드'].astype(str)
csv_data11['상권_코드'] = csv_data11['상권_코드'].astype(str)

csv_data1 = pd.concat([csv_data1,csv_data2],axis=0).reset_index(drop=True) #'20~'21년 점포 데이터 결합
csv_data3 = pd.concat([csv_data3,csv_data4],axis=0).reset_index(drop=True) #'20~'21년 추정매출 데이터 결합
csv_data5 = pd.concat([csv_data5,csv_data6],axis=0).reset_index(drop=True) #'20~'21년 생활인구 데이터 결합


# In[4]:


################# 점포 데이터 정제 #################
# 1. 불필요 컬럼 제거
# 2. 업종 통합(N종 → 3종)
# 3. group by sum
# 4. 폐업률 컬럼 계산

key_col = ['기준_년_코드','기준_분기_코드','상권_구분_코드','상권_구분_코드_명','상권_코드','상권_코드_명']
sel_col = key_col+['점포_수','폐업_점포_수']

# # 업종코드 통합
# # CS1 : 외식, CS2 : 서비스, CS3 : 소매   (출처 : https://golmok.seoul.go.kr/introduce.do)
# csv_data1['서비스_업종_코드'] = csv_data1['서비스_업종_코드'].str[:3]
# csv_data1

# 필요한 컬럼만 선택(keys, 점포수, 폐업점포수)
csv_data1 = csv_data1.loc[:,sel_col]

# 기준년, 기준분기, 상권, 업종(외식,서비스,소매) 단위로 Group by 수행
# as_index=False : group by 할때 컬럼들을 index 컬럼으로 설정하지 않는 옵션
data1 = csv_data1.groupby(key_col,as_index=False).sum()

# 점포수가 너무 작은(10건 미만) 상권&업종은 제거
data1 = data1[data1['점포_수']>10]

# 폐업률 계산
data1['폐업률'] = data1['폐업_점포_수'].astype('float') / data1['점포_수'].astype('float')

# 폐업이 있는데, 점포수가 0인 경우 (inf)
data1 = data1.copy().replace([np.inf, -np.inf], np.nan) # inf, -inf를 nan으로 대체

# 0 나누기 예외처리(나누기0, 0나누기0)
data1['폐업률'] = data1['폐업률'].fillna(0)

# 폐업률이 1 이상인 데이터가 있다면, 1로 변경 (ex. 점포수 2, 폐업점포수 4)
data1['폐업률'] = data1['폐업률'].apply(lambda x : 1 if x>=1 else x)


# 불필요컬럼(폐업점포수) 컬럼 제거
data1 = data1.loc[:,~data1.columns.isin(['폐업_점포_수','폐업률'])]
# sel_col = key_col+['폐업률상위_상권업종','점포_수']
sel_col = key_col+['점포_수']
data1 = data1.loc[:,sel_col]
data1

data1.describe() # 요약통계
data1.info() # Null 체크

#data1.to_csv(path_or_buf=file_path+'\\data1.csv', encoding='EUC-KR')
################# 점포 데이터 정제 끝 #################


# In[5]:


################# 추정매출 데이터 정제 #################
key_col = ['기준_년_코드','기준_분기_코드','상권_구분_코드','상권_구분_코드_명','상권_코드','상권_코드_명']
sel_col = key_col+['분기당_매출_금액','분기당_매출_건수','주말_매출_금액','점포수']

# CS1 : 외식, CS2 : 서비스, CS3 : 소매   (출처 : https://golmok.seoul.go.kr/introduce.do)
# csv_data3['서비스_업종_코드'] = csv_data3['서비스_업종_코드'].str[:3]
# csv_data3


# 필요한 컬럼만 선택
csv_data3 = csv_data3.loc[:,sel_col]

# 기준년, 기준분기, 상권, 업종(외식,서비스,소매) 단위로 Group by 수행
# as_index=False : group by 할때 컬럼들을 index 컬럼으로 설정하지 않는 옵션
data3 = csv_data3.groupby(key_col,as_index=False).sum()

# 신규컬럼추가, 기존컬럼제거
data3['주말_매출_비중'] = data3['주말_매출_금액'].astype('float') / data3['분기당_매출_금액'].astype('float')
data3['주말_매출_비중'] = data3['주말_매출_비중'].fillna(0)

# 점포당 월평균 매출액 산출(단위 : 원)
data3['점포당_월평균_매출액'] = (data3['분기당_매출_금액'].astype('float')/3)/ data3['점포수'].astype('float')
data3['점포당_월평균_매출액'] = data3['점포당_월평균_매출액'].fillna(0)

# 점포수가0인데, 매출이 있는 경우 (inf) 제거
data3 = data3.copy().replace([np.inf, -np.inf], np.nan) # inf, -inf를 nan으로 대체
data3.dropna(inplace=True)

#####################################################################
## 점포당 월평균_매출액 상위 30%는 1로 라벨링(Child Model's 종속변수)
#####################################################################
top_30p = data3['점포당_월평균_매출액'].quantile(q=0.70, interpolation='nearest')
data3['점포매출상위_상권'] = data3['점포당_월평균_매출액'].apply(lambda x : 1 if x>=top_30p else 0)

# 불필요 컬럼 제거
data3 = data3.loc[:,~data3.columns.isin(['주말_매출_금액','분기당_매출_금액','점포수','점포당_월평균_매출액'])]

data3.describe() # 요약통계
data3.info() # Null 체크 

################# 추정매출 데이터 정제 끝 #################


# In[6]:


################# 생활인구 데이터 정제 #################
# 총생활인구, 남성/여성, 시간대만 남기고 모든 컬럼 제거(성별 x 연령대 x 시간대 복합컬럼 제거, 요일별 데이터 제거)
data5 = csv_data5.loc[:
                          ,~csv_data5.columns.str.startswith('남성연령대')
                          & ~csv_data5.columns.str.startswith('여성연령대')
                          & ~csv_data5.columns.str.contains('요일')]

# key_col = ['기준_년_코드','기준_분기_코드','상권_구분_코드','상권_구분_코드_명','상권_코드','상권_코드_명']

#이상 컬럼명 변경
data5.rename(columns = {' 상권_구분_코드_명': '상권_구분_코드_명','기준 년코드': '기준_년_코드'}, inplace = True)

data5.info()

################# 생활인구 데이터 정제 끝 #################


# In[7]:


################# 직장인구 데이터 정제 #################
data7 = csv_data7.loc[:
                          ,~csv_data7.columns.str.startswith('남성연령대')
                          & ~csv_data7.columns.str.startswith('여성연령대')
                          & ~csv_data7.columns.str.contains('요일')]

#이상 컬럼명 변경
data7.rename(columns = {'기준_년월_코드': '기준_년_코드'}, inplace = True)

data7 = data7[(data7['기준_년_코드'] == 2020) | (data7['기준_년_코드'] == 2021)]

data7.info()
################# 직장인구 데이터 정제 끝 #################


# In[8]:


################# 집객시설 데이터 정제 #################
key_col = ['기준_년_코드','기준_분기_코드','상권_구분_코드','상권_구분_코드_명','상권_코드','상권_코드_명']
sel_col = key_col+['집객시설_수','지하철_역_수','버스_정거장_수']

# 필요한 컬럼만 선택
data8 = csv_data8.loc[:,sel_col]

data8 = data8.fillna(0)

data8 = data8[(data8['기준_년_코드'] == 2020) | (data8['기준_년_코드'] == 2021)]
data8.info()
################# 집객시설 데이터 정제 끝 #################


# In[9]:


# ################# 상권배후지_소득소비 데이터 정제 #################
# 결측치가 60% 이상으로 제거(미사용)
# key_col = ['기준_년_코드','기준_분기_코드','상권_구분_코드','상권_구분_코드_명','상권_코드','상권_코드_명']
# sel_col = key_col+['월_평균_소득_금액','지출_총금액']

# #이상 컬럼명 변경
# csv_data9.rename(columns = {'기준 년 코드': '기준_년_코드'}, inplace = True)

# # 필요한 컬럼만 선택
# data9 = csv_data9.loc[:,sel_col]

# data9 = data9[(data9['기준_년_코드'] == 2020) | (data9['기준_년_코드'] == 2021)]
# data9.info()

# ################# 상권배후지_소득소비 데이터 정제 끝 #################


# In[10]:


################# 상주인구 데이터 정제 #################

data10 = csv_data10.loc[:
                          ,~csv_data10.columns.str.startswith('남성연령대')
                          & ~csv_data10.columns.str.startswith('여성연령대')
                          & ~csv_data10.columns.str.contains('가구')                        
                     ]
#이상 컬럼명 변경
data10.rename(columns = {'상권 코드 명': '상권_코드_명'}, inplace = True)

data10 = data10[(data10['기준_년_코드'] == 2020) | (data10['기준_년_코드'] == 2021)]

data10.info()

################# 상주인구 데이터 정제 끝 #################


# In[11]:


################# 배후지 상주인구 데이터 정제 #################
data11 = csv_data11.loc[:
                          ,~csv_data11.columns.str.startswith('남성연령대')
                          & ~csv_data11.columns.str.startswith('여성연령대')
                          & ~csv_data11.columns.str.contains('가구')                        
                     ]

data11 = data11[(data11['기준_년_코드'] == 2020) | (data11['기준_년_코드'] == 2021)]

data11.rename(columns = lambda x: x.replace('상주인구','배후지_상주인구'), inplace = True)

data11.info()
################# 배후지 상주인구 데이터 정제 끝 #################


# In[12]:


################# 데이터 마트 생성 및 최종 전처리 #################

key_col = ['기준_년_코드','기준_분기_코드','상권_구분_코드','상권_구분_코드_명','상권_코드','상권_코드_명']
mart = pd.merge(data1, data3, how='left', on = key_col,suffixes=('','_right'))

key_col = ['기준_년_코드','기준_분기_코드','상권_구분_코드','상권_구분_코드_명','상권_코드','상권_코드_명']
mart = pd.merge(mart, data5, how='left', on = key_col)
mart = pd.merge(mart, data7, how='left', on = key_col)
mart = pd.merge(mart, data8, how='left', on = key_col)
# mart = pd.merge(mart, data9, how='left', on = key_col) # 결측치 60%이상으로 전체 제거
mart = pd.merge(mart, data10, how='left', left_on = key_col, right_on = key_col)
mart = pd.merge(mart, data11, how='left', left_on = key_col, right_on = key_col)

# 점포수가 확인되지 않는 상권 데이터 제거
#mart[mart['점포수'].isnull()]
mart = mart.dropna(subset=['점포_수'])

# 결측치처리 : 집객시설이 없는 지역은 0으로 입력
mart['집객시설_수'] = mart['집객시설_수'].fillna(0)
mart['지하철_역_수'] = mart['지하철_역_수'].fillna(0)
mart['버스_정거장_수'] = mart['버스_정거장_수'].fillna(0)

# 이외 생활/상주 인구수 데이터 누락 상권 제외
mart.dropna(axis=0, inplace=True)

# 마트 데이터 확인
print(mart.info())
print(mart.describe())

# train set, test set 분리
# 21년 4Q 데이터를 Test set으로 사용
train = mart[(mart['기준_년_코드'] == 2020) | ((mart['기준_년_코드'] == 2021) & (mart['기준_분기_코드'] < 4))]
test = mart[(mart['기준_년_코드'] == 2021) & (mart['기준_분기_코드'] == 4)]

# 인덱스 초기화
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# X, Y 분리
# X : 기준_년_코드 ~ 폐업률 제거
train_X = train.iloc[:,7:]
train_X = train_X.loc[:,~train_X.columns.isin(['점포매출상위_상권'])]
train_y = train['점포매출상위_상권']

test_X = test.iloc[:,7:] # 기준_년_코드 ~ 폐업률 제거
test_X = test_X.loc[:,~test_X.columns.isin(['점포매출상위_상권'])]
test_y = test['점포매출상위_상권'] 

# Min-max Scaling 
scaler = MinMaxScaler(feature_range=(0,1)) # 0~1로 변환
scaler.fit(train_X) # 각 칼럼별 변환 함수 생성
train_X_scaled = scaler.transform(train_X) # fit에서 만든 함수로 데이터 변환(train)
train_X = pd.DataFrame(train_X_scaled, columns = train_X.columns)
test_X_scaled = scaler.transform(test_X) # fit에서 만든 함수로 데이터 변환(test)
test_X = pd.DataFrame(test_X_scaled, columns = test_X.columns)

# Validation set 분리
# train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size = 0.2, random_state = 111)


################# 데이터 마트 및 최종 전처리 끝 #################


# In[13]:


########################### ① 기본 모델 ##################################

# 시간측정 시작
start = time.time()

rfc = RandomForestClassifier(max_depth=2, n_estimators=100, min_samples_leaf=1, random_state=333) # 531

# rfc.fin()에 훈련 데이터를 입력해 Random Forest 모듈을 학습
rfc.fit(train_X, train_y)

#Test data를 입력해 target data를 예측 (매번 달라짐)
pred_y_basic = rfc.predict(test_X)

# 시간측정 종료 ()
end = time.time()
print(f"{end - start:.5f} sec")

# 랜덤포레스트 모델 성능
print('Accuracy : ',metrics.accuracy_score(test_y, pred_y_basic))
print('precision : ',metrics.precision_score(test_y, pred_y_basic, pos_label=1))
print('recall : ',metrics.recall_score(test_y, pred_y_basic, pos_label=1))
print('f1 : ',metrics.f1_score(test_y, pred_y_basic, pos_label=1))

########################### ① 기본 모델 끝 ##################################
# 모델 저장
joblib.dump(rfc, './modle1.pkl')


# In[15]:


########################### ② GridSearchCV 모델 ##################################
# 시간측정 시작
start = time.time()

# 모델 생성
# n_estimators : 모델에서 사용할 트리 갯수(학습시 생성할 트리 갯수)
# max_depth : 트리의 최대 깊이
# min_samples_leaf : 리프 노드에 있어야 할 최소 샘플 수 (default : 1)
rfc = RandomForestClassifier(max_depth=2, n_estimators=100, min_samples_leaf=1, random_state=333)

# Grid Search 수행(5 x 5 x 3 Case)

# 데이터 불균형이 존재하여, F1 SCORE를 기준으로 학습
param1=[100, 200, 400, 600, 1000, 1400]
param2=[2, 4, 6, 8, 10, 12]
param3=[1, 2, 4, 6]

params = {'n_estimators' : param1,'max_depth' : param2,'min_samples_leaf': param3}
gscv_rfc = GridSearchCV(estimator = rfc, param_grid = params, scoring ='f1', cv = 5, n_jobs=1)
gscv_rfc.fit(train_X, train_y)

#Test data를 입력해 target data를 예측 
pred_y = gscv_rfc.predict(test_X)

# rfc.feature_importances_.round(3) # 변수중요도

# 시간측정 종료 (1925.94726 sec, 약 32분)
end = time.time()
print(f"학습 소요시간 : {end - start:.5f} sec")

# 최적의 성능을 가진 모델
print(gscv_rfc.best_estimator_)

# 최적일 경우의 값
print(gscv_rfc.best_score_)

# 최적일 경우의 parameters
print('최적 parameters : ', gscv_rfc.best_params_)
#gscv_rfc.cv_results_.keys()

# 랜덤포레스트 모델 성능
print('Accuracy : ',metrics.accuracy_score(test_y, pred_y))
print('precision : ',metrics.precision_score(test_y, pred_y, pos_label=1))
print('recall : ',metrics.recall_score(test_y, pred_y, pos_label=1))
print('f1 : ',metrics.f1_score(test_y, pred_y, pos_label=1))

########################### ② GridSearchCV 모델 끝 ##################################
# 모델 저장
joblib.dump(gscv_rfc, './modle2.pkl')


# In[16]:


# hyperparameter별 스코어 랭킹 출력
print(pd.DataFrame(gscv_rfc.cv_results_).loc[:,['param_n_estimators','param_max_depth','param_min_samples_leaf','rank_test_score','mean_test_score']].sort_values(by='rank_test_score'))

# learning 스코어 그래프 
plt.plot(gscv_rfc.cv_results_['mean_test_score'])


# In[17]:


class BanditEnv(gym.Env):

    def __init__(self, acion_list, r_dist):
        if len(acion_list) != len(r_dist):
            raise ValueError("Probability and Reward distribution must be the same length")

        for reward in r_dist:
            if isinstance(reward, list) and reward[1] <= 0:
                raise ValueError("Standard deviation in rewards must all be greater than 0")

        self.acion_list = acion_list
        self.r_dist = r_dist

        self.n_bandits = len(acion_list)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(1)

    def step(self, action, score, before_score): # 어떤 action을 했을때, 성능이 과거대비 증가했다면 action(arm)에 보상을 진행한다
        assert self.action_space.contains(action)

        reward = 0
        done = True

        # 스코어가 좋아지면 보상 주기
        if score >= before_score:
            reward = self.r_dist[action]
        elif score < before_score:
            reward = 0

        return [0], reward, done

    def reset(self):
        return [0]

class BanditArmed(BanditEnv):

    def __init__(self,acion_list):
        BanditEnv.__init__(self, acion_list=acion_list, r_dist=[1]*len(acion_list))
        


# In[18]:


class thomsonArmed:

    def __init__(self, acion_list):       
        self.env = BanditArmed(acion_list)
        self.acion_list = acion_list        
        
        self.count = np.zeros(self.env.n_bandits)
        self.sum_rewards = np.zeros(self.env.n_bandits)
        self.Q = np.zeros(self.env.n_bandits)
        self.alpha = np.ones(self.env.n_bandits)
        self.beta = np.ones(self.env.n_bandits)

    def thompson_sampling(self, alpha, beta):
        samples = [np.random.beta(self.alpha[i]+1, self.beta[i]+1) for i in range(self.env.n_bandits)]
        return np.argmax(samples)
    
    # arm을 sampling하고, 해당 arm이 어떤 action(parameter)인지 리턴
    def thompson_sampling_action(self):
        arm = self.thompson_sampling(self.alpha,self.beta)
        return arm, self.acion_list[arm]

    # Arm one step 수행, 직전에 action을 수행하여 input으로 전달해야함
    # score가 개선되었다면 1점 추가
    def thomson_step(self, arm, score, before_score):
        next_state, reward, done = self.env.step(arm, score, before_score)
        self.count[arm] += 1
        self.sum_rewards[arm] += reward
        self.Q[arm] = self.sum_rewards[arm] / self.count[arm]
        
        if reward == 1:
            self.alpha[arm] = self.alpha[arm] + 1
        else:
            self.beta[arm] = self.beta[arm] + 1
        
        return [0]

    
class softArmed:

    def __init__(self, acion_list, T):
        self.env = BanditArmed(acion_list)
        self.acion_list = acion_list        
        
        self.count = np.zeros(self.env.n_bandits)
        self.sum_rewards = np.zeros(self.env.n_bandits)
        self.Q = np.zeros(self.env.n_bandits)
        self.T = T

    def softmax(self, T):
        denom = sum([np.exp(i/T) for i in self.Q])
        probs = [np.exp(i/T)/denom for i in self.Q]
        arm = np.random.choice(self.env.action_space.n, p=probs)
        return arm
    
    def softmax_action(self):
        arm = self.softmax(self.T)
        return arm, self.acion_list[arm]

    # Arm one step 수행, 직전에 action을 수행하여 input으로 전달해야함
    # score가 개선되었다면 1점 추가    
    def softmax_step(self, arm, score, before_score):
        next_state, reward, done = self.env.step(arm, score, before_score)
        self.count[arm] += 1
        self.sum_rewards[arm] += reward
        self.Q[arm] = self.sum_rewards[arm] / self.count[arm]
        self.T = self.T * 0.9999            
        
        return [0]


# In[ ]:





# In[ ]:





# In[19]:


########################## ③ MAB Model 기반 학습 끝 ##########################
####### grid-search CV와 동일하게 (Cross validation) k-fold를 5로 설정######
param1=[100, 200, 400, 600, 1000, 1400]
param2=[2, 4, 6, 8, 10, 12]
param3=[1, 2, 4, 6]

soft1 = softArmed([1,2,3],50) # 어떤 파라메터를 건들이면 성능이 향상되는지 학습
thom1 = thomsonArmed(param1) # 1번 파라메터 교체 학습
thom2 = thomsonArmed(param2) # 2번 파라메터 교체 학습
thom3 = thomsonArmed(param3) # 3번 파라메터 교체 학습

start = time.time()  # 시간측정 시작

p1 = 100     # hyperparameter1 : n_estimator
p2 = 2       # hyperparameter2 : max_depth
p3 = 1       # hyperparameter3 : min_samples_leaf
action = 0   # acion initialize

### 기본 모델 초기화
rfc = RandomForestClassifier(n_estimators=p1,
                             max_depth=p2,
                             min_samples_leaf=p3,
                             random_state=531)
rfc.fit(train_X, train_y)
preds = rfc.predict(test_X)
score = metrics.f1_score(test_y, preds)

best_rfc = rfc
best_score=score
before_score=score

SPLITS = 5 # k-fold split 수
skf = StratifiedKFold(n_splits = SPLITS)
n_iter = 0

score_list = []
time_list = [] 

while((time.time()-start)<60*30): # 최대 30분 동안 학습 수행
    
    for train_idx, test_idx in skf.split(train_X, train_y): # k-fold split 
        n_iter += 1
        
        # k-fold k번째 train, test set 분리
        X_train, X_test = train_X.iloc[train_idx, :], train_X.iloc[test_idx, :]
        y_train, y_test = train_y.iloc[train_idx], train_y.iloc[test_idx]

        # Softmax exploration으로 어떤 hyperprameter를 변화시킬지 선택
        pram_action, pram_num = soft1.softmax_action() 
        
        # hyperparameter 1번이 선택되었다면, thompson sampling으로 parameter 선택
        if pram_num == 1:
            action, p1 = thom1.thompson_sampling_action()
            rfc = RandomForestClassifier(n_estimators=p1,
                                         max_depth=p2,                              
                                         min_samples_leaf=p3,
                                         random_state=531)
            rfc.fit(X_train, y_train)
            preds = rfc.predict(X_test)
            score = metrics.f1_score(y_test, preds)

            thom1.thomson_step(action, score, before_score)
        
        # hyperparameter 2번이 선택되었다면, thompson sampling으로 parameter 선택
        elif pram_num == 2:
            action, p2 = thom2.thompson_sampling_action()
            rfc = RandomForestClassifier(n_estimators=p1,
                                         max_depth=p2,                              
                                         min_samples_leaf=p3,
                                         random_state=531)
            rfc.fit(X_train, y_train)
            preds = rfc.predict(X_test)
            score = metrics.f1_score(y_test, preds)

            thom2.thomson_step(action, score, before_score)
        
        # hyperparameter 3번이 선택되었다면, thompson sampling으로 parameter 선택
        elif pram_num == 3:
            action, p3 = thom3.thompson_sampling_action()
            rfc = RandomForestClassifier(n_estimators=p1,
                                         max_depth=p2,                              
                                         min_samples_leaf=p3,
                                         random_state=531)
            rfc.fit(X_train, y_train)
            preds = rfc.predict(X_test)
            score = metrics.f1_score(y_test, preds)

            thom3.thomson_step(action, score, before_score)    
        
        # Softmax exploration Step
        soft1.softmax_step(pram_action, score, before_score)        
        before_score = score
        
        # 이전 최고스코어보다 크면, 스코어와 모델 저장
        if score > best_score:
            best_score = score
            best_rfc = rfc
        
        print(f'{n_iter}번째 KFold-----------------------------------------')
        print(f'train_idx_len : {len(train_idx)} / pram_number : {pram_num} / action : {action}')           
        print(f'f1_score:{score}')
        score_list.append(score)
        time_list.append(time.time()-start)
    

print('======================================================')
print(f'최종 평균 f1_score : {sum(score_list)/len(score_list)}')
print(f'최종 Best f1_score : {best_score}')
print(f'최종 Best Model : {best_rfc}')

########################## ③ MAB Model 기반 학습 끝 ##########################


# In[20]:


print("## 모델3 학습 결과 ##")
print('parameter selection : ',soft1.count)
print('parameter 1 : ',thom1.count)
print('parameter 2 : : ',thom2.count)
print('parameter 3 : : ',thom3.count)


# In[21]:


# 모델 성능 확인
preds = best_rfc.predict(test_X)

print('Accuracy : ',metrics.accuracy_score(test_y, preds))
print('precision : ',metrics.precision_score(test_y, preds, pos_label=1))
print('recall : ',metrics.recall_score(test_y, preds, pos_label=1))
print('f1 : ',metrics.f1_score(test_y, preds, pos_label=1))

# 모델 저장
joblib.dump(best_rfc, './modle3.pkl')


# In[22]:


plt.plot(time_list[::3], score_list[::3])


# In[23]:


# load_model = joblib.load('./model1.pkl')
# preds =load_model.predict(test_X)

# print('Model1 Accuracy : ',metrics.accuracy_score(test_y, preds))
# print('Model1 precision : ',metrics.precision_score(test_y, preds, pos_label=1))
# print('Model1 recall : ',metrics.recall_score(test_y, preds, pos_label=1))
# print('Model1 f1 : ',metrics.f1_score(test_y, preds, pos_label=1))


# load_model = joblib.load('./model2.pkl')
# preds =load_model.predict(test_X)

# print('Model2 Accuracy : ',metrics.accuracy_score(test_y, preds))
# print('Model2 precision : ',metrics.precision_score(test_y, preds, pos_label=1))
# print('Model2 recall : ',metrics.recall_score(test_y, preds, pos_label=1))
# print('Model2 f1 : ',metrics.f1_score(test_y, preds, pos_label=1))



# load_model = joblib.load('./model3.pkl')
# preds =load_model.predict(test_X)

# print('Model3 Accuracy : ',metrics.accuracy_score(test_y, preds))
# print('Model3 precision : ',metrics.precision_score(test_y, preds, pos_label=1))
# print('Model3 recall : ',metrics.recall_score(test_y, preds, pos_label=1))
# print('Model3 f1 : ',metrics.f1_score(test_y, preds, pos_label=1))


# In[ ]:




