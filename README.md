# reinforece project

## Project 개요
MAB를 기반으로 Hyperparameter Tunining 속도를 개선하는 Project입니다.
Target Model은 Randomforest Classifier이며, Tunning Target Hyperparameter는 총 3종입니다.
① Number of Estimators 
② Max depth 
③ Minimum Samples of leaf 

![image](https://user-images.githubusercontent.com/95091156/206911109-0ad87a37-31fb-4eb5-998d-1daf7377a12a.png)

***

## Python code 구성
### PART1 : MAB CALSS 정의
### PART2 : Data LOAD & 전처리 
### PART3 : 데이터 통합마트 생성 및 학습 준비
### PART4 : 모델학습 및 비교
### PART5 : 모델 다운로드 및 테스트
PART4 학습에 장시간 시간이 소요되므로 이부분을 SKIP하고 MODEL1~3을 확인하려는 경우 PART5 코드를 이용한다.
그런 경우 model1~3 pkl 파일을 다운받아서 사용할 수 있으며, 코드 수행시 Project와 동일 폴더에 pkl 파일이 존재해야한다.

***

## Data Set 및 Preprocessing

분류보델의 데이터셋은 '서울시 우리마을가게 골목상권' Data를 사용합니다.
Data Source : https://data.seoul.go.kr/dataList/3/literacyView.do

Dataset_public 폴더에 있는 총 10개의 csv 파일을 사용합니다.

![image](https://user-images.githubusercontent.com/95091156/206910447-27d76db2-6061-44d4-9a77-c027ba8b8b98.png)

최종 분석 마트

![image](https://user-images.githubusercontent.com/95091156/206910594-254e8e57-a354-4a44-a20e-3d42b3961034.png)

***

