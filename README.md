# GAN_Beginner  
DCGAN -> Wasserstein GAN(WGAN) -> Conditional GAN (cGAN) [sGAN, ACGAN] -> 

![image](https://user-images.githubusercontent.com/72767245/104844851-d228b300-5915-11eb-9dba-658b1013e53f.png)

| **Conditional** | **Unconditional**|
|:----------------:|:----------------:|
|Examples from the classes you want|Examples from random classes|
|Training datasets needs to be labeled|Training dataset dosen't need to be labeled|


### 생성모델링
- 분포에 가깝게 흉내내는 모델 제작
- 이 모델 분포에서 샘플링하여 원본 훈련세트에 있을 것 같은 새롭고 완전히 다른 샘플 생성
- 원본 훈련 데이터셋에 속할 가능성이 높은 픽셀을 출력하는 생성모델 훈련

생성모델 | 판별 모델
------------ | -------------
 P(x &#124; y) or P(x) | P(y &#124; x)

**Pmodel은 Pdata의 추정이다 : Pmodel의 분포로부터 샘플링 하여 new data 생성**
1. 샘플 데이터셋 X을 가지고 있음
2. 샘플이 알려지지 않은 어떤 Pdata 분포로 생성되었다는 가정
3. Pmodel이 Pdata를 흉내내려고함: Pmodel에서 샘플링하여 Pdata에서 뽑은 것과 같이 샘플 생성
4. Pmodel의 규칙
  - Pdata에서 뽑은 것과 같은 샘플 생성 가능
  - X에 있는 샘플과 다른 샘플 생성할 수 있음(즉, 재생산 X)
  
**수학적 개념**
1. 표본공간: 샘플 x가 가질 수 있는 모든 값의 집합 
2. 확률밀도함수 P(x): 표본공간의 포인트 x를 0과 1사이의 숫자에 매핑하는 함수
3. 모수모델 Pθ(X): 한정된 갯수의 파라미터 θ를 사용하여 묘사하는 "확률 밀도 함수"의 한종류
4. 최대가능도 추정: L(θ|x) 값이 최대가 되는 모수 모델 찾기- 최대가능도 추정 hat_θ

**표현학습**
: 의미 있는 잠재공간에서 이미지의 고수준 속성에 영향을 미치는 연산 수행 가능

고차원 표본 공간을 직접 모델링하는 것이 아닌 저차원의 잠재공간을 사용해 훈련세트의 각 샘플을 표현하고 이를 워논 층간의 포인트에 매핑
=> 잠재공간의 각 포인트는 어떤 고차원 이미지에 대한 표현이다

### 심층 신경망

정형데이터 | 비정형데이터
------------ | -------------
 테이블 형식 | 이미지, 소리, 텍스트
 로지스틱 회귀, 랜덤 포레스트, XGBoost훈련 | 딥러닝
 ##### 딥러닝 라이브러리
 - Theano, Tensorflow : 파이썬 기반 (텐서플로우는 **텐서**(데이터 저장하고 네트워크를 통해 전달되는 다차원 배열) 조작에 강하다)
 - Keras : Theano, Tensorflow, CNTK(back-end engine)에서 사용할 수 있도록 함
  - Keras는 고수준의 구성요소를 가지고 있으며 저수준 연산은 다루지 않는다. (백엔드 엔진을 사용)
 **frame work**: 응용 프로그램 개발하기 위한 여러 라이브러리나 모듈 등을 효율적으로 사용하도록 묶은 패키지 <br>
 
**심층 신경망 제작**
(데이터 적재, 모델 만들기, 모델 컴파일, 모델 훈련, 모델 평가)
 **모델 성능 향상**
(합성곱 층 사용, 배치 정규화, 드롭아웃)

### VAE - 변이형 오토인코더
### GAN: DCGAN
### WP-GAN

# GAN [실습]
## 그리기(Style Transfer)
Style Image가 주어졌을 때 base Image를 변환하는 모델을 훈련
- Style Image에 내재된 분포를 모델링 하는 것이 아니라 이미지에서 스타일을 결정하는 요소만 추출하여 베이스 이미지에 주입시킨다 <br>
   * Style Image와 Base Image를 합쳐서는 안됨
- Style Transfer 모델은 두가지<br>
   * **CycleGAN**
   * **Neural Style Transpose**
   
#### Reference
**[미술관에 GAN 딥러닝 실전 프로젝트]을 통해 스터디 진행** <br>
<img src="https://user-images.githubusercontent.com/72767245/98833307-36944580-2481-11eb-8c58-5bb9d022ca67.png" width="20%">
