# GAN
합성곱층 대신 완전 연결 층을 사용

# DCGAN
합성곱층을 사용하여 판별자의 예층 성능 분석

✔ 잠재공간을 원본 차원으로 매핑하는 개념: 생성 모델링에서 매우 일반적  
- 잠재공간의 벡터를 조작하여 원본 차원에 있는 이미지의 고수준 특성으로 바꿀 수 있다

✔ Upsampling
- Deconvolution
  - 입력 텐서의 각 pixel 마다 주변에 zero-paddig 후 conv
- Unpooling
  - Max pooling할때 Index를 기록해두었다가 해당 Index를 기반으로 pooling을 역으로 수행  
- transposed convolution
  - 일반적인 Convolution 연산
  - Convolution 연산에 Transpose  

padding에 따라 계산이 조금씩 달라진다  

```python
padding = 'valid'
```
output = (Input-1) * stride + filter

```python
padding = 'same'
```
output = Input * stride

# WGAN
- 진동하는 손실
- 판별자와 생성자의 균형 문제(Vanishing Gradient)
- 모드 붕괴(Mode Collapse)

Discriminator은 진짜와 가짜를 판별하기 위해 sigmoid를 사용하고, output은 진짜/가짜의 예측 확률값  
Critic은 EM(Earth Mover) distance로 부터 얻은 scalar 값을 이용(y=0,1이 아닌 -1,1사용)    
EM distance는 확률 분포간의 거리를 측정하는 척도 < KL divergence/JS divergence : strict하여 continuous 하지 않다 >  
(ex. KL/JS divergence나 TV의 경우 두 분포가 서로 겹치지 않은 경우에는 0, 겹치면 무한대나 상수로 극단적, 초반에는 실제 데이터의 분포와 겹치지 않으므로 무한대 또는 일정한 상수값을 갖다가, 갑자기 0으로 변하여 gradient가 제대로 전달되지 않음)
### WGAN은 Wasserstein Loss 사용
### WGAN은 진짜에는 레이블 1, 가짜에는 레이블 -1을 사용하여 훈련
### WGAN 비평자의 마지막 층은 시그모이드 활성화 함수가 필요하지 않음
### 매 업데이트 후에 판별자의 가중치를 클리핑
### 생성자를 업데이트할때마다 판별자를 여러번 훈련
### 주요단점: 비평자에서 가중치를 클리핑했기 때문에 학습속도가 크게 감소
#### 립시츠 제약을 두기 위한 가중치 클리핑은 좋지 않은 방법

# WGAN-GP
