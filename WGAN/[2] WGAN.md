# WGAN(Wasserstein GAN)
안정적인 GAN 훈련을 위한 첫 번째 큰 발전

- 생성자가 수렴하는 것과 샘플의 품질을 연관 짓는 의미있는 손실 측정 방법
- 최적화 과정의 안정성 향상

**새로운 손실 함수 소개**

## Wassertein Loss
### 기존 손실함수
#### 이진 크로스 엔트로피 함수
![image](https://user-images.githubusercontent.com/72767245/104926533-a2e07780-59e3-11eb-9dec-c48de98b71aa.png)
- GAN의 판별자D를 훈련하기 위해 진짜 이미지에 대한 예측 ```p = D(x)```와 타깃 ```y = 1```
- 생성된 이미지에 대한 예측 ```p = D(G(z))```와 타깃 ```y = 0```

#### KL-Divergence

#### JS-Divergence

#### GAN 판별자와 생성자 손실 최소화
![image](https://user-images.githubusercontent.com/72767245/104924284-90187380-59e0-11eb-8cb4-b7e638f1577d.png)

가치함수 Pr과 Pg사이의 JS Divergence를 최소화할 수 있지만 그렇게 하면 discriminator가 포화되는 vanishing gradients 문제 유발

### WGAN Loss
- 와서스테인 손실은 1과 0 대신 **y = 1, y= -1** 사용 
- 판별자의 마지막 층에서 시그모이드 활성화 함수를 제거하여 예측 p가 [0,1]범위에 국한되지 않고 [-무한, 무한]범위의 어떤 숫자도 될 수 있도록 함
- WGAN의 판별자는 보통 비평자 라고 부름

### 와서스테인 손실

### WGAN 비평자 손실 최소화

### WGAN 생성자 손실 최소화

## 립시츠 제약

## 가중치 클리핑

## WGAN 분석