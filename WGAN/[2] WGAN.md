# WGAN(Wasserstein GAN)
안정적인 GAN 훈련을 위한 첫 번째 큰 발전  
WGAN에서는 새로운 loss를 사용하는 판별자 D를 비평자 ```C(Critic)```이라고 함

- 생성자가 수렴하는 것과 샘플의 품질을 연관 짓는 의미있는 손실 측정 방법
- 최적화 과정의 안정성 향상

**새로운 손실 함수 소개**

## Wassertein Loss
### 기존 손실함수
![image](https://user-images.githubusercontent.com/72767245/104926656-cacfdb00-59e3-11eb-918a-5b8aaaf26aa8.png)


#### 이진 크로스 엔트로피 함수
![image](https://user-images.githubusercontent.com/72767245/104926533-a2e07780-59e3-11eb-9dec-c48de98b71aa.png)
- GAN의 판별자D를 훈련하기 위해 진짜 이미지에 대한 예측 ```p = D(x)```와 타깃 ```y = 1```
- 생성된 이미지에 대한 예측 ```p = D(G(z))```와 타깃 ```y = 0```
![image](https://user-images.githubusercontent.com/72767245/104936492-f3f66880-59ef-11eb-929c-cd03f5bf086e.png)

#### KL-Divergence
![image](https://user-images.githubusercontent.com/72767245/104936798-5a7b8680-59f0-11eb-8a1a-033b55df85ba.png)

#### JS-Divergence
![image](https://user-images.githubusercontent.com/72767245/104936822-636c5800-59f0-11eb-826a-2e0ea347d24b.png)

- EM distance(=Wasserstein 거리)의 경우 추정하는 모수에 상관없이 일정한 수식을 가지고 있으나, 다른 경우 모수에 따라 거리가 달라질 뿐만 아니라 그 값이 상수 또는 무한대의 값을 가지게 됨
- TV,KL/JS Divergence는 두 분포가 서로 겹치는 경우에 0, 겹치지 않는 경우에는 무한대 또는 상수로 극단적인 값을 가지게됨
- 결론적으로 TV, KL/JS Divergence을 loss로 사용한다면 gradient가 제대로 전달되지 않아 학습이 어려워짐

#### Earth Mover's Distance
- GAN의 목표인 ```Pdata(x)와 동일하도록 Pmodel(x)을 학습```은 Pdata(x) 와 Pmodel(x) **두 분포 사이의 거리를 줄이는 것**
- Earth Mover's Distance는 두 분포가 얼마나 다른지를 나타내는 수치
- Pmodel(x)을 Pdata(x)와 동일하게 만들기 위해 이동해야 하는 거리와 양을 의미
<br><br>
- Earth Mover's Distance의 결과는 0과 1의 한계가 없음
- 분포가 얼마나 멀리 떨어져있든 상관없이 의미있는 gradient(0이 아닌 기울기)를 전달 가능
![image](https://user-images.githubusercontent.com/72767245/104936517-fbb60d00-59ef-11eb-8ca3-dd8aa48322df.png)

- **Vanishing Gradient** 해결, **Model collapse** 가능성 감소

![image](https://user-images.githubusercontent.com/72767245/104936900-7ed76300-59f0-11eb-8640-b5d505ced736.png)


#### GAN 판별자와 생성자 손실 최소화
Earth Mover's Distance 사용

![image](https://user-images.githubusercontent.com/72767245/104924284-90187380-59e0-11eb-8cb4-b7e638f1577d.png)

가치함수 Pr과 Pg사이의 JS Divergence를 최소화할 수 있지만 그렇게 하면 discriminator가 포화되는 vanishing gradients 문제 유발

### WGAN Loss

- 1. 와서스테인 손실은 1과 0 대신 **y = 1, y= -1** 사용 
- 2. 판별자의 마지막 층에서 **시그모이드 활성화 함수를 제거**하여 예측 p가 [0,1]범위에 국한되지 않고 [-무한, 무한]범위의 어떤 숫자도 될 수 있도록 함
- 3. WGAN의 판별자는 보통 ```Critic(비평자)``` 라고 부름

### 와서스테인 손실
![image](https://user-images.githubusercontent.com/72767245/104938656-b6dfa580-59f2-11eb-9474-10476a7bbf0a.png)

![image](https://user-images.githubusercontent.com/72767245/104938900-1047d480-59f3-11eb-837c-5a06cd0c3e89.png)

- Earth Mover's Distance의 출력이 [0,1]로 제한되지 않아 loss에 사용하는 것이 적적하다는 것을 알아보았는데, 일반적으로 신경망에서 너무 큰 숫자는 피해야 하기 때문에 Lipshitz 제약이라는 제약조건을 걸어줌
- 비평가(C)가 1- Lipshiz continuous Function(1-립시츠 연속함수)이어야함

## 립시츠 제약
![image](https://user-images.githubusercontent.com/72767245/104939083-51d87f80-59f3-11eb-9736-b721063bf9b3.png)

비평가 C 예측간의 차이의 절댓값 / 두 이미지의 픽셀의 평균값 차이의 절댓값  
1-립시츠 연속함수의 경우: 기울기의 절댓값의 최대는 1이다.  
함수 위 어느 점에 원뿔을 놓더라도 하얀색 원뿔에 들어가는 영역이 없다  

**가중치 클리핑**을 통해 립시츠 제약을 부과할 수 있다  
WGAN논문에서는 비평가C의 가중치를 [-0.01, 0.01]안에 놓이도록, 훈련배치가 끝난 후 가중치 클리핑을 통해 립시츠 제약을 부과하는 방법을 보임

## 가중치 클리핑

## WGAN 분석
![image](https://user-images.githubusercontent.com/72767245/104939347-a7149100-59f3-11eb-8e6e-e5d7ac449d46.png)
