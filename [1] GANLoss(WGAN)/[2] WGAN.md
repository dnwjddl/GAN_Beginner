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

### Earth Mover's Distance
- GAN의 목표인 ```Pdata(x)와 동일하도록 Pmodel(x)을 학습```은 Pdata(x) 와 Pmodel(x) **두 분포 사이의 거리를 줄이는 것**
- Earth Mover's Distance는 두 분포가 얼마나 다른지를 나타내는 수치
- Pmodel(x)을 Pdata(x)와 동일하게 만들기 위해 이동해야 하는 거리와 양을 의미
<br><br>
- Earth Mover's Distance의 결과는 0과 1의 한계가 없음
- 분포가 얼마나 멀리 떨어져있든 상관없이 의미있는 gradient(0이 아닌 기울기)를 전달 가능
![image](https://user-images.githubusercontent.com/72767245/104936517-fbb60d00-59ef-11eb-8ca3-dd8aa48322df.png)

##### KL/JS divergence 나 TV의 경우 두 분포가 서로 겹치지 않은 경우에는 0, 겹치는 경우에는 무한대 또는 상수로 극단적인 거리의 값을 나타냄
##### 이는 discriminator와 generator가 분포를 학습할 때 위 세가지 distance 기반으로 학습을 하게 되면 굉장히 어려움을 겪을 것 (초반에는 실제 데이터의 분포와 겹치지 않으므로 무한대 또는 일정한 상수값을 갖다가, 갑자기 0으로 변해버리므로 gradient 가 제대로 전달되지 않음)
##### 반면, EM의 경우 분포가 겹치던 겹치지 않던 간에 |θ|를 유지하므로, 학습에 사용하기 쉽다는 것
- **Vanishing Gradient** 해결, **Model collapse** 가능성 감소

![image](https://user-images.githubusercontent.com/72767245/104936900-7ed76300-59f0-11eb-8640-b5d505ced736.png)


#### GAN 판별자와 생성자 손실 최소화
Earth Mover's Distance 사용

![image](https://user-images.githubusercontent.com/72767245/104924284-90187380-59e0-11eb-8cb4-b7e638f1577d.png)

가치함수 Pr과 Pg사이의 JS Divergence를 최소화할 수 있지만 그렇게 하면 discriminator가 포화되는 vanishing gradients 문제 유발

### WGAN Loss
**Discriminator는 진짜와 가짜를 판별하기 위해 sigmoid를 사용하고, output은 진짜/가짜에 대한 예측 확률값이다. 반면 Critic은 EM(Earth Mover)distance로부터 얻은 scalar값을 이용
EM distance는 확률 분포 간의 거리를 측정하는 척도 중 하나인데, 그동안 일반적으로 사용된 척도 KL divergence는 매우 strict 하게 거리를 측정하는 법이라서 continuous 하지 않은 경우가 있고 학습이 어렵다**
- 1. 와서스테인 손실은 1과 0 대신 **y = 1, y= -1** 사용 
- 2. 판별자의 마지막 층에서 **시그모이드 활성화 함수를 제거**하여 예측 p가 [0,1]범위에 국한되지 않고 [-무한, 무한]범위의 어떤 숫자도 될 수 있도록 함
- 3. WGAN의 판별자는 보통 ```Critic(비평자)``` 라고 부름

### 와서스테인 손실
![image](https://user-images.githubusercontent.com/72767245/104938656-b6dfa580-59f2-11eb-9474-10476a7bbf0a.png)

![image](https://user-images.githubusercontent.com/72767245/104938900-1047d480-59f3-11eb-837c-5a06cd0c3e89.png)


WGAN은 GAN 보다 더 낮은 학습률을 사용

```python
def (y_true, y_pred):
  return -K.mean(y_true * y_pred)

critic.compile(
  optimizer = RMSprop(lr = 0.00005),loss = wasserstein
  )
model.compile(
  optimizer = RMSprop(lr = 0.00005), loss = wasserstein
)
```
- Earth Mover's Distance의 출력이 [0,1]로 제한되지 않아 loss에 사용하는 것이 적적하다는 것을 알아보았는데, 일반적으로 신경망에서 너무 큰 숫자는 피해야 하기 때문에 Lipshitz 제약이라는 제약조건을 걸어줌
- 비평가(C)가 1- Lipshiz continuous Function(1-립시츠 연속함수)이어야함

## 립시츠 제약
![image](https://user-images.githubusercontent.com/72767245/104939083-51d87f80-59f3-11eb-9736-b721063bf9b3.png)  
**립시츠 함수는 임의의 두 지점의 기울기가 어떤 상숫값 이상 증가하지 않는 함수, 이 상수가 1일때 1-립시츠 함수라고 함**  
비평가 C 예측간의 차이의 절댓값 / 두 이미지의 픽셀의 평균값 차이의 절댓값  
1-립시츠 연속함수의 경우: 기울기의 절댓값의 최대는 1이다.  

![image](https://user-images.githubusercontent.com/72767245/104942113-5c951380-59f7-11eb-977e-2bac6102a64f.png)  
함수 위 어느 점에 원뿔을 놓더라도 하얀색 원뿔에 들어가는 영역이 없다  

## 가중치 클리핑
**가중치 클리핑**을 통해 립시츠 제약을 부과할 수 있다  
WGAN논문에서는 비평가C의 가중치를 [-0.01, 0.01]안에 놓이도록, 훈련배치가 끝난 후 가중치 클리핑을 통해 립시츠 제약을 부과하는 방법을 보임

```python
def train_critic(x_train, batch_size, clip_threshold):
  valid = np.ones((batch_size, 1))
  fake = - np.ones((batch_size, 1))
  
  #진짜 이미지로 훈련
  idx = np.random.randint(0, x_train.shape[0], batch_size)
  true_imgs = x_train[idx]
  self.critic.train_on_batch(true_imgs, valid)
  
  #생성된 이미지로 훈련
  noise = np.random.normal(0, 1, (batch_size, self.z_dim))
  gen_imgs = self.generator.predict(noise)
  self.critic.train_on_batch(gen_imgs, fake)
  
  for l in critic.layers:
    weights = l.get_weights()
    weights = [np.clip(w, -clip_threshold, clip_threshold) for w in weights]
    l.set_weights(weights)
```

## WGAN 분석
![image](https://user-images.githubusercontent.com/72767245/104939347-a7149100-59f3-11eb-8e6e-e5d7ac449d46.png)

생성자와 판별자의 균형을 맞추기 위하여 WGAN은 생성자를 업데이트 하는 중간에 판별자를 여러 번 훈련하고 수렴에 가깝게 만듦  
**일반적으로 생성자를 한 번 업데이트할 때 판별자를 다섯 번 업데이트**   
적대적관계이기 떄문에 판별자를 강하게 만들어주고 생성자도 뒤따라서 강하게 훈련

```python
for epoch in range(epochs):
  for _ in range(5):
    train_critic(x_train, batch_size = 128, clip_threshold = 0.01)
  train_generator(batch_size)
```

# 정리
## WGAN은 Wasserstein Loss을 사용
## WGAN은 진짜에는 레이블이 1, 가짜에는 레이블이 -1을 사용하여 훈련
## WGAN 비평자의 마지막 층은 시그모이드 활성화함수가 필요 하지 않음
## 매 업데이트 후에 판별자의 가중치를 클리핑
## 생성자를 업데이트할때마다 판별자를 여러번 훈련
## 주요 단점: 비평자에서 가중치를 클리핑했기 때문에 학습속도가 크게 감소
### 립시츠 제약을 두기 위한 가중치 클리핑은 안좋은 방법
