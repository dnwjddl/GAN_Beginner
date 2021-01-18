# WGAN-GP

## WGAN과 WGAN-GP의 차이점
**비평자의 정의와 컴파일 단계**  
- 비평자 손실 함수에 그레이디언트 페널티 항을 포함
- 비평자의 가중치를 클리핑하지 않음
- 비평자에 배치 정규화 층을 사용하지 않음

비평자에게 립시츠 제약을 강제하는 다른 방법을 찾아서 제한  
비평자의 가중치를 클리핑하는 대신 비평자의 그레이디언트 NORM이 1에서 크게 벗어날 때 모델에 페널티를 부과하는 항을 손실 함수에 포함하여 제약을 직접 부과

### 그레이디언트 패널티 손실

전체 손실함수 = 진짜 이미지와 가짜 이미지에 대한 **와서스테인 손실** + **그레이디언트 페널티 손실**


- 그레이디언트 패널티 손실은 입력 이미지에 대한 예측의 그레이디언트 노름과 1사이의 차이를 제곱한 것
- 이것은 자연스럽게 그레이디언트 페널티 항을 최소화하는 중치를 찾으려고 함
  - 립시츠 제약을 따름
- 모든 곳에서 그레이디언트를 계산하는 것이 아닌 일부지점에서만 그레이디언트 계산
  - 진짜 이미지와 가짜 이미지 쌍을 연결한 직선을 따라 무작위로 포인트를 선택해 **보간한 이미지**사용
 
![image](https://user-images.githubusercontent.com/72767245/104947898-c6192000-59ff-11eb-9800-f98eb8c8af36.png)

```python
#### 랜덤한 보간을 수행하는 층 ####
class RandomWeightedAverage(_Merge):
   def __init__(self, batch_size):
      super().__init__()
      self.batch_size = batch_size
   def _merge_function(self, inputs):
      alpha = K.random_uniform((self.batch_size, 1, 1, 1)) #배치에 있는 각 이미지는 0과 1사이의 랜덤한 수치를 얻어 alpha벡터에 저장
      return (alpha * inputs[0]) + ((1-alpha) * inputs[1]) # 이 층은 진짜 이미지(inputs[0])와 가짜 이미지(inputs[1])쌍을 연결하는 직선 위에 놓인 픽셀 기준의 보간된 이미지를 반환
      #각 쌍의 가중치는 alpha 값으로 결정
```

```python
#### 그레이디언트 페널티 손실함수 ####

## 기울기가 1에 가까우면서 weight clip 방지
## interpolated_samples :보간된 이미지
def gradient_penalty_loss(y_true, y_pred, interpolated_samples):
  gradients = K.gradients(y_pred, interpolated_samples)[0]
  
  # l2 norm : 유클리드 거리 계산
  gradient_l2_norm = K.sqrt(
    K.sum(
      K.square(gradients),
      axis = [1:len(gradients.shape)]
    )
  )
  gradient_penalty = K.square(1- gradient_l2_norm)
  return K.mean(gradient_penalty) #l2 norm과 1사이의 거리 제곱 반환

```

## WGAN-GP 컴파일
예측과 진짜 레이블 두 개의 매개변수만 허용 > 이 이슈를 해결하기 위한 partial 함수 사용
```python
from functools import partial

## 비평자 모델 컴파일
self.generator.trainable = False #생성자의 가중치를 동결

real_img = Input(shape = self.input_dim)
z_disc = Input(shape = (self.z_dim,))
fake_img = self.generator(z_disc)

# 비평자 통과
fake = self.critic(fake_img)
valid = self.critic(real_img)

# 보간된 이미지를 만들고 다시 비평자에 통과
interpolated_img = RandomWeightedAverage(self.batch_size)([real_img, fake_img])
validity_interpolated = self.critic(interpolated_img)

# 케라스의 손실함수는 예측과 진짜 레이블 두 개의 입력만을 기대함
## partial 함수를 사용해 보간될 이미지를 gradient_penalty_loss 함수에 적용한 사용자 정의 함수 partial_gp_loss 정의
partial_gp_loss = partial(self.gradient_penalty_loss, interpolated_samples = interpolated_img)

partial_gp_loss.__name__ = "gradient_penalty"

# 비평자는 훈련하기 위한 모델에 두개의 입력을 정의, 하나는 진짜 이미지의 배치, 또하나는 가짜 이미지를 생성하는 랜덤 벡터
# 출력 세가지(진짜이미지는 1, 가짜 이미지는 -1, 더미  0 벡터)
# 0 벡터는 케라스의 모든 손실 함수가 반드시 출력에 매핑되어야 하기 때문에 필요하지만 실제로 사용되지는 않음
# 따라서 partial_gp_loss 함수에 매핑되는 더미 0벡터를 만듦
self.critic_model = Model(inputs = [real_img, z_disc], outputs = [valid, fake, validity_interpolated])


# 진짜 이미지와 가짜 이미지에 대한 두개의 와서스테인 손실과 그레이디언트 페널티 손실 총 세개의 손실함수로 비평자를 컴파일
# 전체 손실은 이 세가지 손실의 합
# 논문에 따라, 그레이디언트 손실에 10배 가중치를 부여
# WGAN-GP 모델에 가장 적합하다고 알려진 Adam Optimizer 사용
self.critic_model.compile(
  loss = [self.wasserstein, self.wasserstein, partial_gp_loss],
  optimizer = Adam(lr = self.critic_learning_rate, beta_1 = 0.5),
  loss_weight = [1,1,self.grad_weight]
  )
```
- 배치 정규화는 배치 안의 이미지 사이에 상관관계를 만들기 때문에 그레이디언트 페널티 손실의 효과가 떨어진다
- 비평자에 배치 정규화 사용해선 안된다

