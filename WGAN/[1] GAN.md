# 첫번째 GAN
## 판별자
판별자의 목표: 이미지가 진짜인지 가짜인지 예측

```
gan = GAN(input_dim=(28,28,1),
        discriminator_conv_filters = [64,64, 128, 128]
        ...
```

```python
discriminator_input = Input(shape = self.input_dim, name = 'discriminator_input')
x = discriminator_input

for i in range(self.n_layers_discriminator):
  filters = self.discriminator_conv_filters[i],
  kernel_size = self.discriminator_conv_kernel_size[i],
  strides = self.discrimiator_conv_strides[i],
  padding = 'same',
  name = 'discriminator_conv_' + str(i)
  
if self.discriminator_batch_norm_momentum and i > 0:
  x = BatchNormalization(momentum = self.discriminator_batch_norm_momentum)(x)

x = Activation(self.discriminator_activation)(x)

if self.discriminator_dropout_rate:
  x = Dropout(rate = self.discriminator_dropout_rate)(x)

x = Flatten()(x)
discriminator_ouptut = Dense(1, activation = 'sigmoid',kernel_initializer = self.weight_init)(x) 
#[0,1사이의 값 하나의 유닛을 반환

discriminator = Model(discriminator_input, discriminator_output) 
## 케라스 모델로 판별자 정의, 이 모델은 이미지를 입력받아 0과 1사이의 숫자 하나를 출력하게 된다
```

## 생성자
생성자의 입력: 다변수 표준 정규분포에서 추출한 벡터(z 벡터)  
출력의 크기: 원본 훈련 데이터의 이미지와 동일한 크기의 이미지  

✔ **잠재공간을 원본 차원으로 매핑하는 개념: 생성 모델링에서 매우 일반적** > 잠재공간의 벡터를 조작하여 원본 차원에 있는 이미지의 고수준 특성을 바꿀 수 있음  


```python
generator_input = Input(shape=(self.z_dim,), name = 'generator_input') #100랜덤 벡터
x = generator_input

x = Dense(np.prod(self.generator_initial_dense_layer_size)(x) #3136 유닛을 가진 Dense Layer
if self.generator_batch_norm_momentum:
        x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)
x = Activation(self.generator_activation)(x)
 
# 배치 정규화와 ReLU 함수를 거침

x = Reshape(self.generator_initial_dense_layer_size)(x) # 7x7x64 텐서로 reshape

if self.generator_dropout_rate:
        x = Dropout(rate = self.generator_dropout_rate)(x)
for i in range(self.n_layers_generator):
        x = UpSampling2D()(x)
        x = Conv2D(
                filters = self.generator_conv_filters[i],
                kernel_size = self.generator_conv_kernel_size[i],
                padding = 'same',
                name = 'generator_conv_' + str(i)
                )(x)
        if i < n_layers_generator -1:
                if self.generator_batch_norm_momentum:
                        x = BatchNormalization(momentum = self.generator_batch_norm_momentum))(x)
                x = Activation('relu')(x)
        else: 
                x = Activation('tanh')(x)
                # 마지막 출력에서 tanh() 활성화함수 적용 [-1, 1]사이 값 반환
generator_output = x
generator = Model(generator_input, generator_output)
# 케라스 모델로 생성자 정의 100인 벡터를 받아 [28,28,1] 크기의 텐서 출력
```
# GAN 훈련
- 훈련세트에서 진짜 샘플을 랜덤하게 선택하고 생성자의 출력을 합쳐서 훈련 세트를 만들어 판별자를 훈련
- 입력은 랜덤하게 생성한 100차원 잠재공간 벡터
- 출력은 1인 훈련 배치를 만들어 전체 모델을 훈련 
- 손실함수는 **이진 크로스 엔트로피 손실**  
<br>
- 전체 모델을 훈련할때 생성자의 가중치만 업데이트 되도록 **판별자의 가중치를 동결**해야 함

### GAN 컴파일
```python
## 판별자 컴파일 ##
# 이진크로스엔트로피 함수를 사용하여 loss 계산
self.discriminator.compile(
        optimizer = RMSprop(lr = 0.008),
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
)

## 생성자 컴파일 ##
self.discriminator.trainable = False # 판별자 가중치 동결

model_input = Input(shape=(self.z_dim,), name = 'model_input')
model_output = discriminator(self.generator(model_input))
self.model = Model(model_input, model_output) 

# 이진 크로스엔트로피 함수를 사용하여 전체 모델을 컴파일
# 일반적으로 판별자가 생성자보다 강해야되므로 학습률이 판별자 보다 느림
self.model.compile(
        optimizer = RMSprop(lr = 0.0004),
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
```

### GAN 훈련
- 판별자와 생성자의 교대로 훈련

```python
def train_discriminator(x_train, batch_size):
        # discriminator의 정답 값
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        #진짜 이미지로 훈련
        idx = np.random.randint(0, x_train.shape[0], batch_size) #batch_size 만큼 random index 추출
        true_imgs = x_train[idx]
        self.discriminator.train_on_batch(true_imgs, valid)
        
        #생성된 이미지로 훈련
        noise = np.random.normal(0, 1, (batch_size, z_dim)) 
        ## np.random.normal(0,1은 평균 0, 표준편차 1인 (batch_size, 100) size의 numpy 생성
        
        gen_imgs = generator.predict(noise)
        self.discriminator.train_on_batch(gen_imgs, fake 
        
def (batch_size):
        valid = np.ones((batch_size, 1))
        
        noise - np.random.normal(0,1(batch)size, z_dim))
        self.model.train_on_batch(noise, valid)
epochs = 2000
batch_size = 64

for epoch in range(epochs):
        train_discriminator(x_train, batch_size)
        train_generator(batch_size)
```
 두이미지 사이의 거리를 재는 ```L1노름```
 
 ```python
 def l1_compare_images(img1, img2):
        return np.mean(np.abs(img1 - img2))
 ```
✔ Upsampling
- Upsampling: 기존 픽셀 값 사용
- TransposedConv: 픽셀 사이 공간 zero


# GAN의 문제점

## 진동하는 손실
판별자와 생성자의 손실이 장기간 안정된 모습을 보여주지 못하고 큰 폭으로 진동하기 시작할 수 있음  

**손실은 장기적으로 출렁이는 것이 아닌 손실이 안정되거나 점진적으로 증가하거나 감소하는 형태를 보여야 함**


## 모드 붕괴 (생성자가 local minimum에 빠진 것)
- mode: 최빈값, 가장 빈도가 높은 값의미  
여러개의 mode 중에서 하나로만 치우쳐서 변환(ex. mnist 1,2,3,4 mode 존재, 3의 모드에만 치우쳐서 변환)  
사용자가 다양한 이미지를 만들어내지 못하고 비슷한 이미지만 계속해서 생성하는 경우  
**생성자가 판별자를 속이는 적은 수의 샘플을 찾을 때 일어남**
- 한정된 샘플 외에는 다른 샘플을 생성하지 못함

- 판별자가 가중치를 업데이트하지 않고 몇 번의 배치를 하는 동안 생성자를 훈련
- 생성자는 판별자를 항상 속이는 ```하나의 샘플(mode)```을 찾으려는 경향이 있고 잠재 공간의 모든 포인트를 이 샘플에 매핑할 수 있음
  - '손실함수의 그레이디언트가 0에 가까운 값으로 무너진다'라는 뜻  
  
하나의 포인트에 속아 넘어가지 못하도록 판별자를 다시 훈련하더라도 생성자는 판별자를 속이는 또 다른 모드를 쉽게 찾을 것임  
생성자가 이미 입력에 무감각해져서 다양한 출력을 만들 이유가 없기 때문

## 유용하지 않은 손실(Vanishing Gradient:기울기 소실)
- 생성자는 BCE를 줄이는 방향으로, 판별자는 BCE를 최대화하는 방향으로 학습을 진행하고 있음
- 판별자 D가 생성자 G보다 학습이 쉽다
<br><br>
- 생성자는 현재 판별자에 의해서만 평가되고 판별자는 계속 향상되기 때문에 훈련과정의 다른 지점에서 평가된 손실을 비교할 수 없음  
- 실제로 시간이 갈수록 이미지 품질은 확실히 향상됨에도 불구하고 생성자의 손실함수는 증가
- ```생성자의 손실과 이미지 품질 사이의 연관성의 부족```
<br><br>
- 판별자 D가 잘 학습될수록 0에 가까운 gradient을 넘겨주게 되고, 이는 생성자 G입장에서 유익하지 않은 피드백
- 생성자는 학습을 종료

## 하이퍼파라미터
판별자와 생성자의 전체구조 & 배치정규화, 드롭아웃, 학습률, 활성화 층, 합성곱 필터, 커널 크기, 스트라이드, 배치 크기, 잠재공간의 크기  
GAN은 이런 파라미터의 작은 변화에도 민감   

## GAN의 도전 과제 해결!
Wassersten GAN (WGAN) + Wasserstein GAN - Gradient Penalty(WGAN-GP)
