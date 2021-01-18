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
