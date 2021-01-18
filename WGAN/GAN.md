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
generator_input = Input(shape=(self.z_dim,), name = 'generator_input')
x = generator_input

x = Dense(np.prod(self.generator_initial_dense_layer_size)(x)
if self.generator_batch_norm_momentum:
        x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)
x = Activation(self.generator_activation)(x)

x = Reshape(self.generator_initial_dense_layer_size)(x)

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
generator_output = x
generator = Model(generator_input, generator_output)
```

✔ Upsampling
