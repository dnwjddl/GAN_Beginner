# pix2pix 
각 이미지가 source와 target 도메인에 모두 존재해야 한다. <br>
데이터셋이 **쌍을 이룬 이미지** 여야함, 한방향으로(source to target)으로 작동 <br>
paired 이미지가 discriminator에 입력되고 discriminator는 fake와 real를 구분함 <br>
<br>
네트워크의 구조는 **U-Net**을 사용하였고, discriminator는 **PatchGAN**의 방법을 사용<br>
PatchGAN: 이미지 전체를 보고 판단하는 것이 아닌 patch형태로 나눠서 분석<br>
patch형태로 본다면 어떤 patch가 loss가 큰지를 generator는 알고 그 부분을 더 잘 학습함<br>
<br>
Loss Function: L1 loss + GAN loss 사용

# CycleGAN
데이터셋이 **쌍을 이루지 않은 이미지**이다 <br>
양방향으로 동시에 모델 훈련이 가능<br>

## CycleGAN 개요
#### 4개의 모델(2개의 생성자, 2개의 판별자)
###### 생성자
- g_AB : 도메인 A의 이미지를 도메인 B로 바꿈 
- g_BA : 도메인 B의 이미지를 도메인 A로 바꿈

###### 판별자
- d_A : 도메인A의 진짜 이미지와 g_BA의 비교
- d_B : 도메인B의 진짜 이미지와 g_AB의 비교

![image](https://user-images.githubusercontent.com/72767245/103436168-17818b00-4c5c-11eb-85fa-d16989def2b0.png)


## 생성자 모델
### U-Net (pix2pix에서는 이를 생성자 모델로 사용)
VAE와 비슷한 방식
- down sampling : 입력 이미지를 공간방향으로 압축, 채널방향으로 확장
- up sampling : 공간방향으로 표현을 확장, 채널방향으로는 압축

VAE와 다른 방식
- U-Net은 down sampling과 up sampling 사이에 **skip connection**이 존재함
- down sampling은 각 층 모델의 이미지가 점차 무엇인지 감지하는 반면, 위치 정보는 잃음
- up sampling에서 down sampling내 잃었던 공간정보를 되돌림 <br>
(image segmentation이나 style transfer는 upsampling이 중요)

![image](https://user-images.githubusercontent.com/72767245/103436583-88c33d00-4c60-11eb-880a-8fe545b17333.png)

#### ```Skip connection```
- 이 연결은 다운샘플링 과정에서 감지된 고수준 추상 정보(image style)를 네트워크의 앞쪽 층으로부터 전달된 구체적인 공간 정보(image contents)을 섞음
- **Concatenate** 층
   - 층을 접합하는 역할로 학습되는 가중치는 없음
   - 업샘플링과 동일한 크기의 출력을 내는 다운샘플링층 연결
   - 채널의 차원에서 합쳐지므로 채널의 수가 두배가 됨 (공간방향은 동일하게 유지)
   
#### ```InstanceNormalization층```
- BatchNormalization층 대신 사용 Instance Normalization층 <br>
(StyleTransfer 문제에서 더 만족스러운 결과)

- Instance Normalization층은 배치 단위가 아니라 **개별 샘플을 각각 정규화**
   - 스케일(gamma)와 이동(beta) 파라미터 불필요 (학습되는 가중치도 없음)
   - 각 층을 정규화하기 위하여 사용되는 평균과 표준편차는 채널별로 나누어 샘플별로 계산됨

- **정규화 네가지 방법**
   - 배치 정규화 : 배치 단위로 정규화
      - feature들의 평균 및 분산 값을 Batch 단위로 계산, batch크기가 작아지면 구하는 평균과 분산은 dataset전체에 대표할 수 없다
   - 층 정규화 : 각 채널과 영상전체를 normalize
   - 샘플 정규화 : 각 채널 단위로 normalize
   - 그룹 정규화 : 각 채널을 N개 단위로 group화하여 normalize
      - LN과 IN은 batch 크기에 독립적으로 동작할 수 있고, 특정 모델들에 대해서는 상당히 잘 동작하는 기술이지만 visual recognition 분야에서는 좋은 성능을 보이지 않음
      - Batch 크기가 작은 상황에서 기존의 BN보다 GN이 더 좋은 성능을 보여줌
![image](https://user-images.githubusercontent.com/72767245/103438684-bb792f80-4c78-11eb-8681-78ce9ca525c6.png)
###### N: 배치 축, C: 채널 축, (H, W): 공간 축 
<br>
U-Net 생성자 만들기<br>

```python
def build_generator_unet(self):
   def downsample(layer_input, filters, f_size = 4):
      d = Conv2D(filters, kernel_size = f_size, strides = 2, padding = 'same')(layer_input)
      d = InstanceNormalization(axis = -1, center = False, scale = False)(d)
      d = Activation('relu')(d)
      return d
   
   def upsample(layer_input, skip_input, filters, f_size = 4, dropout_rate = 0):
      u = UpSampling2D(size = 2)(layer_input) #채널의 크기는 불변
      u = Conv2D(filters, kernel_size = f_size, strides = 1, padding = 'same')(u) #첫번째 업샘플링, 채널 128개로 줄어듦
      u = InstanceNormalization(axis = -1, center = False, scale = False)(u) 
      u = Activation('relu')(u)
      if dropout_rate:
         u = Dropout(dropout_rate)(u)
      u = Concatenate()([u, skip_input]) #첫번째 업샘플링, 채널 256개로 늘어남
      return u
   
   #이미지 입력
   img = Input(shape = self.img_shape)
   
   #다운샘플링
   d1 = downsample(img, self.gen_n_filters)
   d2 = downsample(d1, self.gen_n_filters*2)
   d3 = downsample(d2, self.gen_n_filters*4)
   d4 = downsample(d3, self.gen_n_filters*8)   
   
   #업샘플링
   u1 = upsample(d4, d3, self.gen_n_filters*4) #256 -> 128 -> 256
   u2 = upsample(u1, d2, self.gen_n_filters*2) #128 -> 64 -> 128
   u3 = upsample(u2, d1, self.gen_n_filters) #64 -> 32 -> 64
   u4 = UpSampling2D(size = 2)(u3) #3개의 층
   
   output = Conv2D(self.channels, kernel_size = 4, stride = 1, padding = 'same', activation = 'tanh')(u4)
   
   return Model(img, output)

```


### ResNet (CycleGAN에서는 이를 생성자 모델로 사용)


## 판별자
- 보통 하나의 숫자 출력, 입력이미지가 진짜일 예측 확률
- CycleGAN의 판별자는 숫자 하나가 아니라 **16x16크기의 채널 하나를 가진 텐서**를 출력
<br>

- **Pix2pix**에서는 판별자로 PatchGAN사용
- **CycleGAN**은 PatchGAN의 판별자 구조를 승계함
   - PatchGAN은 이미지 전체에 대한 예측이 아닌 중첩된 "patch"로 나누어 각 패치에 대한 진위여부 결정
   - 그러므로 하나의 출력값이 아닌 **각 패치에 대한 예측 확률을 담은 텐서**가 출력됨
      - 네트워크에 이미지를 전달하면 패치들은 한꺼번에 예측 (합성곱 구조로 인해 자동으로 이미지가 패치로 나뉨)
      
 **PatchGAN**
 - 내용이 아닌 스타일을 기반으로 판별자가 얼마나 잘 구별하는지 손실함수가 측정 가능
 - 판별자 예측의 개별 원소는 이미지의 일부 영역을 기반으로 함
```python
def build_discriminator(self):
   def conv4(layer_input, filters, stride = 2, norm = True):
      y = Conv2D(filters, kernel_size = 4, strides = stride, padding = 'same')(layer_input)
      
      if norm:
         y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
      
      y = LeakyReLU(0.2)(y)
      
   img = Input(shape = self.img_shape)
   
   y = conv4(img, self.disc_n_filters, stride = 2, norm = False)
   y = conv4(y, self.disc_n_filters*2, stride = 2)
   y = conv4(y, self.disc_n_filters*4, stride = 2)
   y = conv4(y, self.disc_n_filters*8, stride = 1)
   
   output = Conv2D(1, kernel_size = 4, strides = 1, padding = 'same')(y)
   
   return Model(img, output)
```
###### CycleGAN의 판별자는 연속된 합성곱 신경망 (첫번째층 제외하고) 모두 샘플정규화를 사용
###### 마지막 합성곱 층은 하나의 필터를 사용하고 활성화함수는 사용하지 않음

## CycleGAN 컴파일

```python
self.d_A = self.build_discriminator()
self.d_B = self.build_discriminator()
self.d_A.compile(loss = 'mse', optimizer = Adam(self.learning_rate, 0.5), metrics = ['accuracy'])
self.d_B.compile(loss = 'mse', optimizer = Adam(self.learning_rate, 0.5), metrics = ['accuracy'])
```

**세가지 조건으로 생성자를 동시에 평가**
- 1. **유효성** : 각 생성자에서 만든 이미지가 대응되는 판별자를 속이는가?
- 2. **재구성** : 두 생성자를 교대로 적용하면(양방향 모두에서) 원본 이미지를 얻는가?
- 3. **동일성** : 각 생성자를 자신의 타깃 도메인에 있는 이미지에 적용했을 때 이미지가 바뀌지 않고 그대로 남아 있는가?

생성자 훈련하기 위해 결합된 모델 만들기

- 6개의 출력이 만들어짐, 판별자의 가중치를 동결. 따라서 판별자가 모델에 관여하지만 결합된 모델은 ```생성자의 가중치```만 훈련한다
- 전체 손실은 ```각 조건에 대한 손실의 가중치 합```이다
- **평균 제곱 오차**는 유효성 조건에 사용, 진짜(1)과 가짜(0) 타깃에 대한 판별자의 출력을 확인
- **평균 절댓값 오차**는 이미지대 이미지 조건에 사용(재구성과 동일성 조건)

```python
self.g_AB = self.build_generator_unet()
self.g_BA = self.build_generator_unet()

self.d_A.trainable = False #이게 뭐지
self.d_B.trainable = False

img_A = Input(shape = self.img_shape)
img_B = Input(shape = self.img_shape)

fake_A = self.g_BA(img_B)
fake_B = self.g_AB(img_A)

valid_A = self.d_A(fake_A)
valid_B = self.d_B(fake_B) #유효성

reconstr_A = self.g_BA(fake_B)
reconstr_B = self.g_AB(fake_A) #재구성

img_A_id = self.g_BA(img_A)
img_B_id = self.g_AB(img_B) #동일성 (배경, 큰 틀을 유지하기 위해 제약을 둠, 이미지의 변환이 필요한 부분 이외에는 바꾸지 않도록)

self.combined = Model(inputs = [img_A, img_B], outputs = [valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id])

self.combined.compile(loss = ['mse','mse', 'mae', 'mae', 'mae', 'mae'], 
                     loss_weights = [self.lambda_validation,
                                      self.lambda_validation,
                                      self.lambda_reconstr,
                                      self.lambda_reconstr,
                                      self.lambda_id,
                                      self.lambda_id ],
                      optimizer = optimizer)       
                              
```

## CycleGAN 훈련


```python
batch_size = 1 #일반적으로 CycleGAN의 batch 사이즈는 1임
patch = int(self.img_rows/2**4)
self.disc_patch = (patch, patch, 1)

valid = np.ones((batch_size,) + self.disc_patch)  #진짜 이미지들은 타깃이 1
fake = np.ones((batch_size,) + self.disc_patch)  #가짜 이미지들은 타깃이 0

for epoch in range(self.epoch, epochs):
   for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(batch_size)):
      #생성자를 사용하여 가짜 이미지 배치 생성
      fake_B = self.g_AB.predict(imgs_A)
      fake_A = self.g_BA.predict(imgs_B)
      
      #가짜 이미지와 진짜 이미지 배치로 각 판별자를 훈련
      dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
      dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
      dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
      
      dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
      dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
      dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
      
      d_loss = 0.5 * np.add(dA_loss, dB_loss)
      
      #생성자는 앞서 컴파일된 결합 모델을 통해 동시에 훈련됨. 6개의 출력은 컴파일 단계에서 정의한 6개의 손실함수에 대응
      g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B])
```
**순환 일관성 손실함수(Cycle Consistency Loss Function)사용**
원본 이미지 ```a``` <br>
유효성 확인 ```G_AB(a)``` [판별자]0일때 ```D_B(G_AB(a))```, 1일때 ```D_B(b)``` <br>
재구성 확인 ```G_BA(G_AB(a))``` <br>
동일성 확인 ```G_BA(a)``` <br>

## ResNet


```잔차블록(Residual block)```을 쌓아서 구성
각 블록은 다음층으로 출력을 전달하기 전에 입력과 출력을 합하는 스킵 연결을 가지고 있음

```python
from keras.layers.merge import add

def residual(layer_input, filters):
   shortcut = layer_input
   y = Conv2D(filters, kernel_size = (3,3), strides = 1, padding = 'same')(layer_input)
   y = InstanceNormalization(axis = -1, center = False, scale =False)(y)
   y = Activation('relu')(y)
   
   y = Conv2D(filters, kernel_size = (3,3), strides = 1, padding = 'same')(y)
   y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
   
   return add([shortcut, y])
```

```ResNet```구조는 앞쪽의 층에도 도달하는 그레이디언트가 작아져 매우 느리게 훈련되는 **그레이디언트 소실** 문제가 없다.  
오차 그레이디언트가 Residual Block의 Skip connection을 통해 네트워크에 그대로 역전파 되기때문  
층을 추가해도 모델의 정확도를 떨어뜨리지 않음  
추가적인 특성이 추출되지 않는다면 skip connection으로 인해 언제든지 이전 층의 특성이 항등사상(identity mapping)을 통과하기 때문

✔```항등 사상(identity mapping)```  
CycleGAN의 잔차 블록은 skip connection을 합친 후에 적용하는 활성화 함수가 없어서 이전 층의 feature map이 그대로 다음 층으로 전달됨

# Neural Style Transfer
훈련세트를 사용하지 않고 이미지의 스타일을 다른 이미지로 전달  

- 콘텐츠 손실(content loss)
합성된 이미지는 베이스 이미지의 콘텐츠를 동일하게 포함해야된다
- 스타일 손실(style loss)
합성된 이미지는 스타일 이미지와 동일한 일반적인 스타일을 가져야 합니다
- 총 변위 손실(total variation loss)
합성된 이미지는 픽셀처럼 격자 문의가 나타나지 않고 부드러워야함

## 콘텐츠 손실
```python
from keras.applications import vgg19
from keras import backend as K

base_image_path = '/path_to_images/base_image.jpg'
style_reference_image_path = '/path_to/images/styled_image.jpg'
content_weight = 0.01

base_image = K.variable(preprocess_image(base_image_path))
style_reference_imiage = K.variable(preprocess_image(style_reference_image_path))
combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

input_tensor = K.concatenate([base_image, style_reference_image, combination_image], axis=0)
model = vgg19.VGG19(input_tensor = input_tensor, weights='imagenet',include_top=False)

output_dict=dict([(layer.name, layer.output) for layer in model.layers])
layer_features = outputs_dict['block5_conv2']

base_image_features = layer_features[0,:,:,;]
combination_features = layer_features[2, :,:,:]

def content_loss(content, gen):
   return K.sum(K.square(gen-content))
   
content_loss= content_weight * content_loss(base_image_features, combination_features)
```
## 스타일 손실

```python
style_loss = 0.0
def gram_matrix(x):
   features = K.batch_flatten(K.permute_dimensions(x, (2,0,1)))
   gram = K.dot(features, K.transpose(features))
   return gram
   
def style_loss(style, combination):
   S = gram_matirx(style)
   C = gram_matrix(combination)
   channels = 3
   size = img_nrows * img_ncols
   return K.sum(K.square(S - C))/(4.0 * (channels ** 2) * (size**2))
   
feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

for layer_name in feature_layers:
   layer_features = outputs_dict[layer_name]
   style_reference_features = layer_features[1, :,:,:]
   combination_features = layer_features[2,:,:,:]
   sl = style_loss(style_reference_features, combination_features)
   style_loss +=(style_weight/len(feature_layers)) * sl

```
## 총 변위 손실

```python
def total_variation_loss(x):
   a = K.square(
      x[:, :img_nrows -1, :img_ncols -1, :] -x[:, 1:, :img_ncols-1, :])
   b = K.square(
      x[:, :img_nrows -1, :img_ncols -1, :] -x[:, :img_nrows-1, 1:, :])
   return K.sum(K.pow(a+b, 1.25))

tv_loss = total_variation_weight * total_variation_loss(combination_image)
loss = content_loss + style_loss + tv_loss
```

# Neural Transfer 실행
```python
from scipy.optimize import fmin_l_bfgs_b

iterations = 1000
x = preprocess_image(base_image_path)

for i in range(iterations):
   x, min_val, info = fmin_l_bfgs_b(
      evaluator.loss,
      x.flatten(),
      fprime = evaluator.grads(),
      maxfun = 20
   )
```
