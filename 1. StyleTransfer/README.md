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

#### Skip connection
- 이 연결은 다운샘플링 과정에서 감지된 고수준 추상 정보(image style)를 네트워크의 앞쪽 층으로부터 전달된 구체적인 공간 정보(image contents)을 섞음
- **Concatenate** 층
   - 층을 접합하는 역할로 학습되는 가중치는 없음
   - 업샘플링과 동일한 크기의 출력을 내는 다운샘플링층 연결
   - 채널의 차원에서 합쳐지므로 채널의 수가 두배가 됨 (공간방향은 동일하게 유지)
   
#### InstanceNormalization층
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
 - 내용이 닌 스타일을 기반으로 판별자가 얼마나 잘 구별하는지 손실함수가 측정 가능
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
