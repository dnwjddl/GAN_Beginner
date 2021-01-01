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
   - 층 정규화 : 각 채널과 영상전체를 normalize
   - 샘플 정규화 : 각 채널 단위로 normalize
   - 그룹 정규화 : 각 채널을 N개 단위로 group화하여 normalize
      - LN과 IN은 batch 크기에 독립적으로 동작할 수 있고, 특정 모델들에 대해서는 상당히 잘 동작하는 기술이지만 visual recognition 분야에서는 좋은 성능을 보이지 않음
      - Batch 크기가 작은 상황에서 기존의 BN보다 GN이 더 좋은 성능을 보여줌
   ![image](https://user-images.githubusercontent.com/72767245/103438684-bb792f80-4c78-11eb-8681-78ce9ca525c6.png)

### ResNet (CycleGAN에서는 이를 생성자 모델로 사용)
