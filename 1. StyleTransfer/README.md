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

##CycleGAN 개요
#### 4개의 모델(2개의 생성자, 2개의 판별자)
###### 생성자
- g_AB : 도메인 A의 이미지를 도메인 B로 바꿈 
- g_BA : 도메인 B의 이미지를 도메인 A로 바꿈

###### 판별자
- d_A : 도메인A의 진짜 이미지와 g_BA의 비교
- d_B : 도메인B의 진짜 이미지와 g_AB의 비교

## 생성자 모델
#### U-Net (pix2pix에서는 이를 생성자 모델로 사용)
VAE와 비슷한 방식
- down sampling
- up sampling
#### ResNet (CycleGAN에서는 이를 생성자 모델로 사용)
