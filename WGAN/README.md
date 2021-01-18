# GAN
합성곱층 대신 완전 연결 층을 사용

# DCGAN
합성곱층을 사용하여 판별자의 예층 성능 분석

✔ 잠재공간을 원본 차원으로 매핑하는 개념: 생성 모델링에서 매우 일반적  
- 잠재공간의 벡터를 조작하여 원본 차원에 있는 이미지의 고수준 특성으로 바꿀 수 있다

✔ Upsampling
- Deconvolution
  - 입력 텐서의 각 pixel 마다 주변에 zero-paddig 후 conv
- transposed convolution
  - 일반적인 Convolution 연산
  - Convolution 연산에 Transpose
- Unpooling
  - Max pooling할때 Index를 기록해두었다가 해당 Index를 기반으로 pooling을 역으로 수행  
     
* Conv2DTranspose  
padding에 따라 계산이 조금씩 달라진다  
```python
padding = 'valid'
```
output = (Input-1) * stride + filter

```python
padding = 'same'
```
output = Input * stride
