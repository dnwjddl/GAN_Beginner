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

discriminator = Model(discriminator_input, discriminator_output)

```
