import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Conv2DTranspose, GlobalAveragePooling2D, Dense, Reshape, multiply, Concatenate, BatchNormalization, Flatten, Activation, Dropout, LeakyReLU
from keras.models import Model

def dl_model():
  input_layer = Input(shape=(32, 32, 3))

  # Increase dimensionality of input channels
  conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

  # Extract initial features with depthwise separable convolution
  depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(conv_1x1)
  batch_norm = BatchNormalization()(depthwise_conv)
  activation = Activation('relu')(batch_norm)

  # Compute channel attention weights
  avg_pool = GlobalAveragePooling2D()(activation)
  dense_1 = Dense(units=64, activation='relu')(avg_pool)
  dense_2 = Dense(units=64, activation='sigmoid')(dense_1)

  # Reshape weights to match initial features
  reshape_weights = Reshape((1, 1, 64))(dense_2)

  # Channel attention weighting
  multiply_output = multiply([activation, reshape_weights])

  # Reduce dimensionality and combine with initial input
  conv_1x1_attn = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(multiply_output)
  concat = Concatenate()([conv_1x1, input_layer])

  # Classification layers
  flatten_layer = Flatten()(concat)
  dense3 = Dense(units=256, activation='relu')(flatten_layer)
  dense4 = Dense(units=128, activation='relu')(dense3)
  output_layer = Dense(units=10, activation='softmax')(dense4)

  model = Model(inputs=input_layer, outputs=output_layer)

  return model