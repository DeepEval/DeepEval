import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D, concatenate, Dense, Flatten
from tensorflow.keras.models import Model

def dl_model():

  # Input layer
  inputs = Input(shape=(32, 32, 3))

  # Initial convolutional layer
  conv_init = Conv2D(16, (1, 1), activation='relu')(inputs)

  # Local feature extraction branch
  conv_local = Conv2D(16, (3, 3), activation='relu')(conv_init)
  conv_local = Conv2D(16, (3, 3), activation='relu')(conv_local)

  # First downsampling branch
  pool_first = AveragePooling2D((2, 2))(conv_init)
  conv_first = Conv2D(16, (3, 3), activation='relu')(pool_first)

  # Second downsampling branch
  pool_second = AveragePooling2D((2, 2))(conv_init)
  conv_second = Conv2D(16, (3, 3), activation='relu')(pool_second)

  # Second upsampling branch
  upsample_second = UpSampling2D((2, 2))(conv_second)
  concat_second = concatenate([upsample_second, conv_first])
  conv_second_upsample = Conv2D(16, (3, 3), activation='relu')(concat_second)

  # Third upsampling branch
  upsample_third = UpSampling2D((2, 2))(conv_second)
  concat_third = concatenate([upsample_third, conv_init])
  conv_third_upsample = Conv2D(16, (3, 3), activation='relu')(concat_third)

  # Concatenate and refine outputs
  concat_all = concatenate([conv_local, conv_second_upsample, conv_third_upsample])
  conv_concat = Conv2D(16, (1, 1), activation='relu')(concat_all)

  # Fully connected layer
  flatten = Flatten()(conv_concat)
  outputs = Dense(10, activation='softmax')(flatten)

  # Create the model
  model = Model(inputs=inputs, outputs=outputs)

  return model