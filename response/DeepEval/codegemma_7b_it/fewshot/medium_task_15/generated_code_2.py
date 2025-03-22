import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Reshape, concatenate

def dl_model():
  # Input layer
  input_layer = Input(shape=(32, 32, 3))

  # Convolutional layer, batch normalization, and ReLU activation
  conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid')(input_layer)
  batch_norm = BatchNormalization()(conv)
  relu = Activation('relu')(batch_norm)

  # Global average pooling and fully connected layers
  gap = GlobalAveragePooling2D()(relu)
  flatten = Flatten()(gap)
  dense1 = Dense(units=64, activation='relu')(flatten)
  dense2 = Dense(units=32, activation='relu')(dense1)

  # Reshape output to match initial feature maps
  reshaped = Reshape(target_shape=(1, 1, 64))(dense2)

  # Weighted feature maps generation
  weighted_features = multiply([reshaped, conv])

  # Concatenate weighted features with input layer
  concat = concatenate([weighted_features, input_layer])

  # 1x1 convolution and average pooling for dimensionality reduction and downsampling
  conv_down = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid')(concat)
  avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_down)

  # Single fully connected layer for classification
  flatten_down = Flatten()(avg_pool)
  output_layer = Dense(units=10, activation='softmax')(flatten_down)

  # Model construction
  model = keras.Model(inputs=input_layer, outputs=output_layer)

  return model