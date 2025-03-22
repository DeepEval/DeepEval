import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Dropout, BatchNormalization, concatenate

def dl_model():
  input_layer = Input(shape=(32, 32, 64))

  # Compress input channels with 1x1 convolutional layer
  conv_compress = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

  # Expand features through parallel convolutional layers
  conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_compress)
  conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_compress)

  # Concatenate outputs of parallel convolutional layers
  concat_features = concatenate([conv1, conv2])

  # Flatten the concatenated feature map
  flatten_layer = Flatten()(concat_features)

  # Fully connected layers for classification
  dense1 = Dense(units=128, activation='relu')(flatten_layer)
  dropout = Dropout(0.5)(dense1)
  dense2 = Dense(units=64, activation='relu')(dropout)
  output_layer = Dense(units=10, activation='softmax')(dense2)

  model = keras.Model(inputs=input_layer, outputs=output_layer)

  return model