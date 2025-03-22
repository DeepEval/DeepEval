import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Flatten, Dense, Dropout, AveragePooling2D

def dl_model():

  # Input layer
  inputs = Input(shape=(32, 32, 3))

  # Path 1
  path_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)

  # Path 2
  path_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)
  path_2 = Conv2D(64, (1, 7), padding='same', activation='relu')(path_2)
  path_2 = Conv2D(64, (7, 1), padding='same', activation='relu')(path_2)

  # Path 3
  path_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)
  path_3 = Conv2D(64, (1, 7), padding='same', activation='relu')(path_3)
  path_3 = Conv2D(64, (7, 1), padding='same', activation='relu')(path_3)
  path_3 = Conv2D(64, (1, 7), padding='same', activation='relu')(path_3)
  path_3 = Conv2D(64, (7, 1), padding='same', activation='relu')(path_3)

  # Path 4
  path_4 = AveragePooling2D((2, 2), padding='same')(inputs)
  path_4 = Conv2D(64, (1, 1), padding='same', activation='relu')(path_4)

  # Fusion
  fusion = concatenate([path_1, path_2, path_3, path_4])

  # Fully connected layer
  fusion = Flatten()(fusion)
  fusion = Dense(512, activation='relu')(fusion)
  fusion = Dropout(0.5)(fusion)
  outputs = Dense(10, activation='softmax')(fusion)

  # Model creation
  model = Model(inputs=inputs, outputs=outputs)

  return model