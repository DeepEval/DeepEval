from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, concatenate, Input, Dense, add
from tensorflow.keras.models import Model

def dl_model():

  inputs = Input(shape=(32, 32, 3))

  # Main Path
  x1 = Conv2D(64, (1, 1), padding='same')(inputs)
  x1 = Conv2D(64, (3, 3), padding='same')(x1)

  # Branch Path
  x2 = Conv2D(64, (1, 1), padding='same')(inputs)
  x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x2)
  x2 = Conv2D(64, (3, 3), padding='same')(x2)

  x3 = Conv2D(64, (1, 1), padding='same')(inputs)
  x3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(x3)
  x3 = Conv2D(64, (3, 3), padding='same')(x3)

  # Concatenate outputs of branches
  x = concatenate([x1, x2, x3])

  # Main Path Output
  x = Conv2D(64, (1, 1), padding='same')(x)

  # Branch Path Output
  x2 = Conv2D(64, (1, 1), padding='same')(x2)
  x2 = UpSampling2D(size=(2, 2))(x2)

  x3 = Conv2D(64, (1, 1), padding='same')(x3)
  x3 = UpSampling2D(size=(4, 4))(x3)

  x = add([x, x2, x3])

  # Fully Connected Layer for Classification
  outputs = Dense(10, activation='softmax')(x)

  # Create Model
  model = Model(inputs=inputs, outputs=outputs)

  return model