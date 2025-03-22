import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add

def dl_model():
  # Input layer for CIFAR-10 images
  input_layer = Input(shape=(32, 32, 3))

  # Main Path
  x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
  x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
  x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)

  # Branch Path
  y = GlobalAveragePooling2D()(input_layer)
  y = Dense(64, activation='relu')(y)
  y = Dense(32, activation='relu')(y)
  y = Reshape((1, 1, 32))(y)

  # Combining paths
  combined = Add()([x, y])
  combined = Dense(64, activation='relu')(combined)
  combined = Dense(32, activation='relu')(combined)
  output_layer = Dense(10, activation='softmax')(combined)

  # Create the model
  model = keras.Model(inputs=input_layer, outputs=output_layer)

  return model