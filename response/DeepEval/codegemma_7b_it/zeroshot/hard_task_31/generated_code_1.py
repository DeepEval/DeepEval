from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2

def dl_model():
  # Input layer
  inputs = keras.Input(shape=(32, 32, 3))

  # First block
  x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-4))(inputs)
  x = layers.Dropout(rate=0.2)(x)
  x = layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-4))(x)
  branch_path = keras.Input(shape=(32, 32, 3))
  branch_path = layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-4))(branch_path)
  main_path = layers.Add()([x, branch_path])
  main_path = layers.BatchNormalization()(main_path)
  main_path = layers.Activation('relu')(main_path)

  # Second block
  x = layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-4))(main_path)
  x = layers.Dropout(rate=0.2)(x)
  x = layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-4))(x)
  x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-4))(x)
  x = layers.Dropout(rate=0.2)(x)
  x = layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-4))(x)
  x = layers.Conv2D(filters=64, kernel_size=7, strides=1, padding='same', activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-4))(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)

  # Output layer
  outputs = layers.Flatten()(x)
  outputs = layers.Dense(units=10, activation='softmax')(outputs)

  # Model creation
  model = Model(inputs=[inputs, branch_path], outputs=outputs)

  return model