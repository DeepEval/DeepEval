import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

  inputs = keras.Input(shape=(32, 32, 3))

  # Stage 1 - Downsampling
  x = layers.Conv2D(filters=16, kernel_size=3, activation='relu')(inputs)
  x = layers.MaxPooling2D()(x)
  x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
  x = layers.MaxPooling2D()(x)

  # Stage 2 - Feature Processing
  x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
  x = layers.Dropout(rate=0.3)(x)
  x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
  x = layers.MaxPooling2D()(x)
  x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
  x = layers.Dropout(rate=0.3)(x)
  x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)

  # Stage 3 - Upsampling and Skip Connections
  x = layers.UpSampling2D()(x)
  skip_1 = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
  x = layers.Add()([skip_1, x])

  x = layers.UpSampling2D()(x)
  skip_2 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
  x = layers.Add()([skip_2, x])

  # Final Stage - Output Layer
  outputs = layers.Conv2D(filters=10, kernel_size=1, activation='softmax')(x)

  model = keras.Model(inputs=inputs, outputs=outputs)
  return model

model = dl_model()
model.summary()