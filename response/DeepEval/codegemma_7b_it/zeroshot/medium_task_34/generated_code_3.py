import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
  inputs = tf.keras.Input(shape=(32, 32, 3))

  # Feature Extraction
  x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
  x = layers.MaxPooling2D()(x)
  x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  x = layers.MaxPooling2D()(x)
  features = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

  # Feature Enhancement
  x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(features)
  x = layers.Dropout(0.5)(x)
  x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

  # Upsampling
  x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
  x = layers.UpSampling2D()(x)
  x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
  x = layers.UpSampling2D()(x)
  outputs = layers.Conv2DTranspose(10, (1, 1), activation='softmax', padding='same')(x)

  # Model Creation
  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  return model