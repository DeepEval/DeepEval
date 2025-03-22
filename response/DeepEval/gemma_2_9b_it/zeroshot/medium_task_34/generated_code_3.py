import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
  model = tf.keras.Sequential([
      # Feature Extraction
      layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
      layers.MaxPooling2D(pool_size=(2, 2)),

      # Generalization Enhancement
      layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
      layers.Dropout(0.5),
      layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),

      # Upsampling and Spatial Information Recovery
      layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=2, activation='relu', padding='same'),
      layers.Add()([layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=2, activation='relu', padding='same'), layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')]),
      layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=2, activation='relu', padding='same'),
      layers.Add()([layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=2, activation='relu', padding='same'), layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')]),
      layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=2, activation='relu', padding='same'),
      layers.Add()([layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=2, activation='relu', padding='same'), layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')]),

      # Output Layer
      layers.Conv2D(10, kernel_size=(1, 1), activation='softmax') 
  ])
  return model