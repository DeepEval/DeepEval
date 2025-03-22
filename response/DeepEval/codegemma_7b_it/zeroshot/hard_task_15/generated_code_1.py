import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
  inputs = keras.Input(shape=(32, 32, 3))

  # Main path
  x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
  x = layers.MaxPooling2D()(x)
  x = layers.Conv2D(64, (3, 3), activation='relu')(x)
  x = layers.MaxPooling2D()(x)
  pooled_features = layers.GlobalAveragePooling2D()(x)

  # Branch path
  branch_output = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
  branch_output = layers.MaxPooling2D()(branch_output)
  branch_output = layers.Conv2D(64, (3, 3), activation='relu')(branch_output)
  branch_output = layers.MaxPooling2D()(branch_output)

  # Combine main and branch paths
  combined = layers.add([pooled_features, branch_output])

  # Fully connected layers
  combined = layers.Dense(64, activation='relu')(combined)
  outputs = layers.Dense(10, activation='softmax')(combined)

  model = keras.Model(inputs=inputs, outputs=outputs)

  return model