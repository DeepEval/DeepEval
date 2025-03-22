import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
  input_tensor = layers.Input(shape=(28, 28, 1))

  # Block 1: Main path and branch path
  x_main = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
  x_main = layers.Conv2D(16, (3, 3), activation='relu')(x_main)
  x_branch = layers.Conv2D(16, (1, 1), activation='relu')(input_tensor)
  x = layers.Add()([x_main, x_branch])

  # Block 2: Max pooling with varying scales
  x_pool1 = layers.MaxPooling2D((1, 1), strides=(1, 1))(x)
  x_pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
  x_pool4 = layers.MaxPooling2D((4, 4), strides=(4, 4))(x)

  x_flat = layers.Flatten()(x_pool1)
  x_flat = layers.Flatten()(x_pool2)
  x_flat = layers.Flatten()(x_pool4)
  x = layers.concatenate([x_flat, x_flat, x_flat])

  # Fully connected layers
  x = layers.Dense(128, activation='relu')(x)
  output = layers.Dense(10, activation='softmax')(x)

  model = tf.keras.Model(inputs=input_tensor, outputs=output)
  return model