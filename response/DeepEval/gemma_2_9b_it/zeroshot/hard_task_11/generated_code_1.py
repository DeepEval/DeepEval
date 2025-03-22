import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
  input_tensor = layers.Input(shape=(32, 32, 3))

  # Main pathway
  x = layers.Conv2D(32, 1, activation='relu')(input_tensor)

  # Parallel branch
  branch1 = layers.Conv2D(32, 1, activation='relu')(input_tensor)
  branch2 = layers.Conv2D(32, 1, activation='relu')(input_tensor)
  branch3 = layers.Conv2D(32, 3, activation='relu')(input_tensor)
  branch_concat = layers.concatenate([branch1, branch2, branch3], axis=-1)

  # Concatenate and further process
  x = layers.concatenate([x, branch_concat], axis=-1)
  x = layers.Conv2D(32, 1, activation='relu')(x)

  # Shortcut connection
  x = layers.add([x, input_tensor]) 

  # Classification layers
  x = layers.Flatten()(x)
  x = layers.Dense(128, activation='relu')(x)
  output_tensor = layers.Dense(10, activation='softmax')(x)

  model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
  return model