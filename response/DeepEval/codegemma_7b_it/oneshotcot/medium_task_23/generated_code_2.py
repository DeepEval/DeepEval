import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

  input_layer = layers.Input(shape=(32, 32, 3))
  
  # Path 1
  path_1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
  
  # Path 2
  path_2 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
  path_2 = layers.Conv2D(filters=32, kernel_size=(1, 7), activation='relu')(path_2)
  path_2 = layers.Conv2D(filters=32, kernel_size=(7, 1), activation='relu')(path_2)
  
  # Path 3
  path_3 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
  path_3 = layers.Conv2D(filters=32, kernel_size=(1, 7), activation='relu')(path_3)
  path_3 = layers.Conv2D(filters=32, kernel_size=(7, 1), activation='relu')(path_3)
  path_3 = layers.Conv2D(filters=32, kernel_size=(1, 7), activation='relu')(path_3)
  path_3 = layers.Conv2D(filters=32, kernel_size=(7, 1), activation='relu')(path_3)
  
  # Path 4
  path_4 = layers.AveragePooling2D(pool_size=(2, 2))(input_layer)
  path_4 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(path_4)
  
  # Concatenate the paths
  concat_layer = layers.Concatenate()([path_1, path_2, path_3, path_4])
  
  # Batch normalization
  batch_norm_layer = layers.BatchNormalization()(concat_layer)
  
  # Flatten the output
  flatten_layer = layers.Flatten()(batch_norm_layer)
  
  # Fully connected layer
  dense_layer = layers.Dense(units=10, activation='softmax')(flatten_layer)
  
  model = keras.Model(inputs=input_layer, outputs=dense_layer)
  
  return model