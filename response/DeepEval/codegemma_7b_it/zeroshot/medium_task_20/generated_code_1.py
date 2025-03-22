import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
  # Define the input shape
  input_shape = (32, 32, 3)

  # Create the input layer
  input_layer = layers.Input(shape=input_shape)

  # Define the four parallel convolutional paths
  path1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
  path2 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
  path2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path2)
  path2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path2)
  path3 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
  path3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path3)
  path4 = layers.MaxPooling2D(pool_size=(2, 2))(input_layer)
  path4 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(path4)

  # Concatenate the outputs from the four paths
  concat = layers.concatenate([path1, path2, path3, path4])

  # Flatten and pass through a dense layer
  flatten = layers.Flatten()(concat)
  dense = layers.Dense(units=128, activation='relu')(flatten)

  # Define the output layer
  output_layer = layers.Dense(units=10, activation='softmax')(dense)

  # Create the model
  model = models.Model(inputs=input_layer, outputs=output_layer)

  # Compile the model
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  return model