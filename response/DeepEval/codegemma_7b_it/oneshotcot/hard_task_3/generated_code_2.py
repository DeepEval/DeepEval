import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, Flatten, Dense, Concatenate, Lambda

def dl_model():
  # Input layer
  input_tensor = Input(shape=(32, 32, 3))

  # Split input into three groups along channel dimension
  x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_tensor)

  # Main pathway
  main_path = []
  for i in range(3):
    # Extract deep features
    main_path.append(Conv2D(filters=64, kernel_size=(1, 1), padding='valid')(x[i]))
    main_path.append(Activation('relu')(main_path[-1]))
    main_path.append(Conv2D(filters=64, kernel_size=(3, 3), padding='same')(main_path[-1]))
    main_path.append(Activation('relu')(main_path[-1]))

    # Dropout for feature selection
    main_path.append(Dropout(0.25)(main_path[-1]))

  # Concatenate outputs from main pathway
  main_path_output = Concatenate(axis=-1)(main_path)

  # Branch pathway
  branch_path = Conv2D(filters=64, kernel_size=(1, 1), padding='valid')(input_tensor)
  branch_path = Activation('relu')(branch_path)

  # Combine outputs from both pathways
  combined_output = Add()([main_path_output, branch_path])

  # Fully connected layer for classification
  output_layer = Flatten()(combined_output)
  output_layer = Dense(units=10, activation='softmax')(output_layer)

  # Model definition
  model = keras.Model(inputs=input_tensor, outputs=output_layer)

  return model

# Example usage:
model = dl_model()
model.summary()