import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Add

def dl_model():
  # Define the input layer
  input_layer = Input(shape=(32, 32, 3))

  # Define the main pathway
  main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
  main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
  main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_path)

  # Define the branch pathway
  branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
  branch_path = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
  branch_path = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)

  # Concatenate the outputs of the main and branch pathways
  concat_path = Concatenate()([main_path, branch_path])

  # Pass the concatenated output through a 1x1 convolution to produce the main output
  main_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_path)

  # Create a direct connection from the input to the branch pathway
  direct_connection = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

  # Add the main output and the direct connection
  fused_output = Add()([main_output, direct_connection])

  # Pass the fused output through two fully connected layers
  flatten_layer = Flatten()(fused_output)
  output_layer = Dense(units=10, activation='softmax')(flatten_layer)

  # Define the model
  model = keras.Model(inputs=input_layer, outputs=output_layer)

  return model