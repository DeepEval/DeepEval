import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute

def dl_model():
  # Input layer
  input_layer = Input(shape=(32, 32, 3))

  # Block 1
  block1_output = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
  block1_outputs = []
  for block1_output_ in block1_output:
    block1_outputs.append(Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(block1_output_))
  block1_output = Concatenate(axis=-1)(block1_outputs)

  # Block 2
  block2_output = Reshape((32, 32, 3, 32))(block1_output)
  block2_output = Permute((0, 1, 3, 2))(block2_output)
  block2_output = Reshape((32, 32, 32))(block2_output)

  # Block 3
  block3_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=False)(block2_output)
  block3_output = BatchNormalization()(block3_output)

  # Branch
  branch_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

  # Main path and branch output
  output = keras.layers.add([block3_output, branch_output])

  # Fully connected layer
  output = Flatten()(output)
  output = Dense(units=10, activation='softmax')(output)

  # Model
  model = keras.Model(inputs=input_layer, outputs=output)

  return model