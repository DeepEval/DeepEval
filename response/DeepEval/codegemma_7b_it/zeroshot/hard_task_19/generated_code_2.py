import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, multiply
from tensorflow.keras.models import Model
from tensorflow.keras import Input

def dl_model():

  # Input layer
  img_input = Input(shape=(32, 32, 3))

  # Main path
  x = Conv2D(64, (3, 3), activation='relu')(img_input)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Conv2D(128, (3, 3), activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Conv2D(256, (3, 3), activation='relu')(x)

  # Branch path
  branch_avg = GlobalAveragePooling2D()(x)
  branch_avg = Dense(256, activation='relu')(branch_avg)
  branch_avg = Dense(256, activation='sigmoid')(branch_avg)
  branch_avg = Reshape((1, 1, 256))(branch_avg)

  # Main and branch outputs
  main_output = Conv2D(256, (1, 1), activation='relu')(x)
  output = multiply([main_output, branch_avg])
  output = Conv2D(256, (1, 1), activation='relu')(output)

  # Classification layer
  output = Dense(10, activation='softmax')(output)

  # Create the model
  model = Model(inputs=img_input, outputs=output)

  return model