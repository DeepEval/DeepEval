from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, concatenate, add, Dense

def dl_model():

  # Input layer
  inputs = Input(shape=(32, 32, 3))

  # Main path
  x = Conv2D(16, (1, 1), padding='same')(inputs)
  branch_1 = Conv2D(16, (3, 3), padding='same')(x)
  branch_2 = AveragePooling2D((2, 2), padding='same')(x)
  branch_2 = Conv2D(16, (3, 3), padding='same')(branch_2)
  branch_3 = AveragePooling2D((4, 4), padding='same')(x)
  branch_3 = Conv2D(16, (3, 3), padding='same')(branch_3)

  # Concatenate branches
  merged = concatenate([branch_1, branch_2, branch_3], axis=-1)

  # Main path output
  main_path_output = Conv2D(16, (1, 1), padding='same')(merged)

  # Branch path
  branch_path_output = Conv2D(16, (1, 1), padding='same')(inputs)
  branch_path_output = Conv2D(16, (3, 3), padding='same')(branch_path_output)

  # Fusion
  fused = add([main_path_output, branch_path_output])

  # Final output
  outputs = Dense(10, activation='softmax')(fused)

  # Create the model
  model = Model(inputs=inputs, outputs=outputs)

  return model