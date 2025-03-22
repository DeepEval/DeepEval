from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Conv2DTranspose, Concatenate, Reshape, Permute, Dense
from tensorflow.keras.models import Model

def dl_model():
  # Input layer
  x = Input(shape=(28, 28, 1))

  # Block 1
  primary_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
  primary_path = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(primary_path)
  primary_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(primary_path)

  branch_path = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x)
  branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch_path)

  # Concatenation
  concat_path = Concatenate()([primary_path, branch_path])

  # Block 2
  shape = concat_path.shape
  concat_path = Reshape((shape[1], shape[2], shape[3] * 2))(concat_path)
  concat_path = Permute((1, 2, 3))(concat_path)
  concat_path = Reshape((shape[1], shape[2], shape[3], 2))(concat_path)

  # Output layer
  output = Dense(units=10, activation='softmax')(concat_path)

  # Model
  model = Model(inputs=x, outputs=output)

  return model