from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, add

def dl_model():
  # Input layer
  img_input = Input(shape=(28, 28, 1))

  # Branch 1
  branch1 = img_input
  for _ in range(3):
    branch1 = Conv2D(32, (3, 3), activation='relu')(branch1)

  # Branch 2
  branch2 = img_input
  branch2 = Conv2D(32, (5, 5), activation='relu')(branch2)
  branch2 = MaxPooling2D()(branch2)
  for _ in range(2):
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)

  # Branch 3
  branch3 = img_input
  branch3 = Conv2D(32, (7, 7), activation='relu')(branch3)
  branch3 = MaxPooling2D()(branch3)
  for _ in range(2):
    branch3 = Conv2D(64, (5, 5), activation='relu')(branch3)

  # Concatenate branches
  concat_layer = concatenate([branch1, branch2, branch3])

  # Fully connected layer
  flatten_layer = Flatten()(concat_layer)
  dense_layer = Dense(10, activation='softmax')(flatten_layer)

  # Model definition
  model = Model(img_input, dense_layer)

  return model