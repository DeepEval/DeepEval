from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def dl_model():

  inputs = Input(shape=(32, 32, 3))

  # Split the input into three groups along the channel dimension
  x_1 = Lambda(lambda x: K.expand_dims(x[:, :, :, 0], axis=-1))(inputs)
  x_2 = Lambda(lambda x: K.expand_dims(x[:, :, :, 1], axis=-1))(inputs)
  x_3 = Lambda(lambda x: K.expand_dims(x[:, :, :, 2], axis=-1))(inputs)

  # Define the main path
  main_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x_1)
  main_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(main_path)
  main_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(main_path)

  # Define the side path 1
  side_path_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x_2)
  side_path_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(side_path_1)
  side_path_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(side_path_1)

  # Define the side path 2
  side_path_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x_3)
  side_path_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(side_path_2)
  side_path_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(side_path_2)

  # Combine the main and side paths
  main_path = Lambda(lambda x: K.concatenate(x, axis=-1))([main_path, side_path_1, side_path_2])

  # Fuse with the original input
  main_path = Lambda(lambda x: x + inputs)([main_path, inputs])

  # Flatten and classify
  main_path = Flatten()(main_path)
  outputs = Dense(units=10, activation='softmax')(main_path)

  model = Model(inputs=inputs, outputs=outputs)

  return model