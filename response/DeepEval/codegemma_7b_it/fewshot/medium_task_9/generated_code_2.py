import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor):
  main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
  main_path = BatchNormalization()(main_path)
  main_path = ReLU()(main_path)
  main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
  main_path = BatchNormalization()(main_path)

  branch_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
  branch_path = BatchNormalization()(branch_path)

  return Add()([main_path, branch_path])

def dl_model():
  input_layer = Input(shape=(32, 32, 3))

  # Initial convolutional layer to reduce dimensionality
  conv_down = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
  conv_down = BatchNormalization()(conv_down)

  # Two basic blocks
  block1 = basic_block(conv_down)
  block2 = basic_block(block1)

  # Feature fusion through addition
  fusion = Add()([block2, conv_down])

  # Average pooling to downsample feature map
  avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2)(fusion)

  # Flatten and fully connected layer for classification
  flatten = Flatten()(avg_pool)
  output_layer = Dense(units=10, activation='softmax')(flatten)

  model = keras.Model(inputs=input_layer, outputs=output_layer)

  return model