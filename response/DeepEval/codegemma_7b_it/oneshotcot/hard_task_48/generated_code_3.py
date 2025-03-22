import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, BatchNormalization, Flatten, Dense, concatenate

def dl_model():

  input_layer = Input(shape=(32, 32, 3))
  
  # Block 1
  block1_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
  block1_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_conv1)
  block1_conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block1_conv1)
  block1_output = concatenate([block1_conv1, block1_conv2, block1_conv3])
  block1_bn = BatchNormalization()(block1_output)

  # Block 2
  path1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_bn)
  path2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(block1_bn)
  path2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
  path3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_bn)
  path3_branch1 = Conv2D(filters=128, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3)
  path3_branch2 = Conv2D(filters=128, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3)
  path3_output = concatenate([path3_branch1, path3_branch2])
  path4 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_bn)
  path4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
  path4_branch1 = Conv2D(filters=128, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4)
  path4_branch2 = Conv2D(filters=128, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4)
  path4_output = concatenate([path4_branch1, path4_branch2])

  block2_output = concatenate([path1, path2, path3_output, path4_output])
  block2_bn = BatchNormalization()(block2_output)

  # Output layer
  flatten_layer = Flatten()(block2_bn)
  output_layer = Dense(units=10, activation='softmax')(flatten_layer)

  model = keras.Model(inputs=input_layer, outputs=output_layer)

  return model